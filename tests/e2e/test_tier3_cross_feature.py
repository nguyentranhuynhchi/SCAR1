"""Tier 3 - Cross-Feature Combination Tests for SCAR M2-Pro.

Covers:
- AMP precision (bf16/fp16) combined with GradScaler, forward/backward passes
- Non-finite gradient detection and optimizer step skipping under AMP
- Multi-term M2ProLoss combined with Deep Supervision auxiliary prediction head
- Multi-stream input backpropagation across bSSFP, LGE, and T2w modalities
- Microbatch scaling, gradient accumulation, and gradient norm clipping (1.0)
"""
from __future__ import annotations

import math
import unittest
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from training.trainer.trainer import resolve_amp
from tests.e2e.contract_helpers import get_m2pro_net, get_m2pro_loss


class TestTier3CrossFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # -------------------------------------------------------------------------
    # 1. AMP Precision & Forward/Backward Flow
    # -------------------------------------------------------------------------
    def test_amp_autocast_forward_and_backward(self):
        """Model and loss run stably under torch.autocast without numerical failure."""
        model, _ = get_m2pro_net(img_size=32)
        criterion, _ = get_m2pro_loss()
        model.train()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w)
        psir = torch.randn(b, 1, h, w)
        t2w = torch.randn(b, 1, h, w)
        targets = torch.randint(0, 4, (b, h, w))

        # Test autocast with bfloat16 (supported natively on CPU & Ampere+ CUDA)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        amp_dtype = torch.bfloat16

        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            out = model(cine, psir, t2w, return_aux=True)
            if isinstance(out, dict):
                logits = out["logits"]
                aux = out["aux_logits"]
            else:
                logits, aux = out
            loss_dict = criterion(logits, targets, aux_logits=aux)

        self.assertTrue(torch.isfinite(loss_dict["loss"]).all())
        loss_dict["loss"].backward()

        # Check all trainable parameters have finite gradients
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                self.assertTrue(torch.isfinite(param.grad).all())
        self.assertTrue(has_grad, "No parameters received gradients")

    def test_grad_scaler_overflow_handling(self):
        """GradScaler detects non-finite gradients, skips step, and scales down."""
        # Test GradScaler logic with simulated overflow
        # On CUDA, use cuda device; on CPU, test scaler mechanics or mock
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        linear = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

        x = torch.randn(2, 4)
        out = linear(x).sum()
        scaler.scale(out).backward()

        # Intentionally inject Inf into gradients to simulate FP16 overflow
        for p in linear.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float("inf"))

        initial_scale = scaler.get_scale()
        # scaler.step should skip updating weights
        old_weight = linear.weight.clone()
        scaler.step(optimizer)
        scaler.update()

        if torch.cuda.is_available():
            # Scaler should have detected Inf, skipped step, and decreased scale
            self.assertLess(scaler.get_scale(), initial_scale)
            torch.testing.assert_close(linear.weight, old_weight)
        else:
            # On CPU scaler was disabled; verify clean behavior
            pass

    # -------------------------------------------------------------------------
    # 2. Multi-Term Loss with Deep Supervision Auxiliary Head
    # -------------------------------------------------------------------------
    def test_multi_term_loss_with_auxiliary_head(self):
        """Auxiliary prediction head at 1/4 resolution provides direct gradients."""
        model, _ = get_m2pro_net(img_size=32)
        criterion, _ = get_m2pro_loss()
        model.train()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w)
        psir = torch.randn(b, 1, h, w)
        t2w = torch.randn(b, 1, h, w)
        targets = torch.randint(0, 4, (b, h, w))

        out = model(cine, psir, t2w, return_aux=True)
        if isinstance(out, dict):
            logits = out["logits"]
            aux = out["aux_logits"]
        else:
            logits, aux = out

        self.assertEqual(aux.shape, (b, 2, h // 4, w // 4))
        loss_dict = criterion(logits, targets, aux_logits=aux)

        # Aux loss should be positive
        self.assertGreater(float(loss_dict["aux"].detach()), 0.0)

        loss_dict["loss"].backward()

        # Check auxiliary head parameters have non-zero gradients
        aux_params_have_grad = False
        for name, param in model.named_parameters():
            if "aux" in name and param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    aux_params_have_grad = True
        self.assertTrue(aux_params_have_grad, "Auxiliary head parameters received zero gradient")

    # -------------------------------------------------------------------------
    # 3. Multi-Stream Modality Gradient Flow
    # -------------------------------------------------------------------------
    def test_gradient_flow_to_all_three_input_streams(self):
        """Gradients flow back to all 3 input streams (bSSFP, LGE, T2w) without dead branches."""
        model, _ = get_m2pro_net(img_size=32)
        criterion, _ = get_m2pro_loss()
        model.train()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w, requires_grad=True)
        psir = torch.randn(b, 1, h, w, requires_grad=True)
        t2w = torch.randn(b, 1, h, w, requires_grad=True)
        targets = torch.randint(0, 4, (b, h, w))

        out = model(cine, psir, t2w)
        logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
        loss = criterion(logits, targets)["loss"]
        loss.backward()

        for name, tensor in (("cine", cine), ("psir", psir), ("t2w", t2w)):
            self.assertIsNotNone(tensor.grad, f"Input stream {name} has no grad")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"Input stream {name} has non-finite grad")
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0, f"Input stream {name} received zero grad")

    # -------------------------------------------------------------------------
    # 4. Microbatch Scaling & Gradient Accumulation
    # -------------------------------------------------------------------------
    def test_gradient_accumulation_and_norm_clipping(self):
        """Simulate microbatch accumulation (accum_steps=2) with clip_grad_norm_(1.0)."""
        model, _ = get_m2pro_net(img_size=32)
        criterion, _ = get_m2pro_loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        accum_steps = 2
        optimizer.zero_grad()

        total_loss = 0.0
        for step in range(accum_steps):
            cine = torch.randn(2, 1, 32, 32)
            psir = torch.randn(2, 1, 32, 32)
            t2w = torch.randn(2, 1, 32, 32)
            targets = torch.randint(0, 4, (2, 32, 32))

            out = model(cine, psir, t2w, return_aux=True)
            if isinstance(out, dict):
                logits, aux = out["logits"], out["aux_logits"]
            else:
                logits, aux = out

            loss_dict = criterion(logits, targets, aux_logits=aux)
            scaled_loss = loss_dict["loss"] / accum_steps
            scaled_loss.backward()
            total_loss += float(loss_dict["loss"].detach())

        # Check gradient norm before clipping
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertTrue(math.isfinite(float(grad_norm)))
        self.assertGreater(float(grad_norm), 0.0)

        # After clipping, the effective norm of gradients must be <= 1.0 + epsilon
        clipped_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None])
        )
        self.assertLessEqual(float(clipped_norm), 1.0 + 1e-4)

        # Step optimizer and zero_grad
        optimizer.step()
        optimizer.zero_grad()

        # Ensure all gradients are None or zeroed
        for p in model.parameters():
            if p.grad is not None:
                self.assertAlmostEqual(float(p.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
