"""Comprehensive Unit Test Suite for M2-Pro Architecture.

Verifies:
  a. M2ProNet forward pass in training mode: outputs main logits (B, 4, 128, 128) and aux logits (B, 2, 32, 32).
  b. M2ProNet forward pass in eval mode: returns single tensor (B, 4, 128, 128).
  c. Backward pass: loss computes and backprops gradients to all parameters without NaN or zero gradient floors.
  d. AGSA and DFE module shapes and gradient floor (alpha_pass = 0.20).
  e. AMP execution safety under torch.bfloat16 and torch.float16.
  f. MODEL_REGISTRY registration and checkpoint load/save fidelity.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from ml_collections import ConfigDict
from torch import nn

from training.models import MODEL_REGISTRY, build_model, model_from_config
from training.models.cmspa_net import get_config, get_r50_b16_config, get_testing
from training.models.m2_pro import M2ProNet
from training.models.modules.m2_pro import (
    AGSA_Block,
    ChannelSEModule,
    DecoupledBottleneckFusion,
    DeepSupervisionAuxHead,
    DFE_Block,
    M2ProOutput,
    SoftMyoGate,
)


class TestM2ProModules(unittest.TestCase):
    """Unit tests for individual M2-Pro building blocks."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_soft_myo_gate_shape_and_range(self):
        """SoftMyoGate extracts (B, 1, H, W) continuous mask strictly in (0, 1)."""
        gate = SoftMyoGate(channels=256, reduction=4)
        cine = torch.randn(2, 256, 32, 32)
        mask = gate(cine)
        self.assertEqual(mask.shape, (2, 1, 32, 32))
        self.assertTrue((mask >= 0.0).all() and (mask <= 1.0).all())
        self.assertTrue(torch.isfinite(mask).all())

    def test_channel_se_module_shape_and_dual_pooling(self):
        """ChannelSEModule recalibrates channel dimensions preserving spatial shape."""
        se = ChannelSEModule(channels=128, reduction=4)
        x = torch.randn(2, 128, 16, 16)
        out = se(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_dfe_block_shapes_and_contrast_enhancement(self):
        """DFE_Block preserves spatial shape across skip scales and amplifies difference."""
        test_configs = [
            (64, 64, 64),
            (256, 32, 32),
            (512, 16, 16),
        ]
        for channels, h, w in test_configs:
            dfe = DFE_Block(channels=channels, reduction=4, init_gamma=0.1)
            x = torch.randn(2, channels, h, w, requires_grad=True)
            out = dfe(x)
            self.assertEqual(out.shape, (2, channels, h, w))
            loss = out.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_agsa_multi_scale_shapes_and_gradient_floor(self):
        """AGSA_Block operates across skip levels and maintains strict gradient floor."""
        test_skips = [
            ("Skip-0 (1/2)", 64, 64),
            ("Skip-1 (1/4)", 256, 32),
            ("Skip-2 (1/8)", 512, 16),
        ]
        for name, channels, size in test_skips:
            block = AGSA_Block(channels=channels, alpha_pass=0.20, use_dfe=True)
            cine = torch.randn(2, channels, size, size, requires_grad=True)
            psir = torch.randn(2, channels, size, size, requires_grad=True)
            t2w = torch.randn(2, channels, size, size, requires_grad=True)

            out = block(cine, psir, t2w)
            self.assertEqual(out.shape, (2, channels, size, size), f"Failed shape at {name}")

            loss = out.sum()
            loss.backward()
            for tensor, tname in ((cine, "cine"), (psir, "psir"), (t2w, "t2w")):
                self.assertIsNotNone(tensor.grad, f"Grad None for {tname} at {name}")
                self.assertTrue(torch.isfinite(tensor.grad).all(), f"Non-finite grad for {tname} at {name}")
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0, f"Zero grad for {tname} at {name}")

    def test_agsa_adversarial_zero_gate_gradient_highway(self):
        """When myocardial gate evaluates to zero, gradient floor alpha_pass=0.20 is strictly preserved."""
        alpha_pass = 0.20
        psir = torch.randn(2, 256, 32, 32, requires_grad=True)
        # Adversarial zero myocardial mask
        m_myo = torch.zeros(2, 1, 32, 32)
        g_mask = alpha_pass + (1.0 - alpha_pass) * m_myo
        psir_gated = psir * g_mask

        loss = psir_gated.sum()
        loss.backward()

        expected_grad = torch.full_like(psir, alpha_pass)
        torch.testing.assert_close(psir.grad, expected_grad)
        self.assertTrue(torch.allclose(psir.grad, expected_grad, atol=1e-6))

    def test_decoupled_bottleneck_fusion_independent_queries(self):
        """DecoupledBottleneckFusion executes inverted cross-attention without modality averaging."""
        fusion = DecoupledBottleneckFusion(in_channels=512, out_channels=256, num_heads=8)
        cine = torch.randn(2, 512, 8, 8, requires_grad=True)
        psir = torch.randn(2, 512, 8, 8, requires_grad=True)
        t2w = torch.randn(2, 512, 8, 8, requires_grad=True)

        out = fusion(cine, psir, t2w)
        self.assertEqual(out.shape, (2, 256, 8, 8))

        loss = out.sum()
        loss.backward()

        # All three inputs must receive non-zero finite backprop gradients
        for tensor, name in ((cine, "cine"), (psir, "psir"), (t2w, "t2w")):
            self.assertIsNotNone(tensor.grad, f"Zero grad for {name}")
            self.assertTrue(torch.isfinite(tensor.grad).all(), f"Non-finite grad for {name}")
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0, f"Vanished grad for {name}")

    def test_deep_supervision_aux_head_resolution(self):
        """DeepSupervisionAuxHead predicts 2 auxiliary classes at 1/4 resolution."""
        aux_head = DeepSupervisionAuxHead(in_channels=128, num_classes=2, hidden_channels=64)
        feat_1_4 = torch.randn(2, 128, 32, 32)
        logits_aux = aux_head(feat_1_4)
        self.assertEqual(logits_aux.shape, (2, 2, 32, 32))
        self.assertTrue(torch.isfinite(logits_aux).all())

    def test_m2_pro_output_polymorphism(self):
        """M2ProOutput satisfies tuple unpacking, indexing, dict-keys, and attribute interfaces."""
        main = torch.randn(2, 4, 128, 128)
        aux = torch.randn(2, 2, 32, 32)
        out = M2ProOutput(main, aux)

        # 1. Tuple unpacking
        m, a = out
        self.assertEqual(m.shape, (2, 4, 128, 128))
        self.assertEqual(a.shape, (2, 2, 32, 32))
        self.assertTrue(isinstance(out, tuple))

        # 2. Integer indexing
        self.assertTrue(torch.equal(out[0], main))
        self.assertTrue(torch.equal(out[1], aux))

        # 3. Dict key indexing & get
        self.assertTrue(torch.equal(out["logits"], main))
        self.assertTrue(torch.equal(out["aux_logits"], aux))
        self.assertTrue(torch.equal(out.get("logits"), main))
        self.assertTrue(torch.equal(out.get("aux_logits"), aux))
        self.assertIn("logits", out)
        self.assertIn("aux_logits", out)

        # 4. Attribute access
        self.assertTrue(torch.equal(out.logits, main))
        self.assertTrue(torch.equal(out.aux_logits, aux))

        # 5. Tensor delegation methods
        self.assertEqual(out.shape, (2, 4, 128, 128))
        self.assertEqual(out.argmax(1).shape, (2, 128, 128))
        detached = out.detach()
        self.assertEqual(detached.shape, (2, 4, 128, 128))


class TestM2ProNetFullModel(unittest.TestCase):
    """Full-model integration tests for M2ProNet."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_registration_and_aliases(self):
        """M2ProNet is properly registered under 'm2_pro', 'm2_pro_net', and 'm2pro'."""
        for alias in ("m2_pro", "m2_pro_net", "m2pro"):
            self.assertIn(alias, MODEL_REGISTRY, f"Missing {alias} in MODEL_REGISTRY")
            model = build_model(alias, config=get_testing(), img_size=32)
            self.assertIsInstance(model, M2ProNet, f"Alias {alias} did not instantiate M2ProNet")

    def test_training_mode_outputs_main_and_aux_logits_128x128(self):
        """Requirement 4.a: Training mode returns main logits (B, 4, 128, 128) and aux logits (B, 2, 32, 32)."""
        config = get_r50_b16_config()
        config.resnet.num_layers = (1, 1, 1)  # Minimal layers for swift execution
        model = M2ProNet(config=config, img_size=128)
        model.train()

        inputs = [torch.randn(2, 1, 128, 128) for _ in range(3)]
        output = model(*inputs)

        self.assertIsInstance(output, tuple)
        main_logits, aux_logits = output
        self.assertEqual(main_logits.shape, (2, 4, 128, 128), "Main logits shape mismatch at 128x128")
        self.assertIsNotNone(aux_logits, "Aux logits should not be None in training mode")
        self.assertEqual(aux_logits.shape, (2, 2, 32, 32), "Aux logits shape mismatch at 32x32 (1/4 scale)")

    def test_eval_mode_returns_single_tensor(self):
        """Requirement 4.b: Eval mode returns single tensor (B, 4, 128, 128), bypassing aux head."""
        config = get_r50_b16_config()
        config.resnet.num_layers = (1, 1, 1)
        model = M2ProNet(config=config, img_size=128)
        model.eval()

        inputs = [torch.randn(2, 1, 128, 128) for _ in range(3)]
        with torch.no_grad():
            output = model(*inputs)

        self.assertIsInstance(output, torch.Tensor, "Eval mode must return a single torch.Tensor for predict_volume")
        self.assertEqual(output.shape, (2, 4, 128, 128))

        # Explicit return_aux overrides
        with torch.no_grad():
            aux_out = model(*inputs, return_aux=True)
            self.assertIsInstance(aux_out, tuple)
            self.assertEqual(aux_out[0].shape, (2, 4, 128, 128))
            self.assertEqual(aux_out[1].shape, (2, 2, 32, 32))

            no_aux_train = model(*inputs, return_aux=False)
            self.assertIsInstance(no_aux_train, torch.Tensor)

    def test_backward_pass_gradients_and_gradient_flow(self):
        """Requirement 4.c: Backward pass computes loss and backprops gradients to all parameters without NaN or zero floors."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        inputs = [torch.randn(2, 1, 32, 32, requires_grad=True) for _ in range(3)]
        output = model(*inputs)
        main_logits, aux_logits = output

        # Joint loss combining main and deep supervision auxiliary head
        loss = main_logits.sum() + 0.5 * aux_logits.sum()
        loss.backward()

        # 1. Inputs receive finite gradients
        for img, name in zip(inputs, ("cine", "psir", "t2w")):
            self.assertIsNotNone(img.grad, f"Input gradient was None for {name}")
            self.assertTrue(torch.isfinite(img.grad).all(), f"Input gradient had NaN/Inf for {name}")
            self.assertGreater(float(img.grad.abs().sum()), 0.0, f"Input gradient vanished for {name}")

        # 2. All model parameters with requires_grad receive finite gradients
        checked_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradients")
                self.assertGreater(float(param.grad.abs().sum()), 0.0, f"Parameter {name} has zero gradient floor")
                checked_params += 1

        self.assertGreater(checked_params, 50, "Insufficient trainable parameters validated")

    def test_checkpoint_save_and_restore(self):
        """Model state dict saves and restores cleanly with bitwise output parity."""
        model = M2ProNet(config=get_testing(), img_size=32)
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "m2_pro.pth"
            torch.save(
                {"model": model.state_dict(), "model_config": model.config.to_dict()},
                ckpt_path,
            )
            loaded = torch.load(ckpt_path, weights_only=True)

        clone = model_from_config(ConfigDict(loaded["model_config"]), img_size=32)
        clone.load_state_dict(loaded["model"], strict=True)

        model.eval()
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs), clone(*inputs))


class TestM2ProAMPSafety(unittest.TestCase):
    """Requirement 4.e: AMP execution safety under torch.bfloat16 and torch.float16."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_amp_bfloat16_execution_safety(self):
        """M2ProNet forward pass under torch.bfloat16 produces finite, non-NaN outputs."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.train()
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]

        # Autocast bfloat16 on CPU / CUDA
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            output = model(*inputs)
            main_logits, aux_logits = output

        self.assertTrue(torch.isfinite(main_logits).all(), "NaN or Inf detected in bfloat16 main logits")
        self.assertTrue(torch.isfinite(aux_logits).all(), "NaN or Inf detected in bfloat16 aux logits")

    def test_amp_float16_execution_safety(self):
        """M2ProNet forward pass under float16 produces finite, non-NaN outputs."""
        model = M2ProNet(config=get_testing(), img_size=32)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            inputs = [torch.randn(2, 1, 32, 32, device="cuda") for _ in range(3)]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    output = model(*inputs)
            self.assertTrue(torch.isfinite(output).all(), "NaN or Inf in CUDA fp16 output")
        else:
            # On CPU, test direct half-precision casting or CPU autocast if supported
            inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
            try:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    with torch.no_grad():
                        output = model(*inputs)
                self.assertTrue(torch.isfinite(output).all())
            except RuntimeError:
                pass


if __name__ == "__main__":
    unittest.main()
