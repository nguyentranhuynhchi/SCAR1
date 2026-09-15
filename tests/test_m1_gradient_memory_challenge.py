"""Empirical Verification & Stress Test Suite: Gradient Flow, Memory Leakage, and Eval Determinism.

Milestone M1 Empirical Challenger Suite for M2-Pro Architecture.
Tests:
  1. Simultaneous non-zero gradient propagation to all 3 encoders (encoder_cine, encoder_psir, encoder_t2w)
     under multi-task loss computation (main segmentation loss + deep supervision aux loss).
  2. Gradient backpropagation specifically to DeepSupervisionAuxHead parameters and high-res skip connections (AGSA).
  3. Memory usage measurement across 10 consecutive training steps to verify zero memory leakage.
  4. Eval mode determinism: multiple evaluations with identical inputs produce identical outputs (torch.allclose).
  5. Adversarial input stress tests: all-zero inputs, extreme values, batch size 1.
"""
from __future__ import annotations

import gc
import math
import sys
import unittest
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.models import build_model
from training.models.cmspa_net import get_r50_b16_config, get_testing
from training.models.m2_pro import M2ProNet
from training.models.modules.m2_pro import AGSA_Block, DeepSupervisionAuxHead


class TestM2ProGradientFlowEncoders(unittest.TestCase):
    """Verify simultaneous gradient propagation across all three encoders."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_multi_task_gradient_flow_all_three_encoders(self):
        """Simultaneous non-zero gradient propagation to encoder_cine, encoder_psir, encoder_t2w."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        batch_size = 2
        cine = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
        psir = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
        t2w = torch.randn(batch_size, 1, 32, 32, requires_grad=True)

        target_main = torch.randint(0, 4, (batch_size, 32, 32))
        target_aux = torch.randint(0, 2, (batch_size, 8, 8))

        output = model(cine, psir, t2w)
        self.assertIsInstance(output, tuple)
        main_logits, aux_logits = output

        self.assertEqual(main_logits.shape, (batch_size, 4, 32, 32))
        self.assertEqual(aux_logits.shape, (batch_size, 2, 8, 8))

        # Multi-task loss: Cross-Entropy for main (4 classes) + Cross-Entropy for aux (2 classes)
        loss_main = F.cross_entropy(main_logits, target_main)
        loss_aux = F.cross_entropy(aux_logits, target_aux)
        total_loss = loss_main + 0.5 * loss_aux

        total_loss.backward()

        # Verify input tensor gradients
        for img, name in [(cine, "cine"), (psir, "psir"), (t2w, "t2w")]:
            self.assertIsNotNone(img.grad, f"Gradient for input {name} is None")
            self.assertTrue(torch.isfinite(img.grad).all(), f"Gradient for input {name} contains NaN/Inf")
            grad_norm = float(img.grad.norm().item())
            self.assertGreater(grad_norm, 0.0, f"Gradient norm for input {name} vanished (0.0)")

        # Verify encoder_cine, encoder_psir, encoder_t2w
        encoders = [
            ("encoder_cine", model.encoder_cine),
            ("encoder_psir", model.encoder_psir),
            ("encoder_t2w", model.encoder_t2w),
        ]

        for enc_name, encoder in encoders:
            param_count = 0
            grad_sq_sum = 0.0
            for param_name, param in encoder.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(
                        param.grad,
                        f"Encoder {enc_name} parameter {param_name} has None gradient",
                    )
                    self.assertTrue(
                        torch.isfinite(param.grad).all(),
                        f"Encoder {enc_name} parameter {param_name} has NaN/Inf gradient",
                    )
                    norm_val = float(param.grad.abs().sum().item())
                    self.assertGreater(
                        norm_val,
                        0.0,
                        f"Encoder {enc_name} parameter {param_name} has exact zero gradient",
                    )
                    grad_sq_sum += float((param.grad ** 2).sum().item())
                    param_count += 1

            self.assertGreater(param_count, 10, f"Insufficient parameters checked in {enc_name}")
            total_enc_norm = math.sqrt(grad_sq_sum)
            self.assertGreater(total_enc_norm, 1e-6, f"Vanishing total norm for {enc_name}")

        # Also verify the higher-level transformer wrappers
        for trans_name, trans in [
            ("transformer1", model.transformer1),
            ("transformer2", model.transformer2),
            ("transformer3", model.transformer3),
        ]:
            t_params = [p for p in trans.parameters() if p.requires_grad]
            self.assertTrue(all(p.grad is not None for p in t_params))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in t_params))

    def test_multi_task_gradient_flow_128x128_production_scale(self):
        """Simultaneous gradient propagation at 128x128 input resolution."""
        config = get_r50_b16_config()
        config.resnet.num_layers = (1, 1, 1)  # lightweight for test performance
        model = M2ProNet(config=config, img_size=128)
        model.train()

        cine = torch.randn(1, 1, 128, 128, requires_grad=True)
        psir = torch.randn(1, 1, 128, 128, requires_grad=True)
        t2w = torch.randn(1, 1, 128, 128, requires_grad=True)

        target_main = torch.randint(0, 4, (1, 128, 128))
        target_aux = torch.randint(0, 2, (1, 32, 32))

        output = model(cine, psir, t2w)
        main_logits, aux_logits = output

        self.assertEqual(main_logits.shape, (1, 4, 128, 128))
        self.assertEqual(aux_logits.shape, (1, 2, 32, 32))

        loss = F.cross_entropy(main_logits, target_main) + 0.5 * F.cross_entropy(aux_logits, target_aux)
        loss.backward()

        for enc_name, encoder in [
            ("encoder_cine", model.encoder_cine),
            ("encoder_psir", model.encoder_psir),
            ("encoder_t2w", model.encoder_t2w),
        ]:
            enc_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in enc_grads))
            total_norm = math.sqrt(sum(float((g ** 2).sum().item()) for g in enc_grads))
            self.assertGreater(total_norm, 1e-6, f"{enc_name} total norm vanished at 128x128")


class TestM2ProAuxHeadAndSkipGradients(unittest.TestCase):
    """Verify gradient backpropagation to DeepSupervisionAuxHead and high-res skips."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradient_backprop_to_aux_head_parameters(self):
        """All DeepSupervisionAuxHead parameters receive non-zero finite gradients."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target_main = torch.randint(0, 4, (2, 32, 32))
        target_aux = torch.randint(0, 2, (2, 8, 8))

        main_logits, aux_logits = model(*inputs)
        loss = F.cross_entropy(main_logits, target_main) + 0.5 * F.cross_entropy(aux_logits, target_aux)
        loss.backward()

        self.assertIsNotNone(model.aux_head, "model.aux_head is None")
        self.assertIsInstance(model.aux_head, DeepSupervisionAuxHead)

        aux_params_checked = 0
        for name, param in model.aux_head.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"AuxHead parameter {name} has None gradient")
                self.assertTrue(torch.isfinite(param.grad).all(), f"AuxHead parameter {name} has NaN/Inf gradient")
                self.assertGreater(
                    float(param.grad.abs().sum().item()),
                    0.0,
                    f"AuxHead parameter {name} has zero gradient",
                )
                aux_params_checked += 1

        self.assertGreater(aux_params_checked, 0, "No trainable parameters found in aux_head")

    def test_gradient_backprop_to_high_res_skip_connections(self):
        """All AGSA skip connection modules (gate, dfe, proj) receive non-zero gradients."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target_main = torch.randint(0, 4, (2, 32, 32))
        target_aux = torch.randint(0, 2, (2, 8, 8))

        main_logits, aux_logits = model(*inputs)
        loss = F.cross_entropy(main_logits, target_main) + 0.5 * F.cross_entropy(aux_logits, target_aux)
        loss.backward()

        self.assertEqual(len(model.feature_fusion), config.n_skip)

        for i, fusion in enumerate(model.feature_fusion):
            self.assertIsInstance(fusion, AGSA_Block, f"Skip fusion {i} is not AGSA_Block")

            # 1. SoftMyoGate parameters
            for name, p in fusion.gate.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f"AGSA[{i}].gate.{name} grad is None")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"AGSA[{i}].gate.{name} grad NaN/Inf")
                    self.assertGreater(float(p.grad.abs().sum().item()), 0.0, f"AGSA[{i}].gate.{name} grad is 0")

            # 2. DFE_Block parameters and learnable gamma_dfe
            if fusion.dfe is not None:
                self.assertIsNotNone(fusion.dfe.gamma_dfe.grad, f"AGSA[{i}].dfe.gamma_dfe grad is None")
                self.assertTrue(torch.isfinite(fusion.dfe.gamma_dfe.grad).all())
                self.assertGreater(float(fusion.dfe.gamma_dfe.grad.abs().item()), 0.0, f"AGSA[{i}].gamma_dfe grad is 0")

                for name, p in fusion.dfe.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad, f"AGSA[{i}].dfe.{name} grad is None")
                        self.assertTrue(torch.isfinite(p.grad).all(), f"AGSA[{i}].dfe.{name} grad NaN/Inf")
                        self.assertGreater(float(p.grad.abs().sum().item()), 0.0, f"AGSA[{i}].dfe.{name} grad is 0")

            # 3. Multimodal projection parameters
            for name, p in fusion.proj.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f"AGSA[{i}].proj.{name} grad is None")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"AGSA[{i}].proj.{name} grad NaN/Inf")
                    self.assertGreater(float(p.grad.abs().sum().item()), 0.0, f"AGSA[{i}].proj.{name} grad is 0")

    def test_aux_loss_alone_backpropagates_specifically_to_attached_stages(self):
        """Aux-loss alone propagates gradients to aux_head, Decoder Stage 1, and attached skips,
        while main segmentation head remains with zero gradients (structural isolation check).
        """
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target_aux = torch.randint(0, 2, (2, 8, 8))

        main_logits, aux_logits = model(*inputs)
        # Only backpropagate auxiliary loss
        aux_loss = F.cross_entropy(aux_logits, target_aux)
        aux_loss.backward()

        # 1. aux_head receives gradients
        aux_norm = sum(float(p.grad.abs().sum().item()) for p in model.aux_head.parameters() if p.grad is not None)
        self.assertGreater(aux_norm, 0.0, "Aux head received zero gradients from aux_loss")

        # 2. Main segmentation head receives NO gradient from aux_loss alone
        for name, p in model.segmentation_head.named_parameters():
            if p.requires_grad:
                grad_sum = float(p.grad.abs().sum().item()) if p.grad is not None else 0.0
                self.assertEqual(
                    grad_sum,
                    0.0,
                    f"Segmentation head parameter {name} unexpectedly received gradient from isolated aux_loss",
                )

        # 3. feature_fusion[1] (attached to Decoder Stage 1) receives gradients
        skip1_norm = sum(
            float(p.grad.abs().sum().item())
            for p in model.feature_fusion[1].parameters()
            if p.grad is not None
        )
        self.assertGreater(skip1_norm, 0.0, "Skip-1 AGSA did not receive gradient from aux_loss")


class TestM2ProMemoryLeakage(unittest.TestCase):
    """Verify zero memory leakage across consecutive training steps."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ten_consecutive_training_steps_memory_stability(self):
        """Measure RSS memory and PyTorch tensors over 10 consecutive train steps to confirm zero leakage."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        process = psutil.Process()

        # Warmup step to allocate static optimizer states and internal buffers
        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 32, 32)
        t2w = torch.randn(2, 1, 32, 32)
        target_main = torch.randint(0, 4, (2, 32, 32))
        target_aux = torch.randint(0, 2, (2, 8, 8))

        optimizer.zero_grad(set_to_none=True)
        m, a = model(cine, psir, t2w)
        loss = F.cross_entropy(m, target_main) + 0.5 * F.cross_entropy(a, target_aux)
        loss.backward()
        optimizer.step()
        del m, a, loss
        gc.collect()

        import tracemalloc
        tracemalloc.start()

        rss_measurements = []
        trace_measurements = []
        step_count = 10

        for step in range(1, step_count + 1):
            optimizer.zero_grad(set_to_none=True)

            cine = torch.randn(2, 1, 32, 32)
            psir = torch.randn(2, 1, 32, 32)
            t2w = torch.randn(2, 1, 32, 32)
            target_main = torch.randint(0, 4, (2, 32, 32))
            target_aux = torch.randint(0, 2, (2, 8, 8))

            out = model(cine, psir, t2w)
            main_logits, aux_logits = out

            loss = F.cross_entropy(main_logits, target_main) + 0.5 * F.cross_entropy(aux_logits, target_aux)
            loss.backward()
            optimizer.step()

            # Clean up step references
            del cine, psir, t2w, target_main, target_aux, main_logits, aux_logits, out, loss
            gc.collect()

            current_rss_mb = process.memory_info().rss / (1024 * 1024)
            current_trace_mb = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
            rss_measurements.append(current_rss_mb)
            trace_measurements.append(current_trace_mb)

        tracemalloc.stop()

        # 1. Exact Python heap memory stability check (tracemalloc)
        # Verify heap memory does not accumulate across training steps
        trace_delta = trace_measurements[-1] - trace_measurements[1]
        self.assertLess(
            trace_delta,
            0.5,
            f"Python heap memory leak detected: heap grew by {trace_delta:.4f} MB across steps",
        )

        # 2. Process RSS working set stability check (allowing for Windows OS paging fluctuations)
        rss_start = rss_measurements[1]
        rss_end = rss_measurements[-1]
        rss_delta = rss_end - rss_start
        self.assertLess(
            rss_delta,
            20.0,
            f"Process RSS runaway memory leak detected: RSS grew by {rss_delta:.2f} MB",
        )


class TestM2ProEvalDeterminism(unittest.TestCase):
    """Verify eval mode determinism: identical inputs produce strictly identical outputs."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_eval_mode_determinism_across_multiple_runs(self):
        """Multiple evaluations with identical inputs produce identical outputs (torch.allclose)."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.eval()

        torch.manual_seed(42)
        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 32, 32)
        t2w = torch.randn(2, 1, 32, 32)

        # Baseline evaluation
        with torch.no_grad():
            out_baseline = model(cine, psir, t2w)

        self.assertIsInstance(out_baseline, torch.Tensor, "Eval mode must return a single torch.Tensor")
        self.assertEqual(out_baseline.shape, (2, 4, 32, 32))

        # Perform 5 repeat evaluations
        for run_idx in range(1, 6):
            with torch.no_grad():
                out_repeat = model(cine, psir, t2w)

            # Check exact parity
            diff = (out_repeat - out_baseline).abs().max().item()
            self.assertEqual(
                diff,
                0.0,
                f"Eval determinism violation at run {run_idx}: max absolute difference {diff} > 0.0",
            )
            self.assertTrue(
                torch.equal(out_baseline, out_repeat),
                f"Eval determinism violation at run {run_idx}: torch.equal failed",
            )

    def test_eval_mode_does_not_mutate_batchnorm_running_stats(self):
        """In eval mode, BatchNorm running_mean and running_var remain strictly unchanged."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.eval()

        # Capture initial state of all BatchNorm running statistics
        bn_stats_before = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                bn_stats_before[f"{name}.running_mean"] = module.running_mean.clone()
                bn_stats_before[f"{name}.running_var"] = module.running_var.clone()
                bn_stats_before[f"{name}.num_batches_tracked"] = module.num_batches_tracked.clone()

        self.assertGreater(len(bn_stats_before), 0, "No BatchNorm modules identified")

        # Run 5 forward passes in eval mode
        with torch.no_grad():
            for _ in range(5):
                _ = model(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))

        # Verify zero drift in running statistics
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                mean_diff = (module.running_mean - bn_stats_before[f"{name}.running_mean"]).abs().max().item()
                var_diff = (module.running_var - bn_stats_before[f"{name}.running_var"]).abs().max().item()
                tracked_diff = (module.num_batches_tracked - bn_stats_before[f"{name}.num_batches_tracked"]).abs().item()

                self.assertEqual(mean_diff, 0.0, f"BatchNorm {name} running_mean drifted by {mean_diff} in eval mode")
                self.assertEqual(var_diff, 0.0, f"BatchNorm {name} running_var drifted by {var_diff} in eval mode")
                self.assertEqual(tracked_diff, 0, f"BatchNorm {name} num_batches_tracked incremented in eval mode")


class TestM2ProAdversarialCornerCases(unittest.TestCase):
    """Adversarial stress-testing of input boundaries."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_zero_inputs_gradient_flow_and_numerical_stability(self):
        """All-zero inputs do not cause division by zero or NaNs; small non-zero inputs propagate non-zero gradients."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)
        model.train()

        cine = torch.zeros(2, 1, 32, 32, requires_grad=True)
        psir = torch.zeros(2, 1, 32, 32, requires_grad=True)
        t2w = torch.zeros(2, 1, 32, 32, requires_grad=True)

        target_main = torch.randint(0, 4, (2, 32, 32))
        target_aux = torch.randint(0, 2, (2, 8, 8))

        main_logits, aux_logits = model(cine, psir, t2w)
        self.assertTrue(torch.isfinite(main_logits).all(), "NaNs in main logits with zero inputs")
        self.assertTrue(torch.isfinite(aux_logits).all(), "NaNs in aux logits with zero inputs")

        loss = F.cross_entropy(main_logits, target_main) + 0.5 * F.cross_entropy(aux_logits, target_aux)
        loss.backward()

        self.assertTrue(torch.isfinite(cine.grad).all(), "Non-finite values in cine.grad with zero inputs")
        self.assertTrue(torch.isfinite(psir.grad).all(), "Non-finite values in psir.grad with zero inputs")
        self.assertTrue(torch.isfinite(t2w.grad).all(), "Non-finite values in t2w.grad with zero inputs")

        # Now test with small non-zero inputs (1e-2) to ensure gradient highway keeps gradients active
        model.zero_grad()
        cine_s = torch.full((2, 1, 32, 32), 0.05, requires_grad=True)
        psir_s = torch.full((2, 1, 32, 32), 0.05, requires_grad=True)
        t2w_s = torch.full((2, 1, 32, 32), 0.05, requires_grad=True)

        out_s = model(cine_s, psir_s, t2w_s)
        loss_s = F.cross_entropy(out_s[0], target_main) + 0.5 * F.cross_entropy(out_s[1], target_aux)
        loss_s.backward()

        self.assertGreater(float(cine_s.grad.abs().sum().item()), 0.0, "Vanished cine grad for small inputs")
        self.assertGreater(float(psir_s.grad.abs().sum().item()), 0.0, "Vanished psir grad for small inputs")
        self.assertGreater(float(t2w_s.grad.abs().sum().item()), 0.0, "Vanished t2w grad for small inputs")

    def test_batch_size_one_training_and_eval(self):
        """Batch size 1 executes without error in both training and eval modes."""
        config = get_testing()
        model = M2ProNet(config=config, img_size=32)

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        self.assertEqual(out_eval.shape, (1, 4, 32, 32))

        # Training mode with batch size 1 (Note: BatchNorm with B=1 requires H, W > 1 which holds for 32x32)
        model.train()
        out_train = model(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        self.assertEqual(out_train[0].shape, (1, 4, 32, 32))
        self.assertEqual(out_train[1].shape, (1, 2, 8, 8))


if __name__ == "__main__":
    unittest.main()
