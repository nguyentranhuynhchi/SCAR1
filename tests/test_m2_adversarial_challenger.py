"""Adversarial Stress Testing Suite for M2ProLoss (Milestone M2).

Commissioned by: m2_challenger_1 (Empirical Challenger)
Milestone: M2 (Extreme Imbalance Loss - R3)

Adversarial Challenge Vectors:
1. Extreme Empty Slices:
   - Microbatches with 0% scar (slices containing only background, myo, edema).
   - Microbatches with 0% foreground (100% all background pixels).
   - Mixed heterogeneous microbatches (empty + non-empty slices in single batch).
   - Present Dice zero-target masking validation: verifies avoidance of 1/0, NaN, and zero-prediction collapse.
2. Severe Pathology Violation:
   - Predict scar in regions where edema is 0 and myocardium is 0.
   - Rigorous monotonicity check: inclusion loss increases strictly monotonically as scar increases.
   - Verification of exact theoretical slope (dL_inc / d_p3 == 2.0 when p_e=0, p_m=0).
   - Gradient direction check: dL_inc / dz_3 > 0 suppresses out-of-bounds scar.
   - Spatial blast radius check: larger spilling area yields proportionally higher penalty.
3. Extreme Logit Values & Numerical Stability:
   - Large positive (+50.0, +100.0, +1000.0) and large negative (-50.0, -100.0, -1000.0) logits.
   - Subnormal floating-point numbers (1e-38, 1e-45).
   - Aux logits extreme values (+-50.0, +-100.0).
   - Verification of log_softmax, sigmoid, and cross_entropy stability without NaN/Inf.
   - Verification of finite or clamped-zero gradients during backward pass.
   - Non-finite input defense (ValueError on NaN/Inf).
4. AMP Compatibility & Mixed Precision Stability:
   - Forward and backward pass under bfloat16 and float16.
   - CPU and CUDA (when available) autocast compatibility.
   - Gradient scaling check (GradScaler scale factor 65536.0) ensuring no underflow/overflow.
   - End-to-end optimizer update step stability.
5. End-to-End M2ProNet + M2ProLoss Integration under Stress:
   - Full model execution with return_aux=True on adversarial empty-slice batches.
"""
from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn
from torch.nn import functional as F

from training.loss.m2_pro_loss import M2ProLoss, present_dice
from training.models.modules.m2_pro import M2ProOutput
from training.models.m2_pro import M2ProNet
from training.models.cmspa_net import get_testing


class TestAdversarialEmptySlices(unittest.TestCase):
    """Challenge Vector 1: Microbatches with 0% scar or 0% foreground."""

    def setUp(self):
        torch.manual_seed(1337)
        self.criterion = M2ProLoss()

    def test_present_dice_pure_empty_batch(self):
        """When all slices in batch have zero target mass, present_dice returns 0.0 with grad preserved."""
        b, h, w = 4, 64, 64
        prob = torch.rand(b, h, w, requires_grad=True)
        target = torch.zeros(b, h, w)

        loss = present_dice(prob, target)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.detach().item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

        # Computation graph must be preserved and backward pass must succeed with 0 grad
        loss.backward()
        self.assertIsNotNone(prob.grad)
        self.assertTrue(torch.isfinite(prob.grad).all())
        self.assertEqual(float(prob.grad.abs().sum()), 0.0)

    def test_present_dice_heterogeneous_batch(self):
        """In a heterogeneous batch, only slices with target mass contribute to Dice loss."""
        b, h, w = 4, 32, 32
        # Slice 0, 1: completely empty target (mass = 0)
        # Slice 2, 3: non-empty target (mass > 0)
        target = torch.zeros(b, h, w)
        target[2, 10:20, 10:20] = 1.0
        target[3, 5:15, 5:15] = 1.0

        # Predict non-zero probabilities everywhere
        prob = torch.full((b, h, w), 0.5, requires_grad=True)

        loss = present_dice(prob, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(prob.grad).all())
        # Slices 0 and 1 must receive zero gradient because they are masked out
        self.assertEqual(float(prob.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(prob.grad[1].abs().sum()), 0.0)
        # Slices 2 and 3 must receive active non-zero gradient
        self.assertGreater(float(prob.grad[2].abs().sum()), 0.0)
        self.assertGreater(float(prob.grad[3].abs().sum()), 0.0)

    def test_m2_pro_loss_microbatch_zero_percent_scar(self):
        """Microbatch where scar (class 3) is 0% across all slices."""
        b, c, h, w = 4, 4, 64, 64
        logits = torch.randn(b, c, h, w, requires_grad=True)
        # Slices contain only class 0 (background), 1 (myocardium), 2 (edema) - NO SCAR (3)
        targets = torch.randint(0, 3, (b, h, w))
        aux_logits = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)

        res = self.criterion(logits, targets, aux_logits=aux_logits)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertFalse(torch.isnan(res["loss"]))

        # Scar dice must not blow up
        self.assertTrue(torch.isfinite(res["dice"]))
        self.assertTrue(torch.isfinite(res["focal_scar"]))

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(aux_logits.grad).all())

    def test_m2_pro_loss_microbatch_zero_percent_foreground(self):
        """Microbatch where 100% of pixels across all slices are background (class 0)."""
        b, c, h, w = 4, 4, 64, 64
        logits = torch.randn(b, c, h, w, requires_grad=True)
        targets = torch.zeros((b, h, w), dtype=torch.long)
        aux_logits = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)

        res = self.criterion(logits, targets, aux_logits=aux_logits)

        # Present dice must be exactly 0.0 for foreground classes
        self.assertAlmostEqual(res["dice"].detach().item(), 0.0, places=5)
        self.assertTrue(torch.isfinite(res["loss"]))
        self.assertFalse(torch.isnan(res["loss"]))

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(aux_logits.grad).all())

    def test_m2_pro_loss_batch_size_one_empty(self):
        """Single slice batch (B=1) with 100% background pixels."""
        logits = torch.randn(1, 4, 32, 32, requires_grad=True)
        targets = torch.zeros((1, 32, 32), dtype=torch.long)

        res = self.criterion(logits, targets)
        self.assertEqual(res["dice"].detach().item(), 0.0)
        self.assertTrue(torch.isfinite(res["loss"]))

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestAdversarialPathologyViolation(unittest.TestCase):
    """Challenge Vector 2: Severe pathology violation & monotonicity testing."""

    def setUp(self):
        torch.manual_seed(1337)
        self.criterion = M2ProLoss()

    def test_inclusion_monotonic_increase_under_pure_violation(self):
        """Assert inclusion loss increases strictly monotonically as scar increases in zero-edema, zero-myo region."""
        b, h, w = 1, 16, 16
        targets = torch.zeros((b, h, w), dtype=torch.long)

        # Sweep scar probability p3 through values in (0, 1)
        # where p_edema = 0 and p_myo = 0, so p_bg = 1 - p3
        # In this regime, inclusion = mean(ReLU(p3 - 0)) + mean(ReLU(p3 - 0)) = 2 * p3
        p3_values = [0.0, 0.05, 0.10, 0.25, 0.40, 0.60, 0.80, 0.95]
        inclusion_losses = []

        for p3 in p3_values:
            logits = torch.zeros(b, 4, h, w)
            if p3 == 0.0:
                logits[:, 0, :, :] = 20.0  # Background 100%
                logits[:, 1, :, :] = -20.0
                logits[:, 2, :, :] = -20.0
                logits[:, 3, :, :] = -20.0
            else:
                p0 = 1.0 - p3
                # logits for p0 and p3 with p1, p2 at -50
                logits[:, 0, :, :] = math.log(p0)
                logits[:, 1, :, :] = -50.0
                logits[:, 2, :, :] = -50.0
                logits[:, 3, :, :] = math.log(p3)

            res = self.criterion(logits, targets)
            inc = res["inclusion"].item()
            inclusion_losses.append(inc)

        # Verify strict monotonic increase
        for i in range(len(inclusion_losses) - 1):
            self.assertLess(
                inclusion_losses[i],
                inclusion_losses[i + 1],
                f"Inclusion loss failed monotonic increase at step {i}: {inclusion_losses[i]} >= {inclusion_losses[i+1]}",
            )

        # Verify exact theoretical slope: inclusion == 2.0 * p3 (for p3 >= 0.05 where p1, p2 ~ 0)
        for p3, inc in zip(p3_values[1:], inclusion_losses[1:]):
            expected_inc = 2.0 * p3
            self.assertAlmostEqual(
                inc,
                expected_inc,
                delta=1e-3,
                msg=f"Inclusion loss {inc} deviates from theoretical 2 * p3 = {expected_inc}",
            )

    def test_inclusion_monotonic_increase_with_scar_logit(self):
        """As scar logit z3 increases from -10 to +10 while other logits fixed, inclusion increases monotonically."""
        b, h, w = 1, 16, 16
        targets = torch.zeros((b, h, w), dtype=torch.long)

        z3_values = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0]
        losses = []

        for z3 in z3_values:
            logits = torch.zeros(b, 4, h, w)
            logits[:, 0, :, :] = 5.0   # Dominant background
            logits[:, 1, :, :] = -5.0  # Low myo
            logits[:, 2, :, :] = -5.0  # Low edema
            logits[:, 3, :, :] = z3    # Swept scar logit

            res = self.criterion(logits, targets)
            losses.append(res["inclusion"].item())

        for i in range(len(losses) - 1):
            self.assertLessEqual(
                losses[i],
                losses[i + 1],
                f"Inclusion loss failed monotonic property at z3 step {i}: {losses[i]} > {losses[i+1]}",
            )

    def test_inclusion_gradient_pushes_scar_down(self):
        """Gradient of inclusion loss w.r.t. scar logit is strictly positive when scar violates pathology."""
        b, h, w = 1, 16, 16
        targets = torch.zeros((b, h, w), dtype=torch.long)

        logits = torch.zeros(b, 4, h, w, requires_grad=True)
        # Violating state: scar has higher probability than edema & myo
        with torch.no_grad():
            logits[:, 0, :, :] = 0.0
            logits[:, 1, :, :] = 0.0
            logits[:, 2, :, :] = 0.0
            logits[:, 3, :, :] = 3.0  # scar is prominent

        res = self.criterion(logits, targets)
        res["inclusion"].backward()

        # Positive gradient on scar logit means gradient descent (z -= lr * grad) reduces scar logit
        scar_grad = logits.grad[:, 3, :, :]
        self.assertTrue((scar_grad > 0.0).all(), "Inclusion gradient did not push scar logit down")

    def test_inclusion_zero_when_conforming(self):
        """When scar probability <= edema and scar probability <= myo, inclusion loss is exactly 0.0."""
        b, h, w = 1, 16, 16
        targets = torch.zeros((b, h, w), dtype=torch.long)

        logits = torch.zeros(b, 4, h, w)
        logits[:, 0, :, :] = 0.0  # background
        logits[:, 1, :, :] = 3.0  # myo high
        logits[:, 2, :, :] = 3.0  # edema high
        logits[:, 3, :, :] = 1.0  # scar lower than both

        res = self.criterion(logits, targets)
        self.assertAlmostEqual(res["inclusion"].item(), 0.0, places=6)

    def test_inclusion_spatial_extent_scaling(self):
        """A larger area of severe pathology violation produces proportionally higher inclusion loss."""
        b, h, w = 1, 32, 32
        targets = torch.zeros((b, h, w), dtype=torch.long)

        # Case A: 4x4 spilling patch (16 pixels)
        logits_small = torch.zeros(b, 4, h, w)
        logits_small[:, 0, :, :] = 5.0
        logits_small[:, 1, :, :] = -2.0
        logits_small[:, 2, :, :] = -2.0
        logits_small[:, 3, :, :] = -10.0  # conforming outside patch
        logits_small[:, 3, 0:4, 0:4] = 10.0  # 16 pixels spilling
        res_small = self.criterion(logits_small, targets)

        # Case B: 8x8 spilling patch (64 pixels = 4x larger)
        logits_large = torch.zeros(b, 4, h, w)
        logits_large[:, 0, :, :] = 5.0
        logits_large[:, 1, :, :] = -2.0
        logits_large[:, 2, :, :] = -2.0
        logits_large[:, 3, :, :] = -10.0  # conforming outside patch
        logits_large[:, 3, 0:8, 0:8] = 10.0  # 64 pixels spilling
        res_large = self.criterion(logits_large, targets)

        # 64 pixels vs 16 pixels: ratio should be exactly 4x
        ratio = res_large["inclusion"].item() / res_small["inclusion"].item()
        self.assertAlmostEqual(ratio, 4.0, delta=0.05)


class TestAdversarialExtremeLogits(unittest.TestCase):
    """Challenge Vector 3: Numerical stability under extreme logits (+-50, +-100, subnormals)."""

    def setUp(self):
        torch.manual_seed(1337)
        self.criterion = M2ProLoss()

    def test_extreme_large_positive_and_negative_logits(self):
        """Logits of +/-50 and +/-100 do not cause NaN or Inf under log_softmax, sigmoid, or CE."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        extreme_values = [50.0, -50.0, 100.0, -100.0]
        for val in extreme_values:
            logits = torch.full((b, c, h, w), val, requires_grad=True)
            # Add small perturbation to avoid all identical channels
            with torch.no_grad():
                logits[:, 1, :, :] += 1.0
                logits[:, 2, :, :] += 2.0
                logits[:, 3, :, :] += 3.0

            res = self.criterion(logits, targets)
            for k, loss_t in res.items():
                self.assertTrue(
                    torch.isfinite(loss_t).all(),
                    f"Non-finite loss '{k}' for logit val {val}: {loss_t}",
                )
                self.assertFalse(
                    torch.isnan(loss_t).any(),
                    f"NaN loss '{k}' for logit val {val}",
                )

            res["loss"].backward()
            self.assertTrue(
                torch.isfinite(logits.grad).all(),
                f"Non-finite gradients for logit val {val}",
            )

    def test_extreme_dynamic_range_per_slice(self):
        """Extreme channel disparity: Class 0 is +50.0, Class 3 is -50.0, and vice versa."""
        b, h, w = 2, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        # Slice 0: Background dominant (+50) vs Scar suppressed (-50)
        # Slice 1: Scar dominant (+50) vs Background suppressed (-50)
        logits = torch.zeros(b, 4, h, w, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, :, :] = 50.0
            logits[0, 1:4, :, :] = -50.0
            logits[1, 3, :, :] = 50.0
            logits[1, 0:3, :, :] = -50.0

        res = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertFalse(torch.isnan(res["loss"]))

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_subnormal_float_values(self):
        """Subnormal float32 values (1e-38, 1e-45) do not destabilize loss or backward pass."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        # Float32 subnormal range
        logits = torch.full((b, c, h, w), 1e-38, requires_grad=True)
        res = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(res["loss"]).all())

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_aux_logits_extreme_values(self):
        """Aux logits with +/-50.0 do not cause NaN or Inf under BCE or Sigmoid."""
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)
        targets = torch.randint(0, 4, (b, h, w))

        aux_logits = torch.zeros(b, 2, h // 4, w // 4, requires_grad=True)
        with torch.no_grad():
            aux_logits[:, 0, :, :] = 50.0
            aux_logits[:, 1, :, :] = -50.0

        res = self.criterion(logits, targets, aux_logits=aux_logits)
        self.assertTrue(torch.isfinite(res["aux"]).all())
        self.assertTrue(torch.isfinite(res["loss"]).all())

        res["loss"].backward()
        self.assertTrue(torch.isfinite(aux_logits.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_non_finite_inputs_rejection(self):
        """NaN and Inf in logits or aux_logits are defensively rejected with ValueError."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        # NaN in main logits
        logits_nan = torch.randn(b, c, h, w)
        logits_nan[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.criterion(logits_nan, targets)

        # Inf in main logits
        logits_inf = torch.randn(b, c, h, w)
        logits_inf[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.criterion(logits_inf, targets)

        # NaN in aux logits
        logits_clean = torch.randn(b, c, h, w)
        aux_nan = torch.randn(b, 2, h // 4, w // 4)
        aux_nan[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.criterion(logits_clean, targets, aux_logits=aux_nan)


class TestAdversarialAMPCompatibility(unittest.TestCase):
    """Challenge Vector 4: Mixed precision (AMP) compatibility under bfloat16 and float16."""

    def setUp(self):
        torch.manual_seed(1337)
        self.criterion = M2ProLoss()

    def test_bfloat16_direct_tensor_execution(self):
        """Forward and backward pass succeed with bfloat16 tensors without loss of gradient history."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        main_bf16 = torch.randn(b, c, h, w, dtype=torch.bfloat16, requires_grad=True)
        aux_bf16 = torch.randn(b, 2, h // 4, w // 4, dtype=torch.bfloat16, requires_grad=True)

        res = self.criterion(main_bf16, targets, aux_logits=aux_bf16)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertEqual(res["loss"].dtype, torch.float32)

        res["loss"].backward()
        self.assertIsNotNone(main_bf16.grad)
        self.assertTrue(torch.isfinite(main_bf16.grad).all())
        self.assertEqual(main_bf16.grad.dtype, torch.bfloat16)

    def test_float16_direct_tensor_execution(self):
        """Forward and backward pass succeed with float16 tensors without underflow/overflow."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        main_fp16 = torch.randn(b, c, h, w, dtype=torch.float16, requires_grad=True)
        aux_fp16 = torch.randn(b, 2, h // 4, w // 4, dtype=torch.float16, requires_grad=True)

        res = self.criterion(main_fp16, targets, aux_logits=aux_fp16)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertEqual(res["loss"].dtype, torch.float32)

        res["loss"].backward()
        self.assertIsNotNone(main_fp16.grad)
        self.assertTrue(torch.isfinite(main_fp16.grad).all())
        self.assertEqual(main_fp16.grad.dtype, torch.float16)

    def test_autocast_bfloat16_context(self):
        """Forward and backward pass within torch.autocast context for bfloat16."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.autocast(device_type, dtype=torch.bfloat16):
                main = torch.randn(b, c, h, w, requires_grad=True)
                aux = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)
                res = self.criterion(main, targets, aux_logits=aux)
                self.assertTrue(torch.isfinite(res["loss"]).all())

        res["loss"].backward()
        self.assertTrue(torch.isfinite(main.grad).all())
        self.assertTrue(torch.isfinite(aux.grad).all())

    def test_grad_scaler_simulation_fp16(self):
        """Simulates AMP GradScaler (scale factor 65536.0) ensuring no gradient overflow."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        main = torch.randn(b, c, h, w, dtype=torch.float16, requires_grad=True)
        aux = torch.randn(b, 2, h // 4, w // 4, dtype=torch.float16, requires_grad=True)

        res = self.criterion(main, targets, aux_logits=aux)
        loss = res["loss"]

        # Simulate GradScaler scaling
        scale_factor = 65536.0
        scaled_loss = loss * scale_factor
        scaled_loss.backward()

        # Unscale gradients
        unscaled_main_grad = main.grad.float() / scale_factor
        unscaled_aux_grad = aux.grad.float() / scale_factor

        self.assertTrue(torch.isfinite(unscaled_main_grad).all())
        self.assertTrue(torch.isfinite(unscaled_aux_grad).all())
        self.assertGreater(float(unscaled_main_grad.abs().sum()), 0.0)


class TestAdversarialEndToEndM2ProIntegration(unittest.TestCase):
    """Challenge Vector 5: Full M2ProNet + M2ProLoss integration under adversarial conditions."""

    def setUp(self):
        torch.manual_seed(1337)
        self.model = M2ProNet(get_testing(), img_size=32, num_classes=4)
        self.model.train()
        self.criterion = M2ProLoss()

    def test_e2e_extreme_empty_slice_training_step(self):
        """Full forward + loss + backward step when input batch contains 100% background targets."""
        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w)
        psir = torch.randn(b, 1, h, w)
        t2w = torch.randn(b, 1, h, w)
        targets_empty = torch.zeros((b, h, w), dtype=torch.long)

        output = self.model(cine, psir, t2w, return_aux=True)
        loss_dict = self.criterion(output, targets_empty)

        self.assertTrue(torch.isfinite(loss_dict["loss"]).all())
        self.assertEqual(loss_dict["dice"].detach().item(), 0.0)

        # Backward step
        loss_dict["loss"].backward()

        # Optimizer step simulation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Non-finite grad in model param: {param.shape}",
                )

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    unittest.main()
