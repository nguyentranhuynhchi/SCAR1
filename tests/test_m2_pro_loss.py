"""Comprehensive Unit Test Suite for M2ProLoss (Milestone M2).

Verifies:
  1. Shape compatibility: inputs (B, 4, 128, 128) and targets (B, 128, 128).
  2. Output dictionary keys ("loss", "ce", "dice", "focal_scar", "inclusion", "aux") and scalar shapes.
  3. Polymorphic input unwrapping (Tensor, tuple, M2ProOutput, dict).
  4. Finite gradient backpropagation to both main logits and aux_logits.
  5. Edge case numerical stability (zero-scar slices, all-background slices, all-class slices).
  6. Pathology & anatomy inclusion constraint behavior: assert higher loss when scar spills outside edema.
  7. Focal scar loss behavior: assert higher loss on hard misclassified scar pixels.
  8. AMP execution safety under torch.bfloat16 and torch.float16.
  9. Class-weighted cross-entropy penalization ratios (30x scar vs background).
  10. Present dice zero-target avoidance: eliminates pathological collapse on negative slices.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.nn import functional as F

from training.loss.m2_pro_loss import M2ProLoss, present_dice
from training.models.modules.m2_pro import M2ProOutput
from training.models.m2_pro import M2ProNet
from training.models.cmspa_net import get_testing


class TestM2ProLoss(unittest.TestCase):
    """Unit tests for M2ProLoss compound extreme imbalance loss."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.criterion = M2ProLoss()

    # -------------------------------------------------------------------------
    # 1. Shape Compatibility and Dimension Handling
    # -------------------------------------------------------------------------
    def test_shape_compatibility_standard_128x128(self):
        """Inputs (B, 4, 128, 128) and targets (B, 128, 128) execute cleanly with and without aux."""
        b, c, h, w = 2, 4, 128, 128
        logits = torch.randn(b, c, h, w, requires_grad=True)
        targets = torch.randint(0, 4, (b, h, w))
        aux_logits = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)

        # Without aux
        res_no_aux = self.criterion(logits, targets)
        self.assertEqual(res_no_aux["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(res_no_aux["loss"]).all())
        self.assertEqual(res_no_aux["aux"].item(), 0.0)

        # With aux
        res_with_aux = self.criterion(logits, targets, aux_logits=aux_logits)
        self.assertEqual(res_with_aux["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(res_with_aux["loss"]).all())
        self.assertGreater(res_with_aux["aux"].item(), 0.0)

    def test_4d_targets_auto_squeeze(self):
        """Targets with shape (B, 1, 128, 128) are automatically squeezed to (B, 128, 128)."""
        logits = torch.randn(2, 4, 128, 128)
        targets_4d = torch.randint(0, 4, (2, 1, 128, 128))
        res = self.criterion(logits, targets_4d)
        self.assertTrue(torch.isfinite(res["loss"]).all())

    def test_shape_mismatch_and_invalid_labels_rejection(self):
        """Rejects spatial mismatch, channel mismatch, and out-of-bounds class labels."""
        logits = torch.randn(2, 4, 64, 64)
        targets_wrong_spatial = torch.randint(0, 4, (2, 32, 32))
        with self.assertRaises(ValueError):
            self.criterion(logits, targets_wrong_spatial)

        targets_wrong_batch = torch.randint(0, 4, (3, 64, 64))
        with self.assertRaises(ValueError):
            self.criterion(logits, targets_wrong_batch)

        targets_invalid_class = torch.full((2, 64, 64), 5, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.criterion(logits, targets_invalid_class)

    # -------------------------------------------------------------------------
    # 2. Output Dictionary Keys and Scalar Properties
    # -------------------------------------------------------------------------
    def test_output_dictionary_keys_and_scalar_properties(self):
        """Returns exact dictionary with required 6 keys, each being a 0-dim finite scalar."""
        expected_keys = {"loss", "ce", "dice", "focal_scar", "inclusion", "aux"}
        logits = torch.randn(2, 4, 32, 32, requires_grad=True)
        targets = torch.randint(0, 4, (2, 32, 32))
        aux = torch.randn(2, 2, 8, 8, requires_grad=True)

        res = self.criterion(logits, targets, aux_logits=aux)
        self.assertEqual(set(res.keys()), expected_keys)

        for key in expected_keys:
            val = res[key]
            self.assertIsInstance(val, torch.Tensor, f"{key} is not a torch.Tensor")
            self.assertEqual(val.ndim, 0, f"{key} must be a 0-dim scalar tensor, got {val.shape}")
            self.assertTrue(torch.isfinite(val).item(), f"{key} is non-finite: {val}")

        # Mathematically verify total weighted sum
        computed_total = (
            self.criterion.lambda_ce * res["ce"]
            + self.criterion.lambda_dice * res["dice"]
            + self.criterion.lambda_focal * res["focal_scar"]
            + self.criterion.lambda_inc * res["inclusion"]
            + self.criterion.lambda_aux * res["aux"]
        )
        torch.testing.assert_close(res["loss"], computed_total)

    # -------------------------------------------------------------------------
    # 3. Polymorphic Input Unwrapping
    # -------------------------------------------------------------------------
    def test_polymorphic_inputs(self):
        """Seamlessly accepts Tensor, tuple (main, aux), M2ProOutput, and dict."""
        b, c, h, w = 2, 4, 32, 32
        main = torch.randn(b, c, h, w)
        aux = torch.randn(b, 2, h // 4, w // 4)
        targets = torch.randint(0, 4, (b, h, w))

        # 1. Direct Tensor + keyword aux
        res_tensor = self.criterion(main, targets, aux_logits=aux)

        # 2. Tuple (main, aux)
        res_tuple = self.criterion((main, aux), targets)

        # 3. M2ProOutput instance
        output_obj = M2ProOutput(main, aux)
        res_m2pro_output = self.criterion(output_obj, targets)

        # 4. Dict with main_logits and aux_logits
        res_dict_main = self.criterion({"main_logits": main, "aux_logits": aux}, targets)

        # 5. Dict with logits and aux_logits
        res_dict_logits = self.criterion({"logits": main, "aux_logits": aux}, targets)

        # All output losses must match exactly
        torch.testing.assert_close(res_tensor["loss"], res_tuple["loss"])
        torch.testing.assert_close(res_tensor["loss"], res_m2pro_output["loss"])
        torch.testing.assert_close(res_tensor["loss"], res_dict_main["loss"])
        torch.testing.assert_close(res_tensor["loss"], res_dict_logits["loss"])

    # -------------------------------------------------------------------------
    # 4. Finite Gradient Backpropagation
    # -------------------------------------------------------------------------
    def test_finite_gradient_backpropagation(self):
        """Loss backpropagates finite, non-zero gradients to both main logits and aux logits."""
        b, c, h, w = 2, 4, 64, 64
        main = torch.randn(b, c, h, w, requires_grad=True)
        aux = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)
        targets = torch.randint(0, 4, (b, h, w))

        res = self.criterion(main, targets, aux_logits=aux)
        res["loss"].backward()

        self.assertIsNotNone(main.grad)
        self.assertTrue(torch.isfinite(main.grad).all())
        self.assertGreater(float(main.grad.abs().sum()), 0.0)

        self.assertIsNotNone(aux.grad)
        self.assertTrue(torch.isfinite(aux.grad).all())
        self.assertGreater(float(aux.grad.abs().sum()), 0.0)

    def test_m2pro_net_e2e_gradient_flow(self):
        """Gradients flow back into all trainable parameters of M2ProNet."""
        model = M2ProNet(get_testing(), img_size=32, num_classes=4)
        model.train()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w)
        psir = torch.randn(b, 1, h, w)
        t2w = torch.randn(b, 1, h, w)
        targets = torch.randint(0, 4, (b, h, w))

        output = model(cine, psir, t2w, return_aux=True)
        loss_dict = self.criterion(output, targets)
        loss_dict["loss"].backward()

        # Verify main head and aux head parameters receive finite gradients
        has_main_grad = False
        has_aux_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite grad on {name}")
                if "segmentation_head" in name or "out_proj" in name:
                    has_main_grad = True
                if "aux_head" in name:
                    has_aux_grad = True

        self.assertTrue(has_main_grad, "Main head received no gradient")
        self.assertTrue(has_aux_grad, "Auxiliary head received no gradient")

    # -------------------------------------------------------------------------
    # 5. Edge Case Stability: Zero Scar, Only Background, All Classes
    # -------------------------------------------------------------------------
    def test_edge_case_slices_with_zero_scar(self):
        """Handles slices containing only normal myocardium (1) and edema (2) with zero scar (3)."""
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)
        # Only class 0, 1, 2 (no scar 3)
        targets = torch.randint(0, 3, (b, h, w))

        res = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertFalse(torch.isnan(res["loss"]).any())

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_edge_case_slices_with_only_background(self):
        """Handles completely negative slices (all pixels class 0) without NaN or zero division."""
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)
        targets = torch.zeros((b, h, w), dtype=torch.long)

        res = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        # Present dice on all-background should evaluate to 0.0 because no foreground target exists
        self.assertAlmostEqual(res["dice"].detach().item(), 0.0, places=5)

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_edge_case_all_classes_present(self):
        """Handles slices where all 4 classes (0, 1, 2, 3) are present simultaneously."""
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)
        targets = torch.zeros((b, h, w), dtype=torch.long)
        targets[:, 0:8, :] = 0
        targets[:, 8:16, :] = 1
        targets[:, 16:24, :] = 2
        targets[:, 24:32, :] = 3

        res = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertGreater(res["dice"].detach().item(), 0.0)
        self.assertGreater(res["focal_scar"].detach().item(), 0.0)

        res["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    # -------------------------------------------------------------------------
    # 6. Pathology & Anatomy Inclusion Constraint Behavior
    # -------------------------------------------------------------------------
    def test_inclusion_constraint_penalizes_scar_spilling_outside_edema_and_myo(self):
        """Inclusion loss is significantly higher when scar probability spills outside edema."""
        b, h, w = 1, 32, 32
        targets = torch.zeros((b, h, w), dtype=torch.long)

        # Case 1: Conforming prediction
        # Scar (3) has lower probability than Edema (2) and Myo (1): P(scar) <= P(edema), P(scar) <= P(myo)
        logits_conforming = torch.zeros((b, 4, h, w))
        logits_conforming[:, 0, :, :] = 0.0   # Background
        logits_conforming[:, 1, :, :] = 2.0   # Myo prob high
        logits_conforming[:, 2, :, :] = 3.0   # Edema prob highest
        logits_conforming[:, 3, :, :] = 1.0   # Scar prob lowest
        res_conforming = self.criterion(logits_conforming, targets)

        # Case 2: Spilling prediction
        # Scar (3) has much higher probability than Edema (2) and Myo (1): P(scar) >> P(edema), P(scar) >> P(myo)
        logits_spilling = torch.zeros((b, 4, h, w))
        logits_spilling[:, 0, :, :] = 0.0    # Background
        logits_spilling[:, 1, :, :] = 0.5    # Myo prob low
        logits_spilling[:, 2, :, :] = 0.5    # Edema prob low
        logits_spilling[:, 3, :, :] = 5.0    # Scar prob dominant
        res_spilling = self.criterion(logits_spilling, targets)

        self.assertAlmostEqual(res_conforming["inclusion"].detach().item(), 0.0, places=5)
        self.assertGreater(res_spilling["inclusion"].detach().item(), 0.5)
        self.assertGreater(res_spilling["inclusion"].detach().item(), res_conforming["inclusion"].detach().item())

    # -------------------------------------------------------------------------
    # 7. Focal Scar Loss Behavior
    # -------------------------------------------------------------------------
    def test_focal_scar_loss_behavior_on_hard_vs_easy_pixels(self):
        """Focal loss concentrates gradient on hard misclassified scar pixels compared to easy ones."""
        criterion = M2ProLoss(focal_gamma=2.0, focal_alpha=0.75)
        b, h, w = 1, 16, 16

        # Ground truth: pixel at (8, 8) is true scar (class 3)
        targets = torch.zeros((b, h, w), dtype=torch.long)
        targets[0, 8, 8] = 3

        # Scenario A: Easy scar prediction (model predicts high scar logit at (8, 8))
        logits_easy = torch.zeros((b, 4, h, w))
        logits_easy[:, 0, :, :] = 2.0  # background elsewhere
        logits_easy[0, 3, 8, 8] = 10.0  # high confidence scar at target
        res_easy = criterion(logits_easy, targets)

        # Scenario B: Hard scar prediction (model misclassifies scar target as background)
        logits_hard = torch.zeros((b, 4, h, w))
        logits_hard[:, 0, :, :] = 2.0
        logits_hard[0, 0, 8, 8] = 10.0  # predicting background on true scar
        logits_hard[0, 3, 8, 8] = -5.0  # very low scar logit on true scar
        res_hard = criterion(logits_hard, targets)

        # Hard misclassification must produce vastly larger focal loss
        self.assertGreater(res_hard["focal_scar"].detach().item(), res_easy["focal_scar"].detach().item() * 5.0)

    # -------------------------------------------------------------------------
    # 8. AMP Safety (bfloat16 & float16)
    # -------------------------------------------------------------------------
    def test_amp_safety_bfloat16_and_float16(self):
        """Executes stably under autocast with bfloat16 and float16 without NaN."""
        b, c, h, w = 2, 4, 32, 32
        targets = torch.randint(0, 4, (b, h, w))

        # Test bfloat16 autocast
        with torch.autocast("cpu", dtype=torch.bfloat16):
            main_bf16 = torch.randn(b, c, h, w, requires_grad=True)
            aux_bf16 = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)
            res_bf16 = self.criterion(main_bf16, targets, aux_logits=aux_bf16)
            self.assertTrue(torch.isfinite(res_bf16["loss"]).all())
            res_bf16["loss"].backward()
            self.assertTrue(torch.isfinite(main_bf16.grad).all())

        # Test explicit half-precision tensors
        main_fp16 = torch.randn(b, c, h, w, dtype=torch.float16, requires_grad=True)
        aux_fp16 = torch.randn(b, 2, h // 4, w // 4, dtype=torch.float16, requires_grad=True)
        res_fp16 = self.criterion(main_fp16, targets, aux_logits=aux_fp16)
        self.assertTrue(torch.isfinite(res_fp16["loss"]).all())
        res_fp16["loss"].backward()
        self.assertTrue(torch.isfinite(main_fp16.grad).all())

    # -------------------------------------------------------------------------
    # 9. Class-Weighted Cross-Entropy Penalization Ratios
    # -------------------------------------------------------------------------
    def test_class_weighted_cross_entropy_ratios(self):
        """Weights [0.1, 1.0, 2.0, 3.0] heavily penalize scar errors 30x more than background."""
        ce_weights = self.criterion.ce_weights
        self.assertAlmostEqual(float(ce_weights[0]), 0.1, places=5)
        self.assertAlmostEqual(float(ce_weights[1]), 1.0, places=5)
        self.assertAlmostEqual(float(ce_weights[2]), 2.0, places=5)
        self.assertAlmostEqual(float(ce_weights[3]), 3.0, places=5)

        # Ratio of scar weight (3.0) to background weight (0.1) is exactly 30
        self.assertAlmostEqual(float(ce_weights[3] / ce_weights[0]), 30.0, places=4)

    # -------------------------------------------------------------------------
    # 10. Present Dice Eliminates Negative Slice Penalties
    # -------------------------------------------------------------------------
    def test_present_dice_prevents_zero_prediction_collapse(self):
        """On a slice with no scar, present_dice returns 0.0 and does not penalize exploratory predictions."""
        b, h, w = 1, 32, 32
        prob = torch.full((b, h, w), 0.3)  # exploratory non-zero probability
        target_empty = torch.zeros((b, h, w))  # no scar

        loss_empty = present_dice(prob, target_empty)
        # Empty slice should evaluate to 0.0 loss rather than 1.0
        self.assertEqual(loss_empty.detach().item(), 0.0)

        # When target is present, dice loss is strictly active
        target_positive = torch.zeros((b, h, w))
        target_positive[:, 10:20, 10:20] = 1.0
        loss_positive = present_dice(prob, target_positive)
        self.assertGreater(loss_positive.detach().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
