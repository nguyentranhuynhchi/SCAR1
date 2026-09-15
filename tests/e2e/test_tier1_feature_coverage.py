"""Tier 1 - Feature Coverage Tests for SCAR M2-Pro.

Covers:
- Canonical label order semantics & normalization
- Multimodal input handling (bSSFP, LGE, T2w)
- Model output tensor shapes across training and evaluation modes
- Loss computation on valid slices and empty lesion slices
- 5-region benchmark metrics calculation and pathology aggregation
"""
from __future__ import annotations

import unittest
import numpy as np
import torch

from training.dataset.data_contract import (
    CLASS_NAMES,
    CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER,
    canonicalize_label,
    resolve_label_order,
)
from training.metrics.surface_distance import (
    BENCHMARK_PROTOCOL,
    benchmark_rows,
    summarize_rows,
    binary_metrics,
)
from tests.e2e.contract_helpers import (
    get_m2pro_net,
    get_m2pro_loss,
    create_synthetic_volume,
)


class TestTier1FeatureCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # -------------------------------------------------------------------------
    # 1. Canonical Label Order Normalization
    # -------------------------------------------------------------------------
    def test_canonical_label_order_semantics(self):
        """Verify canonical and legacy label order resolution and names."""
        self.assertEqual(CLASS_NAMES, ("background", "normal_myocardium", "edema", "scar"))
        self.assertEqual(resolve_label_order("canonical"), CANONICAL_LABEL_ORDER)
        self.assertEqual(resolve_label_order("legacy"), LEGACY_LABEL_ORDER)
        self.assertEqual(resolve_label_order(CANONICAL_LABEL_ORDER), CANONICAL_LABEL_ORDER)
        self.assertEqual(resolve_label_order(LEGACY_LABEL_ORDER), LEGACY_LABEL_ORDER)
        with self.assertRaises(ValueError):
            resolve_label_order("unsupported_order")

    def test_canonicalize_label_mapping(self):
        """Verify label mapping vector [0, 1, 3, 2] from legacy to canonical."""
        # Canonical: 0=bg, 1=myo, 2=edema, 3=scar
        # Legacy:    0=bg, 1=myo, 2=scar,  3=edema
        legacy_input = np.array([[0, 1], [2, 3]], dtype=np.int32)
        canonical_output = canonicalize_label(legacy_input, "legacy")
        # Legacy 2 (scar) should become 3 (scar)
        # Legacy 3 (edema) should become 2 (edema)
        expected = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(canonical_output, expected)

        # Canonical input remains unchanged
        canonical_input = np.array([[0, 1], [2, 3]], dtype=np.int32)
        canonical_output2 = canonicalize_label(canonical_input, "canonical")
        np.testing.assert_array_equal(canonical_output2, canonical_input.astype(np.uint8))

    def test_canonicalize_label_immutability_and_validation(self):
        """Verify canonicalize_label never mutates input and rejects invalid class IDs."""
        original = np.array([[0, 1], [2, 3]], dtype=np.int32)
        original_copy = original.copy()
        _ = canonicalize_label(original, "legacy")
        np.testing.assert_array_equal(original, original_copy)

        # Non-integer IDs
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([[0.5, 1.2]]), "canonical")
        # Negative IDs
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([[-1, 1]]), "canonical")
        # Out-of-bounds IDs (> 3)
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([[0, 4]]), "canonical")

    # -------------------------------------------------------------------------
    # 2. Multimodal Input Handling
    # -------------------------------------------------------------------------
    def test_multimodal_three_stream_inputs(self):
        """Model accepts 3 separate tensors (B, 1, H, W) for cine, psir, t2w."""
        model, is_live = get_m2pro_net(img_size=32)
        model.eval()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w, dtype=torch.float32)
        psir = torch.randn(b, 1, h, w, dtype=torch.float32)
        t2w = torch.randn(b, 1, h, w, dtype=torch.float32)

        with torch.no_grad():
            output = model(cine, psir, t2w)

        if isinstance(output, dict):
            logits = output["logits"]
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        self.assertEqual(logits.shape, (b, 4, h, w))
        self.assertTrue(torch.isfinite(logits).all())

    def test_multimodal_shape_mismatch_rejection(self):
        """Model rejects inputs where modalities have mismatched spatial dimensions."""
        model, _ = get_m2pro_net(img_size=32)
        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 64, 64)  # Mismatched spatial size
        t2w = torch.randn(2, 1, 32, 32)

        with self.assertRaises(Exception):
            model(cine, psir, t2w)

    # -------------------------------------------------------------------------
    # 3. Model Output Shapes & Modes (Train vs Eval)
    # -------------------------------------------------------------------------
    def test_model_output_shapes_eval_mode(self):
        """In eval mode, model outputs main_logits of shape (B, 4, H, W)."""
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        for (h, w) in ((32, 32), (64, 64)):
            cine = torch.randn(1, 1, h, w)
            psir = torch.randn(1, 1, h, w)
            t2w = torch.randn(1, 1, h, w)
            with torch.no_grad():
                out = model(cine, psir, t2w)
            logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
            self.assertEqual(logits.shape, (1, 4, h, w))

    def test_model_output_shapes_train_mode_with_aux(self):
        """In train mode, model outputs main_logits (B, 4, H, W) and aux_logits (B, 2, H/4, W/4)."""
        model, _ = get_m2pro_net(img_size=32)
        model.train()

        b, h, w = 2, 32, 32
        cine = torch.randn(b, 1, h, w)
        psir = torch.randn(b, 1, h, w)
        t2w = torch.randn(b, 1, h, w)

        out = model(cine, psir, t2w, return_aux=True)
        if isinstance(out, dict):
            main_logits = out["logits"]
            aux_logits = out["aux_logits"]
        else:
            self.assertIsInstance(out, tuple)
            self.assertEqual(len(out), 2)
            main_logits, aux_logits = out

        self.assertEqual(main_logits.shape, (b, 4, h, w))
        self.assertEqual(aux_logits.shape, (b, 2, h // 4, w // 4))

    # -------------------------------------------------------------------------
    # 4. Loss Computation on Valid & Empty Slices
    # -------------------------------------------------------------------------
    def test_loss_computation_contract(self):
        """Loss function returns dict with required keys: loss, ce, dice, focal_scar, inclusion, aux."""
        criterion, _ = get_m2pro_loss()
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)
        aux_logits = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)
        targets = torch.randint(0, 4, (b, h, w))

        res = criterion(logits, targets, aux_logits=aux_logits)
        self.assertIsInstance(res, dict)
        for key in ("loss", "ce", "dice", "focal_scar", "inclusion", "aux"):
            self.assertIn(key, res, f"Missing required loss key: {key}")
            self.assertTrue(torch.isfinite(res[key]).all(), f"Non-finite loss value for {key}")

        # Backward pass gradient flow
        res["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNotNone(aux_logits.grad)
        self.assertTrue(torch.isfinite(aux_logits.grad).all())

    def test_loss_on_empty_lesion_slices(self):
        """Loss handles slices with 0 scar voxels or 0 edema voxels without division-by-zero or NaN."""
        criterion, _ = get_m2pro_loss()
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)

        # Target containing only background (0) and normal myocardium (1) - zero scar, zero edema
        targets_no_lesion = torch.randint(0, 2, (b, h, w))
        res = criterion(logits, targets_no_lesion)
        self.assertTrue(torch.isfinite(res["loss"]).all())
        self.assertFalse(torch.isnan(res["loss"]).any())

        # Target containing only background
        targets_all_bg = torch.zeros((b, h, w), dtype=torch.long)
        res_bg = criterion(logits, targets_all_bg)
        self.assertTrue(torch.isfinite(res_bg["loss"]).all())

    # -------------------------------------------------------------------------
    # 5. 5-Region Benchmark Metrics Calculation
    # -------------------------------------------------------------------------
    def test_benchmark_rows_five_regions(self):
        """benchmark_rows must produce exactly the 5 canonical benchmark regions."""
        h, w, d = 16, 16, 2
        pred = np.zeros((h, w, d), dtype=np.uint8)
        target = np.zeros((h, w, d), dtype=np.uint8)

        # Region assignment: normal myo=1, edema=2, scar=3
        target[4:12, 4:12, :] = 1
        target[6:10, 6:10, :] = 2
        target[7:9, 7:9, :] = 3

        pred[4:12, 4:12, :] = 1
        pred[6:10, 6:10, :] = 2
        pred[7:9, 7:9, :] = 3

        rows = benchmark_rows(pred, target, case="case0001", compute_distance=True)
        self.assertEqual(len(rows), 5)
        regions = [r["region"] for r in rows]
        expected_regions = ["normal_myocardium", "edema", "scar", "edema_inclusive", "myocardial_ring"]
        self.assertEqual(regions, expected_regions)

        # On perfect match, all Dice scores should be 1.0 and HD95 0.0
        for r in rows:
            self.assertEqual(r["status"], "ok")
            self.assertAlmostEqual(r["dice"], 1.0)
            self.assertAlmostEqual(r["iou"], 1.0)
            self.assertAlmostEqual(r["hd95_voxel"], 0.0)
            self.assertAlmostEqual(r["asd_voxel"], 0.0)

    def test_summarize_rows_avg_pathology_dice(self):
        """summarize_rows computes avg_pathology_dice as arithmetic mean of scar and edema."""
        synth = create_synthetic_volume(h=32, w=32, d=4)
        target = synth["label"]
        pred = target.copy()

        rows = benchmark_rows(pred, target, case="case0001", compute_distance=False)
        summary = summarize_rows(rows)

        self.assertIn("avg_pathology_dice", summary)
        expected_avg = (summary["scar"]["mean_dice"] + summary["edema"]["mean_dice"]) / 2.0
        self.assertAlmostEqual(summary["avg_pathology_dice"], expected_avg)
        self.assertAlmostEqual(summary["avg_pathology_dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
