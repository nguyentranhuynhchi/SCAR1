"""Tier 2 - Boundary & Corner Case Tests for SCAR M2-Pro.

Covers:
- Extreme empty masks (both empty, prediction empty, target empty)
- All-zero and all-foreground extreme segmentation predictions
- NaN / Inf checks and non-finite input rejection
- Zero-size scar volumes and single-voxel lesions
- Single-slice 3D volume inputs (D=1) and arbitrary in-plane resolutions
"""
from __future__ import annotations

import unittest
import numpy as np
import torch

from training.metrics.surface_distance import (
    binary_metrics,
    benchmark_rows,
    summarize_rows,
    calculate_metric_percase,
)
from training.predict import predict_volume
from tests.e2e.contract_helpers import get_m2pro_net, get_m2pro_loss


class TestTier2BoundaryCorner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # -------------------------------------------------------------------------
    # 1. Extreme Empty Masks
    # -------------------------------------------------------------------------
    def test_both_masks_empty(self):
        """When both prediction and target are empty, primary metric is None and official is 1.0."""
        pred = np.zeros((16, 16, 2), dtype=bool)
        truth = np.zeros((16, 16, 2), dtype=bool)

        res = binary_metrics(pred, truth)
        self.assertEqual(res["status"], "both_empty")
        self.assertIsNone(res["dice"])
        self.assertIsNone(res["iou"])
        self.assertIsNone(res["hd95"])
        self.assertIsNone(res["asd"])

        # In benchmark_rows
        p_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        t_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        rows = benchmark_rows(p_arr, t_arr, case="case_empty")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertEqual(scar_row["status"], "both_empty")
        self.assertIsNone(scar_row["dice"])
        self.assertEqual(scar_row["official_dice"], 1.0)
        self.assertEqual(scar_row["official_hd95_voxel"], 0.0)

        # In summarize_rows, undefined cases are excluded from the primary mean
        summary = summarize_rows(rows)
        self.assertIsNone(summary["scar"]["mean_dice"])
        self.assertEqual(summary["scar"]["dice_undefined_cases"], 1)
        self.assertEqual(summary["scar"]["dice_defined_cases"], 0)
        self.assertEqual(summary["scar"]["mean_official_dice"], 1.0)

    def test_prediction_empty_target_positive(self):
        """When target is present but prediction is completely empty (missed lesion)."""
        pred = np.zeros((16, 16, 2), dtype=bool)
        truth = np.zeros((16, 16, 2), dtype=bool)
        truth[5:10, 5:10, :] = True

        res = binary_metrics(pred, truth)
        self.assertEqual(res["status"], "prediction_empty")
        self.assertEqual(res["dice"], 0.0)
        self.assertEqual(res["iou"], 0.0)
        self.assertIsNone(res["hd95"])
        self.assertIsNone(res["asd"])

        # In benchmark_rows: precision is None (0/0), recall is 0.0
        p_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        t_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        t_arr[5:10, 5:10, :] = 3  # scar
        rows = benchmark_rows(p_arr, t_arr, case="case_miss")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertEqual(scar_row["dice"], 0.0)
        self.assertIsNone(scar_row["precision"])
        self.assertEqual(scar_row["recall"], 0.0)

    def test_target_empty_prediction_positive(self):
        """When target is empty but prediction hallucinates a lesion."""
        pred = np.zeros((16, 16, 2), dtype=bool)
        truth = np.zeros((16, 16, 2), dtype=bool)
        pred[5:10, 5:10, :] = True

        res = binary_metrics(pred, truth)
        self.assertEqual(res["status"], "target_empty")
        self.assertEqual(res["dice"], 0.0)
        self.assertEqual(res["iou"], 0.0)
        self.assertIsNone(res["hd95"])

        # In benchmark_rows: precision is 0.0, recall is None (0/0)
        p_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        t_arr = np.zeros((16, 16, 2), dtype=np.uint8)
        p_arr[5:10, 5:10, :] = 3
        rows = benchmark_rows(p_arr, t_arr, case="case_hallucination")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertEqual(scar_row["dice"], 0.0)
        self.assertEqual(scar_row["precision"], 0.0)
        self.assertIsNone(scar_row["recall"])

    # -------------------------------------------------------------------------
    # 2. All-Zero and All-Foreground Extremes
    # -------------------------------------------------------------------------
    def test_all_background_prediction(self):
        """Prediction with only background (class 0) evaluated on realistic volume."""
        h, w, d = 32, 32, 2
        pred = np.zeros((h, w, d), dtype=np.uint8)
        target = np.zeros((h, w, d), dtype=np.uint8)
        target[10:20, 10:20, :] = 1  # myocardium
        target[12:16, 12:16, :] = 2  # edema
        target[14:16, 14:16, :] = 3  # scar

        rows = benchmark_rows(pred, target, case="case_all_bg")
        for r in rows:
            self.assertEqual(r["status"], "prediction_empty")
            self.assertEqual(r["dice"], 0.0)
            self.assertEqual(r["recall"], 0.0)
            self.assertIsNone(r["precision"])

    def test_all_foreground_scar_prediction(self):
        """Prediction predicting entirely class 3 (scar)."""
        h, w, d = 16, 16, 2
        pred = np.full((h, w, d), 3, dtype=np.uint8)
        target = np.zeros((h, w, d), dtype=np.uint8)
        target[6:10, 6:10, :] = 3  # scar (4*4*2 = 32 voxels)

        rows = benchmark_rows(pred, target, case="case_all_scar")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertEqual(scar_row["status"], "ok")
        self.assertEqual(scar_row["recall"], 1.0)
        self.assertGreater(scar_row["precision"], 0.0)
        self.assertLess(scar_row["precision"], 1.0)
        # 2 * 32 / (512 + 32) = 64 / 544 ~ 0.1176
        expected_dice = 2 * 32 / (16 * 16 * 2 + 32)
        self.assertAlmostEqual(scar_row["dice"], expected_dice, places=4)

    # -------------------------------------------------------------------------
    # 3. NaN and Non-Finite Input Protection
    # -------------------------------------------------------------------------
    def test_predict_volume_rejects_nan_and_inf(self):
        """predict_volume raises ValueError on non-finite inputs."""
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        # NaN input
        nan_img = np.ones((32, 32, 2), dtype=np.float32)
        nan_img[5, 5, 0] = np.nan
        valid_img = np.ones((32, 32, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Non-finite inference inputs"):
            predict_volume(model, [nan_img, valid_img, valid_img], img_size=32)

        # Inf input
        inf_img = np.ones((32, 32, 2), dtype=np.float32)
        inf_img[5, 5, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Non-finite inference inputs"):
            predict_volume(model, [inf_img, valid_img, valid_img], img_size=32)

    def test_loss_numerical_stability_on_extreme_targets(self):
        """M2ProLoss remains finite and non-NaN even when all pixels belong to background or scar."""
        criterion, _ = get_m2pro_loss()
        b, h, w = 2, 32, 32
        logits = torch.randn(b, 4, h, w, requires_grad=True)

        for fill_class in (0, 1, 2, 3):
            targets = torch.full((b, h, w), fill_class, dtype=torch.long)
            res = criterion(logits, targets)
            self.assertTrue(torch.isfinite(res["loss"]).all())
            self.assertFalse(torch.isnan(res["loss"]).any())

    # -------------------------------------------------------------------------
    # 4. Zero-Size Scars & Sparse Lesions
    # -------------------------------------------------------------------------
    def test_zero_size_scar_volume(self):
        """A clinical volume with no scar tissue evaluates correctly without crashing."""
        h, w, d = 32, 32, 4
        target = np.zeros((h, w, d), dtype=np.uint8)
        target[10:22, 10:22, :] = 1  # normal myocardium
        target[12:18, 12:18, :] = 2  # edema, but ZERO scar

        pred = target.copy()  # model also predicted zero scar
        rows = benchmark_rows(pred, target, case="case_no_scar")
        scar_row = next(r for r in rows if r["region"] == "scar")
        self.assertEqual(scar_row["status"], "both_empty")
        self.assertIsNone(scar_row["dice"])
        self.assertEqual(scar_row["official_dice"], 1.0)
        self.assertEqual(scar_row["official_hd95_voxel"], 0.0)

    def test_single_voxel_scar(self):
        """Single isolated voxel lesion (N=1) calculates finite HD95 and ASD."""
        pred = np.zeros((16, 16, 2), dtype=bool)
        truth = np.zeros((16, 16, 2), dtype=bool)
        pred[8, 8, 0] = True
        truth[8, 9, 0] = True  # adjacent voxel

        res = binary_metrics(pred, truth, compute_distance=True)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["dice"], 0.0)  # no overlap
        self.assertIsNotNone(res["hd95"])
        self.assertIsNotNone(res["asd"])
        self.assertTrue(np.isfinite(res["hd95"]))
        self.assertTrue(np.isfinite(res["asd"]))
        self.assertAlmostEqual(res["hd95"], 1.0)  # Euclidean distance between (8,8) and (8,9) is 1.0

    # -------------------------------------------------------------------------
    # 5. Single-Slice 3D Inputs (D=1) & Arbitrary Resolution
    # -------------------------------------------------------------------------
    def test_single_slice_volume_inference_and_metrics(self):
        """Volume with depth D=1 runs correctly through predict_volume and surface distance."""
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        h, w, d = 32, 32, 1
        images = [np.random.default_rng(i).uniform(0.0, 1.0, (h, w, d)).astype(np.float32) for i in range(3)]
        pred = predict_volume(model, images, img_size=32, batch_size=2)
        self.assertEqual(pred.shape, (h, w, 1))

        # Test metrics on single slice
        target = np.zeros((h, w, 1), dtype=np.uint8)
        target[10:20, 10:20, 0] = 1
        target[12:16, 12:16, 0] = 2
        target[14:15, 14:15, 0] = 3

        rows = benchmark_rows(pred, target, case="case_single_slice")
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertIn(r["status"], ("ok", "both_empty", "prediction_empty", "target_empty"))

    def test_arbitrary_in_plane_resolution_resizing(self):
        """predict_volume seamlessly resizes arbitrary resolutions (e.g. 47x53) to model grid and back."""
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        h, w, d = 47, 53, 2  # Non-power-of-two, non-square
        images = [np.random.default_rng(i).uniform(0.0, 1.0, (h, w, d)).astype(np.float32) for i in range(3)]
        pred = predict_volume(model, images, img_size=32, batch_size=2)
        self.assertEqual(pred.shape, (47, 53, 2))
        self.assertTrue(np.isin(pred, [0, 1, 2, 3]).all())


if __name__ == "__main__":
    unittest.main()
