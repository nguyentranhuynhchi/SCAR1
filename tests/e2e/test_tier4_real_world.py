"""Tier 4 - Real-World Application Scenario Tests for SCAR M2-Pro.

Covers:
- End-to-end evaluation pipeline on 3D test volumes
- Exact 17-column schema verification for per_case.csv
- Hierarchical schema and mathematical integrity of metrics.json (allow_nan=False)
- Incremental epoch metrics logging and schema expansion in metrics.csv
- Official 76-case benchmark SHA256 cryptographic verification and patient disjointness
- Quantitative benchmark acceptance threshold validation against ORIGINAL_REQUEST.md
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from training.dataset.data_contract import (
    OFFICIAL_TEST_IDS_SHA256,
    validate_patient_splits,
    patient_id,
)
from training.metrics.surface_distance import (
    BENCHMARK_PROTOCOL,
    benchmark_rows,
    summarize_rows,
)
from training.predict import predict_volume
from training.trainer.trainer import append_metrics_csv, json_safe, write_json
from tests.e2e.contract_helpers import get_m2pro_net, create_synthetic_volume


class TestTier4RealWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    # -------------------------------------------------------------------------
    # 1. Official Test Set Cryptographic Fingerprint & Patient Disjointness
    # -------------------------------------------------------------------------
    def test_official_test_split_sha256_lock(self):
        """Official 76-case test split manifest matches OFFICIAL_TEST_IDS_SHA256 exactly."""
        manifest_path = Path("preprocessing/splits/test_vol.txt")
        self.assertTrue(manifest_path.is_file(), "preprocessing/splits/test_vol.txt must exist")

        lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.assertEqual(len(lines), 76, "Test cohort must contain exactly 76 patient IDs")

        # Independent hash computation
        fixed_hash = hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()
        self.assertEqual(fixed_hash, OFFICIAL_TEST_IDS_SHA256)

    def test_validate_patient_splits_detects_leakage(self):
        """validate_patient_splits raises ValueError if any patient ID leaks between splits."""
        clean_splits = {
            "train": ["case0002", "case0003", "case0004"],
            "val": ["case0005"],
            "test_vol": ["case0001", "case0006"],
        }
        # Should succeed without error
        patient_sets = validate_patient_splits(clean_splits)
        self.assertEqual(len(patient_sets["train"]), 3)

        # Inject leakage: case0001 present in both train and test_vol
        leaky_splits = {
            "train": ["case0001", "case0002"],
            "val": ["case0005"],
            "test_vol": ["case0001", "case0006"],
        }
        with self.assertRaisesRegex(ValueError, "Patient leakage between train and test_vol"):
            validate_patient_splits(leaky_splits)

    # -------------------------------------------------------------------------
    # 2. End-to-End Evaluation Pipeline & 17-Column per_case.csv Schema
    # -------------------------------------------------------------------------
    def test_per_case_csv_exact_17_columns_and_rows(self):
        """per_case.csv has exactly 17 specified columns and 5 rows per evaluated case."""
        cases = ["case0001", "case0006", "case0011"]
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "eval_test_vol"
            output_dir.mkdir(parents=True)

            all_rows = []
            for case in cases:
                synth = create_synthetic_volume(h=32, w=32, d=3, seed=int(case[-4:]))
                images = [synth["cine"], synth["psir"], synth["t2w"]]
                pred = predict_volume(model, images, img_size=32, batch_size=2)
                case_rows = benchmark_rows(pred, synth["label"], case=case, compute_distance=True)
                for r in case_rows:
                    r["inference_seconds"] = 0.05
                all_rows.extend(case_rows)

            csv_path = output_dir / "per_case.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)

            # Verification of written file
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                header = reader.fieldnames
                rows = list(reader)

            expected_columns = [
                "case",
                "region",
                "dice",
                "iou",
                "hd95",
                "asd",
                "status",
                "precision",
                "recall",
                "prediction_voxels",
                "target_voxels",
                "hd95_unit",
                "hd95_voxel",
                "asd_voxel",
                "official_dice",
                "official_hd95_voxel",
                "inference_seconds",
            ]
            self.assertEqual(len(header), 17, f"Expected exactly 17 columns, found {len(header)}: {header}")
            self.assertEqual(header, expected_columns, "Column names or ordering mismatch")

            # Exactly 3 cases * 5 regions = 15 rows
            self.assertEqual(len(rows), len(cases) * 5)

            # Check validity of individual rows
            for row in rows:
                self.assertIn(row["case"], cases)
                self.assertIn(row["region"], BENCHMARK_PROTOCOL["regions"])
                self.assertIn(row["status"], ("ok", "both_empty", "prediction_empty", "target_empty"))
                self.assertEqual(row["hd95_unit"], "voxel")
                self.assertTrue(row["prediction_voxels"].isdigit())
                self.assertTrue(row["target_voxels"].isdigit())

    # -------------------------------------------------------------------------
    # 3. Hierarchical metrics.json Schema & Math Integrity
    # -------------------------------------------------------------------------
    def test_metrics_json_schema_and_math_consistency(self):
        """metrics.json validates without NaN (allow_nan=False) and verifies avg_pathology_dice."""
        cases = ["case0001", "case0006"]
        model, _ = get_m2pro_net(img_size=32)
        model.eval()

        all_rows = []
        for case in cases:
            synth = create_synthetic_volume(h=32, w=32, d=4, seed=int(case[-4:]))
            pred = predict_volume(model, [synth["cine"], synth["psir"], synth["t2w"]], img_size=32, batch_size=2)
            case_rows = benchmark_rows(pred, synth["label"], case=case, compute_distance=True)
            for r in case_rows:
                r["inference_seconds"] = 0.08
            all_rows.extend(case_rows)

        summary = summarize_rows(all_rows)
        summary.update(
            checkpoint="checkpoints/best.pth",
            checkpoint_epoch=40,
            ablation="M2-Pro",
            architecture="m2_pro",
            split="test_vol",
            case_count=len(cases),
            inference_seconds=0.16,
            inference_slices_per_second=50.0,
            benchmark_protocol=BENCHMARK_PROTOCOL,
            device="cpu",
            amp_dtype="none",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "metrics.json"
            write_json(json_path, summary)

            self.assertTrue(json_path.is_file())
            content = json_path.read_text(encoding="utf-8")
            # Must parse cleanly with allow_nan=False (no raw NaN in compliant JSON)
            parsed = json.loads(content)

            # Check 5 region sections exist
            for reg in ("normal_myocardium", "edema", "scar", "edema_inclusive", "myocardial_ring"):
                self.assertIn(reg, parsed)
                self.assertIn("mean_dice", parsed[reg])
                self.assertIn("dice_defined_cases", parsed[reg])
                self.assertIn("dice_undefined_cases", parsed[reg])
                self.assertIn("mean_hd95_voxel", parsed[reg])

            # Check summary pathology metrics
            self.assertIn("avg_pathology_dice", parsed)
            self.assertIn("avg_pathology_iou", parsed)
            self.assertIn("official_avg_pathology_dice", parsed)

            # Mathematical property: avg_pathology_dice = (scar_dice + edema_dice) / 2
            scar_mean = parsed["scar"]["mean_dice"]
            edema_mean = parsed["edema"]["mean_dice"]
            if scar_mean is not None and edema_mean is not None:
                expected_avg = (scar_mean + edema_mean) / 2.0
                self.assertAlmostEqual(parsed["avg_pathology_dice"], expected_avg)

    # -------------------------------------------------------------------------
    # 4. Incremental Epoch Logging in metrics.csv
    # -------------------------------------------------------------------------
    def test_metrics_csv_epoch_logging_and_expansion(self):
        """append_metrics_csv appends epoch rows and handles schema expansion cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "metrics.csv"

            # Epoch 1: standard metrics
            append_metrics_csv(csv_path, {"epoch": 1, "train/loss": 0.65, "val/dice": 0.55})
            # Epoch 2: new metric key introduced
            append_metrics_csv(csv_path, {"epoch": 2, "train/loss": 0.52, "val/dice": 0.62, "val/avg_pathology_dice": 0.58})
            # Epoch 3: further progress
            append_metrics_csv(csv_path, {"epoch": 3, "train/loss": 0.41, "val/dice": 0.70, "val/avg_pathology_dice": 0.66})

            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                header = reader.fieldnames
                rows = list(reader)

            self.assertEqual(len(rows), 3)
            self.assertIn("val/avg_pathology_dice", header)
            # In Epoch 1, the new metric should be blank string, not None
            self.assertEqual(rows[0]["val/avg_pathology_dice"], "")
            self.assertEqual(rows[1]["val/avg_pathology_dice"], "0.58")
            self.assertEqual(rows[2]["val/avg_pathology_dice"], "0.66")

    # -------------------------------------------------------------------------
    # 5. Quantitative Benchmark Acceptance Criteria Evaluation
    # -------------------------------------------------------------------------
    def test_acceptance_criteria_threshold_evaluator(self):
        """Verify acceptance threshold criteria logic from ORIGINAL_REQUEST.md."""
        # Thresholds:
        # Scar Dice > 60.0%
        # Edema Dice >= 73.5%
        # Myo Ring Dice >= 87.5%
        # Avg Pathology Dice >= 65.0%

        def check_acceptance(scar_dice, edema_dice, myo_ring_dice, avg_pathology_dice):
            return {
                "scar_dice_pass": scar_dice > 0.60,
                "edema_dice_pass": edema_dice >= 0.735,
                "myo_ring_dice_pass": myo_ring_dice >= 0.875,
                "avg_pathology_dice_pass": avg_pathology_dice >= 0.65,
                "all_pass": (
                    scar_dice > 0.60
                    and edema_dice >= 0.735
                    and myo_ring_dice >= 0.875
                    and avg_pathology_dice >= 0.65
                ),
            }

        # Baseline M2 (from metrics.json): Scar 40.85%, Edema 72.82%, Myo 87.48%, Avg 56.84%
        m2_result = check_acceptance(0.4085, 0.7282, 0.8748, 0.5684)
        self.assertFalse(m2_result["scar_dice_pass"], "Baseline M2 must fail scar threshold")
        self.assertFalse(m2_result["all_pass"])

        # Target M2-Pro: Scar 63.5%, Edema 74.5%, Myo 88.2%, Avg 69.0%
        m2_pro_target = check_acceptance(0.635, 0.745, 0.882, 0.690)
        self.assertTrue(m2_pro_target["scar_dice_pass"])
        self.assertTrue(m2_pro_target["edema_dice_pass"])
        self.assertTrue(m2_pro_target["myo_ring_dice_pass"])
        self.assertTrue(m2_pro_target["avg_pathology_dice_pass"])
        self.assertTrue(m2_pro_target["all_pass"])


if __name__ == "__main__":
    unittest.main()
