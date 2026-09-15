"""Milestone M3 Integration Test: SCAR Pipeline, Config, and M2-Pro Architecture."""
import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from training.config.config_utils import load_merged_config
from training.train import parse_args
from training.models import build_model, model_from_config
from training.models.m2_pro import M2ProNet
from training.loss.m2_pro_loss import M2ProLoss
from training.loss import build_loss


class TestM3Integration(unittest.TestCase):
    def test_m2_pro_yaml_config_loading(self):
        config_path = "training/config/models/m2_pro.yaml"
        base_path = "training/config/base.yaml"
        merged = load_merged_config(config_path, base_path)
        self.assertEqual(merged["model"]["model_name"], "M2-Pro")
        self.assertEqual(merged["model"]["architecture"], "m2_pro")
        self.assertEqual(merged["model"]["ablation"], "M2-PRO")

    def test_train_py_parse_args_m2_pro(self):
        args, merged = parse_args(["--config", "training/config/models/m2_pro.yaml"])
        self.assertEqual(args.ablation, "M2-PRO")
        self.assertEqual(merged["model"]["architecture"], "m2_pro")

    def test_model_instantiation(self):
        model = build_model("m2_pro", img_size=128, num_classes=4)
        self.assertIsInstance(model, M2ProNet)

    def test_build_loss_m2_pro(self):
        loss_fn = build_loss("m2_pro_loss")
        self.assertIsInstance(loss_fn, M2ProLoss)
        self.assertTrue(getattr(loss_fn, "requires_aux", False))

    def test_e2e_training_forward_backward(self):
        model = build_model("m2_pro", img_size=128, num_classes=4)
        model.train()
        criterion = M2ProLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        cine = torch.randn(2, 1, 128, 128)
        psir = torch.randn(2, 1, 128, 128)
        t2w = torch.randn(2, 1, 128, 128)
        targets = torch.randint(0, 4, (2, 128, 128))

        outputs = model(cine, psir, t2w, return_aux=True)
        losses = criterion(outputs, targets)

        self.assertIn("loss", losses)
        self.assertIn("ce", losses)
        self.assertIn("dice", losses)
        self.assertIn("aux", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

        optimizer.zero_grad()
        losses["loss"].backward()

        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

        optimizer.step()

    def test_eval_mode_output(self):
        model = build_model("m2_pro", img_size=128, num_classes=4)
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        out = model(x, x, x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 4, 128, 128))


if __name__ == "__main__":
    unittest.main()
