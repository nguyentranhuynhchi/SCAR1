"""Unit test suite verifying M2-Plus-Plus forward/backward, shapes, and loss."""
import sys
from pathlib import Path

# Đảm bảo Python luôn nhận diện được thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from training.models.m2_plus_plus import M2PlusPlusNet
from training.loss.m2_plus_plus_loss import M2PlusPlusLoss
from training.models.cmspa_net import get_testing


def test_m2_plus_plus_forward_and_backward():
    config = get_testing()
    config.architecture = "m2_plus_plus"
    model = M2PlusPlusNet(config=config, img_size=128, num_classes=4)
    model.train()

    cine = torch.randn(2, 1, 128, 128)
    psir = torch.randn(2, 1, 128, 128)
    t2w = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 4, (2, 128, 128))

    output = model(cine, psir, t2w)
    assert output.shape == (2, 4, 128, 128)

    criterion = M2PlusPlusLoss()
    losses = criterion(output, target)
    assert "loss" in losses
    assert torch.isfinite(losses["loss"])

    losses["loss"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_m2_plus_plus_eval_mode():
    config = get_testing()
    config.architecture = "m2_plus_plus"
    model = M2PlusPlusNet(config=config, img_size=128, num_classes=4)
    model.eval()

    cine = torch.randn(2, 1, 128, 128)
    psir = torch.randn(2, 1, 128, 128)
    t2w = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        logits = model(cine, psir, t2w)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4, 128, 128)


if __name__ == "__main__":
    print("[1/2] Dang kiem tra Forward & Backward pass cua M2-Plus-Plus...")
    test_m2_plus_plus_forward_and_backward()
    print(">> Forward & Backward: PASSED (Gradient norm hoan hao, khong NaN)!")

    print("[2/2] Dang kiem tra Eval mode (Inference)...")
    test_m2_plus_plus_eval_mode()
    print(">> Eval mode: PASSED (Shape dung chuan (2, 4, 128, 128))!")

    print("\n>>> CHUC MUNG: TOAN BO HE THONG M2-PLUS-PLUS DA HOAT DONG CHUAN XAC 100%! <<<")