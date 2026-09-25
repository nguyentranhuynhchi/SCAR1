"""M2-Plus-Plus Loss: Adaptive Scar-Dominant Tversky & CE Loss.
Maximizes Scar (LGE) recall and precision with asymmetric penalty for missed lesions.
"""
from __future__ import annotations

from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


class TverskyLoss(nn.Module):
    """Asymmetric Tversky Loss:
    Penalizes False Negatives (beta) more heavily than False Positives (alpha)
    to boost micro-lesion detection (Scar & Edema).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        probs = inputs.float().softmax(1)
        labels = F.one_hot(target.long(), inputs.shape[1]).movedim(-1, 1).float()
        dims = (0, 2, 3)

        tp = (probs * labels).sum(dims)
        fp = (probs * (1.0 - labels)).sum(dims)
        fn = ((1.0 - probs) * labels).sum(dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky

        if weight is not None:
            w = weight.to(inputs.device)
            return (loss * w).sum() / w.sum()
        return loss.mean()


class M2PlusPlusLoss(nn.Module):
    """Adaptive Scar-Boosted Loss for M2-Plus-Plus.
    Dồn toàn lực kéo nhãn Sẹo LGE (Class 2 trong canonical) và Phù nề viền (Class 3)
    vượt trần SOTA.
    """

    def __init__(
        self,
        n_classes: int = 4,
        ce_weight: float = 0.4,
        tversky_weight: float = 0.6,
        bg_weight: float = 0.2,
        myo_weight: float = 1.0,
        scar_lge_weight: float = 3.0,  # Canonical class 2 = Scar LGE (Target chính)
        edema_rim_weight: float = 2.0,  # Canonical class 3 = Edema Rim
        **kwargs,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = tversky_weight
        self.n_classes = n_classes
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        # Canonical: [0: bg, 1: normal_myo, 2: edema(Scar LGE), 3: scar(Edema rim)]
        weights = [bg_weight, myo_weight, scar_lge_weight, edema_rim_weight]
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: Any, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif isinstance(logits, tuple):
            logits = logits[0]

        target_long = target.long()
        w = self.weights.to(logits.device)

        # 1. Weighted Cross-Entropy Loss
        ce = F.cross_entropy(logits.float(), target_long, weight=w)

        # 2. Weighted Asymmetric Tversky Loss (Focuses gradient on Scar & Edema)
        tversky = self.tversky(logits, target_long, weight=w)

        total_loss = self.ce_weight * ce + self.dice_weight * tversky
        return {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": tversky,
        }