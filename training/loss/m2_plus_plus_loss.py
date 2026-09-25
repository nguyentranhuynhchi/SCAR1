"""M2-Plus-Plus Loss: Adaptive Scar & Edema Reweighted Segmentation Loss.
Directly boosts small pathology regions without probability degradation or multi-head conflict.
"""
from __future__ import annotations

import math
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from training.loss.losses import DiceLoss


class M2PlusPlusLoss(nn.Module):
    """Adaptive Pathology-Boosted Loss for M2-Plus-Plus.
    Applies asymmetric class weighting to Cross-Entropy and Soft Dice loss
    to specifically target minority Scar and Edema lesions while preserving Myocardium.
    """

    def __init__(
        self,
        n_classes: int = 4,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        bg_weight: float = 0.2,
        myo_weight: float = 1.0,
        edema_weight: float = 1.5,
        scar_weight: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.n_classes = n_classes
        self.dice = DiceLoss(n_classes)

        # Class weights: [0: background, 1: normal_myocardium, 2: edema, 3: scar]
        weights = [bg_weight, myo_weight, edema_weight, scar_weight]
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

        # 2. Weighted Soft Dice Loss
        dice = self.dice(logits, target_long, weight=w, softmax=True)

        total_loss = self.ce_weight * ce + self.dice_weight * dice
        return {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
        }