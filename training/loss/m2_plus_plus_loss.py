"""M2-Plus-Plus Hierarchical Residual Loss Implementation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any

from training.loss.m2_pro_loss import present_dice


class M2PlusPlusLoss(nn.Module):
    """Hierarchical Decomposition Loss tailored for extreme lesion contrast differences."""

    def __init__(
        self,
        weight_myo: float = 1.0,
        weight_path: float = 1.5,
        weight_scar: float = 2.5,
        weight_penalty: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.w_myo = weight_myo
        self.w_path = weight_path
        self.w_scar = weight_scar
        self.w_pen = weight_penalty
        self.requires_aux = True

    def forward(
        self,
        output: Any,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if hasattr(output, "hierarchical") and output.hierarchical is not None:
            hier = output.hierarchical
        elif isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], dict):
            hier = output[1]
        else:
            # Fallback for standard logits
            canonical = output.logits if hasattr(output, "logits") else output
            ce = F.cross_entropy(canonical, target.long())
            p = F.softmax(canonical, dim=1)
            d = 1.0 - present_dice(p[:, 1:], F.one_hot(target.long(), num_classes=4).permute(0, 3, 1, 2)[:, 1:].float())
            return {"loss": ce + d, "ce": ce, "dice_loss": d}

        target_long = target.long()
        y_myo = (target_long >= 1).float().unsqueeze(1)
        y_path = ((target_long == 2) | (target_long == 3)).float().unsqueeze(1)
        y_scar = (target_long == 2).float().unsqueeze(1)  # Class 2 thực tế là Scar

        # Myocardium branch loss
        loss_myo_bce = F.binary_cross_entropy_with_logits(hier["logit_myo"], y_myo)
        loss_myo_dice = present_dice(hier["p_myo"], y_myo)
        loss_myo = loss_myo_bce + loss_myo_dice

        # Pathology branch loss
        loss_path_bce = F.binary_cross_entropy_with_logits(hier["logit_path"], y_path)
        loss_path_dice = present_dice(hier["p_path"], y_path)
        loss_path = loss_path_bce + loss_path_dice

        # Scar branch loss (Focal + Scar-heavy Dice)
        p_s = hier["p_scar"]
        pt = torch.where(y_scar == 1.0, p_s, 1.0 - p_s).clamp(min=1e-6, max=1.0 - 1e-6)
        focal_weight = (1.0 - pt) ** 2.0
        bce_s = F.binary_cross_entropy_with_logits(hier["logit_scar"], y_scar, reduction="none")
        loss_scar_focal = (focal_weight * bce_s).mean()
        loss_scar_dice = present_dice(p_s, y_scar)
        loss_scar = loss_scar_focal + 1.5 * loss_scar_dice

        # Hierarchical Inclusion Penalty: P_scar <= P_path <= P_myo
        pen_scar = F.relu(hier["p_scar"] - hier["p_path"]).mean()
        pen_path = F.relu(hier["p_path"] - hier["p_myo"]).mean()
        loss_penalty = pen_scar + pen_path

        total_loss = (
            self.w_myo * loss_myo
            + self.w_path * loss_path
            + self.w_scar * loss_scar
            + self.w_pen * loss_penalty
        )

        return {
            "loss": total_loss,
            "ce": loss_myo_bce + loss_path_bce,
            "dice_loss": loss_myo_dice + loss_scar_dice,
            "scar_loss": loss_scar,
            "penalty": loss_penalty,
        }