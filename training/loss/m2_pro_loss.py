"""M2-Pro Extreme Imbalance Loss Implementation.

Addresses severe class imbalance in cardiac MRI scar delineation via:
  a. Class-Weighted Cross-Entropy (Background heavily suppressed, scar/edema amplified)
  b. Scar-Heavy Present Dice Loss (Evaluated strictly on slices with target presence)
  c. Binary One-vs-Rest Focal Loss on Scar (Concentrates gradients on hard positive scar pixels)
  d. Pathology & Anatomy Inclusion Constraint (P(scar) <= P(edema) and P(scar) <= P(myo))
  e. Auxiliary Deep Supervision Loss (BCE + Dice at 1/4 resolution for edema & scar)
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    from training.models.modules.m2_pro import M2ProOutput
except ImportError:
    M2ProOutput = None  # type: ignore


def present_dice(
    prob: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Dice loss evaluated strictly on slices where ground truth target is present.

    Eliminates pathological zero-prediction collapse on negative slices.

    Args:
        prob: (*, H, W) float probabilities in [0, 1].
        target: (*, H, W) binary float ground truth in {0, 1}.
        smooth: Laplace smoothing constant.

    Returns:
        Scalar tensor representing the present dice loss. If no slice in the batch
        contains positive target pixels, returns 0.0 with computation graph preserved.
    """
    dims = tuple(range(1, prob.ndim))
    mass = target.sum(dims)
    intersection = (prob * target).sum(dims)
    union = prob.sum(dims) + mass
    dice = (2.0 * intersection + smooth) / (union + smooth)
    valid = mass > 0
    if valid.any():
        return ((1.0 - dice) * valid.float()).sum() / valid.float().sum()
    return (prob * 0.0).sum()


class M2ProLoss(nn.Module):
    """M2-Pro Extreme Imbalance Loss Module.

    Formulation:
        L_total = lambda_ce * L_ce
                + lambda_dice * L_dice
                + lambda_focal * L_focal_scar
                + lambda_inc * L_inc
                + lambda_aux * L_aux

    Default weights:
        lambda_ce = 1.0, lambda_dice = 1.0, lambda_focal = 1.0,
        lambda_inc = 0.5, lambda_aux = 0.4.

    Args:
        num_classes: Total number of segmentation classes (default: 4).
        ce_weights: Weights per class for cross-entropy (default: [0.1, 1.0, 2.0, 3.0]).
        dice_weights: Weights for foreground classes [1, 2, 3] (default: [0.15, 0.35, 0.50]).
        focal_gamma: Focusing parameter for scar focal loss (default: 2.0).
        focal_alpha: Positive class balance factor for scar focal loss (default: 0.75).
        lambda_ce: Cross-entropy loss multiplier (default: 1.0).
        lambda_dice: Present dice loss multiplier (default: 1.0).
        lambda_focal: Scar focal loss multiplier (default: 1.0).
        lambda_inc: Pathology inclusion loss multiplier (default: 0.5).
        lambda_aux: Auxiliary supervision loss multiplier (default: 0.4).
        smooth: Smoothing epsilon for dice denominators (default: 1e-5).
    """

    requires_aux: bool = True

    def __init__(
        self,
        num_classes: int = 4,
        ce_weights: tuple[float, ...] = (0.1, 1.0, 2.0, 3.0),
        dice_weights: tuple[float, ...] = (0.15, 0.35, 0.50),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        lambda_ce: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_inc: float = 0.5,
        lambda_aux: float = 0.4,
        smooth: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = int(num_classes)

        # Support keyword aliases from various configurations / trainers
        self.lambda_ce = float(kwargs.get("ce_weight", kwargs.get("weight_ce", lambda_ce)))
        self.lambda_dice = float(kwargs.get("dice_weight", kwargs.get("weight_dice", lambda_dice)))
        self.lambda_focal = float(kwargs.get("focal_weight", kwargs.get("weight_focal", lambda_focal)))
        self.lambda_inc = float(
            kwargs.get(
                "inc_weight",
                kwargs.get("weight_inc", kwargs.get("weight_inclusion", kwargs.get("inclusion_weight", lambda_inc))),
            )
        )
        self.lambda_aux = float(kwargs.get("aux_weight", kwargs.get("weight_aux", lambda_aux)))

        self.register_buffer("ce_weights", torch.tensor(ce_weights, dtype=torch.float32))
        self.dice_weights = tuple(float(w) for w in dice_weights)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.smooth = float(smooth)
        self.requires_aux = True

    def forward(
        self,
        logits: torch.Tensor | tuple | list | dict | M2ProOutput,
        targets: torch.Tensor | None = None,
        aux_logits: torch.Tensor | None = None,
        *,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the M2-Pro compound extreme imbalance loss.

        Args:
            logits: Predicted logits tensor of shape (B, 4, H, W), or tuple/list
                (main_logits, aux_logits), or M2ProOutput instance, or dict.
            targets: Ground truth class index tensor of shape (B, H, W) with values in [0, 3].
            aux_logits: Optional auxiliary prediction logits (B, 2, H/4, W/4).
            target: Keyword alias for targets.

        Returns:
            dict containing:
                "loss": scalar total loss tensor with gradient history.
                "ce": scalar class-weighted cross-entropy loss.
                "dice": scalar scar-heavy present dice loss.
                "focal_scar": scalar binary scar focal loss.
                "inclusion": scalar pathology inclusion constraint loss.
                "aux": scalar auxiliary deep supervision loss.
        """
        if targets is None:
            targets = target
        if targets is None:
            raise ValueError("Target tensor must be provided to M2ProLoss forward.")

        # Polymorphic input unwrapping
        main_logits = logits
        extracted_aux = None

        if M2ProOutput is not None and isinstance(logits, M2ProOutput):
            main_logits = logits.main_logits
            extracted_aux = logits.aux_logits
        elif isinstance(logits, (tuple, list)):
            main_logits = logits[0]
            if len(logits) > 1:
                extracted_aux = logits[1]
        elif isinstance(logits, dict):
            main_logits = logits.get("main_logits", logits.get("logits"))
            extracted_aux = logits.get("aux_logits")
        elif hasattr(logits, "main_logits"):
            main_logits = logits.main_logits
            extracted_aux = getattr(logits, "aux_logits", None)
        elif hasattr(logits, "logits"):
            main_logits = logits.logits
            extracted_aux = getattr(logits, "aux_logits", None)

        if aux_logits is None:
            aux_logits = extracted_aux

        if not isinstance(main_logits, torch.Tensor):
            raise TypeError(f"Expected main logits to be a torch.Tensor, got {type(main_logits)}")

        # Enforce float32 for loss reductions under mixed precision (AMP)
        main_logits = main_logits.float()
        targets = torch.as_tensor(targets, device=main_logits.device).long()

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim != 3:
            raise ValueError(f"targets must be 3D (B, H, W), got {targets.shape}")
        if main_logits.ndim != 4:
            raise ValueError(f"logits must be 4D (B, C, H, W), got {main_logits.shape}")
        if targets.shape[0] != main_logits.shape[0] or targets.shape[1:] != main_logits.shape[2:]:
            raise ValueError(
                f"Shape mismatch between logits {main_logits.shape} and targets {targets.shape}"
            )

        # Numerical sanity check
        if not torch.isfinite(main_logits).all():
            raise ValueError("Non-finite values detected in logits")
        if (targets < 0).any() or (targets >= self.num_classes).any():
            raise ValueError(f"Target contains invalid class labels outside [0, {self.num_classes - 1}]")

        # 1. Class-Weighted Cross-Entropy
        ce = F.cross_entropy(main_logits, targets, weight=self.ce_weights.to(main_logits.device))

        # 2. Scar-Heavy Present Dice Loss
        probs = main_logits.softmax(1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice = torch.zeros((), device=main_logits.device, dtype=torch.float32)
        for class_id, weight in zip((1, 2, 3), self.dice_weights):
            dice_c = present_dice(probs[:, class_id], one_hot[:, class_id], smooth=self.smooth)
            dice = dice + weight * dice_c

        # 3. Binary One-vs-Rest Focal Loss on Scar
        p_scar = probs[:, 3].clamp(1e-7, 1.0 - 1e-7)
        y_scar = one_hot[:, 3]
        p_t = torch.where(y_scar == 1.0, p_scar, 1.0 - p_scar)
        alpha_t = torch.where(y_scar == 1.0, self.focal_alpha, 1.0 - self.focal_alpha)
        focal_scar = (-alpha_t * ((1.0 - p_t) ** self.focal_gamma) * torch.log(p_t)).mean()

        # 4. Pathology & Anatomy Inclusion Constraint (P(scar) <= P(edema) and P(scar) <= P(myo))
        p_edema = probs[:, 2]
        p_myo = probs[:, 1]
        inclusion = torch.relu(probs[:, 3] - p_edema).mean() + torch.relu(probs[:, 3] - p_myo).mean()

        # 5. Auxiliary Deep Supervision Loss
        aux = torch.zeros((), device=main_logits.device, dtype=torch.float32)
        if aux_logits is not None:
            aux_logits = aux_logits.float()
            if not torch.isfinite(aux_logits).all():
                raise ValueError("Non-finite values detected in aux_logits")

            num_aux_channels = aux_logits.shape[1]
            if num_aux_channels == 2:
                # Standard M2-Pro aux head: Channel 0 is Edema (2), Channel 1 is Scar (3)
                aux_target_full = one_hot[:, 2:4]
            elif num_aux_channels == self.num_classes:
                aux_target_full = one_hot
            else:
                aux_target_full = one_hot[:, :num_aux_channels]

            aux_target = F.interpolate(aux_target_full, size=aux_logits.shape[2:], mode="area")
            bce_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)

            aux_probs = torch.sigmoid(aux_logits)
            aux_dice_sum = torch.zeros((), device=main_logits.device, dtype=torch.float32)
            for ch in range(num_aux_channels):
                aux_dice_sum = aux_dice_sum + present_dice(
                    aux_probs[:, ch], aux_target[:, ch], smooth=self.smooth
                )
            aux_dice = aux_dice_sum / max(num_aux_channels, 1)
            aux = 0.5 * bce_aux + 0.5 * aux_dice

        # Total combined loss
        total = (
            self.lambda_ce * ce
            + self.lambda_dice * dice
            + self.lambda_focal * focal_scar
            + self.lambda_inc * inclusion
            + self.lambda_aux * aux
        )

        return {
            "loss": total,
            "ce": ce,
            "dice": dice,
            "focal_scar": focal_scar,
            "inclusion": inclusion,
            "aux": aux,
        }


__all__ = ["M2ProLoss", "present_dice"]
