"""Interface contracts, mock oracles, and test fixtures for SCAR M2-Pro E2E tests."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ml_collections import ConfigDict

from training.dataset.data_contract import CLASS_NAMES, CANONICAL_LABEL_ORDER


class ReferenceM2ProNet(nn.Module):
    """Reference contract-compliant model implementing M2ProNet interface specifications.

    Inputs:
        cine: (B, 1, H, W) float32
        psir: (B, 1, H, W) float32 (LGE)
        t2w:  (B, 1, H, W) float32
    Outputs:
        Training: tuple(main_logits, aux_logits) or dict
            main_logits: (B, 4, H, W)
            aux_logits:  (B, 2, H/4, W/4)
        Eval:
            main_logits: (B, 4, H, W)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_cine = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_psir = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_t2w = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.main_head = nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
        # 1/4 resolution auxiliary prediction head for edema (class 2) and scar (class 3)
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, None)),  # resolution matches 1/4 downsampling
            nn.Conv2d(base_channels * 2, 2, kernel_size=1),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor, return_aux: bool | None = None):
        if cine.ndim != 4 or psir.ndim != 4 or t2w.ndim != 4:
            raise ValueError("Expected 4D inputs (B, 1, H, W)")
        if not (cine.shape == psir.shape == t2w.shape):
            raise ValueError("All three input modalities must have identical shapes")
        if cine.shape[1] != 1:
            raise ValueError(f"Expected 1 channel per modality, got {cine.shape[1]}")

        f_cine = self.encoder_cine(cine)
        f_psir = self.encoder_psir(psir)
        f_t2w = self.encoder_t2w(t2w)

        fused = self.fusion(torch.cat([f_cine, f_psir, f_t2w], dim=1))
        main_logits = self.main_head(fused)

        b, _, h, w = cine.shape
        downsampled = F.interpolate(fused, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
        aux_logits = self.aux_head(downsampled)

        # Mode determination:
        # If return_aux is explicitly True, return tuple or dict
        # If return_aux is False, return main_logits
        # If return_aux is None, return tuple when self.training, else main_logits
        should_return_aux = return_aux if return_aux is not None else self.training
        if should_return_aux:
            return main_logits, aux_logits
        return main_logits


class ReferenceM2ProLoss(nn.Module):
    """Reference contract-compliant loss implementing M2ProLoss interface specifications.

    Inputs:
        logits: (B, 4, H, W)
        targets: (B, H, W) canonical labels 0..3
        aux_logits (optional): (B, 2, H/4, W/4)
    Outputs:
        dict: {"loss", "ce", "dice", "focal_scar", "inclusion", "aux"}
    """

    def __init__(
        self,
        ce_weights=(0.1, 1.0, 2.0, 3.0),
        dice_weights=(0.15, 0.35, 0.50),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        aux_weight: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("ce_weights", torch.tensor(ce_weights, dtype=torch.float32))
        self.dice_weights = dice_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.aux_weight = aux_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(logits, (tuple, list)):
            logits, aux_logits = logits[0], (logits[1] if len(logits) > 1 else aux_logits)
        elif isinstance(logits, dict):
            aux_logits = logits.get("aux_logits", aux_logits)
            logits = logits["logits"]

        logits = logits.float()
        targets = targets.long()

        if targets.ndim != 3 or logits.ndim != 4:
            raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
        if targets.shape[0] != logits.shape[0] or targets.shape[1:] != logits.shape[2:]:
            raise ValueError("Spatial dimensions of logits and targets must match")

        b, c, h, w = logits.shape
        # 1. Class-Weighted Cross Entropy (Background suppression)
        ce = F.cross_entropy(logits, targets, weight=self.ce_weights.to(logits.device))

        # 2. Present Dice (evaluates on foreground classes 1, 2, 3)
        probs = logits.softmax(1)
        one_hot = F.one_hot(targets, num_classes=4).permute(0, 3, 1, 2).float()

        dice_total = torch.tensor(0.0, device=logits.device)
        for class_id, weight in zip((1, 2, 3), self.dice_weights):
            p_c = probs[:, class_id]
            y_c = one_hot[:, class_id]
            dims = (1, 2)
            mass = y_c.sum(dims)
            intersect = (p_c * y_c).sum(dims)
            dice_c = (2.0 * intersect + 1e-5) / (p_c.sum(dims) + mass + 1e-5)
            # Present dice: average over nonempty target slices
            valid = mass > 0
            if valid.any():
                loss_c = ((1.0 - dice_c) * valid).sum() / valid.sum().clamp_min(1)
            else:
                loss_c = torch.tensor(0.0, device=logits.device)
            dice_total = dice_total + weight * loss_c

        # 3. Binary Scar Focal Loss (Class 3 one-vs-rest)
        p_scar = probs[:, 3].clamp(1e-6, 1.0 - 1e-6)
        y_scar = one_hot[:, 3]
        focal_weight = torch.where(y_scar == 1, self.focal_alpha, 1.0 - self.focal_alpha)
        p_t = torch.where(y_scar == 1, p_scar, 1.0 - p_scar)
        focal_scar = (-focal_weight * ((1.0 - p_t) ** self.focal_gamma) * torch.log(p_t)).mean()

        # 4. Pathology Inclusion Constraint (Scar <= Edema <= Myo Ring)
        # Penalize if Scar > Edema or Edema > Myo Ring
        # p[:, 3] (scar) vs p[:, 2] (edema) vs p[:, 1] (normal myo)
        # Inclusion penalty: ReLU(p_scar - (p_edema + p_scar)) etc.
        p_myo_ring = probs[:, 1:4].sum(1)
        p_pathology = probs[:, 2:4].sum(1)
        # Scar must be within pathology, pathology must be within myocardial ring
        inclusion_penalty = F.relu(probs[:, 3] - p_pathology).mean() + F.relu(p_pathology - p_myo_ring).mean()

        # 5. Auxiliary Loss (if aux_logits provided)
        aux_loss = torch.tensor(0.0, device=logits.device)
        if aux_logits is not None:
            # Target for aux_head: downsampled edema (2) and scar (3)
            aux_target = F.interpolate(one_hot[:, 2:4], size=aux_logits.shape[2:], mode="area")
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits.float(), aux_target)

        total_loss = ce + dice_total + focal_scar + 0.1 * inclusion_penalty + self.aux_weight * aux_loss

        return {
            "loss": total_loss,
            "ce": ce,
            "dice": dice_total,
            "focal_scar": focal_scar,
            "inclusion": inclusion_penalty,
            "aux": aux_loss,
        }


def get_m2pro_net(img_size: int = 32, num_classes: int = 4):
    """Dynamically discover live M2ProNet or return reference contract model."""
    try:
        mod = importlib.import_module("training.models.m2_pro")
        if hasattr(mod, "M2ProNet"):
            try:
                # Attempt to instantiate with default/testing config if supported
                from training.models.cmspa_net import get_testing
                return mod.M2ProNet(get_testing(), img_size=img_size, num_classes=num_classes), True
            except Exception:
                try:
                    return mod.M2ProNet(img_size=img_size, num_classes=num_classes), True
                except Exception:
                    pass
    except ImportError:
        pass

    # Check model registry
    try:
        from training.models import MODEL_REGISTRY
        if "m2_pro" in MODEL_REGISTRY or "m2pronet" in MODEL_REGISTRY:
            cls_or_fn = MODEL_REGISTRY.get("m2_pro") or MODEL_REGISTRY.get("m2pronet")
            return cls_or_fn(img_size=img_size, num_classes=num_classes), True
    except Exception:
        pass

    # Fall back to interface contract reference implementation
    return ReferenceM2ProNet(in_channels=1, num_classes=num_classes, base_channels=16), False


def get_m2pro_loss():
    """Dynamically discover live M2ProLoss or return reference contract loss."""
    try:
        mod = importlib.import_module("training.loss.m2_pro_loss")
        if hasattr(mod, "M2ProLoss"):
            return mod.M2ProLoss(), True
    except ImportError:
        pass

    try:
        from training.loss import M2ProLoss
        return M2ProLoss(), True
    except (ImportError, AttributeError):
        pass

    # Fall back to interface contract reference implementation
    return ReferenceM2ProLoss(), False


def create_synthetic_volume(h=32, w=32, d=4, seed=42):
    """Create a synthetic 3D multi-modal volume with realistic myocardium, edema, scar."""
    rng = np.random.default_rng(seed)
    # Background images
    cine = rng.normal(0.2, 0.05, size=(h, w, d)).astype(np.float32)
    psir = rng.normal(0.2, 0.05, size=(h, w, d)).astype(np.float32)
    t2w = rng.normal(0.2, 0.05, size=(h, w, d)).astype(np.float32)
    labels = np.zeros((h, w, d), dtype=np.uint8)

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r_outer = min(h, w) // 3
    r_inner = min(h, w) // 5

    # Center myocardial ring across all slices
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring_mask = (dist_from_center >= r_inner) & (dist_from_center <= r_outer)

    for z in range(d):
        # Normal myocardium (class 1)
        labels[:, :, z][ring_mask] = 1
        cine[:, :, z][ring_mask] += 0.4

        # In slice z >= 1, add edema (class 2) in left sector of the ring
        if z >= 1:
            edema_sector = ring_mask & (x < cx)
            labels[:, :, z][edema_sector] = 2
            t2w[:, :, z][edema_sector] += 0.5
            psir[:, :, z][edema_sector] += 0.2

        # In slice z >= 2, add scar (class 3) inside the edema region
        if z >= 2:
            scar_subsector = ring_mask & (x < cx - 2) & (y < cy)
            labels[:, :, z][scar_subsector] = 3
            psir[:, :, z][scar_subsector] += 0.7

    # Clip images to [0, 1]
    cine = np.clip(cine, 0.0, 1.0)
    psir = np.clip(psir, 0.0, 1.0)
    t2w = np.clip(t2w, 0.0, 1.0)

    return {"cine": cine, "psir": psir, "t2w": t2w, "label": labels}
