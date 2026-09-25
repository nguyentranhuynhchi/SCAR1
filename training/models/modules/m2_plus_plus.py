"""M2-Plus-Plus Modules: Dynamic ROI Zoom-in, Deformable Cross-Modal Alignment,
and Hierarchical Residual Decoupled Heads.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DynamicROIExtractor(nn.Module):
    """Stage 1: Differentiable Coarse Myocardium Locator & Dynamic ROI Cropper/Zoomer."""

    def __init__(self, in_channels: int = 64, margin: float = 0.20):
        super().__init__()
        self.margin = margin
        self.coarse_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(
        self,
        cine_feat: torch.Tensor,
        images: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Detect coarse myocardium from CINE features and crop/zoom input images."""
        batch_size, _, orig_h, orig_w = images[0].shape
        coarse_logits = self.coarse_detector(cine_feat)
        coarse_prob = torch.sigmoid(coarse_logits)

        grids = []
        inv_grids = []
        for b in range(batch_size):
            prob = coarse_prob[b, 0]
            mask = prob > 0.35
            coords = torch.nonzero(mask)

            if coords.shape[0] < 16:
                # Fallback to centered conservative crop if no clear myocardium
                y_min, y_max = 0.15 * orig_h, 0.85 * orig_h
                x_min, x_max = 0.15 * orig_w, 0.85 * orig_w
            else:
                y_min, y_max = coords[:, 0].min().float(), coords[:, 0].max().float()
                x_min, x_max = coords[:, 1].min().float(), coords[:, 1].max().float()

            h_box = max(y_max - y_min, 16.0)
            w_box = max(x_max - x_min, 16.0)

            # Apply safe margin padding
            pad_y = h_box * self.margin
            pad_x = w_box * self.margin
            y1 = max(0.0, y_min - pad_y)
            y2 = min(float(orig_h), y_max + pad_y)
            x1 = max(0.0, x_min - pad_x)
            x2 = min(float(orig_w), x_max + pad_x)

            # Normalized coordinates [-1, 1]
            theta = torch.tensor([
                [(x2 - x1) / orig_w, 0.0, (x1 + x2) / orig_w - 1.0],
                [0.0, (y2 - y1) / orig_h, (y1 + y2) / orig_h - 1.0],
            ], dtype=cine_feat.dtype, device=cine_feat.device).unsqueeze(0)

            grid = F.affine_grid(theta, torch.Size([1, 1, orig_h, orig_w]), align_corners=False)
            grids.append(grid)

            # Inverse transform to map fine predictions back to original coordinates
            scale_x = orig_w / max(x2 - x1, 1e-4)
            scale_y = orig_h / max(y2 - y1, 1e-4)
            trans_x = -((x1 + x2) / orig_w - 1.0) * scale_x
            trans_y = -((y1 + y2) / orig_h - 1.0) * scale_y

            theta_inv = torch.tensor([
                [scale_x, 0.0, trans_x],
                [0.0, scale_y, trans_y],
            ], dtype=cine_feat.dtype, device=cine_feat.device).unsqueeze(0)

            inv_grid = F.affine_grid(theta_inv, torch.Size([1, 1, orig_h, orig_w]), align_corners=False)
            inv_grids.append(inv_grid)

        full_grid = torch.cat(grids, dim=0)
        full_inv_grid = torch.cat(inv_grids, dim=0)

        zoomed_images = [
            F.grid_sample(img, full_grid, mode="bilinear", padding_mode="border", align_corners=False)
            for img in images
        ]
        return zoomed_images, full_inv_grid, coarse_logits


class DeformableCrossModalAlignment(nn.Module):
    """Learns dynamic spatial displacement vectors to align LGE & T2w with CINE anatomy."""

    def __init__(self, channels: int):
        super().__init__()
        self.offset_net = nn.Sequential(
            nn.Conv2d(2 * channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, ref_cine: torch.Tensor, target_modality: torch.Tensor) -> torch.Tensor:
        b, c, h, w = target_modality.shape
        cat_feat = torch.cat([ref_cine, target_modality], dim=1)
        offset = self.offset_net(cat_feat) * 0.15  # Max 15% displacement

        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=ref_cine.device, dtype=ref_cine.dtype),
            torch.linspace(-1.0, 1.0, w, device=ref_cine.device, dtype=ref_cine.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        deformed_grid = base_grid + offset.permute(0, 2, 3, 1)

        aligned = F.grid_sample(target_modality, deformed_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return target_modality + self.gamma * aligned


class HierarchicalDecoupledHeads(nn.Module):
    """Bypasses monolithic Softmax by formulating 3 nested anatomical/pathological binary heads.
    
    Inference Subtraction:
      - Class 2 (Scar LGE thực tế): P_scar = P_scar_raw * P_path * P_myo
      - Class 3 (Edema Exclusive thực tế): P_edema = P_path * (1 - P_scar_raw) * P_myo
      - Class 1 (Normal Myocardium): P_normal = P_myo * (1 - P_path)
      - Class 0 (Background): P_bg = 1 - P_myo
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = max(in_channels // 2, 32)
        self.head_myo = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )
        self.head_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )
        self.head_scar = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logit_myo = self.head_myo(x)
        logit_path = self.head_path(x)
        logit_scar = self.head_scar(x)

        p_myo = torch.sigmoid(logit_myo)
        p_path_cond = torch.sigmoid(logit_path)
        p_scar_cond = torch.sigmoid(logit_scar)

        # Hierarchical inclusion
        p_path = p_path_cond * p_myo
        p_scar = p_scar_cond * p_path

        p_edema_exclusive = (p_path - p_scar).clamp(min=0.0)
        p_normal_myo = (p_myo - p_path).clamp(min=0.0)
        p_bg = (1.0 - p_myo).clamp(min=0.0)

        # Canonical format [0:bg, 1:normal_myo, 2:scar_code, 3:edema_code]
        probs = torch.cat([p_bg, p_normal_myo, p_scar, p_edema_exclusive], dim=1)
        eps = 1e-6
        canonical_logits = torch.log(probs.clamp(min=eps))

        hierarchical_dict = {
            "logit_myo": logit_myo,
            "logit_path": logit_path,
            "logit_scar": logit_scar,
            "p_myo": p_myo,
            "p_path": p_path,
            "p_scar": p_scar,
        }
        return canonical_logits, hierarchical_dict


class M2PlusPlusOutput(tuple):
    """Output container for M2PlusPlusNet."""

    def __new__(
        cls,
        canonical_logits: torch.Tensor,
        hierarchical_dict: dict[str, torch.Tensor] | None = None,
        coarse_logits: torch.Tensor | None = None,
    ):
        return super().__new__(cls, (canonical_logits, hierarchical_dict, coarse_logits))

    @property
    def logits(self) -> torch.Tensor:
        return self[0]

    @property
    def hierarchical(self) -> dict[str, torch.Tensor] | None:
        return self[1]

    @property
    def coarse_logits(self) -> torch.Tensor | None:
        return self[2]

    @property
    def shape(self) -> torch.Size:
        return self[0].shape

    def detach(self) -> torch.Tensor:
        return self[0].detach()

    def argmax(self, *args, **kwargs) -> torch.Tensor:
        return self[0].argmax(*args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("logits", "canonical_logits"):
                return self[0]
            elif item in ("hierarchical", "hierarchical_dict"):
                return self[1]
            elif item == "coarse_logits":
                return self[2]
            raise KeyError(f"Invalid M2PlusPlusOutput key {item!r}")
        return super().__getitem__(item)