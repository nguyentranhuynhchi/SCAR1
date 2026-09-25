"""M2-Plus-Plus Modules: Anatomy Mask Gating, Deformable Cross-Modal Alignment,
and Hierarchical Residual Decoupled Heads.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SoftMyoGate(nn.Module):
    """Generates a soft anatomical mask from CINE to gate LGE/T2w features."""
    def __init__(self, channels: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cine_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        mask = self.gate_net(cine_feat)
        return target_feat * mask


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
    ):
        return super().__new__(cls, (canonical_logits, hierarchical_dict))

    @property
    def logits(self) -> torch.Tensor:
        return self[0]

    @property
    def hierarchical(self) -> dict[str, torch.Tensor] | None:
        return self[1]

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
            raise KeyError(f"Invalid M2PlusPlusOutput key {item!r}")
        return super().__getitem__(item)