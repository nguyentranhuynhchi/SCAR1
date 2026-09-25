"""M2-Plus-Plus Modules: Cross-Modal Decoupled Skip Fusion (CMDSF).
Builds upon M2-Plus by extending decoupled attention to high-resolution skip levels
with non-destructive anatomical amplification.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from training.models.modules.m2_plus import M2Plus_Fusion


class DecoupledSkipFusion(nn.Module):
    """Cross-Modal Decoupled Skip Fusion (CMDSF):
    Enhances high-resolution spatial details for Scar (PSIR/LGE) and Edema (T2w)
    guided by CINE cardiac anatomical geometry without zeroing out peripheral lesions.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid_ch = max(channels // 2, 32)
        # CINE guides PSIR (Scar specialist)
        self.gate_scar = nn.Sequential(
            nn.Conv2d(2 * channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        # CINE guides T2W (Edema specialist)
        self.gate_edema = nn.Sequential(
            nn.Conv2d(2 * channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        # Decoupled projection preserving all 3 modalities
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        # Non-destructive amplification: 1.0 + Sigmoid ensures peripheral lesions are NEVER suppressed
        g_scar = 1.0 + self.gate_scar(torch.cat([cine, psir], dim=1))
        g_edema = 1.0 + self.gate_edema(torch.cat([cine, t2w], dim=1))

        psir_enhanced = psir * g_scar
        t2w_enhanced = t2w * g_edema

        fused = self.fuse(torch.cat([cine, psir_enhanced, t2w_enhanced], dim=1))
        return cine + fused  # Stable anatomical residual