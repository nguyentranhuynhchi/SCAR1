"""M2-Pro Modules: Anatomy-Guided Skip Attention (AGSA), Detail-Aware Feature Enhancement (DFE),
Decoupled Bottleneck Fusion, and Deep Supervision Auxiliary Head.

Designed for high-resolution scar feature preservation, gradient highway integrity,
and decoupled multi-modal cardiac pathology representation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SoftMyoGate(nn.Module):
    """Extracts a continuous soft myocardial spatial attention mask from CINE skip features.

    Architecture:
        Conv3x3(channels -> mid_channels, pad=1, bias=False)
        BatchNorm2d(mid_channels)
        GELU()
        Conv1x1(mid_channels -> 1, bias=True)
        Sigmoid()

    Args:
        channels: Input feature channel count.
        in_channels: Alias for channels.
        reduction: Channel reduction ratio for intermediate bottleneck (default: 4).
    """

    def __init__(
        self,
        channels: int | None = None,
        in_channels: int | None = None,
        reduction: int = 4,
    ):
        super().__init__()
        actual_channels = channels if channels is not None else in_channels
        if actual_channels is None:
            raise ValueError("Either channels or in_channels must be specified.")
        self.channels = int(actual_channels)
        self.in_channels = self.channels
        mid_channels = max(self.channels // reduction, 16)

        self.net = nn.Sequential(
            nn.Conv2d(self.channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, cine: torch.Tensor) -> torch.Tensor:
        """Extract continuous soft myocardial mask from CINE features.

        Args:
            cine: CINE feature tensor of shape (B, C, H, W).

        Returns:
            Soft myocardial mask of shape (B, 1, H, W) bounded strictly in (0, 1).
        """
        return self.net(cine)


class ChannelSEModule(nn.Module):
    """Dual-pooling (GAP + GMP) Channel Squeeze-and-Excitation calibration module.

    Captures both global diffuse background distribution (average pooling) and
    focal peak hyper-enhancement spikes (max pooling) across feature channels.

    Args:
        channels: Input feature channel count.
        reduction: Squeeze-and-excitation reduction ratio (default: 4).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = int(channels)
        mid_channels = max(self.channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(self.channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise recalibration using dual-pooled statistics."""
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        weights = self.fc(avg_pool + max_pool)
        return x * weights


class DFE_Block(nn.Module):
    """Detail-Aware Feature Enhancement (DFE) Module.

    Amplifies subtle scar tissue boundaries and cancels uniform myocardial background
    via multi-scale dilated convolutions (r=1 vs r=2, 4), feature subtraction
    (Delta F = ReLU(Local - Context)), Channel SE calibration, and learnable residual scaling (gamma_dfe).

    Mathematical formulation:
        F_mid = ReLU(BN(Conv1x1(F_in)))
        Local_F = ReLU(BN(DWConv3x3_r1(F_mid)))
        Ctx_r2  = ReLU(BN(DWConv3x3_r2(F_mid)))
        Ctx_r4  = ReLU(BN(DWConv3x3_r4(F_mid)))
        Delta_1 = ReLU(Local_F - Ctx_r2)  [Fine-scale boundary difference]
        Delta_2 = ReLU(Local_F - Ctx_r4)  [Contextual lesion difference]
        Delta_cat = Concat(Delta_1, Delta_2)
        Delta_mod = ChannelSE(Delta_cat)
        Enhanced = BN(Conv1x1(Delta_mod))
        F_out = F_in + gamma_dfe * Enhanced

    Args:
        channels: Input and output feature channel count.
        in_channels: Alias for channels.
        mid_channels: Intermediate bottleneck channels. Defaults to max(channels // 4, 16).
        reduction: Squeeze-and-excitation reduction ratio (default: 4).
        init_gamma: Initial value for learnable residual scaling parameter (default: 0.1).
    """

    def __init__(
        self,
        channels: int | None = None,
        in_channels: int | None = None,
        mid_channels: int | None = None,
        reduction: int = 4,
        init_gamma: float = 0.1,
    ):
        super().__init__()
        actual_channels = channels if channels is not None else in_channels
        if actual_channels is None:
            raise ValueError("Either channels or in_channels must be specified.")
        self.channels = int(actual_channels)
        self.in_channels = self.channels
        self.mid_channels = int(mid_channels) if mid_channels is not None else max(self.channels // 4, 16)

        # 1. Bottleneck input projection (1x1 conv)
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Multi-scale depthwise dilated convolutions
        # Local branch: dilation=1, padding=1
        self.local_conv = nn.Sequential(
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # Contextual branch 1: dilation=2, padding=2
        self.context_r2 = nn.Sequential(
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                groups=self.mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # Contextual branch 2: dilation=4, padding=4
        self.context_r4 = nn.Sequential(
            nn.Conv2d(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                groups=self.mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # 3. Channel calibration via Squeeze-and-Excitation on concatenated differences
        cat_channels = 2 * self.mid_channels
        self.se = ChannelSEModule(cat_channels, reduction=reduction)

        # 4. Output projection back to in_channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(cat_channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels),
        )

        # 5. Learnable residual scaling parameter
        self.gamma_dfe = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

    @property
    def gamma(self) -> nn.Parameter:
        """Alias for gamma_dfe for compatibility."""
        return self.gamma_dfe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass enhancing subtle scar hyper-enhancement."""
        feat = self.in_proj(x)

        # Multi-scale receptive field responses
        local_f = self.local_conv(feat)
        ctx_r2 = self.context_r2(feat)
        ctx_r4 = self.context_r4(feat)

        # Feature subtraction: Delta F = ReLU(Local - Context)
        delta_1 = F.relu(local_f - ctx_r2)
        delta_2 = F.relu(local_f - ctx_r4)

        # Dual-scale concatenation & Channel SE calibration
        delta_cat = torch.cat([delta_1, delta_2], dim=1)
        delta_mod = self.se(delta_cat)

        # Projection and residual blend
        enhanced = self.out_proj(delta_mod)
        return x + self.gamma_dfe * enhanced


class AGSA_Block(nn.Module):
    """Anatomy-Guided Skip Attention (AGSA) module.

    Gates pathology skip features (PSIR/LGE and T2w) using a soft myocardial mask
    derived from CINE, with a convex residual gradient highway (alpha_pass=0.20),
    DFE lesion enhancement on LGE, and multimodal fusion projection.

    Drop-in replacement for Fusion_Embed across all skip levels (64, 256, 512 channels).

    Args:
        channels: Number of input channels per modality.
        out_channels: Number of output channels (defaults to channels).
        in_channels: Alias for channels.
        alpha_pass: Guaranteed gradient floor parameter (default: 0.20).
        use_dfe: Whether to apply DFE lesion enhancement on LGE (default: True).
        reduction: Gate and SE bottleneck reduction ratio (default: 4).
        init_gamma: Initial value for DFE learnable residual scale (default: 0.1).
    """

    def __init__(
        self,
        channels: int | None = None,
        out_channels: int | None = None,
        in_channels: int | None = None,
        alpha_pass: float = 0.20,
        use_dfe: bool = True,
        reduction: int = 4,
        init_gamma: float = 0.1,
    ):
        super().__init__()
        actual_channels = channels if channels is not None else in_channels
        if actual_channels is None:
            raise ValueError("Either channels or in_channels must be specified.")
        self.channels = int(actual_channels)
        self.in_channels = self.channels
        self.out_channels = int(out_channels) if out_channels is not None else self.channels
        self.alpha_pass = float(alpha_pass)
        self.use_dfe = bool(use_dfe)

        # 1. Soft Myocardial Gate from CINE
        self.gate = SoftMyoGate(self.channels, reduction=reduction)

        # 2. Detail-Aware Difference Enhancement on LGE
        if self.use_dfe:
            self.dfe = DFE_Block(self.channels, reduction=reduction, init_gamma=init_gamma)
        else:
            self.dfe = None

        # 3. Multimodal Skip Projection: [CINE, LGE_enhanced, T2w_gated] -> out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(3 * self.channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        cine: torch.Tensor,
        psir: torch.Tensor,
        t2w: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for multi-modal skip connection.

        Args:
            cine: CINE skip tensor of shape (B, C, H, W).
            psir: LGE (PSIR) skip tensor of shape (B, C, H, W).
            t2w: T2w skip tensor of shape (B, C, H, W).

        Returns:
            Fused skip tensor of shape (B, out_channels, H, W).
        """
        # Step 1: Compute continuous soft myocardial gate from CINE
        m_myo = self.gate(cine)  # (B, 1, H, W) in (0, 1)

        # Step 2: Convex residual gradient highway
        # Guarantees d(g_mask)/d(psir) >= alpha_pass > 0 everywhere
        g_mask = self.alpha_pass + (1.0 - self.alpha_pass) * m_myo

        # Step 3: Gate pathology features to suppress non-cardiac false alarms
        psir_gated = psir * g_mask
        t2w_gated = t2w * g_mask

        # Step 4: Detail-Aware Feature Enhancement on LGE
        if self.dfe is not None:
            psir_gated = self.dfe(psir_gated)

        # Step 5: Channel concatenation and skip projection
        fused = torch.cat((cine, psir_gated, t2w_gated), dim=1)
        return self.proj(fused)


class DecoupledBottleneckFusion(nn.Module):
    """M2-Pro Decoupled Cross-Attention Bottleneck Fusion.

    Inverted cross-attention architecture at 1/16 bottleneck:
    - LGE (PSIR) queries CINE anatomical keys and values: Q = LGE, K = V = CINE.
    - T2w (Edema) queries CINE anatomical keys and values: Q = T2w, K = V = CINE.
    - Zero modality averaging is performed between LGE and T2w (remedying M2 50% dilution).
    - Direct residual connections preserve raw pathology signal against gradient starvation.

    Args:
        in_channels: Input channels per modality at bottleneck (e.g. 1024 for R50-B16).
        out_channels: Fused bottleneck channels feeding the decoder (e.g. 512).
        num_heads: Number of attention heads in multihead cross-attention.
        dropout: Dropout rate for cross-attention and residual branches.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Ensure num_heads divides in_channels
        if in_channels % num_heads != 0:
            valid_heads = [h for h in (8, 4, 2, 1) if in_channels % h == 0]
            num_heads = valid_heads[0] if valid_heads else 1

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)

        # Independent Multihead Cross-Attention modules
        self.mha_scar = nn.MultiheadAttention(
            self.in_channels, self.num_heads, dropout=dropout, batch_first=True
        )
        self.mha_edema = nn.MultiheadAttention(
            self.in_channels, self.num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for token stabilization under mixed precision
        self.norm_cine = nn.LayerNorm(self.in_channels)
        self.norm_psir = nn.LayerNorm(self.in_channels)
        self.norm_t2w = nn.LayerNorm(self.in_channels)

        # Residual dropouts
        self.drop_scar = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_edema = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Multimodal fusion projection: [CINE, LGE_attended, T2w_attended] -> out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(3 * self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass executing inverted cross-attention.

        Args:
            cine: CINE bottleneck tensor of shape (B, in_channels, H, W).
            psir: LGE (PSIR) bottleneck tensor of shape (B, in_channels, H, W).
            t2w: T2w bottleneck tensor of shape (B, in_channels, H, W).

        Returns:
            Fused bottleneck tensor of shape (B, out_channels, H, W).
        """
        batch, channels, height, width = cine.shape

        # Token projection & layer normalization: (B, H*W, C)
        q_cine = self.norm_cine(cine.flatten(2).transpose(1, 2))
        q_psir = self.norm_psir(psir.flatten(2).transpose(1, 2))
        q_t2w = self.norm_t2w(t2w.flatten(2).transpose(1, 2))

        # 1. LGE (PSIR) queries CINE keys and values
        attn_scar, _ = self.mha_scar(
            query=q_psir, key=q_cine, value=q_cine, need_weights=False
        )
        attn_scar = (
            self.drop_scar(attn_scar)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        feat_scar = psir + attn_scar  # Residual preservation of raw scar contrast

        # 2. T2w (Edema) queries CINE keys and values
        attn_edema, _ = self.mha_edema(
            query=q_t2w, key=q_cine, value=q_cine, need_weights=False
        )
        attn_edema = (
            self.drop_edema(attn_edema)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        feat_edema = t2w + attn_edema  # Residual preservation of raw edema contrast

        # 3. Concatenate anatomical reference with both attended pathology streams
        fused = torch.cat((cine, feat_scar, feat_edema), dim=1)  # (B, 3*C, H, W)
        return self.proj(fused)  # (B, out_channels, H, W)


class DeepSupervisionAuxHead(nn.Module):
    """Deep Supervision Auxiliary Head attached to Decoder Block at 1/4 resolution (32x32).

    Predicts pathology classes (default 2: edema, scar).
    Provides direct gradient flow into high-resolution skip attention (AGSA + DFE)
    during training mode, and is completely bypassed in eval mode.

    Args:
        in_channels: Channels from Decoder Block at 1/4 resolution (default 128).
        num_classes: Number of auxiliary classes (default 2: edema, scar).
        hidden_channels: Intermediate channel width (default: max(in_channels // 2, 32)).
        dropout: Spatial dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 2,
        hidden_channels: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels) if hidden_channels is not None else max(self.in_channels // 2, 32)

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(self.hidden_channels, self.num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce 1/4 resolution auxiliary pathology logits."""
        return self.block(x)


class M2ProOutput(tuple):
    """Polymorphic output container for M2ProNet.

    Acts as a 2-tuple `(main_logits, aux_logits)` while simultaneously
    supporting dictionary access (`output['logits']`, `output['aux_logits']`),
    attribute access (`output.logits`, `output.aux_logits`),
    and tensor delegating methods (`output.detach()`, `output.argmax()`, `output.shape`).
    """

    def __new__(cls, logits: torch.Tensor, aux_logits: torch.Tensor | None = None):
        return super().__new__(cls, (logits, aux_logits))

    @property
    def logits(self) -> torch.Tensor:
        return self[0]

    @property
    def main_logits(self) -> torch.Tensor:
        """Alias for logits."""
        return self[0]

    @property
    def aux_logits(self) -> torch.Tensor | None:
        return self[1]

    @property
    def shape(self) -> torch.Size:
        return self[0].shape

    def detach(self) -> torch.Tensor:
        """Delegate detach to main logits for trainer compatibility."""
        return self[0].detach()

    def argmax(self, *args, **kwargs) -> torch.Tensor:
        """Delegate argmax to main logits for evaluation metrics."""
        return self[0].argmax(*args, **kwargs)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("logits", "main_logits"):
                return self[0]
            elif item == "aux_logits":
                return self[1]
            raise KeyError(
                f"Invalid M2ProOutput key {item!r}; expected 'logits', 'main_logits', or 'aux_logits'"
            )
        return super().__getitem__(item)

    def get(self, key: str, default=None):
        if key in ("logits", "main_logits"):
            return self[0]
        elif key == "aux_logits":
            return self[1]
        return default

    def keys(self):
        return ["logits", "main_logits", "aux_logits"]

    def values(self):
        return [self[0], self[0], self[1]]

    def items(self):
        return [("logits", self[0]), ("main_logits", self[0]), ("aux_logits", self[1])]

    def __contains__(self, key):
        return key in ("logits", "main_logits", "aux_logits")


__all__ = [
    "SoftMyoGate",
    "ChannelSEModule",
    "DFE_Block",
    "AGSA_Block",
    "DecoupledBottleneckFusion",
    "DeepSupervisionAuxHead",
    "M2ProOutput",
]
