"""M2-Plus-Plus Network Architecture (Anatomy Mask Gating & Hierarchical Decoupled Heads)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict
import torch
from torch import nn

from training.models.cmspa_net import CMSPANet, get_config
from training.models.modules.m2_pro import AGSA_Block, DecoupledBottleneckFusion
from training.models.modules.m2_plus_plus import (
    DeformableCrossModalAlignment,
    SoftMyoGate,
    HierarchicalDecoupledHeads,
    M2PlusPlusOutput,
)


class M2PlusPlusNet(CMSPANet):
    """M2-Plus-Plus: Anatomy Mask Gating & Hierarchical Residual Decoupled Architecture."""

    def __init__(
        self,
        config: ConfigDict | dict[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int | None = None,
        zero_head: bool = False,
        vis: bool = False,
        ablation: str | None = None,
        pretrained_path: str | Path | None = None,
        **kwargs,
    ):
        if config is None:
            raw_config = get_config(ablation="M3")
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            raw_config = ConfigDict(config)
        else:
            raw_config = config
        model_config = deepcopy(raw_config)
        model_config.ablation = (ablation or model_config.get("ablation", "M2-PLUS-PLUS")).upper()

        super().__init__(
            config=model_config,
            img_size=img_size,
            num_classes=num_classes,
            zero_head=zero_head,
            vis=vis,
            ablation="M3",
            pretrained_path=None,
            **kwargs,
        )

        self.ablation = "M2-PLUS-PLUS"
        self.config.ablation = "M2-PLUS-PLUS"
        self.config.architecture = "m2_plus_plus"

        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width
        expected_skips = [8 * width, 4 * width, width]

        # 1. Deformable Cross-Modal Alignments at Skip Levels
        self.align_psir = nn.ModuleList([DeformableCrossModalAlignment(c) for c in expected_skips[:self.config.n_skip]])
        self.align_t2w = nn.ModuleList([DeformableCrossModalAlignment(c) for c in expected_skips[:self.config.n_skip]])

        # 2. Soft Anatomy Gating (Use CINE to mask out non-cardiac noise in LGE/T2w)
        self.gate_psir = nn.ModuleList([SoftMyoGate(c) for c in expected_skips[:self.config.n_skip]])
        self.gate_t2w = nn.ModuleList([SoftMyoGate(c) for c in expected_skips[:self.config.n_skip]])

        # 3. Decoupled Bottleneck Fusion
        heads = self.config.get("cross_attention_heads", 8)
        if channels % heads != 0:
            for h in (8, 4, 2, 1):
                if channels % h == 0:
                    heads = h
                    break
        self.cross_fusion = DecoupledBottleneckFusion(
            in_channels=channels,
            out_channels=self.config.fused_channels,
            num_heads=heads,
            dropout=0.1,
        )

        # 4. Anatomy-Guided Skip Attention
        self.feature_fusion = nn.ModuleList([
            AGSA_Block(channels=c)
            for c in expected_skips[:self.config.n_skip]
        ])

        # 5. Hierarchical Decoupled Prediction Heads
        self.hierarchical_heads = HierarchicalDecoupledHeads(in_channels=self.config.decoder_channels[-1])

        if pretrained_path is not None:
            self.load_pretrained_encoders(pretrained_path)

    def forward(
        self,
        cine: torch.Tensor,
        psir: torch.Tensor,
        t2w: torch.Tensor,
        return_aux: bool | None = None,
    ) -> torch.Tensor | M2PlusPlusOutput:
        self._validate_inputs(cine, psir, t2w)
        orig_images = [img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img for img in (cine, psir, t2w)]

        # --- Stage 1: Feature Extraction ---
        cine_f, cine_skips = self.transformer1(orig_images[0])
        psir_f, psir_skips = self.transformer2(orig_images[1])
        t2w_f, t2w_skips = self.transformer3(orig_images[2])

        # --- Stage 2: Spatial Alignment & Anatomy Gating ---
        aligned_psir_skips = []
        aligned_t2w_skips = []
        for i in range(self.config.n_skip):
            # Deformable alignment
            a_psir = self.align_psir[i](cine_skips[i], psir_skips[i])
            a_t2w = self.align_t2w[i](cine_skips[i], t2w_skips[i])
            
            # Anatomy Gating
            g_psir = self.gate_psir[i](cine_skips[i], a_psir)
            g_t2w = self.gate_t2w[i](cine_skips[i], a_t2w)

            aligned_psir_skips.append(g_psir)
            aligned_t2w_skips.append(g_t2w)

        # --- Stage 3: Bottleneck Decoupled Fusion ---
        fused_bottleneck = self.cross_fusion(
            self.sspanet_cine(cine_f),
            self.sspanet_psir(psir_f),
            self.sspanet_t2w(t2w_f),
        )

        # High-res skip fusion
        skips = [
            fusion(cine_skips[i], aligned_psir_skips[i], aligned_t2w_skips[i])
            for i, fusion in enumerate(self.feature_fusion)
        ]

        # Decoder pass
        dec_out = fused_bottleneck
        for i, block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < self.decoder.n_skip else None
            dec_out = block(dec_out, skip)

        # Hierarchical output
        canonical_logits, hier_dict = self.hierarchical_heads(dec_out)

        output = M2PlusPlusOutput(canonical_logits, hier_dict)
        if return_aux is None:
            return output if self.training else canonical_logits
        return output if return_aux else canonical_logits