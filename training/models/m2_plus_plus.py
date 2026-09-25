"""M2-Plus-Plus: Clean High-Performance Architecture.
Combines M2-Plus Decoupled Bottleneck with Multi-Scale Decoupled Skip Fusion (CMDSF).
No probability cascading damping, standard 4-class output.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict
import torch
from torch import nn

from training.models.cmspa_net import CMSPANet, get_config
from training.models.modules.m2_plus import M2Plus_Fusion
from training.models.modules.m2_plus_plus import DecoupledSkipFusion


class M2PlusPlusNet(CMSPANet):
    """M2-Plus-Plus Network:
    - Bottleneck: Decoupled Cross-Attention (M2Plus_Fusion) separating Scar & Edema streams.
    - Skip Connections: Multi-Scale Decoupled Skip Fusion (CMDSF) amplifying high-res lesion boundaries.
    - Direct Canonical 4-class Output.
    """

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
            raw_config = get_config(ablation="M2-PLUS")
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            raw_config = ConfigDict(config)
        else:
            raw_config = config
        model_config = deepcopy(raw_config)
        model_config.ablation = "M2-PLUS"  # Inherit M2-PLUS base setup safely

        super().__init__(
            config=model_config,
            img_size=img_size,
            num_classes=num_classes,
            zero_head=zero_head,
            vis=vis,
            ablation="M2-PLUS",
            pretrained_path=None,
            **kwargs,
        )

        self.ablation = "M2-PLUS-PLUS"
        self.config.ablation = "M2-PLUS-PLUS"
        self.config.architecture = "m2_plus_plus"

        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width
        expected_skips = [8 * width, 4 * width, width]

        # 1. Bottleneck: M2-Plus Decoupled Cross-Attention (Proven SOTA)
        self.cross_fusion = M2Plus_Fusion(
            channels,
            self.config.fused_channels,
            self.config.cross_attention_heads,
        )

        # 2. Skip Connections: Upgrade Fusion_Embed to DecoupledSkipFusion (CMDSF)
        self.feature_fusion = nn.ModuleList(
            DecoupledSkipFusion(c)
            for c in expected_skips[:self.config.n_skip]
        )

        if pretrained_path is not None:
            self.load_pretrained_encoders(pretrained_path)

    def forward(
        self,
        cine: torch.Tensor,
        psir: torch.Tensor,
        t2w: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(cine, psir, t2w)
        images = [
            img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            for img in (cine, psir, t2w)
        ]

        # Encoders
        cine_f, cine_skips = self.transformer1(images[0])
        psir_f, psir_skips = self.transformer2(images[1])
        t2w_f, t2w_skips = self.transformer3(images[2])

        # Bottleneck: Decoupled Cross-Attention
        fused = self.cross_fusion(
            self.sspanet_cine(cine_f),
            self.sspanet_psir(psir_f),
            self.sspanet_t2w(t2w_f),
        )

        # High-res Skips: Decoupled Skip Fusion (CMDSF)
        skips = [
            fusion(cine_skips[i], psir_skips[i], t2w_skips[i])
            for i, fusion in enumerate(self.feature_fusion)
        ]

        # Decoder & Head
        decoded = self.decoder(fused, skips)
        logits = self.segmentation_head(decoded)
        return logits