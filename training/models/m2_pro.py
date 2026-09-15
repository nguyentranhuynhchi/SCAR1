"""M2-Pro Network Architecture: High-Resolution & Anatomy-Guided Scar Attention Network."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict
import torch
from torch import nn

from training.models.cmspa_net import CMSPANet, get_config
from training.models.modules.m2_pro import (
    AGSA_Block,
    DecoupledBottleneckFusion,
    DeepSupervisionAuxHead,
    M2ProOutput,
)


class M2ProNet(CMSPANet):
    """M2-Pro: High-Resolution & Anatomy-Guided Scar Attention Network.

    Key Architectural Enhancements:
      1. Decoupled Bottleneck Fusion: LGE (PSIR) queries CINE keys; T2w queries CINE keys;
         no modality averaging; residual projections on raw pathology streams.
      2. Anatomy-Guided Skip Attention (AGSA): High-resolution CINE soft myocardial mask gating
         LGE features at 1/4 (32x32) and 1/8 (16x16) skip levels with guaranteed gradient floor (0.20).
      3. Detail-Aware Feature Enhancement (DFE): Multiscale dilated conv difference operations
         (Delta F = ReLU(Local - Context)) isolating fine scar boundaries.
      4. Deep Supervision Auxiliary Head: Attached at Decoder Block 1 (1/4 resolution, 32x32)
         predicting edema and scar (2 classes), active during training and bypassed in eval mode.

    Inputs:
      cine, psir, t2w: floating (B, 1, H, W) or (B, 3, H, W) tensors.
    Outputs:
      Training mode: tuple(main_logits, aux_logits) as M2ProOutput where main_logits is (B, 4, H, W)
                     and aux_logits is (B, 2, H/4, W/4).
      Eval mode: main_logits tensor of shape (B, 4, H, W).
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
            raw_config = get_config(ablation="M3")
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            raw_config = ConfigDict(config)
        else:
            raw_config = config
        model_config = deepcopy(raw_config)
        model_config.ablation = (ablation or model_config.get("ablation", "M2-PRO")).upper()

        # Initialize base CMSPANet with backbone encoder/decoder configuration
        super().__init__(
            config=model_config,
            img_size=img_size,
            num_classes=num_classes,
            zero_head=zero_head,
            vis=vis,
            ablation="M3",  # Uses M3 SSPANet backbone weights
            pretrained_path=None,
            **kwargs,
        )

        self.ablation = "M2-PRO"
        self.config.ablation = "M2-PRO"
        self.config.architecture = "m2_pro"

        # Model hyperparameters
        width = int(64 * self.config.resnet.width_factor)
        channels = 16 * width  # 1024 in R50-B16, 512 in testing config
        expected_skips = [8 * width, 4 * width, width]  # [512, 256, 64]
        self.aux_classes = int(self.config.get("aux_classes", 2))
        self.enable_aux_head = bool(self.config.get("enable_aux_head", True))

        # 1. Replace Bottleneck Fusion with Decoupled Cross-Attention (LGE queries CINE)
        heads = self.config.get("cross_attention_heads", 8)
        if channels % heads != 0:
            for h in (8, 4, 2, 1):
                if channels % h == 0:
                    heads = h
                    break
        dropout_rate = 0.1
        if hasattr(self.config, "transformer") and hasattr(self.config.transformer, "get"):
            dropout_rate = float(self.config.transformer.get("dropout_rate", 0.1))
        elif hasattr(self.config, "get"):
            dropout_rate = float(self.config.get("dropout_rate", 0.1))

        self.cross_fusion = DecoupledBottleneckFusion(
            in_channels=channels,
            out_channels=self.config.fused_channels,
            num_heads=heads,
            dropout=dropout_rate,
        )

        # 2. Replace Skip Concat Fusion with Anatomy-Guided Skip Attention (AGSA)
        self.feature_fusion = nn.ModuleList([
            AGSA_Block(channels=c)
            for c in expected_skips[:self.config.n_skip]
        ])

        # 3. Deep Supervision Auxiliary Head attached to Decoder Stage 1 (1/4 resolution, 32x32)
        aux_in_channels = self.config.decoder_channels[1]
        if self.enable_aux_head:
            self.aux_head = DeepSupervisionAuxHead(
                in_channels=aux_in_channels,
                num_classes=self.aux_classes,
                hidden_channels=max(aux_in_channels // 2, 32),
                dropout=0.1,
            )
        else:
            self.aux_head = None

        if zero_head:
            nn.init.zeros_(self.segmentation_head[0].weight)
            nn.init.zeros_(self.segmentation_head[0].bias)
            if self.aux_head is not None:
                nn.init.zeros_(self.aux_head.block[-1].weight)
                nn.init.zeros_(self.aux_head.block[-1].bias)

        if pretrained_path is not None:
            self.load_pretrained_encoders(pretrained_path)

    def forward_decoder(
        self,
        fused: torch.Tensor,
        skips: list[torch.Tensor],
        compute_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Progressively upsample fused features and tap auxiliary head at 1/4 scale."""
        aux_logits = None
        x = fused
        for i, block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < self.decoder.n_skip else None
            x = block(x, skip)
            # Stage i = 1 corresponds to 1/4 resolution (32x32 for 128x128 input)
            if i == 1 and compute_aux and self.aux_head is not None:
                aux_logits = self.aux_head(x)
        return x, aux_logits

    def forward(
        self,
        cine: torch.Tensor,
        psir: torch.Tensor,
        t2w: torch.Tensor,
        return_aux: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_inputs(cine, psir, t2w)

        # 1. Format input channels (repeat 1 -> 3 for ResNetV2 stem)
        images = [image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
                  for image in (cine, psir, t2w)]

        # 2. Independent ResNetV2 Encoders
        cine_f, cine_skips = self.transformer1(images[0])
        psir_f, psir_skips = self.transformer2(images[1])
        t2w_f, t2w_skips = self.transformer3(images[2])

        # 3. SSPANet Bottleneck Enhancement & Decoupled Cross-Attention (LGE queries CINE)
        fused_bottleneck = self.cross_fusion(
            self.sspanet_cine(cine_f),
            self.sspanet_psir(psir_f),
            self.sspanet_t2w(t2w_f),
        )

        # 4. Anatomy-Guided Skip Attention Fusion
        skips = [
            fusion(cine_skips[i], psir_skips[i], t2w_skips[i])
            for i, fusion in enumerate(self.feature_fusion)
        ]

        # 5. Decoder Cascade with Deep Supervision Tap at 1/4 Resolution
        should_compute_aux = (return_aux is True) or (return_aux is None and self.training)
        dec_out, aux_logits = self.forward_decoder(fused_bottleneck, skips, compute_aux=should_compute_aux)

        # 6. Main Segmentation Logits
        main_logits = self.segmentation_head(dec_out)

        # 7. Mode-dependent return
        output = M2ProOutput(main_logits, aux_logits)
        if return_aux is None:
            return output if self.training else main_logits
        elif return_aux:
            return output
        else:
            return main_logits


__all__ = ["M2ProNet"]
