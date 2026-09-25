"""Model architectures and model registry."""
from __future__ import annotations

from functools import partial
from typing import Callable

from torch import nn
from training.models.modules.m2_plus import M2Plus_Fusion
from training.models.backbones.resnet_v2 import PreActBottleneck, ResNetV2, StdConv2d
from training.models.cmspa_net import (
    CONFIGS,
    CMSPANet,
    Embeddings,
    Transformer,
    VisionTransformer,
    get_config,
    get_r50_b16_config,
    get_r50_l16_config,
    get_testing,
)
from training.models.modules.cmspa import CMSPA_Fusion
from training.models.modules.decoder import DecoderCup, SegmentationHead
from training.models.modules.fusion import ConcatFusion, CrossAttention_Fusion, Fusion_Embed
from training.models.modules.sspanet import SSPANet_Block
from training.models.m3_dpf import M3DPF
from training.models.m2_pro import M2ProNet
from training.models.modules.m2_pro import (
    AGSA_Block,
    ChannelSEModule,
    DecoupledBottleneckFusion,
    DeepSupervisionAuxHead,
    DFE_Block,
    M2ProOutput,
    SoftMyoGate,
)
from training.models.m2_plus_plus import M2PlusPlusNet

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "m2_plus_plus": M2PlusPlusNet,
    "m2_plusplus": M2PlusPlusNet,
    "m2++": M2PlusPlusNet,
    "m2_pro": M2ProNet,
    "m2_pro_net": M2ProNet,
    "m2pro": M2ProNet,
    "m3_dpf": M3DPF,
    "cmspa_net": CMSPANet,
    "cmspa": CMSPANet,
    "vision_transformer": CMSPANet,
    "concat_baseline": partial(CMSPANet, ablation="M0"),
    "sspanet_baseline": partial(CMSPANet, ablation="M1"),
    "cross_attn_baseline": partial(CMSPANet, ablation="M2"),
}


def model_from_config(config, **kwargs):
    """Old checkpoints without an architecture field remain CMSPANet."""
    architecture = config.get("architecture", "cmspa_net")
    return build_model(architecture, config=config, **kwargs)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    name = model_name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    expected = {"concat_baseline": "M0", "sspanet_baseline": "M1",
                "cross_attn_baseline": "M2"}.get(name)
    if expected is not None and kwargs.get("ablation", expected).upper() != expected:
        raise ValueError(f"Model {model_name!r} requires ablation {expected}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "M3DPF",
    "model_from_config",
    "CMSPANet",
    "VisionTransformer",
    "CONFIGS",
    "get_config",
    "get_r50_b16_config",
    "get_r50_l16_config",
    "get_testing",
    "Embeddings",
    "Transformer",
    "ResNetV2",
    "StdConv2d",
    "PreActBottleneck",
    "SSPANet_Block",
    "CMSPA_Fusion",
    "ConcatFusion",
    "CrossAttention_Fusion",
    "Fusion_Embed",
    "DecoderCup",
    "SegmentationHead",
    "MODEL_REGISTRY",
    "build_model",
    "M2Plus_Fusion",
    "M2ProNet",
    "AGSA_Block",
    "ChannelSEModule",
    "DecoupledBottleneckFusion",
    "DeepSupervisionAuxHead",
    "DFE_Block",
    "M2ProOutput",
    "SoftMyoGate",
]
