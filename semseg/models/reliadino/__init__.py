"""P34-ReliaDINO: DINOv3-RBMA reliability-gated multimodal segmentation (card A).

Deliberately NOT imported from semseg.models.__init__ — this package requires
timm, and the SAM2 fleet must stay importable without it. Import explicitly:

    from semseg.models.reliadino import ReliaDINO, build_reliadino
"""
from .encoder import FrozenViTEncoder, MultiModalLoRAQKV, SimpleFPN, LayerNorm2d
from .fusion import ReliabilityGatedFusion, CrossModalAttentionLayer, AuxDecoder
from .model import ReliaDINO, FPNSegHead, build_reliadino
from .panoptic_head import MaskClsHead

__all__ = [
    'FrozenViTEncoder', 'MultiModalLoRAQKV', 'SimpleFPN', 'LayerNorm2d',
    'ReliabilityGatedFusion', 'CrossModalAttentionLayer', 'AuxDecoder',
    'ReliaDINO', 'FPNSegHead', 'build_reliadino', 'MaskClsHead',
]
