"""P34-ReliaDINO: DINOv3-RBMA reliability-gated multimodal segmentation (card A).

Deliberately NOT imported from semseg.models.__init__ — this package requires
timm, and the SAM2 fleet must stay importable without it. Import explicitly:

    from semseg.models.reliadino import ReliaDINO, build_reliadino
"""
from .encoder import FrozenViTEncoder, MultiModalLoRAQKV, SimpleFPN, LayerNorm2d
from .fusion import (ReliabilityGatedFusion, CrossModalAttentionLayer, AuxDecoder,
                     XAttnTrunk, XAttnTrunkLayer)
from .model import ReliaDINO, FPNSegHead, build_reliadino
from .panoptic_head import MaskClsHead
from .p46 import (ClassLossEMA, EMATeacher, PrototypeBank, RareClassSampler,
                  compute_class_stats, rcs_base_prob)
from .p47 import OGMGE, UniModalBalance, UniModalHead, resolve_modals
# [P49-AIR] .model 다음에 임포트해야 한다 — p49 가 model.FPNSegHead 를 재사용한다.
from .p49 import (P49AIR, P49ViTEncoder, AuxCNNEncoder, AuxStemEncoder,
                  AuxViTLoRAEncoder, Injector, Extractor, build_p49)

__all__ = [
    'FrozenViTEncoder', 'MultiModalLoRAQKV', 'SimpleFPN', 'LayerNorm2d',
    'ReliabilityGatedFusion', 'CrossModalAttentionLayer', 'AuxDecoder',
    'ReliaDINO', 'FPNSegHead', 'build_reliadino', 'MaskClsHead',
    # [A/B trunk] MODEL.FUSION.TRUNK: xattn
    'XAttnTrunk', 'XAttnTrunkLayer',
    # [P46-CTR] class-transfer recovery (학습 전용 3토글)
    'ClassLossEMA', 'EMATeacher', 'PrototypeBank', 'RareClassSampler',
    'compute_class_stats', 'rcs_base_prob',
    # [P47-2] Uni-modal Balance (구 D-2, 학습 전용)
    'UniModalBalance', 'UniModalHead', 'OGMGE', 'resolve_modals',
    # [P49-AIR] 비대칭 주입 + RGB 주경로
    'P49AIR', 'P49ViTEncoder', 'AuxCNNEncoder', 'AuxStemEncoder',
    'AuxViTLoRAEncoder', 'Injector', 'Extractor', 'build_p49',
]
