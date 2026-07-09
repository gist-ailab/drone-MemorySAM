"""LoRA_Sam 모델 패키지 — 구 sam_lora_image_encoder_seg.py(8.5k줄)를 버전별로 분리.

- base.py   : LoRA_Sam (원본, 구 sam_lora_image_encoder_seg_bkup.py)
- heads.py  : aux head 5종 + energy/entropy confidence 함수
- viz.py    : save_sam2_full_report 등 시각화 헬퍼
- legacy.py : P1~P7, P10~P21, P23~P26(+AblB) — configs 호환 위해 전부 보존
- p08/p09/p22/p27/p28/p29/p30/p31/p32/p33.py : ACTIVE 버전 개별 파일
- det.py    : P29_Det / P30_Det / P31_Det

모델 클래스 조회는 `get_model(name)` 또는 `MODEL_REGISTRY[name]`을 사용할 것
(train/val 스크립트의 구 `eval(lora_model_name)` 대체).
"""
from .base import LoRA_Sam
from .viz import (
    save_sam2_full_report,
    _denormalize,
    _compute_gpu_pca_single,
    _save_image,
    _save_heatmap,
    _save_pca,
)
from .heads import (
    ConfidenceAuxHead,
    ModalAuxDecoder,
    MultiScaleModalAuxDecoder,
    ResNetAuxBackbone,
    ResNetAuxDecoder,
    compute_energy_confidence,
    compute_spatial_energy_confidence,
    compute_spatial_entropy_confidence,
)
from .legacy import (
    LoRA_Sam_P1,
    LoRA_Sam_P2,
    LoRA_Sam_P3,
    LoRA_Sam_P4,
    LoRA_Sam_P5,
    LoRA_Sam_P6,
    LoRA_Sam_P7,
    LoRA_Sam_P10,
    LoRA_Sam_P11,
    LoRA_Sam_P12,
    LoRA_Sam_P13,
    LoRA_Sam_P14,
    LoRA_Sam_P15,
    LoRA_Sam_P16,
    LoRA_Sam_P17,
    LoRA_Sam_P18,
    LoRA_Sam_P19,
    LoRA_Sam_P20,
    LoRA_Sam_P21,
    LoRA_Sam_P23,
    LoRA_Sam_P24,
    LoRA_Sam_P25,
    LoRA_Sam_P26,
    LoRA_Sam_P26_AblB,
)
from .p08 import LoRA_Sam_P8
from .p09 import LoRA_Sam_P9
from .p22 import LoRA_Sam_P22
from .p27 import LoRA_Sam_P27
from .p28 import LoRA_Sam_P28
from .p29 import LoRA_Sam_P29
from .p30 import LoRA_Sam_P30
from .p31 import LoRA_Sam_P31
from .p32 import LoRA_Sam_P32
from .p33 import LoRA_Sam_P33
from .det import LoRA_Sam_P29_Det, LoRA_Sam_P30_Det, LoRA_Sam_P31_Det

#: 클래스명 문자열 → 클래스. configs의 MODEL.LORA_MODEL 값으로 조회한다.
MODEL_REGISTRY = {
    "LoRA_Sam": LoRA_Sam,
    "LoRA_Sam_P1": LoRA_Sam_P1,
    "LoRA_Sam_P2": LoRA_Sam_P2,
    "LoRA_Sam_P3": LoRA_Sam_P3,
    "LoRA_Sam_P4": LoRA_Sam_P4,
    "LoRA_Sam_P5": LoRA_Sam_P5,
    "LoRA_Sam_P6": LoRA_Sam_P6,
    "LoRA_Sam_P7": LoRA_Sam_P7,
    "LoRA_Sam_P8": LoRA_Sam_P8,
    "LoRA_Sam_P9": LoRA_Sam_P9,
    "LoRA_Sam_P10": LoRA_Sam_P10,
    "LoRA_Sam_P11": LoRA_Sam_P11,
    "LoRA_Sam_P12": LoRA_Sam_P12,
    "LoRA_Sam_P13": LoRA_Sam_P13,
    "LoRA_Sam_P14": LoRA_Sam_P14,
    "LoRA_Sam_P15": LoRA_Sam_P15,
    "LoRA_Sam_P16": LoRA_Sam_P16,
    "LoRA_Sam_P17": LoRA_Sam_P17,
    "LoRA_Sam_P18": LoRA_Sam_P18,
    "LoRA_Sam_P19": LoRA_Sam_P19,
    "LoRA_Sam_P20": LoRA_Sam_P20,
    "LoRA_Sam_P21": LoRA_Sam_P21,
    "LoRA_Sam_P22": LoRA_Sam_P22,
    "LoRA_Sam_P23": LoRA_Sam_P23,
    "LoRA_Sam_P24": LoRA_Sam_P24,
    "LoRA_Sam_P25": LoRA_Sam_P25,
    "LoRA_Sam_P26": LoRA_Sam_P26,
    "LoRA_Sam_P26_AblB": LoRA_Sam_P26_AblB,
    "LoRA_Sam_P27": LoRA_Sam_P27,
    "LoRA_Sam_P28": LoRA_Sam_P28,
    "LoRA_Sam_P29": LoRA_Sam_P29,
    "LoRA_Sam_P29_Det": LoRA_Sam_P29_Det,
    "LoRA_Sam_P30": LoRA_Sam_P30,
    "LoRA_Sam_P30_Det": LoRA_Sam_P30_Det,
    "LoRA_Sam_P31": LoRA_Sam_P31,
    "LoRA_Sam_P31_Det": LoRA_Sam_P31_Det,
    "LoRA_Sam_P32": LoRA_Sam_P32,
    "LoRA_Sam_P33": LoRA_Sam_P33,
    # aux head 클래스 (LORA_MODEL로는 쓰이지 않지만 구 메가파일 40개 클래스 전부 등록)
    "ConfidenceAuxHead": ConfidenceAuxHead,
    "ModalAuxDecoder": ModalAuxDecoder,
    "MultiScaleModalAuxDecoder": MultiScaleModalAuxDecoder,
    "ResNetAuxBackbone": ResNetAuxBackbone,
    "ResNetAuxDecoder": ResNetAuxDecoder,
}


def get_model(name):
    """클래스명 문자열로 LoRA_Sam 모델 클래스를 반환한다.

    구 `eval(lora_model_name)` 패턴의 대체. 등록되지 않은 이름이면
    사용 가능한 이름 목록과 함께 KeyError를 발생시킨다.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        available = "\n  ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown LoRA model name: {name!r}. "
            f"MODEL.LORA_MODEL must be one of:\n  {available}"
        ) from None


__all__ = [
    "MODEL_REGISTRY",
    "get_model",
    "save_sam2_full_report",
    "ConfidenceAuxHead",
    "ModalAuxDecoder",
    "MultiScaleModalAuxDecoder",
    "ResNetAuxBackbone",
    "ResNetAuxDecoder",
    "compute_energy_confidence",
    "compute_spatial_energy_confidence",
    "compute_spatial_entropy_confidence",
] + list(MODEL_REGISTRY.keys())
