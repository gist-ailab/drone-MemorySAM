"""DEPRECATED shim — LoRA_Sam 원본은 lora_sam/base.py로 이동 (2026-07 재구조화).

기존 `from .sam_lora_image_encoder_seg_bkup import LoRA_Sam` import를
무중단 유지하기 위한 re-export. 새 코드는 `from .lora_sam import LoRA_Sam`
(또는 `from .lora_sam.base import LoRA_Sam`)을 사용할 것.

이 파일에 있던 MLP_my / _LoRA_qkv / random_element_swap 사본과 시각화 헬퍼는
각각 modules/ 패키지와 lora_sam/viz.py의 단일 구현으로 통합되었다.
"""
from .lora_sam.base import LoRA_Sam  # noqa: F401
from .lora_sam.viz import (  # noqa: F401
    save_sam2_full_report,
    _denormalize,
    _compute_gpu_pca_single,
    _save_image,
    _save_heatmap,
    _save_pca,
)
from .modules.common import MLP_my, random_element_swap  # noqa: F401
from .modules.moe import _LoRA_qkv  # noqa: F401
