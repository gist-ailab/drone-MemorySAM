"""DEPRECATED shim — 실제 구현은 lora_sam/ 패키지로 이동 (2026-07 재구조화).

구 8.5k줄 메가파일. 기존 `from .sam_lora_image_encoder_seg import *` /
`import ... as seg_module` + getattr 사용처를 무중단 유지하기 위한 re-export.

새 코드는 다음을 사용할 것:
    from .lora_sam import get_model, MODEL_REGISTRY
    from .lora_sam.p09 import LoRA_Sam_P9
"""
from .lora_sam import *  # noqa: F401,F403
from .lora_sam import MODEL_REGISTRY, get_model  # noqa: F401
# 구 메가파일이 노출하던 utils 이름들(transitively re-exported)도 유지
from .sam_lola_utils import *  # noqa: F401,F403
