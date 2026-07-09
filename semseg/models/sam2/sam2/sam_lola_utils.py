"""DEPRECATED shim — 실제 구현은 modules/ 패키지로 이동 (2026-07 재구조화).

기존 `from .sam_lola_utils import X` import를 무중단 유지하기 위한 re-export.
새 코드는 `from .modules.moe import SoftMoE_LoRA_Layer` 처럼 직접 import할 것.
"""
from .modules import *  # noqa: F401,F403
from .modules import __all__  # noqa: F401
