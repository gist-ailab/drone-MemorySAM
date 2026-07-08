"""Detection 변형: LoRA_Sam_P29_Det / P30_Det / P31_Det (verbatim 이동)."""
import copy
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as tv_models
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic

from ..modeling.sam2_base import SAM2Base
from ..modules import *  # noqa: F401,F403
from ..modules.moe import _LoRA_qkv, _MoE_LoRA_qkv, _SoftMoE_LoRA_qkv, _MoE_DeBA_BB_qkv  # noqa: F401
from .base import LoRA_Sam  # noqa: F401
from .viz import (save_sam2_full_report, _denormalize, _compute_gpu_pca_single,
                  _save_image, _save_heatmap, _save_pca)  # noqa: F401
from .heads import (ConfidenceAuxHead, ModalAuxDecoder, MultiScaleModalAuxDecoder,
                    ResNetAuxBackbone, ResNetAuxDecoder,
                    compute_energy_confidence, compute_spatial_energy_confidence,
                    compute_spatial_entropy_confidence)  # noqa: F401
from .p28 import LoRA_Sam_P28  # noqa: F401
from .p30 import LoRA_Sam_P30  # noqa: F401
from .p31 import LoRA_Sam_P31  # noqa: F401


class LoRA_Sam_P30_Det(LoRA_Sam_P30):
    """
    LoRA_Sam_P30_Det — P30 segmentation backbone repurposed for object detection.

    계보: `LoRA_Sam_P30_Det(LoRA_Sam_P30)` — RBMA(P27/P28 memory-attn logit bias) +
    SDC(P29 self-derived FiLM gate) + P30의 두 기구(class-token decoder / reliability-
    anchored router)를 **그대로 상속**. P30 seg 의 두 노벨티는 detection 헤드 쪽으로
    번역되며(아래), 백본의 seg-level class_token_decoder / learned_router 는 기본 OFF
    (detection 은 backbone 의 fused seg 출력을 쓰지 않고 detection feature 경로만 구동).

    detection feature 경로는 P27 에서 상속한 `extract_det_features()`:
      fpn0 (32,s4) · fpn1 (64,s8) · mem (256,s16, memory+RBMA) · output (per-modal seg
      logits → training-free reliability 1-H/logC). downstream
      `objdet.models.det_model.MemorySAMDetectorP30` 가 이를 소비:
        ① ReliabilityAnchoredRouter 로 per-modality FPN 융합 (mean 대신 — P30 ② 이식)
        ② Object-Query Decoder 가 융합된 mem(memory-conditioned)에 cross-attend (P30 ① 이식)
        + FCOS dense head (aux, 조기 수렴 안정화)

    백본 fine-tune(indoor domain shift): SAM2 base 는 freeze, LoRA/SQG/per-modal decoder/
    RBMA λ/SDC 는 학습, detection model 이 `sam.memory_attention` 도 unfreeze 가능
    (MemorySAMDetectorP30 `train_memory`).
    """
    pass


class LoRA_Sam_P29_Det(LoRA_Sam_P28):
    """
    LoRA_Sam_P29_Det — P28/RBMA backbone repurposed for object detection (mean-fusion
    baseline). Identical to P28 (SAM2 Hiera-B+ + SoftMoE-LoRA + cross-modal memory
    attention with RBMA logit bias); the only addition is the detection-feature path
    `extract_det_features()` inherited from P27. Used as SEG_MODEL for the P29-Det
    detector (objdet.models.det_model.MemorySAMDetector). Mirrors the same class on the
    worktree-p29-det branch so P29-Det configs resolve on this branch too.
    """
    pass


class LoRA_Sam_P31_Det(LoRA_Sam_P31):
    """
    LoRA_Sam_P31_Det — P31.1 backbone repurposed for object detection.

    계보: `LoRA_Sam_P31_Det(LoRA_Sam_P31)` — RBMA + P31.1의 calibrated reliability
    (per-modal learnable temperature `rbma_log_temp`) + decisive router + backbone
    부분 unfreeze를 그대로 상속. detection feature 경로는 P27에서 상속한
    `extract_det_features()` (fpn0/fpn1/mem + per-modal seg output). downstream
    `MemorySAMDetectorP30`(primary_head='fcos', use_calibrated_reliability=True,
    router_reg_mode='decisive')가 이를 소비한다.

    **calibrated reliability**: detector는 `self.rbma_log_temp`(있으면)를 읽어
    reliability = 1 − H(softmax(seg_output_i / T_i))/logC 로 계산 (P30-Det의 raw
    1−H/logC 대비 event/lidar anti-calibration 수리). T_i는 P31.1 seg 학습에서
    보정되므로 **P31.1 seg 체크포인트 warm-start**(SEG_CHECKPOINT)가 권장된다.

    P31.1 교훈 반영: object-query decoder를 최종 출력으로 쓰지 않음(FCOS primary).
    calibration/CTD aux loss는 seg-GT 의존이라 detection-only 학습에선 비활성
    (backbone은 seg warm-start로 보정 흡수).
    """
    pass
