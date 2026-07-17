"""LoRA_Sam_P28 (verbatim 이동)."""
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
from .p27 import LoRA_Sam_P27  # noqa: F401


class LoRA_Sam_P28(LoRA_Sam_P27):
    """
    LoRA_Sam_P28: RBMA — Reliability-Biased Memory Attention.

    P27의 additive memory-attention logit-bias 기구(softmax(QK^T/√d + λ·B)V)를 그대로
    재사용하되, bias 신호 B를 SpatialQualityGating(학습형 예측기, B-2 진단에서
    underfit·평탄·정적붕괴 확인)에서 **per-modality decoder의 training-free
    예측 불확실성**으로 교체한다.

      reliability_i(x) = 1 - H(softmax(D_i(f_i)))(x) / log(C)
        - D_i = per_modal_decoders[i] (모달리티 단독 디코드, memory 융합 이전 → 순환 없음)
        - H   = per-pixel predictive entropy (GT 불필요, 학습형 quality head 불필요)
      bias_i = λ · (reliability_i - mean_j reliability_j)   # 모달리티 간 상대 신뢰도

    설계 근거 (deep-research 2026-06-15):
      - 신규성 축 = B(attention LOGIT additive bias): 선행연구는 모두 feature-multiply /
        attention-output-scale / loss-level (ReliFusion/READ/UTFNet/HyperDUM 등). logit-bias
        전례 0. 신호(A)는 UTFNet/HyperDUM의 *학습 evidential/HD head* 대비 "training-free"가 차별점.
      - B-2 병목 직격: SQG(frozen feature 예측기)를 bias 경로에서 제거, decoder 자체
        예측분포에서 신뢰도 산출 (Target Q=exp(-CE)의 GT-free 버전).

    SQG/per-modal decoder/aux-CE/KL teacher loss는 학습용으로 유지(점진 제거는 ablation).
    추론 시 bias는 오직 decoder 불확실성에서 나옴. λ(self.lambda_bias)만 학습.
    순수 RBMA 평가를 위해 config에서 AMF_MODE: uniform 권장(출력 융합은 등가중).
    """
    RBMA_EPS = 1e-6

    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        rel_maps = []
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W) class logits
                C = aux_logits.shape[1]
                p = F.softmax(aux_logits, dim=1)
                ent = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)  # (B,1,H,W)
                ent = ent / math.log(C)                       # normalize → [0,1]
                rel = 1.0 - ent                               # reliability (high=confident)
                rel = F.interpolate(rel, size=(fpn_h, fpn_w),
                                    mode='bilinear', align_corners=False)
            rel_maps.append(rel)
        # center across modalities per pixel → zero-mean relative up/down-weight
        rel_stack = torch.stack(rel_maps, dim=0)              # (m, B, 1, H, W)
        rel_stack = rel_stack - rel_stack.mean(dim=0, keepdim=True)
        return [rel_stack[i] for i in range(m)]
