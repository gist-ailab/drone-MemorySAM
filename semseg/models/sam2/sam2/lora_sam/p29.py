"""LoRA_Sam_P29 (verbatim 이동)."""
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


class LoRA_Sam_P29(LoRA_Sam_P28):
    """
    LoRA_Sam_P29: Self-Derived Condition (SDC) routing for the Soft-MoE LoRA gate.

    계보: P29(P28) — RBMA(P27/P28 memory-attn logit bias)·SoftMoE LoRA·per-modal
    decoder·SQG는 그대로 상속. **변경점은 MoE 게이트의 조건화**뿐:

      - P28 게이트 조건 = modal_embed(modality id)만 → day/night/snow가 라우터에 부재.
      - P29: **이미지에서 무감독 자기파생한 조건 latent z_c**(SelfDerivedCondition)를
        modal_embed과 concat → 게이트에 **FiLM(scale+shift)** 로 주입(cond_mode='film',
        zero-init=identity). z_c는 RGB 입력의 채널 통계 → MLP → prototype bank(K) cosine
        soft-assign(라벨/텍스트 0). label-free clustering loss는 self._sdc_loss로 노출
        (trainer가 total loss에 가산).

    구현 방식(최소 침습): super().__init__로 P28(=add 게이트) 빌드 후, 기존 moe 게이트의
    조건 분기만 'film' + (cond_dim+sdc_latent)로 **사후 재구성**. P28 클래스 자체는 불변.

    deviation 노트: 설계의 "fpn[0] feature" 대신 **RGB 입력 이미지**(in_channels=3)에서
    z_c를 계산해 encoder cyclic-dependency를 회피(게이트가 인코딩 전에 조건 필요).
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='uniform',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8, lambda_bias_init=1.0,
                 sdc_enable=True, sdc_K=6, sdc_latent=32, sdc_in_channels=3):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
        )
        self.sdc_enable = sdc_enable
        self.sdc_latent = sdc_latent
        self._z_c = None
        self._sdc_loss = None
        if sdc_enable:
            self.sdc = SelfDerivedCondition(
                in_channels=sdc_in_channels, latent_dim=sdc_latent, K=sdc_K)
            new_cd = cond_dim + sdc_latent
            # Post-hoc reconfigure existing Soft-MoE gates: add→film, widen cond.
            for layer in list(self.moe_layers_q) + list(self.moe_layers_v):
                layer.cond_mode = 'film'
                layer.cond_dim = new_cd
                if hasattr(layer, 'cond_proj'):
                    del layer.cond_proj
                film = nn.Linear(new_cd, 2 * layer.num_experts)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                layer.cond_film = film

    def _encode_single_modality(self, img, modal_idx_tensor):
        """P26 body + SDC: gate condition = [modal_embed ⊕ z_c]."""
        B = img.shape[0]
        modal_cond = self.modal_embed(modal_idx_tensor).unsqueeze(0).expand(B, -1)
        if self.sdc_enable and self._z_c is not None:
            cond = torch.cat([modal_cond, self._z_c], dim=-1)
        else:
            cond = modal_cond
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer.set_condition(cond)

        emb = self.sam.image_encoder(img)
        use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)
        raw_fpn = []
        if self.multi_scale_sqg and use_hr:
            raw_fpn = [f.clone() for f in emb['backbone_fpn']]
        if use_hr:
            emb["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(emb["backbone_fpn"][0])
            emb["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(emb["backbone_fpn"][1])
        return tuple(emb['backbone_fpn']) + tuple(emb['vision_pos_enc']) + tuple(raw_fpn)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        # Compute the self-derived condition once per image from RGB (modality 0),
        # before the per-modality encoding loop (gates read self._z_c).
        if self.sdc_enable:
            rgb = batched_input[0]                      # (B, 3, H, W)
            z_c, sdc_loss = self.sdc(rgb)
            self._z_c = z_c
            self._sdc_loss = sdc_loss
        else:
            self._z_c = None
            self._sdc_loss = None
        return super().forward(batched_input, multimask_output, gt_mask)
