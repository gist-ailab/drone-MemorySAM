"""LoRA_Sam_P30 (verbatim 이동)."""
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
from .p29 import LoRA_Sam_P29  # noqa: F401


class LoRA_Sam_P30(LoRA_Sam_P29):
    """
    LoRA_Sam_P30: rare-class + dead-modality fix, grounded in the P28 failure analysis
    (condclass: Water/Bridge=0.00, Wall/Other/Dynamic/Ground/TrafficLight ≪; ablation:
    event Δ-0.000 / LiDAR Δ+0.001 = UNUSED while depth Δ-0.224 / RGB Δ-0.097 dominate).
    Inherits P29 (SDC) + RBMA. Two config-gated mechanisms (both OFF by default → P28/P29
    byte-identical):

    ① Class-token decoder (rare-class fix): C learnable class queries cross-attend the fused
       cross-modal memory feature m_feat (all modalities + RBMA bias) → per-class masks.
       Ports the SAM3-RBMA class-collapse break (val 8.49→16.27). Applied post-hoc on the
       grad-attached m_feat returned by super().forward (trains end-to-end). APPROXIMATION:
       light MaskFormer/SAM-style decoder, not surgery on sam_mask_decoder weights.
    ② Reliability-anchored learned router (dead-modality fix): replaces the fixed UAMM scalar
       fusion with a learned (optionally per-class) modality router, anchored by the RBMA
       reliability so it can't collapse to a constant (P10–P27 'gate 상수수렴', ISSUE-002/015).
       Implemented by overriding _fuse_outputs (the P26 hook), using grad-attached per-modal
       feats/outputs. Reliability anchor = 1 − H(softmax(output_i))/logC (training-free).
       Exposes self._router_reg (modality-mixing entropy) for an optional diversity term.
    """
    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='uniform',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8, lambda_bias_init=1.0,
                 sdc_enable=True, sdc_K=6, sdc_latent=32, sdc_in_channels=3,
                 class_token_decoder=False, ctd_dim=128,
                 learned_router=False, router_per_class=False, router_anchor_lambda=1.0,
                 num_classes=25):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
            sdc_enable=sdc_enable, sdc_K=sdc_K, sdc_latent=sdc_latent, sdc_in_channels=sdc_in_channels,
        )
        self.class_token_decoder_enable = class_token_decoder
        self.learned_router_enable = learned_router
        self.router_per_class = router_per_class
        self.num_classes_p30 = num_classes
        self._router_reg = None
        feat_ch = 32  # all_backbone_feats[i] = fpn[0] after conv_s0 (32ch)
        if class_token_decoder:
            self.class_decoder = ClassTokenDecoder(feat_ch, num_classes, dim=ctd_dim)
        if learned_router:
            self.router = ReliabilityAnchoredRouter(
                feat_ch, num_modalities, num_classes=num_classes,
                per_class=router_per_class, anchor_lambda=router_anchor_lambda)

    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        if not self.learned_router_enable:
            return super()._fuse_outputs(output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)
        # Reliability anchor: training-free 1 - normalized entropy of each modality's logits
        rels = []
        for i in range(m):
            p = F.softmax(output[i], dim=1)
            ent = -(p * (p + 1e-8).log()).sum(dim=1, keepdim=True) / math.log(num_classes)
            rels.append(1.0 - ent)                                  # (B,1,H_out,W_out)
        rel = torch.stack(rels, dim=0)                              # (m,B,1,H_out,W_out)
        Bn = rel.shape[1]; hf, wf = all_backbone_feats[0].shape[-2:]
        rel_f = F.interpolate(rel.reshape(m * Bn, 1, *rel.shape[-2:]), size=(hf, wf),
                              mode='bilinear', align_corners=False).reshape(m, Bn, 1, hf, wf)
        w, reg = self.router(all_backbone_feats, rel_f)            # w: (m,B,K,hf,wf)
        self._router_reg = reg
        w_feat = w.mean(dim=2, keepdim=True) if self.router_per_class else w  # (m,B,1,hf,wf)
        m_feat = sum(w_feat[i] * all_backbone_feats[i] for i in range(m))
        Hh, Ww = output[0].shape[-2:]
        w_out = F.interpolate(w.reshape(m * Bn, w.shape[2], hf, wf), size=(Hh, Ww),
                              mode='bilinear', align_corners=False).reshape(m, Bn, w.shape[2], Hh, Ww)
        m_output = sum(w_out[i] * output[i] for i in range(m))     # per-class or broadcast scalar
        return m_output, m_feat

    def forward(self, batched_input, multimask_output, gt_mask=None):
        out = super().forward(batched_input, multimask_output, gt_mask)
        if not self.class_token_decoder_enable:
            return out
        m_output, m_feat = out[0], out[1]
        cls_logits = self.class_decoder(m_feat)                    # (B, num_classes, hf, wf)
        cls_logits = F.interpolate(cls_logits, size=m_output.shape[-2:],
                                   mode='bilinear', align_corners=False)
        return (cls_logits, m_feat) + tuple(out[2:])
