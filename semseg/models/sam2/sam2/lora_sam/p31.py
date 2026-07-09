"""LoRA_Sam_P31 (verbatim 이동)."""
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
from .p30 import LoRA_Sam_P30  # noqa: F401


class LoRA_Sam_P31(LoRA_Sam_P30):
    """
    LoRA_Sam_P31: Calibrated Dual-Reliability RBMA + Multi-scale Class-Token Decoding
    (doc 20 P31-Seg core; grounded in the doc 16 §7 module diagnostics).

    확정 진단 (P28/P29 실측):
      - reliability AUROC [img .77, depth .62, event .30, lidar .22] → event/LiDAR는
        anti-calibrated (틀린 곳에서 과확신) → RBMA bias 신호가 geometry 모달에서 무의미.
      - m_feat 단일 저해상(32ch, stride4) 질의 → thin-class(Bridge/Water/Wall) 경계 muffle.
      - AMF uniform → 죽은 event/lidar에 ~45% 질량 낭비.

    네 가지 config-gated 기구 (전부 OFF → P30과 byte-identical):

    ① [Seg-A] RBMA reliability 재보정 (rbma_calibrate):
       - per-modal learnable temperature T_i (rbma_log_temp): reliability = 1 − H(softmax
         (D_i(f_i)/T_i))/logC. _compute_bias_source에 적용 (여전히 training-free 신호;
         T는 스칼라 보정자일 뿐 학습형 quality head가 아님).
       - correctness-contrastive calibration loss (training): 틀린 픽셀 entropy↑ + 맞은
         픽셀 entropy↓ → reliability의 정답 AUROC를 직접 최적화. gate_loss_data
         ['rbma_cal_loss']로 노출 (trainer가 RBMA_CALIB.LAMBDA로 가산). GT는 학습 시에만
         사용 — 추론 시 bias는 여전히 GT-free.
    ② [Seg-B] Consistency 2차 bias (consistency_bias, A 성공 후 조건부 ON):
       Attention = softmax(QKᵀ/√d + λ_ent·B_ent + λ_ent·λ_cons·B_cons).
       B_cons_i = mean_{j≠i} Bhattacharyya(p_i, p_j) — cross-modal 일치도의 training-free
       2번째 additive 항 (RSGMamba의 learned gate와 차별: training-free + pre-softmax
       additive). λ_cons는 학습 스칼라 (hook의 λ_ent=lambda_bias가 전체에 곱해지므로
       유효 계수 = λ_ent·λ_cons — 2-dof로 동일 공간 커버).
    ③ [Seg-A2] Reliability-proportional AMF (amf_reliability; learned_router OFF일 때만):
       출력 융합 가중치 = softmax_modality(reliability/τ). "보정 후에만 전환" 게이트
       (doc 16 §7-2: 미보정 상태에서 비례가중은 악화 위험 → 기본 OFF).
    ④ [Seg-C] Multi-scale HR class-token decoder (ctd_multi_scale):
       self.class_decoder를 ClassTokenDecoderMS로 교체 — simple-FPN {4,8,16,32} 피라미드
       + coarse→fine cross-attend + 학습형 ConvTranspose(×up) 고해상 pixel-embed +
       training-only aux CE head @H/4 (gate_loss_data['ctd_aux_ce']). P30.forward의
       class_decoder(m_feat)+interpolate가 고해상 mask를 투명 처리 (forward 미수정).
       [P31.1] ctd_aux_only=True → CTD 강등: P30의 "cls_logits가 최종 출력 대체"를
       우회하고 최종 출력은 SAM decoder 융합(m_output) 유지, CTD는 학습 시 aux CE
       (gate_loss_data['ctd_seg_ce'])로만 기여(추론 경로 제거). 근거 = P30-seg 실측
       붕괴(Day-Val 49.76/Test 44.10, P29 대비 −13.4/−10.2) + det E0.1(query head 유죄).
       [P31.1 로깅] _calibration_loss가 per-modal reliability AUROC/μ/σ를 stash
       (_last_rel_auroc/_last_rel_stats) → trainer가 epoch마다 tb/wandb 기록
       (doc 20 "AUROC>0.5 선행 게이트"의 측정 구현; σ→0 = 엔트로피 상수화 퇴화 감지).

    학습 레버 (orthogonal, doc 16 ISSUE-008 — frozen-backbone ceiling의 유일한 지렛대):
    ⑤ unfreeze_last_n_blocks: Hiera trunk 마지막 N개 block unfreeze (Bridge/Other
       structural-dead class 직격). optimizer는 UNFREEZE_LR_SCALE로 감쇠 LR 적용.
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
                 num_classes=25,
                 rbma_calibrate=False, consistency_bias=False, lambda_cons_init=0.5,
                 amf_reliability=False, amf_rel_tau=0.25,
                 ctd_multi_scale=False, ctd_up=2, ctd_aux_ce=True, ctd_aux_only=False,
                 unfreeze_last_n_blocks=0, router_reg_mode='diversity'):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
            sdc_enable=sdc_enable, sdc_K=sdc_K, sdc_latent=sdc_latent,
            sdc_in_channels=sdc_in_channels,
            class_token_decoder=class_token_decoder, ctd_dim=ctd_dim,
            learned_router=learned_router, router_per_class=router_per_class,
            router_anchor_lambda=router_anchor_lambda, num_classes=num_classes,
        )
        self.rbma_calibrate = rbma_calibrate
        self.consistency_bias = consistency_bias
        self.amf_reliability = amf_reliability
        self.amf_rel_tau = amf_rel_tau
        self.ctd_multi_scale = ctd_multi_scale
        self.ctd_aux_only = ctd_aux_only
        self._rbma_rel = None            # uncentered reliability stash (m,B,1,hf,wf)
        self._p31_aux_stash = None       # grad-attached per-modal aux logits (training)
        self._last_rel_auroc = None      # [P31 logging] per-modal reliability AUROC (list of m)
        self._last_rel_stats = None      # [P31 logging] (means, stds) of reliability per modality
        if rbma_calibrate:
            self.rbma_log_temp = nn.Parameter(torch.zeros(num_modalities))
        if consistency_bias:
            self.lambda_cons = nn.Parameter(torch.tensor(float(lambda_cons_init)))
        if class_token_decoder and ctd_multi_scale:
            # ④ drop-in swap (P30.forward 호출부는 동일 (feat) 시그니처를 그대로 사용)
            self.class_decoder = ClassTokenDecoderMS(
                32, num_classes, dim=ctd_dim, up=ctd_up, aux_ce=ctd_aux_ce)
        if learned_router and router_reg_mode != 'diversity':
            # ⑥ router 비적응(uniform) 해소: doc 16 §7 — 측정된 융합 가중치가 거의
            # uniform([.27,.28,.23,.23])인데 기존 'diversity' reg(entropy 보상)는 uniform
            # 방향으로 미는 모순 → 'decisive' = per-pixel commit + batch-marginal 다양성.
            self.router.reg_mode = router_reg_mode
        if unfreeze_last_n_blocks > 0:
            # ⑤ P26.__init__이 image_encoder 전체를 freeze한 것을 마지막 N block만 해제.
            # LoRA qkv wrapper 내부의 base linear도 함께 풀림 (의도된 backbone unfreeze).
            blocks = self.sam.image_encoder.trunk.blocks
            for blk in blocks[max(0, len(blocks) - unfreeze_last_n_blocks):]:
                for p_ in blk.parameters():
                    p_.requires_grad = True

    # ── ① aux-decode stash: Phase 2.5의 grad-attached per-modal logits를 붙잡아
    #    calibration loss에 재사용 (재디코딩 없음). _compute_bias_source의 no_grad
    #    호출은 torch.is_grad_enabled()==False라 stash되지 않음.
    def _auxiliary_decode_single(self, decoder, vision_feats, vision_pos_embeds, feat_sizes):
        out = super()._auxiliary_decode_single(decoder, vision_feats, vision_pos_embeds, feat_sizes)
        if self._p31_aux_stash is not None and torch.is_grad_enabled():
            self._p31_aux_stash.append(out)
        return out

    # ── ①/② bias 소스: temperature-보정 reliability (+ 선택적 consistency 2차 항)
    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        if not (self.rbma_calibrate or self.consistency_bias or self.amf_reliability):
            return super()._compute_bias_source(
                quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        rel_maps, prob_maps = [], []
        T = (self.rbma_log_temp.exp().clamp(min=0.05, max=20.0).detach()
             if self.rbma_calibrate else None)
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W)
                C = aux_logits.shape[1]
                lg = aux_logits.float()
                if T is not None:
                    lg = lg / T[i]
                p = F.softmax(lg, dim=1)
                ent = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)
                rel = 1.0 - ent / math.log(C)                 # (B,1,H,W) native res (P28 순서 유지)
                rel = F.interpolate(rel, size=(fpn_h, fpn_w),
                                    mode='bilinear', align_corners=False)
                rel_maps.append(rel)
                if self.consistency_bias:
                    p_f = F.interpolate(p, size=(fpn_h, fpn_w),
                                        mode='bilinear', align_corners=False)
                    prob_maps.append(p_f / p_f.sum(dim=1, keepdim=True).clamp(min=1e-6))
        rel_stack = torch.stack(rel_maps, dim=0)              # (m,B,1,hf,wf)
        self._rbma_rel = rel_stack                            # uncentered stash (③ AMF용)
        bias = rel_stack - rel_stack.mean(dim=0, keepdim=True)
        if self.consistency_bias:
            with torch.no_grad():
                cons_maps = []
                for i in range(m):
                    agree = [(prob_maps[i] * prob_maps[j]).clamp_min(0).sqrt()
                             .sum(dim=1, keepdim=True)
                             for j in range(m) if j != i]     # Bhattacharyya coeff ∈ [0,1]
                    cons_maps.append(torch.stack(agree, dim=0).mean(dim=0))
                cons_stack = torch.stack(cons_maps, dim=0)
                cons_stack = cons_stack - cons_stack.mean(dim=0, keepdim=True)
            # λ_cons 곱은 no_grad 밖 → attention을 통해 λ_cons에 grad 전달
            bias = bias + self.lambda_cons * cons_stack
        return [bias[i] for i in range(m)]

    # ── ③ reliability-proportional AMF (learned_router가 우선; doc 20 A "보정 후 전환")
    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        if (self.learned_router_enable or not self.amf_reliability
                or self._rbma_rel is None):
            return super()._fuse_outputs(
                output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)
        rel = self._rbma_rel                                  # (m,B,1,hf,wf) no-grad
        w = F.softmax(rel / self.amf_rel_tau, dim=0)
        Bn = rel.shape[1]; hf, wf = rel.shape[-2:]
        Hh, Ww = output[0].shape[-2:]
        w_out = F.interpolate(w.reshape(m * Bn, 1, hf, wf), size=(Hh, Ww),
                              mode='bilinear', align_corners=False).reshape(m, Bn, 1, Hh, Ww)
        w_out = w_out / w_out.sum(dim=0, keepdim=True).clamp(min=1e-6)
        m_output = sum(w_out[i] * output[i] for i in range(m))
        m_feat = sum(q_uamm_norm[i] * all_backbone_feats[i] for i in range(m))
        return m_output, m_feat

    # ── ① correctness-contrastive calibration loss (reliability AUROC 직접 최적화).
    #    1/4 해상도에서 계산 (reliability 소비 지점 = fpn 해상도; 메모리 16× 절약).
    def _calibration_loss(self, aux_logits_list, gt_mask):
        gt = gt_mask.long()
        Ht, Wt = max(1, gt.shape[-2] // 4), max(1, gt.shape[-1] // 4)
        gt_ds = F.interpolate(gt.unsqueeze(1).float(), size=(Ht, Wt),
                              mode='nearest').squeeze(1).long()
        valid = gt_ds != 255
        T = self.rbma_log_temp.exp().clamp(min=0.05, max=20.0)
        total = None
        aurocs, rel_mu, rel_sd = [], [], []
        for i, logits in enumerate(aux_logits_list):
            C = logits.shape[1]
            lg = F.interpolate(logits.float(), size=(Ht, Wt),
                               mode='bilinear', align_corners=False) / T[i]
            p = F.softmax(lg, dim=1)
            ent = -(p * (p + 1e-8).log()).sum(dim=1) / math.log(C)   # (B,Ht,Wt) ∈ [0,1]
            pred = lg.argmax(dim=1)
            wrong = (pred != gt_ds) & valid
            correct = (pred == gt_ds) & valid
            l_wrong = ((1.0 - ent) * wrong.float()).sum() / wrong.float().sum().clamp(min=1.0)
            l_correct = (ent * correct.float()).sum() / correct.float().sum().clamp(min=1.0)
            li = l_wrong + l_correct
            total = li if total is None else total + li
            # [P31 logging] doc 20 "AUROC>0.5 선행 게이트"의 측정: reliability(1−H)가
            # 정답을 rank하는지(Mann-Whitney AUROC) + 퇴화 해(엔트로피 상수화) 감지용 μ/σ.
            with torch.no_grad():
                score = (1.0 - ent).detach()[valid].float()
                lab = correct[valid]
                n1 = lab.sum().float(); n0 = lab.numel() - lab.sum().float()
                if n1 > 0 and n0 > 0:
                    order = score.argsort()
                    ranks = torch.zeros_like(score)
                    ranks[order] = torch.arange(1, score.numel() + 1,
                                                device=score.device, dtype=score.dtype)
                    auroc = ((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
                else:
                    auroc = float('nan')
                aurocs.append(auroc)
                rel_mu.append(score.mean().item())
                rel_sd.append(score.std().item())
        self._last_rel_auroc = aurocs
        self._last_rel_stats = (rel_mu, rel_sd)
        return total / len(aux_logits_list)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        self._rbma_rel = None
        self._p31_aux_stash = ([] if (self.training and gt_mask is not None
                                      and self.rbma_calibrate) else None)
        # [P31.1] CTD 강등(ctd_aux_only): P30의 "cls_logits가 최종 출력을 대체" 경로를 우회.
        # 근거: P30-seg 실측 붕괴(Day-Val 49.8/Test 44.1 = P29 대비 −13.4/−10.2) + det
        # E0.1(같은 ckpt에서 query head 0.256 vs FCOS aux 0.431 = head 단독 유죄) →
        # 경량 query decoder가 SAM decoder 출력을 대체하는 구조가 주범 후보.
        # aux-only에서 최종 출력 = SAM decoder 융합(m_output), CTD는 학습 시
        # auxiliary CE(gate_loss_data['ctd_seg_ce'])로만 rare-class gradient를 공급
        # (추론 경로에서 완전 제거 — GOOSE-M2F류 training-only head와 동일 지위).
        bypass = self.ctd_aux_only and self.class_token_decoder_enable
        if bypass:
            self.class_token_decoder_enable = False
        try:
            out = super().forward(batched_input, multimask_output, gt_mask)
        finally:
            if bypass:
                self.class_token_decoder_enable = True
        # 학습 시 반환 = (cls_logits|m_output, m_feat, gate_loss_data)
        if self.training and len(out) == 3 and isinstance(out[2], dict):
            gate_loss_data = out[2]
            if self.rbma_calibrate and self._p31_aux_stash:
                gate_loss_data['rbma_cal_loss'] = self._calibration_loss(
                    self._p31_aux_stash, gt_mask)
            if bypass and gt_mask is not None:
                cls_logits = self.class_decoder(out[1])          # grad-attached m_feat
                cls_logits = F.interpolate(cls_logits.float(), size=gt_mask.shape[-2:],
                                           mode='bilinear', align_corners=False)
                gate_loss_data['ctd_seg_ce'] = F.cross_entropy(
                    cls_logits, gt_mask.long(), ignore_index=255)
            aux_logits = getattr(self.class_decoder, 'last_aux_logits', None) \
                if self.class_token_decoder_enable else None
            if self.ctd_multi_scale and aux_logits is not None and gt_mask is not None:
                al = F.interpolate(aux_logits.float(), size=gt_mask.shape[-2:],
                                   mode='bilinear', align_corners=False)
                gate_loss_data['ctd_aux_ce'] = F.cross_entropy(
                    al, gt_mask.long(), ignore_index=255)
        self._p31_aux_stash = None
        return out
