"""LoRA_Sam_P32 (CoRB) — origin/develop 병합 시 메가파일 append분을 컨벤션에 따라 분리 (2026-07-09)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.sam2_base import SAM2Base
from .p31 import LoRA_Sam_P31


class LoRA_Sam_P32(LoRA_Sam_P31):
    """
    LoRA_Sam_P32: CoRB — Corroboration-Biased Memory Attention (roadmap
    23_seg_arch_proposals_P32 P32-B; Phase 0 검증 = doc 24).

    P28~P31의 RBMA 신뢰도 = self-entropy `1−H(softmax(D_i(f_i)))/logC`는 per-modal
    decoder 용량과 confound → event/LiDAR anti-calibrated (correctness-AUROC .30/.22
    <0.5, doc 16 §7). P32는 RBMA 노벨티(무학습 신호 + memory-attn logit additive bias)를
    그대로 두고 **신호의 의미만 교체**: "이 모달이 스스로 얼마나 확신하나(self-entropy)"
    → "이 모달의 주장이 다른 모달들의 합의와 얼마나 상호검증(corroborate)되나".

    신호형 = **corr_veto** (Phase 0 v2 확정, doc 24 §결과 C — P28/P31 ckpt 둘 다에서
    worst-modality AUROC 최고, 어떤 모달도 anti-calibrated로 남기지 않음):
      p_i        = softmax(D_i(f_i))                            # per-modal posterior (무학습)
      p̄_{−i}     = mean_{j≠i} p_j                               # leave-one-out 합의
      corr_i     = Σ_c √(p_i · p̄_{−i})                          # Bhattacharyya coeff ∈[0,1]
      selfent_i  = 1 − H(p_i)/logC
      g_i        = clamp(selfent_i − max_{j≠i} selfent_j, 0, 1) # unique-info veto gate
      rel_i      = g_i·selfent_i + (1−g_i)·corr_i               # veto blend
      bias_i     = λ·(rel_i − mean_j rel_j)                     # RBMA 배관 그대로 (λ만 학습)

    veto 근거(doc 24): 순수 corroboration은 "다수가 못 보는 곳에서 홀로 confident한
    모달"(P31 depth workhorse)을 합의 불일치로 벌해 AUROC .90→.28 붕괴 → g_i가 그런
    uniquely-confident 모달을 self-confidence 쪽으로 보호(threshold-free). corr_veto가
    depth를 .71로 회복하면서 event/LiDAR도 >.6 유지.

    수식은 tools/eval_reliability_auroc.py의 corr_veto와 동일(무학습 검증본). **temperature-
    free**(진단과 정확 매칭). rbma_calibrate(loss)는 orthogonal 병행 가능(decoder logit
    shaping으로 selfent/veto-gate 품질↑). corroboration_bias=False → P31/P30/P28 byte-identical.
    선행연구 차별: RSGMamba consistency gate=learned MLP; 우리는 무학습 통계 + attention
    logit bias. corr는 P31 consistency_bias(2차 항)를 1차 신호로 승격한 것.
    """
    CORRB_EPS = 1e-6

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
                 unfreeze_last_n_blocks=0, router_reg_mode='diversity',
                 corroboration_bias=False, corrb_veto=True):
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
            rbma_calibrate=rbma_calibrate, consistency_bias=consistency_bias,
            lambda_cons_init=lambda_cons_init, amf_reliability=amf_reliability,
            amf_rel_tau=amf_rel_tau, ctd_multi_scale=ctd_multi_scale, ctd_up=ctd_up,
            ctd_aux_ce=ctd_aux_ce, ctd_aux_only=ctd_aux_only,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks, router_reg_mode=router_reg_mode,
        )
        self.corroboration_bias = corroboration_bias
        self.corrb_veto = corrb_veto

    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        # OFF or single-modality → P31/P30/P28 self-entropy path (byte-identical).
        if not self.corroboration_bias or m < 2:
            return super()._compute_bias_source(
                quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        probs, se = [], []
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W) — temperature-free (matches validated diagnostic)
                C = aux_logits.shape[1]
                p = F.softmax(aux_logits.float(), dim=1)
                ent = -(p * torch.log(p + self.CORRB_EPS)).sum(dim=1, keepdim=True)
                rel = 1.0 - ent / math.log(C)                 # self-entropy reliability (B,1,H,W)
                p = F.interpolate(p, size=(fpn_h, fpn_w), mode='bilinear', align_corners=False)
                p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-6)   # renorm after interp
                rel = F.interpolate(rel, size=(fpn_h, fpn_w), mode='bilinear', align_corners=False)
            probs.append(p)
            se.append(rel)
        with torch.no_grad():
            p_sum = torch.stack(probs, dim=0).sum(dim=0)      # (B,C,hf,wf)
            rel_maps = []
            for i in range(m):
                cons = ((p_sum - probs[i]) / (m - 1)).clamp_min(0)   # leave-one-out consensus
                corr = (probs[i] * cons).clamp_min(0).sqrt().sum(dim=1, keepdim=True)  # BC ∈[0,1]
                if self.corrb_veto:
                    others_max = torch.stack([se[j] for j in range(m) if j != i],
                                             dim=0).amax(dim=0)      # (B,1,hf,wf)
                    g = (se[i] - others_max).clamp(0, 1)             # unique-confidence gate
                    rel_i = g * se[i] + (1.0 - g) * corr
                else:
                    rel_i = corr
                rel_maps.append(rel_i.clamp(0, 1))
            rel_stack = torch.stack(rel_maps, dim=0)          # (m,B,1,hf,wf)
        self._rbma_rel = rel_stack                            # ③ AMF reuse (now corroboration-based)
        bias = rel_stack - rel_stack.mean(dim=0, keepdim=True)   # relative up/down-weight per pixel
        return [bias[i] for i in range(m)]


