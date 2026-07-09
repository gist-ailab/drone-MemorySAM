"""LoRA_Sam_P33 (CG-MoD) — origin/develop 병합 시 메가파일 append분을 컨벤션에 따라 분리 (2026-07-09)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.sam2_base import SAM2Base
from .p32 import LoRA_Sam_P32


class LoRA_Sam_P33(LoRA_Sam_P32):
    """
    LoRA_Sam_P33: Competence-Weighted Fusion + Asymmetric Modality Dropout + Calibration
    (roadmap P33_CGMoD; grounded in doc 24/25 P32-B 검증 + module_diagnostics §D/§E).

    P32(CoRB)는 **memory-attn bias** 신호를 self-entropy → corroboration(corr_veto)으로
    교체해 event/LiDAR anti-calibration을 무학습으로 반전했다. 그러나 P32 검증에서 확인된
    두 잔여 병목은 bias가 아니라 **출력 융합(AMF)** 과 **모달 활용도**에 있었다:
      - F1: AMF uniform → 측정 misallocation 51.6%. depth가 단독 competence 최고(43.7)인데
            등가중 융합이 이를 무시. (module_diagnostics §D)
      - F2: event/LiDAR는 사실상 죽음(competence≈16, drop-ΔmIoU≈0). 모델이 img/depth에만
            의존 → 약모달에서 정보를 추출하도록 강제할 학습 압력 부재. (§E)

    세 가지 config-gated 기구 (전부 OFF → P32와 byte-identical):

    ① [M1] Competence-Weighted Fusion (competence_fusion; AMF_MODE:competence):
       출력 융합 가중치 = softmax_modality(s_i / comp_tau). **s_i = per-modal CALIBRATED
       self-entropy reliability** `1 − H(softmax(D_i(f_i)/T_i))/logC` (M3의 T_i 보정) —
       corr_veto가 아니다. 근거(doc 24/25): reliability-value는 competent depth를 최상위
       (0.94, 단독 competence 43.7과 일치)로 rank하지만 corr_veto는 **죽은 lidar**를 0.85로
       오상향(쉬운 영역에서 합의와 일치하기 때문) → corr_veto로 융합하면 죽은 모달을
       up-weight. 따라서 M1의 융합 신호는 self-entropy이며 P32의 corr_veto(`_rbma_rel`,
       memory-bias source)와 **별도 attribute `_comp_rel`** 에 stash된다.
         - hard top-k(comp_topk>0): 픽셀별 상위 k개 모달만 유지·재정규화.
         - anti-collapse 엔트로피 정칙(comp_entropy_reg>0): 융합 혼합분포 엔트로피를
           보상(−reg·H)해 단일 모달 붕괴 억제. gradient는 comp_entropy_reg>0일 때
           T_i(보정 온도)를 통해서만 흐른다(가중치 자체는 무학습). gate_loss_data
           ['comp_entropy']로 노출. 신호 평탄 시 ~uniform으로 퇴화(softmax 성질).
       특징 융합(m_feat)은 P26 기본(UAMM) 유지 — 출력 융합만 competence로 교체(P31
       amf_reliability 경로와 동일한 설계).

    ② [M2] Asymmetric Modality Dropout (modal_dropout; MODAL_DROPOUT):
       **학습 전용**. 매 step 확률 p(WARMUP_EP epoch에 걸쳐 0→P 선형 warmup)로 TARGETS
       중 한 모달의 입력을 zero로 치환(module_diagnostics의 drop 실험과 동일 zeros_like).
       기본 TARGETS = [img, depth](지배 모달) — event/lidar는 **절대 drop하지 않음**.
       추론 시 전 모달 present(self.training=False → no-op). 목적: 죽은 약모달에서
       정보를 추출하도록 강제(§E competence≈16, drop-Δ≈0 직격).

    ③ [M3] Calibration restore (rbma_calibrate; RBMA_CALIB.ENABLE — P31 재사용):
       per-modal learnable temperature T_i(rbma_log_temp) + correctness-contrastive
       calibration loss(P31._calibration_loss)를 그대로 상속. M1의 `_comp_rel`은 이
       **temperature-보정 self-entropy**를 사용하므로 M3가 M1의 전제(competence-correlated
       reliability)를 공급한다.

    선행연구 차별(P32 상속): memory-attn bias는 무학습 corroboration logit-bias 유지.
    M1의 융합 가중치도 무학습 통계(calibrated self-entropy)이며 learned gate가 아님 —
    ReliFusion/RSGMamba류 learned reliability MLP와 차별.
    competence_fusion=False & modal_dropout=False → P32 byte-identical(조기 super() 반환).
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
                 unfreeze_last_n_blocks=0, router_reg_mode='diversity',
                 corroboration_bias=False, corrb_veto=True,
                 competence_fusion=False, comp_tau=0.25, comp_topk=0, comp_entropy_reg=0.0,
                 modal_dropout=False, modal_dropout_p=0.3,
                 modal_dropout_targets=(0, 1), modal_dropout_warmup_ep=20):
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
            corroboration_bias=corroboration_bias, corrb_veto=corrb_veto,
        )
        # ① M1 competence-weighted fusion
        self.competence_fusion = competence_fusion
        self.comp_tau = comp_tau
        self.comp_topk = int(comp_topk)          # 0 or >=m → top-k 비활성 (전 모달 사용)
        self.comp_entropy_reg = comp_entropy_reg
        self._comp_rel = None                    # calibrated self-entropy stash (m,B,1,hf,wf)
        self._comp_entropy_loss = None           # anti-collapse 정칙 (training)
        # ② M2 asymmetric modality dropout
        self.modal_dropout = modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.modal_dropout_targets = tuple(int(t) for t in modal_dropout_targets)
        self.modal_dropout_warmup_ep = int(modal_dropout_warmup_ep)
        self._last_dropped_modality = None       # 로깅용
        # 학습 스크립트가 매 epoch 설정 (train_sam2_lora_paper.py: hasattr('_current_epoch'))
        self._current_epoch = 0

    # ── M2: 학습 시에만 TARGETS 중 한 모달의 입력을 zero로 치환 (warmup된 확률 p).
    #    event/LiDAR(비-TARGETS)는 절대 drop하지 않음. 추론(self.training=False)=전 모달.
    def _pick_dropout_modality(self, m):
        targets = [t for t in self.modal_dropout_targets if 0 <= t < m]
        if not targets:
            return None
        return targets[int(torch.randint(len(targets), (1,)).item())]

    def _maybe_drop_modality(self, batched_input):
        if not (self.modal_dropout and self.training):
            return batched_input
        m = len(batched_input)
        if m < 2:
            return batched_input
        ep = getattr(self, '_current_epoch', None)
        if ep is None:
            p = self.modal_dropout_p                          # epoch 미상 → full p (학습 중만 도달)
        elif self.modal_dropout_warmup_ep > 0:
            p = self.modal_dropout_p * min(1.0, float(ep) / self.modal_dropout_warmup_ep)
        else:
            p = self.modal_dropout_p
        if float(torch.rand(1).item()) >= p:
            return batched_input
        j = self._pick_dropout_modality(m)
        if j is None:
            return batched_input
        self._last_dropped_modality = j
        new_input = list(batched_input)
        new_input[j] = torch.zeros_like(new_input[j])         # drop = zero 입력 (module_diagnostics 규약)
        return new_input

    # ── M1 stash: memory-bias(P32 corr_veto)와 별도로, 융합용 CALIBRATED self-entropy를
    #    `_comp_rel`에 stash. corr_veto(_rbma_rel)는 P32가 그대로 memory-attn bias로 사용.
    def _stash_comp_rel(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        # comp_entropy_reg>0에서만 T에 grad 허용(무학습 가중치에 anti-collapse 지렛대 제공);
        # 기본은 detach → 순수 무학습 신호(P28/P31 self-entropy와 동일 성질).
        grad_T = bool(self.rbma_calibrate and self.training and self.comp_entropy_reg > 0)
        T = None
        if self.rbma_calibrate:
            T = self.rbma_log_temp.exp().clamp(min=0.05, max=20.0)
            if not grad_T:
                T = T.detach()
        rels = []
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W)
            C = aux_logits.shape[1]
            lg = aux_logits.float().detach()                  # decoder logit은 무학습(detach)
            if T is not None:
                lg = lg / T[i]                                # T만 grad_T일 때 grad 전달
            p = F.softmax(lg, dim=1)
            ent = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)
            rel = 1.0 - ent / math.log(C)                     # calibrated self-entropy reliability
            rel = F.interpolate(rel, size=(fpn_h, fpn_w),
                                mode='bilinear', align_corners=False)
            rels.append(rel)
        self._comp_rel = torch.stack(rels, dim=0)             # (m,B,1,hf,wf)

    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        # memory-attn bias는 P32(corr_veto)/P31/P28 경로 그대로 — competence는 융합만 건드림.
        bias = super()._compute_bias_source(
            quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
        if self.competence_fusion and m >= 2:
            self._stash_comp_rel(quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
        return bias

    # ── M1: competence-weighted 출력 융합 (OFF → super()=P31/P26 기본, byte-identical).
    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        if not self.competence_fusion or self._comp_rel is None or m < 2:
            return super()._fuse_outputs(
                output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)
        rel = self._comp_rel                                  # (m,B,1,hf,wf) calibrated self-entropy
        w = F.softmax(rel / self.comp_tau, dim=0)             # 모달 축 softmax
        if 0 < self.comp_topk < m:
            # 픽셀별 상위 comp_topk개 모달만 유지 → 재정규화 (hard sparsification)
            topk_idx = w.topk(self.comp_topk, dim=0).indices  # (k,B,1,hf,wf)
            keep = torch.zeros_like(w).scatter_(0, topk_idx, 1.0)
            w = w * keep
            w = w / w.sum(dim=0, keepdim=True).clamp(min=1e-6)
        if self.training and self.comp_entropy_reg > 0:
            # anti-collapse: 혼합분포 엔트로피를 보상(−reg·H) → 단일 모달 붕괴 억제.
            H = -(w * (w + 1e-8).log()).sum(dim=0).mean()
            self._comp_entropy_loss = -self.comp_entropy_reg * H
        Bn = w.shape[1]
        hf, wf = w.shape[-2:]
        Hh, Ww = output[0].shape[-2:]
        w_out = F.interpolate(w.reshape(m * Bn, 1, hf, wf), size=(Hh, Ww),
                              mode='bilinear', align_corners=False).reshape(m, Bn, 1, Hh, Ww)
        w_out = w_out / w_out.sum(dim=0, keepdim=True).clamp(min=1e-6)
        m_output = sum(w_out[i] * output[i] for i in range(m))
        m_feat = sum(q_uamm_norm[i] * all_backbone_feats[i] for i in range(m))   # 특징 융합 = P26 기본
        return m_output, m_feat

    def forward(self, batched_input, multimask_output, gt_mask=None):
        self._comp_rel = None
        self._comp_entropy_loss = None
        self._last_dropped_modality = None
        batched_input = self._maybe_drop_modality(batched_input)   # M2 (학습 전용)
        out = super().forward(batched_input, multimask_output, gt_mask)  # P31.forward = M3 cal loss
        if (self.training and isinstance(out, tuple) and len(out) == 3
                and isinstance(out[2], dict) and self._comp_entropy_loss is not None):
            out[2]['comp_entropy'] = self._comp_entropy_loss       # M1 anti-collapse 정칙 노출
        return out


