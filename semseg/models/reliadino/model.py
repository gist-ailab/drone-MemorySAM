"""P34-ReliaDINO — DINOv3-RBMA multimodal semantic segmentation (card A).

ReliaDINO = FrozenViTEncoder (shared frozen DINOv3/DINOv2 ViT, per-modality
LoRA) -> ReliabilityGatedFusion (cross-modal memory-style attention, RBMA-v2
bias + competence gate) -> SimpleFPN on the FUSED stride-16 map -> light
query-free conv head to num_classes at stride 4 (Mask2Former-lite is a staged
TODO — speed first; the head is deliberately dumb so backbone/fusion effects
are readable).

Trainer contract (compatible with the P24+ gate_loss_data pattern):
  training : forward(inputs, multimask_output, gt_mask) ->
             (logits(B,K,H,W), m_feat(B,C,H/4,W/4), gate_loss_data: dict)
  eval     : forward(inputs, multimask_output) -> (logits, m_feat)
so `output, _ = model(images, True)` (val_mm_sam.evaluate) works unchanged.
No SAM2 imports anywhere in this package.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classtoken import ClassTokenLiteHead
from .encoder import FrozenViTEncoder, SimpleFPN, LayerNorm2d
from .fusion import ReliabilityGatedFusion
from .m2f_head import MaskQueryLiteHead


class FPNSegHead(nn.Module):
    """Light query-free head: upsample every pyramid level to stride 4, sum,
    two 3x3 conv blocks, 1x1 classifier. (GOOSE/SemanticFPN-style.)"""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )
        self.cls = nn.Conv2d(dim, num_classes, 1)

    def forward(self, pyramid: List[torch.Tensor]):
        tgt = pyramid[0].shape[-2:]                       # stride-4 level
        x = pyramid[0]
        for p in pyramid[1:]:
            x = x + F.interpolate(p, size=tgt, mode='bilinear', align_corners=False)
        feat = self.fuse(x)                               # (B, dim, H/4, W/4)
        return self.cls(feat), feat


class ReliaDINO(nn.Module):

    def __init__(self,
                 num_classes: int = 25,
                 modalities: Sequence[str] = ('img', 'depth', 'event', 'lidar'),
                 backbone: str = 'vit_large_patch16_dinov3',
                 backbone_fallback: str = 'vit_large_patch14_reg4_dinov2',
                 pretrained: bool = True,
                 img_size: int = 1024,
                 lora_r: int = 8,
                 lora_alpha: Optional[float] = None,
                 fpn_dim: int = 256,
                 fusion_layers: int = 2,
                 fusion_heads: int = 8,
                 fusion_mlp_ratio: float = 4.0,
                 aux_hidden: int = 256,
                 attn_bias: bool = True,
                 lambda1_init: float = 1.0,
                 consistency_bias: bool = True,
                 lambda2_init: float = 0.5,
                 gate_enable: bool = True,
                 gate_tau: float = 0.25,
                 gate_entropy_reg: float = 0.0,
                 gate_entropy_floor: float = 0.5,
                 veto_floor: bool = True,
                 veto_thresh: float = 0.10,
                 veto_cap: float = 0.05,
                 calibrate: bool = True,
                 router_enable: bool = False,
                 router_anchor_lambda: float = 1.0,
                 router_reg_mode: str = 'decisive',
                 router_reg_lambda: float = 0.01,
                 router_alpha_init: float = 0.0,
                 router_hidden: int = 64,
                 cefr_enable: bool = False,
                 cefr_hidden: int = 64,
                 cefr_morph_init: float = -4.0,
                 cefr_anchor_posterior: bool = True,
                 cefr_lambda1: float = 1.0,
                 cefr_lambda2_target: float = 0.5,
                 cefr_lambda2_warmup_ep: int = 10,
                 cefr_reg_lambda: float = 0.01,
                 cefr_entropy_floor: float = 0.5,
                 cefr_hinge_reg: float = 1.0,
                 class_token_enable: bool = False,
                 class_token_layers: int = 3,
                 class_token_dim: int = 256,
                 class_token_heads: int = 8,
                 class_token_mlp_ratio: float = 2.0,
                 class_token_beta_init: float = 0.0,
                 m2f_enable: bool = False,
                 m2f_num_queries: int = 100,
                 m2f_num_layers: int = 6,
                 m2f_dim: int = 256,
                 m2f_num_heads: int = 8,
                 m2f_mlp_ratio: float = 2.0,
                 m2f_beta_init: float = 0.0,
                 m2f_w_cls: float = 2.0,
                 m2f_w_bce: float = 5.0,
                 m2f_w_dice: float = 5.0,
                 m2f_no_obj_w: float = 0.1,
                 m2f_points: int = 12544,
                 m2f_deep_supervision: bool = True,
                 m2f_loss_w: float = 0.5,
                 m2f_src_modal: bool = False,
                 m2f_anchored: bool = False,
                 m2f_point_quota: int = 0,
                 p39_trunk_exp: bool = False,
                 p39_arbiter: bool = False,
                 p39_path_dropout_p: float = 0.25,
                 p39_router_ce_w: float = 0.4,
                 modal_dropout: bool = False,
                 modal_dropout_p: float = 0.3,
                 modal_dropout_targets: Sequence[int] = (0, 1),
                 modal_dropout_warmup_ep: int = 20):
        super().__init__()
        self.modalities = list(modalities)
        self.num_modalities = len(self.modalities)
        self.num_classes = num_classes

        self.encoder = FrozenViTEncoder(
            backbone=backbone, fallback=backbone_fallback, pretrained=pretrained,
            img_size=img_size, num_modalities=self.num_modalities,
            lora_r=lora_r, lora_alpha=lora_alpha)
        dim = self.encoder.embed_dim
        self.fusion = ReliabilityGatedFusion(
            dim=dim, num_classes=num_classes, num_modalities=self.num_modalities,
            num_layers=fusion_layers, num_heads=fusion_heads,
            mlp_ratio=fusion_mlp_ratio, aux_hidden=aux_hidden,
            attn_bias=attn_bias, lambda1_init=lambda1_init,
            consistency_bias=consistency_bias, lambda2_init=lambda2_init,
            gate_enable=gate_enable, gate_tau=gate_tau,
            gate_entropy_reg=gate_entropy_reg, gate_entropy_floor=gate_entropy_floor,
            veto_floor=veto_floor, veto_thresh=veto_thresh, veto_cap=veto_cap,
            calibrate=calibrate,
            router_enable=router_enable, router_anchor_lambda=router_anchor_lambda,
            router_reg_mode=router_reg_mode, router_reg_lambda=router_reg_lambda,
            router_alpha_init=router_alpha_init, router_hidden=router_hidden,
            cefr_enable=cefr_enable, cefr_hidden=cefr_hidden,
            cefr_morph_init=cefr_morph_init,
            cefr_anchor_posterior=cefr_anchor_posterior,
            cefr_lambda1=cefr_lambda1, cefr_lambda2_target=cefr_lambda2_target,
            cefr_lambda2_warmup_ep=cefr_lambda2_warmup_ep,
            cefr_reg_lambda=cefr_reg_lambda,
            cefr_entropy_floor=cefr_entropy_floor,
            cefr_hinge_reg=cefr_hinge_reg)
        self.fpn = SimpleFPN(dim, fpn_dim)
        self.head = FPNSegHead(fpn_dim, num_classes)
        # [P36-Det] Router->detection seam. The seg path adds routed_logits to the
        # head logits, which the detection path (extract_det_pyramid) never sees, so
        # without this the router is dead weight for det. Project routed_logits
        # (num_classes ch) to fpn_dim + zero-init alpha residual on every pyramid
        # level => at init identical to router-off, comparable to P35-Det.
        if router_enable:
            self.det_router_proj = nn.Conv2d(num_classes, fpn_dim, 1)
            nn.init.zeros_(self.det_router_proj.weight)
            nn.init.zeros_(self.det_router_proj.bias)
            self.det_router_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.det_router_proj = None
            self.det_router_alpha = None

        # [P37b] ClassToken-lite-Learned auxiliary head (config-gated, default
        # OFF -> byte-identical to P36). Learned class tokens over the gated
        # fused stride-16 map; residual scale beta is zero-init (collapse-safe).
        self.classtoken = None
        if class_token_enable:
            self.classtoken = ClassTokenLiteHead(
                dim=dim, fpn_dim=fpn_dim, num_classes=num_classes,
                num_layers=class_token_layers, dim_t=class_token_dim,
                num_heads=class_token_heads, mlp_ratio=class_token_mlp_ratio,
                beta_init=class_token_beta_init)

        # [P37b-Det] ClassToken->detection seam (mirror of the P36 router seam). The
        # seg path adds token_logits to the head logits, which detection never sees.
        # Project token_logits (num_classes ch) to fpn_dim + zero-init alpha residual
        # on every pyramid level => at init identical to class-token-off (= P36-Det).
        if class_token_enable:
            self.det_classtoken_proj = nn.Conv2d(num_classes, fpn_dim, 1)
            nn.init.zeros_(self.det_classtoken_proj.weight)
            nn.init.zeros_(self.det_classtoken_proj.bias)
            self.det_classtoken_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.det_classtoken_proj = None
            self.det_classtoken_alpha = None

        # [P37b] ClassToken-lite-Learned auxiliary head (config-gated, default
        # OFF -> byte-identical to P36). Learned class tokens over the gated
        # fused stride-16 map; residual scale beta is zero-init (collapse-safe).
        self.classtoken = None
        if class_token_enable:
            self.classtoken = ClassTokenLiteHead(
                dim=dim, fpn_dim=fpn_dim, num_classes=num_classes,
                num_layers=class_token_layers, dim_t=class_token_dim,
                num_heads=class_token_heads, mlp_ratio=class_token_mlp_ratio,
                beta_init=class_token_beta_init)

        # [P38] Mask2Former-lite query head (config-gated, default OFF ->
        # byte-identical to P36). Learned queries + Hungarian mask-cls losses
        # give a panoptic-capable branch (MUSES PQ); its semantic scores are
        # merged collapse-safe via the ZERO-INIT beta residual.
        self.m2f = None
        self.m2f_loss_w = float(m2f_loss_w)
        if m2f_enable:
            self.m2f = MaskQueryLiteHead(
                dim=dim, fpn_dim=fpn_dim, num_classes=num_classes,
                num_queries=m2f_num_queries, num_layers=m2f_num_layers,
                dim_t=m2f_dim, num_heads=m2f_num_heads,
                mlp_ratio=m2f_mlp_ratio, beta_init=m2f_beta_init,
                w_cls=m2f_w_cls, w_bce=m2f_w_bce, w_dice=m2f_w_dice,
                no_obj_w=m2f_no_obj_w, num_points=m2f_points,
                deep_supervision=m2f_deep_supervision,
                use_modal_src=m2f_src_modal,
                num_modalities=self.num_modalities,
                anchored=m2f_anchored, point_quota=m2f_point_quota)

        # [P39] Dual-Path Compete (실패-키 문서 2026-07-20 반영, 전 항목 토글).
        # V1 trunk rank expansion: fused' = fused + Σ_m P_m(f_m). small-random
        # init (NOT zero — 키1: 소극 잔차 금지), 주 경로 소속이라 첫 스텝부터
        # CE gradient를 받는다. FUSED eff.rank 7/256 병목(키3) 직접 확장.
        self.trunk_exp = None
        if p39_trunk_exp:
            self.trunk_exp = nn.ModuleList(
                nn.Conv2d(dim, dim, 1) for _ in range(self.num_modalities))
            for m in self.trunk_exp:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # V5 compete-and-arbitrate: per-class Λ (softplus(0)=0.69, 죽은 시작
        # 아님) + 학습 시 path dropout 경쟁(dense-only/query-only/combined) +
        # router 직접 CE 감독(의존→기여 전환, 키2). β 잔차(반증 완료)는 arbiter
        # 활성 시 사용하지 않는다.
        self.arb_lambda = nn.Parameter(torch.zeros(num_classes)) if p39_arbiter else None
        self.p39_path_dropout_p = float(p39_path_dropout_p)
        self.p39_router_ce_w = float(p39_router_ce_w)
        # eval-time ablation flags (tools/module_ablation attr_toggle 호환)
        self.p39_query_off = False
        self.p39_trunkexp_off = False

        # M2 seam (asymmetric modality dropout) — default OFF: it helped nothing
        # at mid-run so far (P33 empirical constraint 4); seam kept for P34.2.
        self.modal_dropout = modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.modal_dropout_targets = tuple(int(t) for t in modal_dropout_targets)
        self.modal_dropout_warmup_ep = int(modal_dropout_warmup_ep)
        self._current_epoch = 0          # trainer sets this each epoch
        self._last_dropped_modality = None
        # [analysis] SAM2 표준 훅 미러링 (seg_analysis 도구 무수정 동작; 학습 영향 0)
        self._last_per_modal_feats = None
        self._last_per_modal_outputs = None
        self._last_uamm_spatial = None

    # ── M2: P33._maybe_drop_modality port (zero-input replacement, train only) ─
    def _maybe_drop_modality(self, batched_input):
        if not (self.modal_dropout and self.training):
            return batched_input
        m = len(batched_input)
        if m < 2:
            return batched_input
        p = self.modal_dropout_p
        if self.modal_dropout_warmup_ep > 0:
            p = p * min(1.0, float(self._current_epoch) / self.modal_dropout_warmup_ep)
        if float(torch.rand(1).item()) >= p:
            return batched_input
        targets = [t for t in self.modal_dropout_targets if 0 <= t < m]
        if not targets:
            return batched_input
        j = targets[int(torch.randint(len(targets), (1,)).item())]
        self._last_dropped_modality = j
        new_input = list(batched_input)
        new_input[j] = torch.zeros_like(new_input[j])
        return new_input

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)

    def _decode(self, fused: torch.Tensor, routed: Optional[torch.Tensor]):
        """Shared FPN + head (+ [P36] router residual) → (logits@stride4, feat).

        [P36] router-refined residual: per-class routed aux logits added to
        the head output. router_alpha is zero-init → identical to the
        router-off path at start (collapse-safe); grads reach alpha, the
        router heads AND the aux decoders through this decision path."""
        pyramid = self.fpn(fused)
        logits, m_feat = self.head(pyramid)
        if routed is not None:
            logits = logits + self.fusion.router_alpha * F.interpolate(
                routed, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        return logits, m_feat

    def forward(self, batched_input: List[torch.Tensor], multimask_output: bool = True,
                gt_mask: Optional[torch.Tensor] = None):
        # `multimask_output` kept for call-site compatibility with the SAM2 fleet.
        self._last_dropped_modality = None
        x = self._maybe_drop_modality(batched_input)
        H, W = x[0].shape[-2:]
        feats = [self.encoder(x[i], i) for i in range(self.num_modalities)]
        if not self.training:
            self._last_per_modal_feats = [f.detach() for f in feats]
        fused, aux = self.fusion(feats, gt_mask if self.training else None)
        if not self.training:
            al = getattr(self.fusion, '_last_aux_logits', None)
            self._last_per_modal_outputs = list(al) if al is not None else None
            gs = getattr(self.fusion, '_last_gate_spatial', None)
            self._last_uamm_spatial = (
                [gs[i].float().cpu().numpy() for i in range(gs.shape[0])]
                if gs is not None else None)
        routed = aux.pop('routed_logits', None)     # [P36] consumed here, not by trainer
        cefr_ctx = aux.pop('cefr_ctx', None)        # [P37a] consumed here, not by trainer
        if cefr_ctx is not None:
            # [P37a] two-pass CEFR flow (shared FPN+head weights, two calls):
            #   pass 1 (unchanged P36 path, no_grad — logits1 feeds only the
            #   DETACHED q) → q_k(s)=softmax_k(logits1), avg-pooled to stride 16
            #   → CEFR class-expected routing over POST-attention fused tokens
            #   → feature-level blend fused_final=(1−σ(a))·gate_fused+σ(a)·fused'
            #   (a init −4, σ≈0.018 → byte-near P36 start) → pass 2 = final.
            with torch.no_grad():
                logits1, _ = self._decode(fused, routed)
                q = F.softmax(logits1.float(), dim=1)
                q = F.adaptive_avg_pool2d(q, fused.shape[-2:])
            fused_p, _, cefr_reg = self.fusion.cefr(
                cefr_ctx['fused_tokens'], cefr_ctx['rel_cal'],
                cefr_ctx['log_post'], q, self._current_epoch)
            mix = torch.sigmoid(self.fusion.cefr.a)
            fused = (1.0 - mix) * fused + mix * fused_p
            if self.training and cefr_reg is not None:
                aux['cefr_reg'] = cefr_reg          # trainer adds to total loss
        if self.trunk_exp is not None and not self.p39_trunkexp_off:
            # [P39-V1] modal subspace restoration on the main trunk.
            fused = fused + sum(proj(f) for proj, f in zip(self.trunk_exp, feats))
        logits, m_feat = self._decode(fused, routed)
        token_logits = None
        if self.classtoken is not None:
            # [P37b] class-token residual: mask-embedding dot-product logits at
            # stride 4, added to the head output scaled by the ZERO-INIT beta →
            # exactly equal to the classtoken-off path at init (collapse-safe).
            # CEFR와 합성 시 classtoken은 blend된 fused_final을 본다(의도된 결합).
            token_logits = self.classtoken(fused, m_feat)   # (B, K, H/4, W/4)
            logits = logits + self.classtoken.beta * token_logits
        if self.m2f is not None:
            m2f_out = self.m2f(
                fused, m_feat,
                modal_feats=feats if self.m2f.use_modal_src else None)  # [P39-V2]
            sem_q = self.m2f.semantic_scores(m2f_out)       # (B, K, H/4, W/4)
            if self.arb_lambda is not None:
                # [P39-V5] compete-and-arbitrate (β 잔차 대체): per-class Λ로
                # 스케일한 query semantic을 학습 시 path dropout으로 경쟁시킨다
                # (25% dense-only / 25% query-only / 50% combined) — 어느 경로도
                # 주 손실 무임승차(no-op 고착, 키1)가 불가능. 추론은 항상 결합.
                q_scaled = F.softplus(self.arb_lambda).view(1, -1, 1, 1) * sem_q
                if self.p39_query_off:
                    q_scaled = q_scaled * 0.0
                if self.training:
                    p = self.p39_path_dropout_p
                    r = float(torch.rand(1).item())
                    if r < p:
                        pass                                # dense-only turn
                    elif r < 2.0 * p:
                        logits = q_scaled                   # query-only turn
                    else:
                        logits = logits + q_scaled
                else:
                    logits = logits + q_scaled
            else:
                # [P38] legacy zero-init beta residual (반증됨 — arbiter 권장)
                logits = logits + self.m2f.beta * sem_q
            if self.training and gt_mask is not None:
                aux['m2f_loss'] = self.m2f_loss_w * self.m2f.losses(
                    m2f_out, m_feat, gt_mask)
        if (self.arb_lambda is not None and self.training
                and routed is not None and gt_mask is not None):
            # [P39-V5] router 직접 감독: 결정경로 의존(co-adaptation)이 아니라
            # 자립 기여로 학습시킨다 (키2). routed 해상도에서 CE.
            gt_r = F.interpolate(gt_mask.unsqueeze(1).float(),
                                 size=routed.shape[-2:],
                                 mode='nearest').squeeze(1).long()
            aux['router_ce'] = self.p39_router_ce_w * F.cross_entropy(
                routed.float(), gt_r, ignore_index=255)
        logits = F.interpolate(logits.float(), size=(H, W),
                               mode='bilinear', align_corners=False)
        if self.training:
            if token_logits is not None and gt_mask is not None:
                # [P37b] training-only aux CE on token_logits at 1/4 label res
                # (same downsampling convention as the fusion aux CE). Trainer
                # weights it by MODEL.CLASS_TOKEN.AUX_CE_W (default 0.4).
                gt_ds = F.interpolate(gt_mask.unsqueeze(1).float(),
                                      size=token_logits.shape[-2:],
                                      mode='nearest').squeeze(1).long()
                aux['ctd_ce'] = F.cross_entropy(token_logits.float(), gt_ds,
                                                ignore_index=255)
            return logits, m_feat, aux
        return logits, m_feat


    def extract_det_pyramid(self, batched_input: List[torch.Tensor]) -> List[torch.Tensor]:
        """Multi-scale pyramid for a detection head (RF-DETR / FCOS), CEFR-aware.

        Per-modality frozen-ViT+LoRA encoding -> reliability-gated fusion -> [P37a]
        the SAME two-pass CEFR feature blend as forward() (so the detection pyramid
        is built on CEFR-refined fused tokens) -> SimpleFPN. Returns the ViTDet
        pyramid *before* the seg head: [s4, s8, s16, s32], each (B, fpn_dim, h, w).

        When CEFR is off (cefr_ctx is None) this is byte-identical to the P35/P36-Det
        path. [P36] router->det residual (zero-init) is applied if the router is on.
        """
        feats = [self.encoder(batched_input[i], i) for i in range(self.num_modalities)]
        fused, aux = self.fusion(feats, None)
        routed = aux.get('routed_logits', None) if isinstance(aux, dict) else None
        cefr_ctx = aux.get('cefr_ctx', None) if isinstance(aux, dict) else None
        if cefr_ctx is not None:
            # pass-1 seg decode supplies only the DETACHED posterior q (reliability
            # signal); pass-2 blends CEFR-routed tokens into fused. a init -4 =>
            # sigma(a)~0.018 => at start byte-near the non-CEFR pyramid.
            with torch.no_grad():
                logits1, _ = self._decode(fused, routed)
                q = F.softmax(logits1.float(), dim=1)
                q = F.adaptive_avg_pool2d(q, fused.shape[-2:])
            fused_p, _, _ = self.fusion.cefr(
                cefr_ctx['fused_tokens'], cefr_ctx['rel_cal'],
                cefr_ctx['log_post'], q, self._current_epoch)
            mix = torch.sigmoid(self.fusion.cefr.a)
            fused = (1.0 - mix) * fused + mix * fused_p
        pyramid = self.fpn(fused)
        if routed is not None and getattr(self, 'det_router_proj', None) is not None:
            r = self.det_router_proj(routed)
            pyramid = [
                p + self.det_router_alpha * F.interpolate(
                    r, size=p.shape[-2:], mode='bilinear', align_corners=False)
                for p in pyramid
            ]
        if self.classtoken is not None and getattr(self, 'det_classtoken_proj', None) is not None:
            # [P37b-Det] class-token per-class logits -> pyramid (zero-init residual).
            # feat_s4 = the seg head's stride-4 feature (from _decode), detached: det
            # loss must not train the seg head, but the class tokens + proj do train.
            with torch.no_grad():
                _, feat_s4 = self._decode(fused, routed)
            token_logits = self.classtoken(fused, feat_s4)      # (B, num_classes, H/4, W/4)
            t = self.det_classtoken_proj(token_logits)          # (B, fpn_dim, H/4, W/4)
            pyramid = [
                p + self.det_classtoken_alpha * F.interpolate(
                    t, size=p.shape[-2:], mode='bilinear', align_corners=False)
                for p in pyramid
            ]
        return pyramid


    def extract_m2f_output(self, batched_input):
        """[P38-Det] Run the M2F query head for detection: encoder -> fusion ->
        M2F queries (cls + box per query). feat_s4 (the seg head stride-4 feature)
        is taken under no_grad so det loss never trains the seg head; the M2F
        decoder, in_proj, query, cls_head and box_head train through cls+box loss.
        """
        feats = [self.encoder(batched_input[i], i) for i in range(self.num_modalities)]
        fused, aux = self.fusion(feats, None)
        routed = aux.get('routed_logits', None) if isinstance(aux, dict) else None
        with torch.no_grad():
            _, feat_s4 = self._decode(fused, routed)
        return self.m2f(fused, feat_s4)


def build_reliadino(cfg: dict, num_classes: int) -> ReliaDINO:
    """Map a training-config dict (configs/*_P34_reliadino.yaml) to ReliaDINO."""
    mc = cfg['MODEL']
    fus = mc.get('FUSION', {}) or {}
    ab = fus.get('ATTN_BIAS', {}) or {}
    gate = mc.get('GATE', {}) or {}
    veto = gate.get('VETO_FLOOR', {}) or {}
    cal = mc.get('CALIBRATION', {}) or {}
    cons = mc.get('CONSISTENCY', {}) or {}
    router = mc.get('ROUTER', {}) or {}
    cefr = mc.get('CEFR', {}) or {}
    ctok = mc.get('CLASS_TOKEN', {}) or {}
    m2f = mc.get('M2F', {}) or {}
    p39 = mc.get('P39', {}) or {}
    mdrop = mc.get('MODAL_DROPOUT', {}) or {}
    modals = cfg['DATASET']['MODALS']
    raw_targets = mdrop.get('TARGETS', ['img', 'depth'])
    tgt_idx = [modals.index(t) if isinstance(t, str) else int(t)
               for t in raw_targets if (not isinstance(t, str)) or (t in modals)]
    img_size = cfg['TRAIN']['IMAGE_SIZE'][0]
    return ReliaDINO(
        num_classes=num_classes,
        modalities=modals,
        backbone=mc.get('BACKBONE_TIMM', 'vit_large_patch16_dinov3'),
        backbone_fallback=mc.get('BACKBONE_FALLBACK', 'vit_large_patch14_reg4_dinov2'),
        pretrained=mc.get('PRETRAINED_BACKBONE', True),
        img_size=img_size,
        lora_r=mc.get('LORA_R', 8),
        lora_alpha=mc.get('LORA_ALPHA', None),
        fpn_dim=mc.get('FPN_DIM', 256),
        fusion_layers=fus.get('NUM_LAYERS', 2),
        fusion_heads=fus.get('NUM_HEADS', 8),
        fusion_mlp_ratio=fus.get('MLP_RATIO', 4.0),
        aux_hidden=fus.get('AUX_HIDDEN', 256),
        attn_bias=ab.get('ENABLE', True),
        lambda1_init=ab.get('LAMBDA1_INIT', 1.0),
        consistency_bias=cons.get('ENABLE', True),
        lambda2_init=cons.get('LAMBDA2_INIT', 0.5),
        gate_enable=gate.get('ENABLE', True),
        gate_tau=gate.get('TAU', 0.25),
        gate_entropy_reg=gate.get('ENTROPY_REG', 0.0),
        gate_entropy_floor=gate.get('ENTROPY_FLOOR', 0.5),
        veto_floor=veto.get('ENABLE', True),
        veto_thresh=veto.get('THRESH', 0.10),
        veto_cap=veto.get('CAP', 0.05),
        calibrate=cal.get('ENABLE', True),
        router_enable=router.get('ENABLE', False),
        router_anchor_lambda=router.get('ANCHOR_LAMBDA', 1.0),
        router_reg_mode=router.get('REG_MODE', 'decisive'),
        router_reg_lambda=router.get('REG_LAMBDA', 0.01),
        router_alpha_init=router.get('ALPHA_INIT', 0.0),
        router_hidden=router.get('HIDDEN', 64),
        cefr_enable=cefr.get('ENABLE', False),
        cefr_hidden=cefr.get('HIDDEN', 64),
        cefr_morph_init=cefr.get('MORPH_INIT', -4.0),
        cefr_anchor_posterior=cefr.get('ANCHOR_POSTERIOR', True),
        cefr_lambda1=cefr.get('LAMBDA1', 1.0),
        cefr_lambda2_target=cefr.get('LAMBDA2_TARGET', 0.5),
        cefr_lambda2_warmup_ep=cefr.get('LAMBDA2_WARMUP_EP', 10),
        cefr_reg_lambda=cefr.get('REG_LAMBDA', 0.01),
        cefr_entropy_floor=cefr.get('ENTROPY_FLOOR', 0.5),
        cefr_hinge_reg=cefr.get('HINGE_REG', 1.0),
        class_token_enable=ctok.get('ENABLE', False),
        class_token_layers=ctok.get('NUM_LAYERS', 3),
        class_token_dim=ctok.get('DIM', 256),
        class_token_heads=ctok.get('NUM_HEADS', 8),
        class_token_mlp_ratio=ctok.get('MLP_RATIO', 2.0),
        class_token_beta_init=ctok.get('BETA_INIT', 0.0),
        m2f_enable=m2f.get('ENABLE', False),
        m2f_num_queries=m2f.get('NUM_QUERIES', 100),
        m2f_num_layers=m2f.get('NUM_LAYERS', 6),
        m2f_dim=m2f.get('DIM', 256),
        m2f_num_heads=m2f.get('NUM_HEADS', 8),
        m2f_mlp_ratio=m2f.get('MLP_RATIO', 2.0),
        m2f_beta_init=m2f.get('BETA_INIT', 0.0),
        m2f_w_cls=m2f.get('W_CLS', 2.0),
        m2f_w_bce=m2f.get('W_BCE', 5.0),
        m2f_w_dice=m2f.get('W_DICE', 5.0),
        m2f_no_obj_w=m2f.get('NO_OBJ_W', 0.1),
        m2f_points=m2f.get('POINTS', 12544),
        m2f_deep_supervision=m2f.get('DEEP_SUPERVISION', True),
        m2f_loss_w=m2f.get('LOSS_W', 0.5),
        m2f_src_modal=(str(m2f.get('SRC', 'fused')).lower() == 'modal'),
        m2f_anchored=m2f.get('ANCHORED', False),
        m2f_point_quota=m2f.get('POINT_QUOTA', 0),
        p39_trunk_exp=p39.get('TRUNK_EXP', False),
        p39_arbiter=p39.get('ARBITER', False),
        p39_path_dropout_p=p39.get('PATH_DROPOUT_P', 0.25),
        p39_router_ce_w=p39.get('ROUTER_CE_W', 0.4),
        modal_dropout=mdrop.get('ENABLE', False),
        modal_dropout_p=mdrop.get('P', 0.3),
        modal_dropout_targets=tuple(tgt_idx) if tgt_idx else (0, 1),
        modal_dropout_warmup_ep=mdrop.get('WARMUP_EP', 20),
    )
