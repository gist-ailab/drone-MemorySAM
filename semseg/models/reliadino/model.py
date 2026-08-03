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

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import p44 as P44
from . import p46 as P46
from . import p47 as P47
from .classtoken import ClassTokenLiteHead
from .encoder import FrozenViTEncoder, SimpleFPN, LayerNorm2d
from .fusion import ReliabilityGatedFusion
from .m2f_head import MaskQueryLiteHead
from .panoptic_head import MaskClsHead


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
                 p39_trunk_mode: str = 'linear',
                 p39_trunk_hidden: int = 256,
                 p39_arbiter: bool = False,
                 p39_path_dropout_p: float = 0.25,
                 p39_router_ce_w: float = 0.4,
                 p391_vicreg: bool = False,
                 p391_vicreg_lvar: float = 0.1,
                 p391_vicreg_lcov: float = 0.01,
                 p391_vicreg_tokens: int = 2048,
                 p391_vicreg_lidar_w: float = 1.0,
                 p391_vicreg_other_w: float = 0.25,
                 p41_fcr: bool = False,
                 p41_fcr_lambda: float = 0.1,
                 p42_mask_img: bool = False,
                 p42_mask_frac: float = 0.5,
                 p42_mask_warmup_ep: int = 20,
                 p43_m2f_head: bool = False,
                 p43_lateral: bool = True,
                 p43_num_taps: int = 3,
                 p43_num_queries: int = 100,
                 p43_dec_layers: int = 6,
                 p43_dim: int = 256,
                 p43_num_heads: int = 8,
                 p43_mlp_ratio: float = 2.0,
                 p43_w_cls: float = 2.0,
                 p43_w_bce: float = 5.0,
                 p43_w_dice: float = 5.0,
                 p43_no_obj_w: float = 0.1,
                 p43_num_points: int = 12544,
                 p43_oversample: float = 3.0,
                 p43_importance: float = 0.75,
                 p43_deep_supervision: bool = True,
                 p43_lambda: float = 1.0,
                 p43_lambda_warmup_ep: int = 5,
                 p43_eval_head: bool = False,
                 p43_sem_source: str = 'pixel',
                 p43_thing_ids: Optional[Sequence[int]] = None,
                 p44_local_mask: bool = False,
                 p44_mask_mode: str = 'rect',
                 p44_mask_frac: float = 0.5,
                 p44_mask_warmup_ep: int = 20,
                 p44_area_ratio: Sequence[float] = (0.1, 0.5),
                 p44_num_regions: Sequence[int] = (1, 3),
                 p44_coverage_dilate: int = 31,
                 p44_blob_grid: int = 16,
                 p44_blob_p: float = 0.5,
                 p44_hard_pixel_aux: bool = False,
                 p44_hard_pixel_w: float = 0.5,
                 p44_validity_renorm: bool = False,
                 p44_validity_dilate: int = 1,
                 p44_mutual_kl: bool = False,
                 p44_mkl_w: float = 0.5,
                 p44_mkl_t: float = 1.0,
                 p44_mkl_warmup_ep: int = 10,
                 p44_rel_corr: bool = False,
                 p44_rc_w: float = 0.1,
                 p44_rc_pairs: int = 2048,
                 p44_rc_mode: str = 'mse',
                 p44_rc_warmup_ep: int = 10,
                 p46_proto: bool = False,
                 p46_proto_dim_src: str = 'mfeat',
                 p46_proto_lambda: float = 0.1,
                 p46_proto_ema: float = 0.999,
                 p46_proto_temp: float = 0.1,
                 p46_proto_pixels: int = 4096,
                 p46_proto_warmup_ep: int = 5,
                 p47_2_unibal: bool = False,
                 p47_2_lambda_u: float = 0.4,
                 p47_2_modals='all',
                 p47_2_head: str = 'linear',
                 p47_2_hidden: int = 256,
                 p47_2_warmup_ep: int = 0,
                 p47_2_gt_div: int = 4,
                 p47_2_reduce: str = 'mean',
                 p45_fogstyle: bool = False,
                 p45_prob: float = 0.5,
                 p45_sigma: float = 0.5,
                 p45_weight: float = 0.1,
                 p45_detach_clean: bool = True,
                 rca_enable: bool = False,
                 rca_p_max: float = 0.5,
                 rca_warmup_ep: int = 20,
                 rca_quantile: float = 0.3,
                 rca_alpha_min: float = 0.1,
                 rca_alpha_max: float = 0.5,
                 rca_readout_w: float = 0.5,
                 rca_buf_size: int = 512,
                 rca_min_fill: int = 128,
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
            lora_r=lora_r, lora_alpha=lora_alpha,
            num_taps=(int(p43_num_taps) if p43_lateral else 0))   # [P43-T2]
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
            cefr_hinge_reg=cefr_hinge_reg,
            # [P44-B2/V-1] peer 상호증류 손실 + presence 재정규화 (전부 default off)
            p44_mutual_kl=p44_mutual_kl, p44_mkl_w=p44_mkl_w,
            p44_mkl_t=p44_mkl_t, p44_mkl_warmup_ep=p44_mkl_warmup_ep,
            p44_rel_corr=p44_rel_corr, p44_rc_w=p44_rc_w,
            p44_rc_pairs=p44_rc_pairs, p44_rc_mode=p44_rc_mode,
            p44_rc_warmup_ep=p44_rc_warmup_ep,
            p44_validity_renorm=p44_validity_renorm,
            p44_export_train_aux=bool(p44_hard_pixel_aux or p45_fogstyle))
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

        # (중복 ClassTokenLiteHead 생성 블록 제거 — 감사 2026-07-21: 동일
        # 블록 2회 생성으로 init RNG 스트림이 어긋나 seed 재현 비교를 깨던
        # merge 잔재. 첫 생성(위)만 유지.)

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

        # ── [P43] PanopticDual ──────────────────────────────────────────────
        # T-1 mask-classification head as an INDEPENDENT primary loss (no
        # residual, no gate, no blend into the pixel logits — 실패-키 1) and
        # T-2 PMT-style multi-depth lateral taps into the shared SimpleFPN.
        # Both default OFF -> the model is byte-identical to the P39.1 baseline.
        self.p43 = None
        self.p43_lambda = float(p43_lambda)
        self.p43_lambda_warmup_ep = int(p43_lambda_warmup_ep)
        self.p43_eval_head = bool(p43_eval_head)
        self.p43_sem_source = str(p43_sem_source).lower()
        if self.p43_sem_source not in ('pixel', 'query', 'sum'):
            raise ValueError(f"P43.SEM_SOURCE must be pixel|query|sum, "
                             f"got {p43_sem_source!r}")
        self.p43_thing_ids = list(p43_thing_ids) if p43_thing_ids else []
        if p43_m2f_head:
            self.p43 = MaskClsHead(
                fpn_dim=fpn_dim, num_classes=num_classes,
                num_queries=p43_num_queries, dec_layers=p43_dec_layers,
                dim_t=p43_dim, num_heads=p43_num_heads,
                mlp_ratio=p43_mlp_ratio, w_cls=p43_w_cls, w_bce=p43_w_bce,
                w_dice=p43_w_dice, no_obj_w=p43_no_obj_w,
                num_points=p43_num_points, oversample=p43_oversample,
                importance=p43_importance,
                deep_supervision=p43_deep_supervision)
        # lateral projections: one per tap, injected at the matching pyramid
        # level (shallow tap -> highest resolution). NOT zero-init and NOT
        # gated: they sit on the primary path and must earn gradient from step
        # one (실패-키 1 — "잔차로 살짝 얹기"는 반증 완료).
        self.p43_lateral = None
        self.p43_lateral_levels: List[int] = []
        n_taps = len(self.encoder.tap_layers)
        if p43_lateral and n_taps > 0:
            self.p43_lateral_levels = [min(i, 2) for i in range(n_taps)]
            self.p43_lateral = nn.ModuleList(
                nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, fpn_dim, 1))
                for _ in range(n_taps))
        self._p43_taps = None          # per-forward, modality-averaged taps
        self._last_p43_out = None      # eval-only, for panoptic_inference
        # eval-time ablation flags (tools/module_ablation attr_toggle 호환)
        self.p43_m2f_off = False
        self.p43_lateral_off = False

        # [P39] Dual-Path Compete (실패-키 문서 2026-07-20 반영, 전 항목 토글).
        # V1 trunk rank expansion: fused' = fused + Σ_m P_m(f_m). small-random
        # init (NOT zero — 키1: 소극 잔차 금지), 주 경로 소속이라 첫 스텝부터
        # CE gradient를 받는다. FUSED eff.rank 7/256 병목(키3) 직접 확장.
        self.trunk_exp = None
        self.trunk_gamma = None
        self.p39_trunk_mode = p39_trunk_mode
        if p39_trunk_exp:
            if p39_trunk_mode == 'gated_mlp':
                # [P39.1-R1] 선형 1×1의 암묵적 저rank 편향(deep matrix
                # factorization/DirectCLR — P39에서 lidar rank 4.7 붕괴의 유력
                # 원인)을 제거: LN→1×1→GELU→1×1 비선형 + tanh(γ) 게이트(γ=0
                # init, ReZero/LLaMA-Adapter) — shortcut이 초기 gradient
                # highway가 되어 LoRA를 저rank 코드로 조각하는 것을 막는다.
                # V1의 night +2.50 기여 메커니즘은 보존.
                h = int(p39_trunk_hidden)
                self.trunk_exp = nn.ModuleList(
                    nn.Sequential(LayerNorm2d(dim),
                                  nn.Conv2d(dim, h, 1), nn.GELU(),
                                  nn.Conv2d(h, dim, 1))
                    for _ in range(self.num_modalities))
                # γ init 0.1 (NOT 0): tanh(0)=0이면 MLP가 gradient를 전혀 못
                # 받아 키1(수동 zero-결선 사장, 4연속)의 재판이 된다. 0.1이면
                # 게이트는 사실상 닫혀 있으면서(≈0.0997) 첫 스텝부터 흐른다.
                self.trunk_gamma = nn.Parameter(
                    torch.full((self.num_modalities,), 0.1))
            else:
                self.trunk_exp = nn.ModuleList(
                    nn.Conv2d(dim, dim, 1) for _ in range(self.num_modalities))
                for m in self.trunk_exp:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        # [P39.1-R2] VICReg variance+covariance 정규화 (per-modal 토큰,
        # lidar 가중 강화) — 붕괴 "복원"용. per-GPU 서브샘플, fp32, sync 불요.
        self.p391_vicreg = p391_vicreg
        self.p391_vicreg_lvar = float(p391_vicreg_lvar)
        self.p391_vicreg_lcov = float(p391_vicreg_lcov)
        self.p391_vicreg_tokens = int(p391_vicreg_tokens)
        _lidx = self.modalities.index('lidar') if 'lidar' in self.modalities else -1
        self.p391_vicreg_w = [
            p391_vicreg_lidar_w if i == _lidx else p391_vicreg_other_w
            for i in range(self.num_modalities)]
        # [P41] FCR — fused between-class 분산비 η² 최대화(supervised aux, Phase-0 deficit 교정)
        self.p41_fcr = bool(p41_fcr)
        self.p41_fcr_lambda = float(p41_fcr_lambda)
        # [P42-M1] 조건부 균형 img 마스킹 — fusion이 lidar/event를 쓰도록 강제(train only)
        self.p42_mask_img = bool(p42_mask_img)
        self.p42_mask_frac = float(p42_mask_frac)
        self.p42_mask_warmup_ep = int(p42_mask_warmup_ep)
        self._img_idx = self.modalities.index('img') if 'img' in self.modalities else -1
        self._lidar_idx = self.modalities.index('lidar') if 'lidar' in self.modalities else -1
        self._p42_img_idx = self._img_idx          # P42 시절 이름 (호환 유지)
        self._last_p42_mask = None   # (B,) 진단용(마스킹된 샘플)
        # [P44-B3] 커버리지 패턴 국소 마스킹 — P42 전역 img-drop의 승격.
        # P42와 **직교**: p42_mask_img 경로는 손대지 않았고, MODE 'global'이
        # P42와 같은 의미를 P44 config로 재현한다(ablation 연속성).
        self.p44_local_mask = bool(p44_local_mask)
        self.p44_mask_mode = str(p44_mask_mode)
        self.p44_mask_frac = float(p44_mask_frac)
        self.p44_mask_warmup_ep = int(p44_mask_warmup_ep)
        self.p44_area_ratio = tuple(p44_area_ratio)
        self.p44_num_regions = tuple(int(v) for v in p44_num_regions)
        self.p44_coverage_dilate = int(p44_coverage_dilate)
        self.p44_blob_grid = int(p44_blob_grid)
        self.p44_blob_p = float(p44_blob_p)
        self._last_p44_mask = None   # (B,1,H,W) 마스킹된 영역 (1 = img 제거)
        # [P44-M3] hard-pixel aux: 마스킹 영역에서 fused가 틀린 픽셀에 생존 모달 aux 집중
        self.p44_hard_pixel_aux = bool(p44_hard_pixel_aux)
        self.p44_hard_pixel_w = float(p44_hard_pixel_w)
        # [P44-V1] presence 재정규화 (학습 파라미터 0, 추론 경로에도 적용)
        self.p44_validity_renorm = bool(p44_validity_renorm)
        self.p44_validity_dilate = int(p44_validity_dilate)
        # ── [P46-C3] Domain-Invariant Class-Prototype Consistency ────────────
        # per-class EMA prototype bank + prototype-contrastive CE (학습 전용).
        # feature source = 픽셀 head의 stride-4 m_feat (fpn_dim) — 최종 분류를
        # 실제로 담당하는 표현이라 여기서 클래스 표현을 도메인불변으로 묶는 것이
        # test 전이에 직접 작용한다. 손실은 detach된 prototype을 타깃으로 삼아
        # gradient가 전부 feature 경로로 흐른다(키1, zero-init 잔차 아님).
        self.p46_proto_lambda = float(p46_proto_lambda)
        self.p46_proto_warmup_ep = int(p46_proto_warmup_ep)
        self.p46_proto_src = str(p46_proto_dim_src).lower()
        if self.p46_proto_src not in ('mfeat', 'fused'):
            raise ValueError(f"P46.C3_PROTO.FEATURE must be mfeat|fused, "
                             f"got {p46_proto_dim_src!r}")
        self.p46_proto = None
        if p46_proto:
            _pdim = fpn_dim if self.p46_proto_src == 'mfeat' else dim
            self.p46_proto = P46.PrototypeBank(
                num_classes=num_classes, dim=_pdim, momentum=p46_proto_ema,
                temperature=p46_proto_temp, pixels=p46_proto_pixels)
        # [P46-C2/C3] 보조 branch(마스킹/스타일 2-view) forward가 주 forward와
        # **정확히 같은 파라미터 집합**을 쓰도록 P39 path-dropout 추첨을 재생한다.
        # 이유(DDP): find_unused_parameters=True는 마지막 forward의 그래프로
        # unused 집합을 정하는데, 두 forward의 경로가 갈리면 한쪽에서만 쓰인
        # 파라미터가 "unused로 ready 처리된 뒤 hook이 또 발화" → reducer 사망.
        self._p46_replay_path = False
        self._p46_last_path_r = None
        # [P45-F1] feature-space fog style 일관성 (기본 off)
        self.p45_fogstyle = bool(p45_fogstyle)
        self.p45_prob = float(p45_prob)
        self.p45_sigma = float(p45_sigma)
        self.p45_weight = float(p45_weight)
        self.p45_detach_clean = bool(p45_detach_clean)
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

        # [P40] RCA — Reliability-Conditioned Attenuation (학습 전용).
        # 5세대 반증된 "추론-시 신뢰도 재가중" 대신 같은 신호를 "학습-시
        # 조건화"로 사용: 모델 자신의 rel 추정이 카메라 열화를 가리키는
        # 샘플(배치 하위 분위)의 img FEATURE를 soft 감쇠(hard-zero 금지,
        # 2403.04245의 missing-지름길 역효과 회피) + lidar readout 보조 CE로
        # gradient 출구 제공. 최근접 선행 OPM(T-PAMI'24)/SGMA와의 차별 4축은
        # decisions/2026-07-21 문서 참조.
        self.rca_enable = rca_enable
        self.rca_p_max = float(rca_p_max)
        self.rca_warmup_ep = int(rca_warmup_ep)
        self.rca_quantile = float(rca_quantile)
        self.rca_alpha_min = float(rca_alpha_min)
        self.rca_alpha_max = float(rca_alpha_max)
        self.rca_readout_w = float(rca_readout_w)
        self.rca_img_idx = self.modalities.index('img') if 'img' in self.modalities else -1
        self.rca_lidar_idx = self.modalities.index('lidar') if 'lidar' in self.modalities else -1
        self._rca_pick = None            # (B,) bool, 학습 스텝별
        self._last_lidar_validity = None  # (B,) [P40-C1] 리턴 유효 픽셀 비율
        # [P40-F1] rel(img) 분위 임계값은 "현재 미니배치"가 아니라 최근 관측
        # 분포에서 뽑는다. per-GPU 배치 quantile은 BATCH_SIZE=1(MUSES)에서
        # 1원소의 분위수 = 그 값 자신이 되어 `r_img <= thr`가 **항상 참**이 되고,
        # RCA가 확률 p_t짜리 **무조건** modality dropout으로 퇴화한다 —
        # P33에서 no-op으로 판정났고 2403.04245가 역효과를 실증한 그 설계다.
        # BATCH_SIZE=2에서도 하위 30%가 아니라 하위 50%가 뽑힌다.
        #
        # 🔴 DDP 의미론 주의: 이 버퍼들은 register_buffer라서 DDP의
        # broadcast_buffers=True(기본값)가 **매 forward마다 rank0의 값을 전 rank에
        # 덮어쓴다.** persistent=False는 state_dict에만 영향을 줄 뿐
        # named_buffers()에는 그대로 나오므로 broadcast 대상이다. 따라서 실효
        # 동작은 "rank0이 본 최근 buf_size개"의 분위수를 전 rank가 공유하는 것이고,
        # 윈도는 buf_size×world가 아니라 buf_size다. rank0 스트림도 iid 표본이라
        # 수치적으로는 무해하며, 오히려 rank 간 임계값이 정확히 일치해 유리하다.
        # 진짜 rank-로컬을 원하면 DDP(..., broadcast_buffers=False) 또는
        # _ddp_params_and_buffers_to_ignore 등록이 필요하다.
        # ⚠️ M2F를 끈 DDP config에서는 이 버퍼가 buffer-broadcast collective를
        # 새로 켜게 된다(M2F on이면 empty_weight 버퍼로 이미 켜져 있어 증분 0).
        # train_reliadino.py의 07-12 NCCL desync 이력과 관련 — 그 조합을 새로
        # 만들 때 주의.
        self.rca_buf_size = int(rca_buf_size)
        self.rca_min_fill = int(rca_min_fill)
        if self.rca_buf_size < 1:
            raise ValueError(f"RCA.BUF_SIZE must be >= 1, got {self.rca_buf_size}")
        if self.rca_min_fill > self.rca_buf_size:
            # 이 조합이면 _rca_threshold가 영원히 None -> RCA가 무음 no-op이 된다.
            # 조용히 죽는 대신 시끄럽게 보정한다(ISSUE-024류 무음 실패 방지).
            print(f"[P40][warn] RCA.MIN_FILL({self.rca_min_fill}) > "
                  f"BUF_SIZE({self.rca_buf_size}) — RCA가 영구 비활성화되므로 "
                  f"MIN_FILL을 {self.rca_buf_size}로 낮춘다.")
            self.rca_min_fill = self.rca_buf_size
        self.register_buffer('_rca_buf', torch.zeros(self.rca_buf_size),
                             persistent=False)
        self.register_buffer('_rca_buf_n', torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.register_buffer('_rca_buf_ptr', torch.zeros((), dtype=torch.long),
                             persistent=False)
        # 진단(pick_rate)은 train_reliadino.py가 self._rca_pick으로 이미 집계하므로
        # 여기서 별도 필드를 만들지 않는다 — 중복인 데다 micro-step마다
        # GPU->CPU 동기화를 추가하게 된다.

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
        # [analysis §0.5] 피쳐 특성화 tap (eval-only, plain attr, detach → 학습 영향 0)
        self._last_fused_postfusion = None   # T3: fusion 직후 fused (fused-level 모듈 이전)
        self._last_fused_prehead = None      # T5: _decode 직전 fused (CEFR·trunk_exp 등 fused-level
                                             #     모듈 이후. router/classtoken/m2f는 이 뒤 logit-level이라 미포함)

        # ── [P47-2] Uni-modal Balance (구 D-2) ───────────────────────────────
        # 모달별 **독립** 경량 head + uni-modal CE (학습 전용, 추론 경로 불변).
        # 진단 = modality laziness: 융합 손실만으로 학습하면 지배 모달(RGB)의
        # uni-modal feature가 under-optimize 된다 (2305.01233 / 1905.12681 /
        # 2203.12221). 손실은 주 손실에 직접 합산된다(키1 — zero-init 잔차 아님).
        # 🔴 **가장 마지막에 생성**한다: off일 때 이 블록이 init RNG 스트림을
        #    전혀 건드리지 않아야 P39.1 baseline과 seed 재현이 일치한다(2026-07-21
        #    ClassTokenLiteHead 중복 생성으로 스트림이 어긋났던 사고의 교훈).
        self.p47_2 = None
        if p47_2_unibal:
            self.p47_2 = P47.UniModalBalance(
                dim=dim, num_classes=num_classes,
                num_modalities=self.num_modalities,
                active=P47.resolve_modals(p47_2_modals, self.modalities),
                head=p47_2_head, hidden=p47_2_hidden, lambda_u=p47_2_lambda_u,
                warmup_ep=p47_2_warmup_ep, gt_div=p47_2_gt_div,
                reduce=p47_2_reduce)

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

    def _p42_mask_img(self, batched_input):
        """[P42-M1] 조건부 균형 img 마스킹 (train only). 배치의 일부 샘플에서 img 입력을 0으로 →
        fusion이 그 샘플을 lidar/event로 풀도록 강제 = 지배 모달 의존 완화(MCRM 2603.17705식).
        RCA(P40, 추론-시 감쇠·유해)와 다름: **학습 시에만**, 추론은 항상 full-modality.
        무조건 dropout(P33 실패)과 다름: **균형 분할**(일부만·full 샘플 유지) + 커리큘럼 ramp."""
        self._last_p42_mask = None
        if not (self.p42_mask_img and self.training and self._p42_img_idx >= 0):
            return batched_input
        B = batched_input[0].shape[0]
        frac = self.p42_mask_frac
        if self.p42_mask_warmup_ep > 0:
            frac = frac * min(1.0, float(self._current_epoch) / self.p42_mask_warmup_ep)
        if frac <= 0:
            return batched_input
        # [발견 A] 확률적 per-sample 마스킹 — round(B·frac) 양자화(B=2에서 계단함수·BS1 무음) 회피,
        # ramp가 실제 연속(각 샘플 prob=frac). rank별 독립이나 기대비율은 frac 유지.
        mask = (torch.rand(B, device=batched_input[0].device) < frac).float()
        if float(mask.sum()) < 1.0:
            return batched_input           # 이 배치는 우연히 아무것도 안 뽑힘
        self._last_p42_mask = mask
        new_input = list(batched_input)
        img = new_input[self._p42_img_idx]
        keep = (1.0 - mask).view(-1, *([1] * (img.dim() - 1)))
        new_input[self._p42_img_idx] = img * keep   # 마스킹된 샘플의 img → 0
        return new_input

    def _p44_local_mask(self, batched_input):
        """[P44-B3] 커버리지 패턴 **국소** img 마스킹 (train only, 추론은 항상 full).

        P42(M-1)는 뽑힌 샘플의 img를 통째로 0으로 만들었다 — "img가 아예 없는"
        분포는 추론에 존재하지 않으므로 학습↔추론 정합이 나쁘다. B-3는 img를
        **영역 단위로** 지우고 그 영역의 다른 모달은 그대로 두어, 실제로 일어나는
        상황("이 영역의 카메라 정보가 열화/부재, lidar는 살아 있음")을 연습시킨다.
        MODE 'coverage'는 그 영역을 **같은 샘플의 lidar 리턴 패턴**에서 뽑는다
        (§7-b: partial coverage는 예외가 아니라 기본 상태).
        """
        self._last_p44_mask = None
        if not (self.p44_local_mask and self.training and self._img_idx >= 0):
            return batched_input
        frac = self.p44_mask_frac * P44.ramp(self._current_epoch,
                                             self.p44_mask_warmup_ep)
        if frac <= 0:
            return batched_input
        lidar = batched_input[self._lidar_idx] if self._lidar_idx >= 0 else None
        region = P44.sample_region_mask(
            batched_input[self._img_idx], frac, mode=self.p44_mask_mode,
            lidar=lidar, area_ratio=self.p44_area_ratio,
            num_regions=self.p44_num_regions,
            coverage_dilate=self.p44_coverage_dilate,
            blob_grid=self.p44_blob_grid, blob_p=self.p44_blob_p)
        if region is None:
            return batched_input               # 이 배치는 아무 샘플도 안 뽑힘
        self._last_p44_mask = region
        new_input = list(batched_input)
        new_input[self._img_idx] = new_input[self._img_idx] * (1.0 - region)
        return new_input

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)

    # ── [P43] lateral taps + loss schedule ──────────────────────────────────
    def _encode_all(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Per-modality encoding; also collects the [P43-T2] lateral taps.

        The taps are averaged over modalities with FIXED uniform weights — a
        deterministic reduction, not a learned/inferred modality weighting
        (which is the reverse-engineered failure path C-2). The projections
        that follow are trainable, the backbone stays frozen.
        """
        collect = (self.p43_lateral is not None and not self.p43_lateral_off)
        self.encoder.collect_taps = collect     # don't pay for taps we discard
        feats, acc = [], None
        for i in range(self.num_modalities):
            feats.append(self.encoder(x[i], i))
            # encoder.last_taps is overwritten on every call -> accumulate now
            # (summing in place avoids materializing a stacked (M,B,C,h,w)).
            if collect and self.encoder.last_taps:
                lt = self.encoder.last_taps
                acc = list(lt) if acc is None else [a + b for a, b in zip(acc, lt)]
        self._p43_taps = ([t / float(self.num_modalities) for t in acc]
                          if acc is not None else None)
        return feats

    def _p43_lambda_now(self) -> float:
        """lambda(t): 0.1 -> 1.0 over LAMBDA_WARMUP_EP epochs, scaled by LAMBDA."""
        w = self.p43_lambda_warmup_ep
        if w <= 0:
            return self.p43_lambda
        r = min(1.0, max(0.0, float(self._current_epoch) / float(w)))
        return self.p43_lambda * (0.1 + 0.9 * r)

    def _apply_p43_lateral(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """[P43-T2] inject the frozen-ViT multi-depth taps into the SimpleFPN."""
        taps = self._p43_taps
        if self.p43_lateral is None or self.p43_lateral_off or not taps:
            return pyramid
        out = list(pyramid)
        for j, proj in enumerate(self.p43_lateral):
            lvl = self.p43_lateral_levels[j]
            t = proj(taps[j])
            out[lvl] = out[lvl] + F.interpolate(
                t, size=out[lvl].shape[-2:], mode='bilinear', align_corners=False)
        return out

    def _apply_trunk_exp(self, fused: torch.Tensor,
                         feats: List[torch.Tensor]) -> torch.Tensor:
        """[P39-V1/P39.1-R1] modal subspace restoration — seg/det 공용 단일
        경로. (감사 2026-07-21: det seam 2곳이 gated_mlp 모드에서 tanh(γ)
        게이트를 생략해 seg와 다른 trunk를 만들던 버그의 수정 — 결선을 한
        곳으로 모아 재발 차단.)"""
        if self.trunk_exp is None or self.p39_trunkexp_off:
            return fused
        if self.trunk_gamma is not None:
            return fused + sum(
                torch.tanh(self.trunk_gamma[i]) * self.trunk_exp[i](f)
                for i, f in enumerate(feats))
        return fused + sum(proj(f) for proj, f in zip(self.trunk_exp, feats))

    def _vicreg_loss(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """[P39.1-R2] VICReg var+cov on per-modality tokens (VICRegL-style,
        dense). per-GPU, fp32, token-subsampled — restores collapsed branch
        rank (P39 lidar eff.rank 4.7 vs P38 24.7). Pre-scaled by λ and the
        per-modality weight (lidar emphasized)."""
        total = feats[0].new_zeros((), dtype=torch.float32)
        # 감사 2026-07-21: trainer autocast(bf16) 아래에서 covariance 행렬곱이
        # bf16으로 계산되던 것을 fp32로 강제 (docstring 계약 준수).
        with torch.autocast(device_type=feats[0].device.type, enabled=False):
            for i, f in enumerate(feats):
                w = self.p391_vicreg_w[i]
                if w <= 0:
                    continue
                z = f.flatten(2).transpose(1, 2).reshape(-1, f.shape[1]).float()
                M = z.shape[0]
                k = min(self.p391_vicreg_tokens, M)
                if k < M:
                    z = z[torch.randint(0, M, (k,), device=z.device)]
                z = z - z.mean(0, keepdim=True)
                l_var = F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()
                C = (z.T @ z) / max(z.shape[0] - 1, 1)
                d = C.shape[0]
                l_cov = (C.pow(2).sum() - C.diagonal().pow(2).sum()) / d
                total = total + w * (self.p391_vicreg_lvar * l_var
                                     + self.p391_vicreg_lcov * l_cov)
        return total

    def _fcr_loss(self, fused: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """[P41-F1] Fused Class-alignment Regularizer: fused의 between-class 분산비
        η²=tr(Sb)/tr(St)를 최대화(손실 = −η²). Phase-0에서 fused η²=0.35(저-task정렬) +
        img 과지배로 측정된 deficit을 **주손실 레벨**에서 직접 교정(frozen 백본이라 loss-lever만
        유효). 키1 준수(aux 손실, zero-init 잔차 아님). trainer bf16 autocast 하에서 fp32 강제."""
        B, D, h, w = fused.shape
        with torch.autocast(device_type=fused.device.type, enabled=False):
            f = fused.float().permute(0, 2, 3, 1).reshape(-1, D)          # (B*h*w, D)
            gt = F.interpolate(gt_mask.unsqueeze(1).float(), size=(h, w),
                               mode='nearest').squeeze(1).long().reshape(-1)
            keep = gt != 255
            f, gt = f[keep], gt[keep]
            if f.shape[0] < 2:
                return fused.new_zeros(())
            mu = f.mean(0, keepdim=True)
            tr_st = (f - mu).pow(2).sum()
            tr_sb = fused.new_zeros(())
            for c in torch.unique(gt):
                fc = f[gt == c]
                tr_sb = tr_sb + fc.shape[0] * (fc.mean(0) - mu[0]).pow(2).sum()
            return -(tr_sb / (tr_st + 1e-6))                              # η² 최대화

    # ── [P40-F1] rel(img) 분포 링버퍼 ────────────────────────────────────
    @torch.no_grad()
    def _rca_buf_push(self, v: torch.Tensor) -> None:
        """최근 관측한 r_img 값을 원형 버퍼에 적재.

        비유한 값은 걸러낸다. NaN이 하나라도 들어가면 torch.quantile이 NaN을
        반환하고 `r_img <= NaN`이 전부 False가 되어, 그 NaN이 밀려날 때까지
        (BS1이면 buf_size 스텝) RCA가 **조용히** 꺼진다.
        """
        v = v.detach().flatten().to(self._rca_buf.dtype)
        v = v[torch.isfinite(v)]
        n = int(v.numel())
        if n == 0:
            return
        size = self.rca_buf_size
        if n >= size:                      # 한 스텝이 버퍼보다 크면 최신분만
            self._rca_buf.copy_(v[-size:])
            self._rca_buf_ptr.fill_(0)
            self._rca_buf_n.fill_(size)
            return
        ptr = int(self._rca_buf_ptr)
        idx = (torch.arange(n, device=v.device) + ptr) % size
        self._rca_buf[idx] = v
        self._rca_buf_ptr.fill_((ptr + n) % size)
        self._rca_buf_n.fill_(min(int(self._rca_buf_n) + n, size))

    @torch.no_grad()
    def _rca_threshold(self) -> Optional[torch.Tensor]:
        """버퍼가 충분히 찼을 때만 경험적 분위수를 반환, 아니면 None.

        버퍼는 원형이지만 가득 차기 전에는 앞에서부터 채워지므로 `[:n]`이
        정확히 유효 구간이고, 가득 찬 뒤에는 n == size라 전체가 된다.
        """
        n = int(self._rca_buf_n)
        if n < self.rca_min_fill:
            return None
        return torch.quantile(self._rca_buf[:n], self.rca_quantile)

    def _decode(self, fused: torch.Tensor, routed: Optional[torch.Tensor],
                pyramid_out: Optional[List[torch.Tensor]] = None):
        """Shared FPN + head (+ [P36] router residual) → (logits@stride4, feat).

        [P36] router-refined residual: per-class routed aux logits added to
        the head output. router_alpha is zero-init → identical to the
        router-off path at start (collapse-safe); grads reach alpha, the
        router heads AND the aux decoders through this decision path.

        `pyramid_out`, when given, receives the (P43-lateral-augmented) pyramid
        so the [P43] mask-cls head can read the SAME trunk levels the pixel
        head just consumed — the only thing the two heads share."""
        pyramid = self._apply_p43_lateral(self.fpn(fused))
        if pyramid_out is not None:
            pyramid_out.extend(pyramid)
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
        x = self._p42_mask_img(x)                          # [P42-M1] 조건부 img 마스킹
        x = self._p44_local_mask(x)                        # [P44-B3] 국소 img 마스킹
        H, W = x[0].shape[-2:]
        feats = self._encode_all(x)                        # [P43-T2] taps here
        if not self.training:
            self._last_per_modal_feats = [f.detach() for f in feats]
        # [P40-C1] lidar 리턴 유효성(입력 유도, 내부 신호) — RCA 가드 + 분석용
        self._rca_pick = None
        _rca_scale = None
        if self.rca_lidar_idx >= 0:
            with torch.no_grad():
                self._last_lidar_validity = (
                    x[self.rca_lidar_idx].abs().sum(1) > 1e-6).float().mean((1, 2))
        if (self.rca_enable and self.training and self.rca_img_idx >= 0
                and self.num_modalities >= 2):
            # [P40-C2] reliability-conditioned camera attenuation: 자기-추정
            # rel(img)이 **최근 관측 분포의** 하위 분위인 샘플을 확률 p(t)로
            # soft 감쇠 (α∈[amin,amax]). 임계값 출처는 링버퍼 — 배치 내
            # quantile을 쓰면 BS1에서 조건이 무력화된다(위 [P40-F1] 주석 참조).
            with torch.no_grad():
                lg = self.fusion.aux_decoders[self.rca_img_idx](
                    feats[self.rca_img_idx]).float()
                p = F.softmax(lg, dim=1)
                ent = -(p * (p + 1e-8).log()).sum(1) / math.log(lg.shape[1])
                r_img = 1.0 - ent.mean((1, 2))                      # (B,)
                p_t = self.rca_p_max * min(1.0, float(self._current_epoch)
                                           / max(self.rca_warmup_ep, 1))
                self._rca_buf_push(r_img)
                thr = self._rca_threshold()
                if thr is None:
                    # 버퍼가 덜 찼다 = 분포 추정 불가 -> 아무것도 감쇠하지 않는다.
                    # p_t가 유의미해지기 훨씬 전에 버퍼가 찬다(BS1이면 ep0의
                    # 512 스텝 만에 포화; p_t는 ep0에서만 0이고 이후 선형 증가).
                    pick = torch.zeros_like(r_img, dtype=torch.bool)
                else:
                    pick = (r_img <= thr) & (torch.rand_like(r_img) < p_t)
                if self.rca_lidar_idx >= 0:
                    # C-1 guard: lidar가 사실상 부재한 샘플은 강제하지 않는다
                    pick = pick & (self._last_lidar_validity > 0.05)
                if bool(pick.any()):
                    alpha = torch.empty_like(r_img).uniform_(
                        self.rca_alpha_min, self.rca_alpha_max)
                    _rca_scale = torch.where(pick, alpha, torch.ones_like(alpha))
                    self._rca_pick = pick
            if _rca_scale is not None:
                feats = list(feats)
                feats[self.rca_img_idx] = feats[self.rca_img_idx] \
                    * _rca_scale.view(-1, 1, 1, 1)
        # [P44-V1] 결정론적 presence 마스크 (학습 파라미터 0). 학습/추론 both —
        # 이것만이 허용된 추론 경로 변경이고 기본 off다.
        presence = None
        if self.p44_validity_renorm:
            presence = P44.presence_masks(
                x, size=feats[0].shape[-2:], img_idx=self._img_idx,
                dilate=self.p44_validity_dilate)
        # img 마스킹 정보: P44 국소 마스크(B,1,H,W)가 있으면 그것, 없으면 P42(B,)
        _img_mask = self._last_p44_mask if self._last_p44_mask is not None \
            else self._last_p42_mask
        fused, aux = self.fusion(feats, gt_mask if self.training else None,
                                 img_mask=_img_mask, img_idx=self._img_idx,   # [P42-M1/C][P44-B3]
                                 presence=presence, epoch=self._current_epoch)
        if self.p45_fogstyle and self.training:
            # [P45-F1] img 브랜치 feature의 style을 흔들고 예측 일관성을 요구.
            # 픽셀 공간을 건드리지 않으므로 physaug 공정성 라인을 넘지 않는다.
            _al = getattr(self.fusion, '_train_aux_logits', None)
            if _al is not None and self._img_idx >= 0:
                _pert, _applied = P44.style_perturb(
                    feats[self._img_idx], self.p45_prob, self.p45_sigma)
                if float(_applied.sum()) > 0:
                    _lgp = self.fusion.aux_decoders[self._img_idx](_pert)
                    _lgc = _al[self._img_idx]
                    if self.p45_detach_clean:
                        _lgc = _lgc.detach()
                    with torch.autocast(device_type=fused.device.type, enabled=False):
                        _logp = F.log_softmax(_lgp.float(), dim=1)
                        _logc = F.log_softmax(_lgc.float(), dim=1)
                        _kl = (_logc.exp() * (_logc - _logp)).sum(1)      # (B,h,w)
                        _kl = _kl.mean(dim=(1, 2))                        # (B,)
                        aux['p45_fogstyle'] = self.p45_weight * (
                            (_kl * _applied).sum() / _applied.sum().clamp(min=1.0))
        if not self.training:
            self._last_fused_postfusion = fused.detach()   # [analysis §0.5 T3]
        if self.p41_fcr and self.training and gt_mask is not None:
            aux['fcr'] = self.p41_fcr_lambda * self._fcr_loss(fused, gt_mask)   # [P41-F1]
        if (self._rca_pick is not None and self.training
                and gt_mask is not None and self.rca_lidar_idx >= 0):
            # [P40-C3] 감쇠 샘플 한정 lidar readout 보조 CE — 감쇠만으로는
            # fusion이 "저카메라 모드 암기"로 빠질 수 있어 gradient 출구 필요.
            lg_l = self.fusion.aux_decoders[self.rca_lidar_idx](
                feats[self.rca_lidar_idx]).float()
            gt_ds = F.interpolate(gt_mask.unsqueeze(1).float(),
                                  size=lg_l.shape[-2:],
                                  mode='nearest').squeeze(1).long()
            pk = self._rca_pick
            if bool((gt_ds[pk] != 255).any()):
                aux['rca_readout'] = self.rca_readout_w * F.cross_entropy(
                    lg_l[pk], gt_ds[pk], ignore_index=255)
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
        fused = self._apply_trunk_exp(fused, feats)
        if self.p391_vicreg and self.training:
            aux['vicreg'] = self._vicreg_loss(feats)    # [P39.1-R2] pre-scaled
        if not self.training:
            self._last_fused_prehead = fused.detach()      # [analysis §0.5 T5]
        # [P43] the mask-cls head reads the pyramid the pixel head just used.
        _p43_run = (self.p43 is not None and not self.p43_m2f_off
                    and (self.training or self.p43_eval_head
                         or self.p43_sem_source != 'pixel'))
        _pyr: List[torch.Tensor] = []
        logits, m_feat = self._decode(fused, routed, _pyr if _p43_run else None)
        self._last_p43_out = None
        if _p43_run:
            # levels COARSE FIRST: {1/32, 1/16, 1/8}; mask features = 1/4.
            p43_out = self.p43([_pyr[3], _pyr[2], _pyr[1]], _pyr[0])
            if self.training and gt_mask is not None:
                # INDEPENDENT primary loss. It is NOT added to `logits` — the
                # pixel head's CE never sees the query branch (실패-키 1).
                aux['p43_mask_loss'] = self._p43_lambda_now() * self.p43.losses(
                    p43_out, gt_mask)
            elif not self.training:
                self._last_p43_out = p43_out
                if self.p43_sem_source != 'pixel':
                    # EVAL-ONLY analysis path (T-3). Training always decodes the
                    # pixel head alone, whatever SEM_SOURCE says.
                    sem_q = self.p43.semantic_scores(p43_out)
                    logits = sem_q if self.p43_sem_source == 'query' \
                        else logits + sem_q
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
                    # [P46] 보조 branch는 주 forward의 추첨을 그대로 재생한다
                    # (DDP unused-param 집합 일치 — __init__ 주석 참조).
                    if self._p46_replay_path and self._p46_last_path_r is not None:
                        r = self._p46_last_path_r
                    else:
                        r = float(torch.rand(1).item())
                        self._p46_last_path_r = r
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
        if (self.p44_hard_pixel_aux and self.training and gt_mask is not None
                and self._last_p44_mask is not None):
            # [P44-M3] 마스킹 영역에서 **fused가 지금 틀린 픽셀**에 한해 생존 모달의
            # aux CE를 집중시킨다(MCRM 2603.17705 hard-pixel). correctness 마스크는
            # detach — 손실이 "틀림 판정" 자체를 학습하지 않게.
            _al = getattr(self.fusion, '_train_aux_logits', None)
            if _al is not None:
                Hq, Wq = logits.shape[-2:]
                gt_q = F.interpolate(gt_mask.unsqueeze(1).float(), size=(Hq, Wq),
                                     mode='nearest').squeeze(1).long()
                with torch.no_grad():
                    wrong = (logits.detach().float().argmax(1) != gt_q) & (gt_q != 255)
                    reg = F.interpolate(self._last_p44_mask, size=(Hq, Wq),
                                        mode='nearest')[:, 0] > 0.5
                    sel = (wrong & reg).float()
                if float(sel.sum()) > 0:
                    terms = []
                    for i in range(self.num_modalities):
                        if i == self._img_idx:
                            continue                       # 생존 모달만
                        lg_i = F.interpolate(_al[i].float(), size=(Hq, Wq),
                                             mode='bilinear', align_corners=False)
                        ce_i = F.cross_entropy(lg_i, gt_q, ignore_index=255,
                                               reduction='none')
                        terms.append((ce_i * sel).sum() / sel.sum().clamp(min=1.0))
                    if terms:
                        aux['p44_hard_aux'] = self.p44_hard_pixel_w * (
                            sum(terms) / len(terms))
        if (self.p46_proto is not None and self.training and gt_mask is not None
                and self._current_epoch >= self.p46_proto_warmup_ep):
            # [P46-C3] prototype-contrastive CE. bank 갱신은 **주 forward에서만**
            # — 보조(스타일 2-view/마스킹) branch는 갱신 없이 같은 bank로 당겨야
            # "두 도메인 view가 하나의 prototype으로 수렴"이라는 제약이 된다.
            _pf = m_feat if self.p46_proto_src == 'mfeat' else fused
            aux['p46_proto'] = self.p46_proto_lambda * self.p46_proto(
                _pf, gt_mask, update=(not self._p46_replay_path))
        if self.p47_2 is not None and self.training and gt_mask is not None:
            # [P47-2] uni-modal balance. **추가 forward 없음** — 이 forward가 이미
            # 만든 per-modal feats를 그대로 쓴다(ISSUE-028의 2-forward 문제 무관).
            # img 마스킹(P42/P44)이 걸린 픽셀은 fusion aux_ce와 같은 규약으로
            # ignore 처리한다(발견C: 0-입력에서 GT를 맞히는 장면 prior 환각 방지).
            _u = self.p47_2(feats, gt_mask, epoch=self._current_epoch,
                            img_mask=_img_mask, img_idx=self._img_idx)
            if _u is not None:
                aux['p47_2_uni'] = _u                  # pre-scaled (LAMBDA_U in module)
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


    # ── [P43-T3] inference paths ────────────────────────────────────────────
    @torch.no_grad()
    def _p43_forward_out(self, batched_input: List[torch.Tensor]):
        """Run one eval forward with the mask-cls head forced on; return its raw
        output dict. The semantic path is untouched (SEM_SOURCE is unchanged)."""
        if self.p43 is None:
            raise RuntimeError("[P43] MODEL.P43.M2F_HEAD is off — no mask-cls head")
        if self.training:
            raise RuntimeError("[P43] call model.eval() first")
        prev = self.p43_eval_head
        self.p43_eval_head = True
        try:
            self.forward(batched_input, True)
        finally:
            self.p43_eval_head = prev
        return self._last_p43_out

    @torch.no_grad()
    def panoptic_inference(self, batched_input: List[torch.Tensor],
                           thing_ids: Optional[Sequence[int]] = None,
                           obj_thresh: float = 0.8, overlap_thresh: float = 0.8,
                           size: Optional[Sequence[int]] = None):
        """PQ path: list of (panoptic_seg (h,w) int32, segments_info).

        `thing_ids` defaults to MODEL.P43.THING_IDS (Cityscapes/MUSES trainIds
        11..18); everything else is treated as stuff and merged per class.
        `size` = (H,W) to emit at label resolution.
        """
        out = self._p43_forward_out(batched_input)
        ids = self.p43_thing_ids if thing_ids is None else thing_ids
        return self.p43.panoptic_inference(
            out, ids, obj_thresh=obj_thresh, overlap_thresh=overlap_thresh,
            size=(tuple(size) if size is not None else None))

    @torch.no_grad()
    def semantic_from_queries(self, batched_input: List[torch.Tensor],
                              size: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Analysis path: semantic logits assembled from the queries (B,K,H,W).

        NOT the reported semantic output — the pixel head owns mIoU. This
        exists to measure what the mask branch alone learned (the P30 vs P38 vs
        P43 3-way ablation in the proposal).
        """
        out = self._p43_forward_out(batched_input)
        sem = self.p43.semantic_scores(out)
        if size is not None:
            sem = F.interpolate(sem, size=tuple(size), mode='bilinear',
                                align_corners=False)
        return sem

    def extract_det_pyramid(self, batched_input: List[torch.Tensor]) -> List[torch.Tensor]:
        """Multi-scale pyramid for a detection head (RF-DETR / FCOS), CEFR-aware.

        Per-modality frozen-ViT+LoRA encoding -> reliability-gated fusion -> [P37a]
        the SAME two-pass CEFR feature blend as forward() (so the detection pyramid
        is built on CEFR-refined fused tokens) -> SimpleFPN. Returns the ViTDet
        pyramid *before* the seg head: [s4, s8, s16, s32], each (B, fpn_dim, h, w).

        When CEFR is off (cefr_ctx is None) this is byte-identical to the P35/P36-Det
        path. [P36] router->det residual (zero-init) is applied if the router is on.
        """
        feats = self._encode_all(list(batched_input))
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
        # [P39-V1] seg forward와 동일 순서(CEFR blend 이후)·동일 게이트로 적용
        fused = self._apply_trunk_exp(fused, feats)
        pyramid = self._apply_p43_lateral(self.fpn(fused))   # [P43-T2] (no-op if off)
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
        feats = self._encode_all(list(batched_input))
        fused, aux = self.fusion(feats, None)
        routed = aux.get('routed_logits', None) if isinstance(aux, dict) else None
        fused = self._apply_trunk_exp(fused, feats)   # seg와 동일 게이트 경로
        with torch.no_grad():
            _, feat_s4 = self._decode(fused, routed)
        # [P39-V2] let the queries attend the per-modal token union (bypasses the
        # fused rank bottleneck). Without modal_feats the head silently falls back
        # to the fused-only source, i.e. P38 behaviour.
        return self.m2f(fused, feat_s4,
                        modal_feats=feats if self.m2f.use_modal_src else None)


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
    vic = p39.get('VICREG', {}) or {}
    rca = (mc.get('P40', {}) or {}).get('RCA', {}) or {}
    fcr = (mc.get('P41', {}) or {}).get('FCR', {}) or {}   # [P41] Fused Class-alignment Regularizer
    p42 = (mc.get('P42', {}) or {}).get('MASK_IMG', {}) or {}   # [P42-M1] 조건부 img 마스킹
    p43 = mc.get('P43', {}) or {}                               # [P43] PanopticDual
    p44 = mc.get('P44', {}) or {}                      # [P44-BMR]
    p44_lm = p44.get('LOCAL_MASK', {}) or {}           #   B-3 국소 마스킹
    p44_hp = p44.get('HARD_PIXEL_AUX', {}) or {}       #   M-3 hard-pixel aux
    p44_vr = p44.get('VALIDITY_RENORM', {}) or {}      #   V-1 presence 재정규화
    p44_mk = p44.get('MUTUAL_KL', {}) or {}            #   B-2 peer 상호증류
    p44_rc = p44.get('REL_CORR', {}) or {}             #   B-2 관계형 대응
    p45_fs = (mc.get('P45', {}) or {}).get('FOGSTYLE', {}) or {}   # [P45-F1]
    p46 = mc.get('P46', {}) or {}                      # [P46-CTR] class-transfer recovery
    p46_c3 = p46.get('C3_PROTO', {}) or {}             #   C-3 prototype consistency
    #   C-1(RCS 샘플러)·C-2(EMA teacher masked consistency)는 모델이 아니라
    #   학습 루프의 결선이다 → train_reliadino.py가 MODEL.P46.C1_RCS/C2_MCC를 읽는다.
    p47_2 = mc.get('P47_2', {}) or {}                  # [P47-2] Uni-modal Balance (구 D-2)
    #   OGM_GE(gradient 변조)는 optimizer step 결선이라 train_reliadino.py가
    #   MODEL.P47_2.OGM_GE를 직접 읽는다 (여기서는 head/손실만 만든다).
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
        p39_trunk_mode=p39.get('TRUNK_MODE', 'linear'),
        p39_trunk_hidden=p39.get('TRUNK_HIDDEN', 256),
        p39_arbiter=p39.get('ARBITER', False),
        p39_path_dropout_p=p39.get('PATH_DROPOUT_P', 0.25),
        p39_router_ce_w=p39.get('ROUTER_CE_W', 0.4),
        p391_vicreg=vic.get('ENABLE', False),
        p391_vicreg_lvar=vic.get('LVAR', 0.1),
        p391_vicreg_lcov=vic.get('LCOV', 0.01),
        p391_vicreg_tokens=vic.get('TOKENS', 2048),
        p391_vicreg_lidar_w=vic.get('LIDAR_W', 1.0),
        p391_vicreg_other_w=vic.get('OTHER_W', 0.25),
        p41_fcr=fcr.get('ENABLE', False),
        p41_fcr_lambda=fcr.get('LAMBDA', 0.1),
        p42_mask_img=p42.get('ENABLE', False),
        p42_mask_frac=p42.get('FRAC', 0.5),
        p42_mask_warmup_ep=p42.get('WARMUP_EP', 20),
        p43_m2f_head=p43.get('M2F_HEAD', False),
        # LATERAL is on by default WHEN P43 is on, and follows M2F_HEAD when
        # unset, so `P43: {M2F_HEAD: false}` alone means "P43 fully off = the
        # forward is byte-identical to the lineage baseline". Set LATERAL
        # explicitly to run either half on its own (lateral-only ablation arm).
        p43_lateral=p43.get('LATERAL', p43.get('M2F_HEAD', False)),
        p43_num_taps=p43.get('NUM_TAPS', 3),
        p43_num_queries=p43.get('NUM_QUERIES', 100),
        p43_dec_layers=p43.get('DEC_LAYERS', 6),
        p43_dim=p43.get('DIM', 256),
        p43_num_heads=p43.get('NUM_HEADS', 8),
        p43_mlp_ratio=p43.get('MLP_RATIO', 2.0),
        p43_w_cls=p43.get('W_CLS', 2.0),
        p43_w_bce=p43.get('W_BCE', 5.0),
        p43_w_dice=p43.get('W_DICE', 5.0),
        p43_no_obj_w=p43.get('NO_OBJ_W', 0.1),
        p43_num_points=p43.get('NUM_POINTS', 12544),
        p43_oversample=p43.get('OVERSAMPLE', 3.0),
        p43_importance=p43.get('IMPORTANCE', 0.75),
        p43_deep_supervision=p43.get('DEEP_SUPERVISION', True),
        p43_lambda=p43.get('LAMBDA', 1.0),
        p43_lambda_warmup_ep=p43.get('LAMBDA_WARMUP_EP', 5),
        p43_eval_head=p43.get('EVAL_HEAD', False),
        p43_sem_source=p43.get('SEM_SOURCE', 'pixel'),
        p43_thing_ids=p43.get('THING_IDS', None),
        p44_local_mask=p44_lm.get('ENABLE', False),
        p44_mask_mode=p44_lm.get('MODE', 'rect'),
        p44_mask_frac=p44_lm.get('FRAC', 0.5),
        p44_mask_warmup_ep=p44_lm.get('WARMUP_EP', 20),
        p44_area_ratio=tuple(p44_lm.get('AREA_RATIO', (0.1, 0.5))),
        p44_num_regions=tuple(p44_lm.get('NUM_REGIONS', (1, 3))),
        p44_coverage_dilate=p44_lm.get('COVERAGE_DILATE', 31),
        p44_blob_grid=p44_lm.get('BLOB_GRID', 16),
        p44_blob_p=p44_lm.get('BLOB_P', 0.5),
        p44_hard_pixel_aux=p44_hp.get('ENABLE', False),
        p44_hard_pixel_w=p44_hp.get('WEIGHT', 0.5),
        p44_validity_renorm=p44_vr.get('ENABLE', False),
        p44_validity_dilate=p44_vr.get('DILATE', 1),
        p44_mutual_kl=p44_mk.get('ENABLE', False),
        p44_mkl_w=p44_mk.get('WEIGHT', 0.5),
        p44_mkl_t=p44_mk.get('TEMPERATURE', 1.0),
        p44_mkl_warmup_ep=p44_mk.get('WARMUP_EP', 10),
        p44_rel_corr=p44_rc.get('ENABLE', False),
        p44_rc_w=p44_rc.get('WEIGHT', 0.1),
        p44_rc_pairs=p44_rc.get('PAIRS', 2048),
        p44_rc_mode=p44_rc.get('MODE', 'mse'),
        p44_rc_warmup_ep=p44_rc.get('WARMUP_EP', 10),
        p46_proto=p46_c3.get('ENABLE', False),
        p46_proto_dim_src=p46_c3.get('FEATURE', 'mfeat'),
        p46_proto_lambda=p46_c3.get('LAMBDA', 0.1),
        p46_proto_ema=p46_c3.get('EMA', 0.999),
        p46_proto_temp=p46_c3.get('TEMPERATURE', 0.1),
        p46_proto_pixels=p46_c3.get('PIXELS', 4096),
        p46_proto_warmup_ep=p46_c3.get('WARMUP_EP', 5),
        p47_2_unibal=p47_2.get('ENABLE', False),
        p47_2_lambda_u=p47_2.get('LAMBDA_U', 0.4),
        p47_2_modals=p47_2.get('MODALS', 'all'),
        p47_2_head=p47_2.get('HEAD', 'linear'),
        p47_2_hidden=p47_2.get('HIDDEN', 256),
        p47_2_warmup_ep=p47_2.get('WARMUP_EP', 0),
        p47_2_gt_div=p47_2.get('GT_DIV', 4),
        p47_2_reduce=p47_2.get('REDUCE', 'mean'),
        p45_fogstyle=p45_fs.get('ENABLE', False),
        p45_prob=p45_fs.get('PROB', 0.5),
        p45_sigma=p45_fs.get('SIGMA', 0.5),
        p45_weight=p45_fs.get('WEIGHT', 0.1),
        p45_detach_clean=p45_fs.get('DETACH_CLEAN', True),
        rca_enable=rca.get('ENABLE', False),
        rca_p_max=rca.get('P_MAX', 0.5),
        rca_warmup_ep=rca.get('WARMUP_EP', 20),
        rca_quantile=rca.get('QUANTILE', 0.3),
        rca_alpha_min=rca.get('ALPHA_MIN', 0.1),
        rca_alpha_max=rca.get('ALPHA_MAX', 0.5),
        rca_readout_w=rca.get('READOUT_W', 0.5),
        rca_buf_size=rca.get('BUF_SIZE', 512),
        rca_min_fill=rca.get('MIN_FILL', 128),
        modal_dropout=mdrop.get('ENABLE', False),
        modal_dropout_p=mdrop.get('P', 0.3),
        modal_dropout_targets=tuple(tgt_idx) if tgt_idx else (0, 1),
        modal_dropout_warmup_ep=mdrop.get('WARMUP_EP', 20),
    )
