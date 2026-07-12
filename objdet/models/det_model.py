"""
MemorySAM Detection Model (P29-Det).

기존 MemorySAM segmentation backbone (SAM2 encoder + SoftMoE-LoRA +
cross-modal Memory Attention) 의 **encoder + memory attention 을 그대로 통과**시키고,
segmentation head 대신 FPN neck + FCOS detection head 를 부착한다.

Feature 경로 (사용자 결정: "Encoder FPN + memory-fused"):
  Input (RGB, LiDAR, Thermal)  — 각 모달리티 (B, 3, H, W)
    → SAM2 Encoder (Hiera-B+, SoftMoE-LoRA)               · per modality
    → Memory Attention (cross-modal, RBMA logit-bias)      · track_step loop
    → backbone.extract_det_features() 가 캡처:
         fpn0 (B, 32, H/4,  W/4)   encoder 고해상도 detail   · per modality
         fpn1 (B, 64, H/8,  W/8)   encoder 중해상도 detail   · per modality
         mem  (B,256, H/16, W/16)  memory-conditioned coarse · per modality
    → modality fusion (mean) → [fpn0, fpn1, mem]
    → FPNNeck (lateral 1x1 + top-down) → P3/P4/P5 (모두 256ch, stride 4/8/16)
    → FCOSHead → cls_logits, bbox_pred, centerness

저해상도 coarse level(P5) 은 memory attention + RBMA 를 통과한 cross-modal feature 라
"encoder 와 memory attention 이 결합" 된 요건을 충족하고, 상위 FPN level 은 top-down
경로로 그 cross-modal semantics 를 고해상도 detail 에 주입한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from objdet.models.heads.fcos_head import FCOSHead
from objdet.losses import FCOSLoss
from objdet.utils.nms import batched_nms


class FPNNeck(nn.Module):
    """가벼운 top-down FPN neck.

    입력은 finest→coarsest 순서의 feature list
    (fpn0=stride4, fpn1=stride8, mem=stride16) 이고, 모두 out_channels 로 lateral
    projection 한 뒤 coarsest level 에서 top-down 으로 합쳐 detection FPN 을 만든다.
    """

    def __init__(self, in_channels: List[int] = [32, 64, 256], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])
        for lat in self.lateral_convs:
            nn.init.kaiming_uniform_(lat.weight, a=1)
            nn.init.zeros_(lat.bias)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: [fpn0 (finest), fpn1, mem (coarsest)]
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        # top-down: coarsest → finest
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='nearest',
            )
            laterals[i - 1] = laterals[i - 1] + up
        return [out_conv(l) for out_conv, l in zip(self.output_convs, laterals)]


class MemorySAMDetector(nn.Module):
    """MemorySAM backbone (encoder + memory attention) + FPN neck + FCOS head.

    Args:
        seg_model: extract_det_features() 를 제공하는 LoRA_Sam_P29_Det (또는 P27/P28 호환) 인스턴스.
        modals: 모달리티 순서 리스트 (예: ['img','lidar','thermal']). sample dict 를 이 순서의
            list 로 정렬해 backbone 에 넘긴다 (backbone forward 는 list 입력).
        n_classes: detection 클래스 수.
        fpn_in_channels: backbone 이 노출하는 [fpn0, fpn1, mem] 채널. SAM2 Hiera-B+ = [32,64,256].
        fpn_out: FPN neck / FCOS tower 채널.
        fpn_strides: [P3,P4,P5] stride. 1024 입력 SAM2 = [4,8,16].
        freeze_backbone: True → backbone 전체 freeze (det head 만 학습).
        train_memory: freeze_backbone=False 일 때 sam.memory_attention 도 unfreeze (LoRA+memory fine-tune).
        modality_fuse: 모달리티 융합 방식 ('mean' | 'learned').
    """

    def __init__(
        self,
        seg_model: nn.Module,
        modals: List[str],
        n_classes: int = 2,
        fpn_in_channels: List[int] = [32, 64, 256],
        fpn_out: int = 256,
        fpn_strides: List[int] = [4, 8, 16],
        freeze_backbone: bool = False,
        train_memory: bool = True,
        n_convs: int = 4,
        hidden_dim: int = 256,
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
        assigner: str = 'fcos',
        atss_topk: int = 9,
        atss_scale: float = 8.0,
        modality_fuse: str = 'mean',
    ):
        super().__init__()
        if not hasattr(seg_model, 'extract_det_features'):
            raise TypeError(
                f"{type(seg_model).__name__} has no extract_det_features(); use a "
                "detection-ready backbone such as LoRA_Sam_P29_Det (P27/P28 lineage)."
            )

        self.seg_model = seg_model
        self.modals = list(modals)
        self.n_modals = len(self.modals)
        self.freeze_backbone = freeze_backbone
        self.train_memory = train_memory
        self.modality_fuse = modality_fuse

        # ── Backbone trainability (indoor domain shift → option 2: fine-tune) ──
        if freeze_backbone:
            for p in self.seg_model.parameters():
                p.requires_grad = False
        else:
            # LoRA adapters / decoders / SQG / RBMA λ keep their seg-time requires_grad
            # (SAM2 base is already frozen inside the LoRA_Sam constructor).
            if train_memory and hasattr(self.seg_model.sam, 'memory_attention'):
                for p in self.seg_model.sam.memory_attention.parameters():
                    p.requires_grad = True

        # Learned per-modality fusion weights (optional)
        if modality_fuse == 'learned':
            self.modality_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(self.n_modals)) for _ in range(len(fpn_in_channels))
            ])

        self.neck = FPNNeck(in_channels=fpn_in_channels, out_channels=fpn_out)

        self.det_head = FCOSHead(
            fpn_channels=[fpn_out] * len(fpn_in_channels),
            n_classes=n_classes,
            n_convs=n_convs,
            hidden_dim=hidden_dim,
            fpn_strides=fpn_strides,
            regress_ranges=regress_ranges,
        )

        self.criterion = FCOSLoss(
            n_classes=n_classes,
            fpn_strides=fpn_strides,
            regress_ranges=regress_ranges or [(-1, 64), (64, 128), (128, 1e8)],
            assigner=assigner, atss_topk=atss_topk, atss_scale=atss_scale,
        )

    # ────────────────────────────────────────────────────────────────
    def _ordered_inputs(self, sample: dict) -> List[torch.Tensor]:
        missing = [m for m in self.modals if m not in sample]
        if missing:
            raise KeyError(f"sample missing modalities {missing}; have {list(sample.keys())}")
        return [sample[m] for m in self.modals]

    def _fuse_modalities(self, per_level: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """per_level[level] = list over modalities of (B,C,H,W) → fused (B,C,H,W)."""
        fused = []
        for lvl, feats_m in enumerate(per_level):
            if self.modality_fuse == 'learned':
                w = torch.softmax(self.modality_logits[lvl], dim=0)
                f = sum(w[i] * feats_m[i] for i in range(len(feats_m)))
            else:
                f = torch.stack(feats_m, dim=0).mean(dim=0)
            fused.append(f)
        return fused

    def extract_fpn_features(self, sample: dict) -> List[torch.Tensor]:
        """encoder + memory attention 통과 후 detection FPN [P3,P4,P5] 반환."""
        batched_input = self._ordered_inputs(sample)
        # Only build the backbone graph when actually fine-tuning in train mode;
        # otherwise (frozen backbone, or any eval/inference pass) stay in no_grad
        # so in-training validation of a fine-tuned model doesn't blow up memory.
        backbone_trains = self.training and not self.freeze_backbone
        grad_ctx = torch.enable_grad() if backbone_trains else torch.no_grad()
        with grad_ctx:
            feats = self.seg_model.extract_det_features(batched_input)
        # feats: {'fpn0': [m], 'fpn1': [m], 'mem': [m]}, finest→coarsest
        fused = self._fuse_modalities([feats['fpn0'], feats['fpn1'], feats['mem']])
        return self.neck(fused)

    def forward(
        self,
        sample: dict,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        fpn_features = self.extract_fpn_features(sample)

        head_out = self.det_head(fpn_features)
        locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)

        if self.training and gt_bboxes is not None:
            return self.criterion(
                head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'],
                locations, gt_bboxes, gt_labels,
            )

        results_per_img = self.det_head.decode_predictions(
            head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'], locations,
        )
        final_results = []
        for boxes, scores, cls_ids in results_per_img:
            if boxes.shape[0] > 0:
                keep = batched_nms(boxes, scores, cls_ids, iou_threshold=0.5)
                final_results.append({
                    'boxes': boxes[keep], 'scores': scores[keep], 'class_ids': cls_ids[keep],
                })
            else:
                final_results.append({'boxes': boxes, 'scores': scores, 'class_ids': cls_ids})
        return {'detections': final_results}

    # ────────────────────────────────────────────────────────────────
    def get_trainable_params(self) -> List[nn.Parameter]:
        """det head + FPN neck + (fine-tune 시) backbone LoRA/memory/RBMA."""
        return [p for p in self.parameters() if p.requires_grad]

    def detector_state_dict(self) -> dict:
        """체크포인트용 — det head + neck (+ 학습된 backbone delta 는 별도 저장)."""
        return {
            'det_head': self.det_head.state_dict(),
            'neck': self.neck.state_dict(),
            'modality_logits': (
                [p.detach().cpu() for p in self.modality_logits]
                if self.modality_fuse == 'learned' else None
            ),
        }

    def load_detector_state_dict(self, state: dict):
        self.det_head.load_state_dict(state['det_head'])
        self.neck.load_state_dict(state['neck'])
        if self.modality_fuse == 'learned' and state.get('modality_logits') is not None:
            for p, v in zip(self.modality_logits, state['modality_logits']):
                with torch.no_grad():
                    p.copy_(v.to(p.device))


# ════════════════════════════════════════════════════════════════════════
# P30-Det: Reliability-anchored router fusion + Object-Query decoder + FCOS aux
# ════════════════════════════════════════════════════════════════════════
import math
from objdet.models.heads.query_decoder import (
    ObjectQueryDecoder, HungarianMatcher, SetCriterion, decode_queries,
    box_xyxy_to_cxcywh,
)
from semseg.models.sam2.sam2.sam_lola_utils import ReliabilityAnchoredRouter


class MemorySAMDetectorP30(nn.Module):
    """P30-Det detector (extends the P29-Det feature path with P30's two novelties).

    Backbone = LoRA_Sam_P30_Det (RBMA + SDC). `extract_det_features()` exposes per-modality
    [fpn0(32,s4), fpn1(64,s8), mem(256,s16)] + per-modality seg logits `output`.

      ① Modality fusion = **ReliabilityAnchoredRouter** per FPN level (replaces P29-Det's
         naive mean). Reliability anchor = training-free 1 − H(softmax(seg_output_i))/logCseg,
         resized to each level → the learned router (zero-init) starts reliability-driven and
         can't collapse to a constant (P30 기구 ② ported to detection).
      ② Primary head = **ObjectQueryDecoder** (DETR-style set prediction) cross-attending the
         fused, memory-conditioned `mem` feature (P30 기구 ① ported: class-token → object-token).
      + FCOS dense head as an **aux** for early-training stability (shares the fused FPN).

    Loss (train) = w_query·query_set_loss + w_fcos·fcos_loss − router_reg_lambda·router_entropy.
    Eval returns query detections (primary) as {'detections': [...]} in model-input px space.
    """

    def __init__(
        self,
        seg_model: nn.Module,
        modals: List[str],
        n_classes: int = 2,
        fpn_in_channels: List[int] = [32, 64, 256],
        fpn_out: int = 256,
        fpn_strides: List[int] = [4, 8, 16],
        freeze_backbone: bool = False,
        train_memory: bool = True,
        n_convs: int = 4,
        hidden_dim: int = 256,
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
        assigner: str = 'fcos',
        atss_topk: int = 9,
        atss_scale: float = 8.0,
        img_size: int = 1024,
        # router (P30 ②)
        router_anchor_lambda: float = 1.0,
        router_reg_lambda: float = 0.0,
        # query decoder (P30 ①)
        num_queries: int = 100,
        query_dim: int = 256,
        query_layers: int = 4,
        query_heads: int = 8,
        # aux / loss balance
        use_fcos_aux: bool = True,
        w_query: float = 1.0,
        w_fcos: float = 1.0,
        # ── P31.1 extensions ──
        primary_head: str = 'query',              # 'query' (P30) | 'fcos' (P31.1)
        use_calibrated_reliability: bool = False,  # apply backbone rbma_log_temp (P31.1 Seg-A)
        router_reg_mode: str = 'diversity',       # 'diversity' (P30) | 'decisive' (P31 Lever②)
        use_query_aux: bool = False,              # keep query decoder as aux when fcos primary
    ):
        super().__init__()
        if not hasattr(seg_model, 'extract_det_features'):
            raise TypeError(
                f"{type(seg_model).__name__} has no extract_det_features(); use a "
                "detection-ready P30 backbone (LoRA_Sam_P30_Det).")

        self.seg_model = seg_model
        self.modals = list(modals)
        self.n_modals = len(self.modals)
        self.freeze_backbone = freeze_backbone
        self.train_memory = train_memory
        self.img_size = img_size
        self.n_classes = n_classes
        self.use_fcos_aux = use_fcos_aux
        self.w_query = w_query
        self.w_fcos = w_fcos
        self.router_reg_lambda = router_reg_lambda
        self._levels = len(fpn_in_channels)
        # P31.1
        self.primary_head = primary_head
        self.use_calibrated_reliability = use_calibrated_reliability
        self.use_query_aux = use_query_aux
        self._build_fcos = use_fcos_aux or primary_head == 'fcos'
        self._build_query = primary_head == 'query' or use_query_aux

        # ── Backbone trainability (indoor domain shift → fine-tune) ──
        if freeze_backbone:
            for p in self.seg_model.parameters():
                p.requires_grad = False
        elif train_memory and hasattr(self.seg_model.sam, 'memory_attention'):
            for p in self.seg_model.sam.memory_attention.parameters():
                p.requires_grad = True

        # ① per-level reliability-anchored modality router (P31 Lever②: decisive reg)
        self.routers = nn.ModuleList([
            ReliabilityAnchoredRouter(
                in_ch=c, num_modalities=self.n_modals, num_classes=1,
                per_class=False, anchor_lambda=router_anchor_lambda,
                reg_mode=router_reg_mode)
            for c in fpn_in_channels
        ])

        self.neck = FPNNeck(in_channels=fpn_in_channels, out_channels=fpn_out)

        # object-query decoder — P30 primary / P31.1 optional aux (default OFF for fcos primary)
        if self._build_query:
            self.query_decoder = ObjectQueryDecoder(
                in_ch=fpn_out, n_classes=n_classes, num_queries=num_queries,
                dim=query_dim, heads=query_heads, n_layers=query_layers)
            self.set_criterion = SetCriterion(n_classes, HungarianMatcher())

        # FCOS head + loss — P31.1 PRIMARY (proven, small-object, stable) / P30 aux
        if self._build_fcos:
            self.det_head = FCOSHead(
                fpn_channels=[fpn_out] * self._levels, n_classes=n_classes,
                n_convs=n_convs, hidden_dim=hidden_dim, fpn_strides=fpn_strides,
                regress_ranges=regress_ranges)
            self.criterion = FCOSLoss(
                n_classes=n_classes, fpn_strides=fpn_strides,
                regress_ranges=regress_ranges or [(-1, 64), (64, 128), (128, 1e8)],
                assigner=assigner, atss_topk=atss_topk, atss_scale=atss_scale)

    # ────────────────────────────────────────────────────────────────
    def _ordered_inputs(self, sample: dict) -> List[torch.Tensor]:
        missing = [m for m in self.modals if m not in sample]
        if missing:
            raise KeyError(f"sample missing modalities {missing}; have {list(sample.keys())}")
        return [sample[m] for m in self.modals]

    @staticmethod
    def _reliability(seg_output_i: torch.Tensor, temp=None) -> torch.Tensor:
        """training-free per-modality reliability = 1 − H(softmax(logits/T))/logC → (B,1,H,W).
        temp (P31.1 calibrated): per-modal learned temperature from the backbone."""
        c = seg_output_i.shape[1]
        logits = seg_output_i if temp is None else seg_output_i / temp
        p = F.softmax(logits, dim=1)
        ent = -(p * (p + 1e-8).log()).sum(dim=1, keepdim=True) / math.log(max(c, 2))
        return 1.0 - ent

    def extract_fused_fpn(self, sample: dict):
        """encoder + memory attention → per-level reliability-router fused features.

        Returns (fused: List[(B,C,H,W)] finest→coarsest after neck, router_reg scalar)."""
        batched_input = self._ordered_inputs(sample)
        backbone_trains = self.training and not self.freeze_backbone
        grad_ctx = torch.enable_grad() if backbone_trains else torch.no_grad()
        with grad_ctx:
            feats = self.seg_model.extract_det_features(batched_input)
        per_level = [feats['fpn0'], feats['fpn1'], feats['mem']]   # each: list over modalities
        # per-modality reliability (B,1,Hseg,Wseg) — P31.1: apply calibrated temperature
        temps = None
        if self.use_calibrated_reliability and hasattr(self.seg_model, 'rbma_log_temp'):
            temps = self.seg_model.rbma_log_temp.exp().clamp(0.05, 20.0)   # (m,)
        rels = [self._reliability(o, None if temps is None else temps[i])
                for i, o in enumerate(feats['output'])]                    # m × (B,1,h,w)

        fused, reg_sum = [], 0.0
        for lvl, feats_m in enumerate(per_level):
            hf, wf = feats_m[0].shape[-2:]
            rel_l = torch.stack([
                F.interpolate(r, size=(hf, wf), mode='bilinear', align_corners=False)
                for r in rels], dim=0)                              # (m,B,1,hf,wf)
            w, reg = self.routers[lvl](feats_m, rel_l)              # w: (m,B,1,hf,wf)
            fused.append(sum(w[i] * feats_m[i] for i in range(self.n_modals)))
            reg_sum = reg_sum + reg
        return self.neck(fused), reg_sum / max(self._levels, 1)

    def _query_targets(self, gt_bboxes, gt_labels):
        targets = []
        for b in range(len(gt_bboxes)):
            boxes = gt_bboxes[b].float()
            boxes = box_xyxy_to_cxcywh(boxes) / self.img_size if boxes.numel() > 0 else boxes.reshape(0, 4)
            targets.append({'labels': gt_labels[b].long(), 'boxes': boxes})
        return targets

    def _decode_fcos(self, fpn_features):
        head_out = self.det_head(fpn_features)
        locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)
        results = self.det_head.decode_predictions(
            head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'], locations)
        final = []
        for boxes, scores, cls_ids in results:
            if boxes.shape[0] > 0:
                keep = batched_nms(boxes, scores, cls_ids, iou_threshold=0.6)
                final.append({'boxes': boxes[keep], 'scores': scores[keep], 'class_ids': cls_ids[keep]})
            else:
                final.append({'boxes': boxes, 'scores': scores, 'class_ids': cls_ids})
        return {'detections': final}

    def _forward_fcos_primary(self, fpn_features, router_reg, gt_bboxes, gt_labels):
        """P31.1: FCOS dense head is the PRIMARY output (query decoder off/aux)."""
        if not (self.training and gt_bboxes is not None):
            return self._decode_fcos(fpn_features)
        head_out = self.det_head(fpn_features)
        locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)
        floss = self.criterion(head_out['cls_logits'], head_out['bbox_pred'],
                               head_out['centerness'], locations, gt_bboxes, gt_labels)
        total = self.w_fcos * floss['loss_total']
        losses = {'loss_cls': floss['loss_cls'], 'loss_reg': floss['loss_reg'],
                  'loss_ctr': floss['loss_ctr'], 'n_pos': floss['n_pos']}
        if self.use_query_aux:
            qloss = self.set_criterion(self.query_decoder(fpn_features[-1]),
                                       self._query_targets(gt_bboxes, gt_labels))
            total = total + self.w_query * qloss['loss_query_total']
            losses.update({f'q_{k}': v for k, v in qloss.items()})
        if self.router_reg_lambda > 0:
            total = total - self.router_reg_lambda * router_reg
            losses['loss_router_reg'] = router_reg.detach()
        losses['loss_total'] = total
        return losses

    def forward(self, sample, gt_bboxes=None, gt_labels=None):
        fpn_features, router_reg = self.extract_fused_fpn(sample)

        if self.primary_head == 'fcos':
            return self._forward_fcos_primary(fpn_features, router_reg, gt_bboxes, gt_labels)

        # ② primary query head on the fused coarse (memory-conditioned) level
        query_out = self.query_decoder(fpn_features[-1])

        if self.training and gt_bboxes is not None:
            # query set loss (targets → normalized cxcywh)
            targets = []
            for b in range(len(gt_bboxes)):
                boxes = gt_bboxes[b].float()
                if boxes.numel() > 0:
                    boxes = box_xyxy_to_cxcywh(boxes) / self.img_size
                else:
                    boxes = boxes.reshape(0, 4)
                targets.append({'labels': gt_labels[b].long(), 'boxes': boxes})
            qloss = self.set_criterion(query_out, targets)

            losses = {f'q_{k}': v for k, v in qloss.items()}
            total = self.w_query * qloss['loss_query_total']

            # FCOS aux
            if self.use_fcos_aux:
                head_out = self.det_head(fpn_features)
                locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)
                floss = self.criterion(
                    head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'],
                    locations, gt_bboxes, gt_labels)
                total = total + self.w_fcos * floss['loss_total']
                # surface FCOS keys so train_det's logger (loss_cls/reg/ctr/n_pos) works
                losses.update({
                    'loss_cls': floss['loss_cls'], 'loss_reg': floss['loss_reg'],
                    'loss_ctr': floss['loss_ctr'], 'n_pos': floss['n_pos'],
                })
            else:
                losses.update({
                    'loss_cls': qloss['loss_query_cls'], 'loss_reg': qloss['loss_query_bbox'],
                    'loss_ctr': qloss['loss_query_giou'], 'n_pos': 0,
                })

            if self.router_reg_lambda > 0:
                total = total - self.router_reg_lambda * router_reg   # encourage modality mixing
                losses['loss_router_reg'] = router_reg.detach()
            losses['loss_total'] = total
            return losses

        # ── eval: query detections (primary) ──
        results = decode_queries(query_out, self.img_size, self.img_size)
        final = []
        for r in results:
            if r['boxes'].shape[0] > 0:
                keep = batched_nms(r['boxes'], r['scores'], r['class_ids'], iou_threshold=0.6)
                final.append({'boxes': r['boxes'][keep], 'scores': r['scores'][keep],
                              'class_ids': r['class_ids'][keep]})
            else:
                final.append(r)
        return {'detections': final}

    # ────────────────────────────────────────────────────────────────
    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def detector_state_dict(self) -> dict:
        state = {'routers': self.routers.state_dict(), 'neck': self.neck.state_dict()}
        if self._build_query:
            state['query_decoder'] = self.query_decoder.state_dict()
        if self._build_fcos:
            state['det_head'] = self.det_head.state_dict()
        return state

    def load_detector_state_dict(self, state: dict):
        self.routers.load_state_dict(state['routers'])
        self.neck.load_state_dict(state['neck'])
        if self._build_query and 'query_decoder' in state:
            self.query_decoder.load_state_dict(state['query_decoder'])
        if self._build_fcos and 'det_head' in state:
            self.det_head.load_state_dict(state['det_head'])


# ════════════════════════════════════════════════════════════════════════
# P34-Det: ReliaDINO (frozen DINOv3 + per-modality LoRA + reliability-gated
# fusion + ViTDet SimpleFPN) backbone + FCOS head.
#
# ReliaDINO fuses modalities INTERNALLY and exposes a 4-level pyramid
# (strides 4/8/16/32, all fpn_dim ch) via extract_det_pyramid(). So this
# detector needs neither per-modality fusion nor an FPN neck (unlike the
# SAM2-based MemorySAMDetector). No `.sam` / memory_attention here.
# ════════════════════════════════════════════════════════════════════════
class ReliaDINODetector(nn.Module):
    """ReliaDINO backbone + FCOS head (P34-Det).

    Args:
        seg_model: ReliaDINO instance exposing extract_det_pyramid(list)->[4 levels].
        modals: modality order (e.g. ['img','lidar','thermal']); sample dict is
            ordered to this list before the backbone (backbone takes a list).
        n_classes: detection class count.
        fpn_dim: channel width of every pyramid level (ReliaDINO fpn_dim, =256).
        fpn_strides: pyramid strides (patch16 DINOv3 → [4,8,16,32]).
        freeze_backbone: True → freeze the whole ReliaDINO (train head only).
            (The DINOv3 ViT is already frozen inside FrozenViTEncoder regardless;
            when False only the LoRA adapters + fusion + SimpleFPN stay trainable.)
    """

    def __init__(
        self,
        seg_model: nn.Module,
        modals: List[str],
        n_classes: int = 10,
        fpn_dim: int = 256,
        fpn_strides: List[int] = [4, 8, 16, 32],
        freeze_backbone: bool = False,
        n_convs: int = 4,
        hidden_dim: int = 256,
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
        assigner: str = 'fcos',
        atss_topk: int = 9,
        atss_scale: float = 8.0,
    ):
        super().__init__()
        if not hasattr(seg_model, 'extract_det_pyramid'):
            raise TypeError(
                f"{type(seg_model).__name__} has no extract_det_pyramid(); use a "
                "ReliaDINO backbone for ReliaDINODetector."
            )
        self.seg_model = seg_model
        self.modals = list(modals)
        self.n_modals = len(self.modals)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.seg_model.parameters():
                p.requires_grad = False
        # else: FrozenViTEncoder already froze the ViT; LoRA/fusion/fpn keep their
        # seg-time requires_grad. No memory_attention to unfreeze (no SAM2).

        n_levels = len(fpn_strides)
        default_ranges = [(-1, 64), (64, 128), (128, 256), (256, 1e8)]
        rr = regress_ranges or default_ranges[:n_levels]

        self.det_head = FCOSHead(
            fpn_channels=[fpn_dim] * n_levels,
            n_classes=n_classes,
            n_convs=n_convs,
            hidden_dim=hidden_dim,
            fpn_strides=fpn_strides,
            regress_ranges=rr,
        )
        self.criterion = FCOSLoss(
            n_classes=n_classes,
            fpn_strides=fpn_strides,
            regress_ranges=rr,
            assigner=assigner, atss_topk=atss_topk, atss_scale=atss_scale,
        )

    def _ordered_inputs(self, sample: dict) -> List[torch.Tensor]:
        missing = [m for m in self.modals if m not in sample]
        if missing:
            raise KeyError(f"sample missing modalities {missing}; have {list(sample.keys())}")
        return [sample[m] for m in self.modals]

    def extract_fpn_features(self, sample: dict) -> List[torch.Tensor]:
        batched_input = self._ordered_inputs(sample)
        backbone_trains = self.training and not self.freeze_backbone
        grad_ctx = torch.enable_grad() if backbone_trains else torch.no_grad()
        with grad_ctx:
            pyramid = self.seg_model.extract_det_pyramid(batched_input)
        return pyramid                    # already [fpn_dim]*n_levels, strides 4/8/16/32

    def forward(
        self,
        sample: dict,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        fpn_features = self.extract_fpn_features(sample)
        head_out = self.det_head(fpn_features)
        locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)

        if self.training and gt_bboxes is not None:
            return self.criterion(
                head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'],
                locations, gt_bboxes, gt_labels,
            )

        results_per_img = self.det_head.decode_predictions(
            head_out['cls_logits'], head_out['bbox_pred'], head_out['centerness'], locations,
        )
        final_results = []
        for boxes, scores, cls_ids in results_per_img:
            if boxes.shape[0] > 0:
                keep = batched_nms(boxes, scores, cls_ids, iou_threshold=0.5)
                final_results.append({
                    'boxes': boxes[keep], 'scores': scores[keep], 'class_ids': cls_ids[keep],
                })
            else:
                final_results.append({'boxes': boxes, 'scores': scores, 'class_ids': cls_ids})
        return {'detections': final_results}

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def detector_state_dict(self) -> dict:
        return {'det_head': self.det_head.state_dict()}

    def load_detector_state_dict(self, state: dict):
        self.det_head.load_state_dict(state['det_head'])
