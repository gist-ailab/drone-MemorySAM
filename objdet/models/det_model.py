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
