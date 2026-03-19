"""
MemorySAM Detection Model.

기존 MemorySAM segmentation backbone (SAM2 encoder + MoE LoRA + Memory Attention)을
그대로 재사용하면서, segmentation head 대신 FCOS detection head를 부착.

Architecture:
  Input (RGB, LiDAR, Thermal)
    → SAM2 Encoder (Hiera-B+ with MoE LoRA) × per modality
    → Memory Attention (cross-modal fusion)
    → FPN features: fpn[0]=(B,32,64,64), fpn[1]=(B,64,32,32), fpn[2]=(B,256,16,16)
    → FCOS Head → cls_logits, bbox_pred, centerness
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from objdet.models.heads.fcos_head import FCOSHead
from objdet.losses import FCOSLoss
from objdet.utils.nms import batched_nms


class MemorySAMDetector(nn.Module):
    """
    MemorySAM backbone + FCOS detection head.

    기존 segmentation 모델(P9/P22 등)의 backbone을 freeze 또는 fine-tune하고,
    detection head만 학습하거나 전체를 end-to-end로 학습.

    Args:
        seg_model: 기존 LoRA_Sam_P9 / P22 등 segmentation 모델 인스턴스.
            forward_image() + _prepare_backbone_features()를 통해 FPN feature 제공.
        n_classes: Detection 클래스 수.
        fpn_channels: FPN 채널 dims. SAM2 Hiera-B+ default: [32, 64, 256].
        fpn_strides: FPN stride. Default: [16, 32, 64].
        freeze_backbone: True이면 backbone (encoder + MoE LoRA) 전체 freeze.
        freeze_memory: True이면 memory attention freeze.
        det_head_cfg: FCOS head config overrides.
    """

    def __init__(
        self,
        seg_model: nn.Module,
        n_classes: int = 2,
        fpn_channels: List[int] = [32, 64, 256],
        fpn_strides: List[int] = [16, 32, 64],
        freeze_backbone: bool = True,
        freeze_memory: bool = True,
        n_convs: int = 4,
        hidden_dim: int = 256,
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.seg_model = seg_model
        self.freeze_backbone = freeze_backbone
        self.freeze_memory = freeze_memory

        # Freeze backbone parameters
        if freeze_backbone:
            for name, param in self.seg_model.named_parameters():
                param.requires_grad = False

        # FCOS detection head
        self.det_head = FCOSHead(
            fpn_channels=fpn_channels,
            n_classes=n_classes,
            n_convs=n_convs,
            hidden_dim=hidden_dim,
            fpn_strides=fpn_strides,
            regress_ranges=regress_ranges,
        )

        # Loss
        self.criterion = FCOSLoss(
            n_classes=n_classes,
            fpn_strides=fpn_strides,
            regress_ranges=regress_ranges or [(-1, 64), (64, 128), (128, 1e8)],
        )

    def extract_fpn_features(
        self,
        sample: dict,
        multimask_output: bool = True,
    ) -> List[torch.Tensor]:
        """
        MemorySAM backbone을 통해 cross-modal fused FPN features를 추출.

        기존 seg_model.forward()의 encoder + memory attention 파트만 실행하고,
        FPN features를 반환. Segmentation head는 실행하지 않음.

        Args:
            sample: Dict with modality keys → (B, 3, H, W) tensors.

        Returns:
            fpn_features: List of (B, C, H, W) per FPN level.
                fpn[0]: (B, 32, 64, 64)
                fpn[1]: (B, 64, 32, 32)
                fpn[2]: (B, 256, 16, 16)
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            # seg_model.forward()를 호출하되, 내부적으로 FPN features를 캐시하도록 hook 사용
            # 또는 seg_model이 _last_fpn_features 속성을 노출하도록 수정
            #
            # 현재 구현: seg_model.forward()를 full로 실행하고
            # backbone_fpn을 중간에 추출
            model_out = self.seg_model(sample, multimask_output=multimask_output)

        # seg_model이 forward 시 _last_backbone_fpn을 저장하도록 해야 함
        # 또는 직접 encoder를 호출
        fpn_features = getattr(self.seg_model, '_last_backbone_fpn', None)
        if fpn_features is None:
            raise RuntimeError(
                "seg_model must expose _last_backbone_fpn after forward(). "
                "Add `self._last_backbone_fpn = [img_emb['backbone_fpn'][0] for ...]` "
                "in the seg_model's forward method."
            )
        return fpn_features

    def forward(
        self,
        sample: dict,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        multimask_output: bool = True,
    ) -> dict:
        """
        Args:
            sample: Dict with modality keys → (B, 3, H, W).
            gt_bboxes: List of (N_i, 4) per image (training only).
            gt_labels: List of (N_i,) per image (training only).

        Returns:
            Training: dict with losses.
            Inference: dict with 'boxes', 'scores', 'class_ids' per image.
        """
        fpn_features = self.extract_fpn_features(sample, multimask_output)

        # FCOS head
        head_out = self.det_head(fpn_features)
        locations = self.det_head.get_locations(fpn_features, fpn_features[0].device)

        if self.training and gt_bboxes is not None:
            # Compute loss
            losses = self.criterion(
                head_out['cls_logits'],
                head_out['bbox_pred'],
                head_out['centerness'],
                locations,
                gt_bboxes,
                gt_labels,
            )
            return losses
        else:
            # Decode and NMS
            results_per_img = self.det_head.decode_predictions(
                head_out['cls_logits'],
                head_out['bbox_pred'],
                head_out['centerness'],
                locations,
            )

            final_results = []
            for boxes, scores, cls_ids in results_per_img:
                if boxes.shape[0] > 0:
                    keep = batched_nms(boxes, scores, cls_ids, iou_threshold=0.5)
                    final_results.append({
                        'boxes': boxes[keep],
                        'scores': scores[keep],
                        'class_ids': cls_ids[keep],
                    })
                else:
                    final_results.append({
                        'boxes': boxes,
                        'scores': scores,
                        'class_ids': cls_ids,
                    })
            return {'detections': final_results}

    def get_trainable_params(self) -> List[nn.Parameter]:
        """학습 가능한 파라미터만 반환 (det_head + unfrozen backbone)."""
        return [p for p in self.parameters() if p.requires_grad]
