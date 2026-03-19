"""
Detection losses for FCOS: Focal Loss + GIoU Loss + Centerness BCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'sum',
) -> torch.Tensor:
    """
    Focal Loss for classification.

    Args:
        logits: (N, C) raw class logits.
        targets: (N,) integer class labels.
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter.
    """
    n_classes = logits.shape[1]

    # One-hot encode targets
    target_onehot = torch.zeros_like(logits)
    target_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target_onehot, reduction='none')

    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = focal_weight * ce

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def giou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'sum',
) -> torch.Tensor:
    """
    Generalized IoU loss.

    Args:
        pred: (N, 4) predicted boxes (x1, y1, x2, y2).
        target: (N, 4) target boxes (x1, y1, x2, y2).
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=1)
    gt_x1, gt_y1, gt_x2, gt_y2 = target.unbind(dim=1)

    # Intersection
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
    gt_area = (gt_x2 - gt_x1).clamp(min=0) * (gt_y2 - gt_y1).clamp(min=0)
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_y1 = torch.min(pred_y1, gt_y1)
    enclose_x2 = torch.max(pred_x2, gt_x2)
    enclose_y2 = torch.max(pred_y2, gt_y2)
    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)

    loss = 1 - giou

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


class FCOSLoss(nn.Module):
    """
    Combined FCOS loss: Focal (cls) + GIoU (reg) + BCE (centerness).

    Target assignment follows FCOS convention:
      - A pixel (cx, cy) at FPN level l is positive if it falls inside a GT box
        AND the max(l, t, r, b) regression target falls within level l's regress_range.
      - If multiple GT boxes match, the smallest area box is assigned.

    Args:
        n_classes: Number of categories.
        fpn_strides: Stride per FPN level.
        regress_ranges: (min, max) per level.
        focal_alpha: Alpha for focal loss.
        focal_gamma: Gamma for focal loss.
        loss_weights: Dict of loss component weights.
    """

    def __init__(
        self,
        n_classes: int = 2,
        fpn_strides: List[int] = [16, 32, 64],
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        ctr_weight: float = 1.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.fpn_strides = fpn_strides
        self.regress_ranges = regress_ranges or [(-1, 64), (64, 128), (128, 1e8)]
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ctr_weight = ctr_weight

    def _compute_targets(
        self,
        locations: List[torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign FCOS targets for one image.

        Args:
            locations: List of (H*W, 2) per level.
            gt_bboxes: (M, 4) x1y1x2y2.
            gt_labels: (M,).

        Returns:
            cls_targets: (total_points,) — 0..n_classes-1 for positive, n_classes for negative
            reg_targets: (total_points, 4) — (l, t, r, b)
            ctr_targets: (total_points,) — centerness
        """
        n_levels = len(locations)
        all_locs = torch.cat(locations, dim=0)  # (total, 2)
        n_points = all_locs.shape[0]
        n_gts = gt_bboxes.shape[0]

        # Level indices for regress range check
        level_sizes = [loc.shape[0] for loc in locations]

        # Default: all negative
        cls_targets = torch.full((n_points,), self.n_classes, dtype=torch.int64, device=all_locs.device)
        reg_targets = torch.zeros((n_points, 4), dtype=torch.float32, device=all_locs.device)
        ctr_targets = torch.zeros((n_points,), dtype=torch.float32, device=all_locs.device)

        if n_gts == 0:
            return cls_targets, reg_targets, ctr_targets

        # Compute ltrb from each point to each GT box
        # all_locs: (P, 2), gt_bboxes: (M, 4)
        cx = all_locs[:, 0].unsqueeze(1)  # (P, 1)
        cy = all_locs[:, 1].unsqueeze(1)  # (P, 1)

        l = cx - gt_bboxes[:, 0].unsqueeze(0)  # (P, M)
        t = cy - gt_bboxes[:, 1].unsqueeze(0)
        r = gt_bboxes[:, 2].unsqueeze(0) - cx
        b = gt_bboxes[:, 3].unsqueeze(0) - cy

        reg_all = torch.stack([l, t, r, b], dim=2)  # (P, M, 4)

        # Point inside GT box: all ltrb > 0
        inside = reg_all.min(dim=2).values > 0  # (P, M)

        # Max regression distance for regress range check
        max_reg = reg_all.max(dim=2).values  # (P, M)

        # Build level mask for regress range
        level_mask = torch.zeros_like(inside)
        offset = 0
        for level_idx, n_pts in enumerate(level_sizes):
            lo, hi = self.regress_ranges[level_idx]
            level_mask[offset:offset + n_pts] = (max_reg[offset:offset + n_pts] >= lo) & (max_reg[offset:offset + n_pts] <= hi)
            offset += n_pts

        # Valid = inside + within regress range
        valid = inside & level_mask  # (P, M)

        # For each point, pick the GT with smallest area
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])  # (M,)
        areas_expanded = gt_areas.unsqueeze(0).expand(n_points, n_gts)  # (P, M)
        areas_expanded = torch.where(valid, areas_expanded, torch.tensor(1e8, device=all_locs.device))

        min_area, min_gt_idx = areas_expanded.min(dim=1)  # (P,), (P,)
        positive = min_area < 1e8  # (P,)

        # Assign targets for positive points
        cls_targets[positive] = gt_labels[min_gt_idx[positive]]

        # Regression targets
        pos_reg = reg_all[torch.arange(n_points, device=all_locs.device), min_gt_idx]  # (P, 4)
        reg_targets[positive] = pos_reg[positive]

        # Centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
        pos_l, pos_t, pos_r, pos_b = reg_targets[positive].unbind(dim=1)
        lr_min = torch.min(pos_l, pos_r)
        lr_max = torch.max(pos_l, pos_r)
        tb_min = torch.min(pos_t, pos_b)
        tb_max = torch.max(pos_t, pos_b)
        ctr_targets[positive] = torch.sqrt(
            (lr_min / lr_max.clamp(min=1e-6)) * (tb_min / tb_max.clamp(min=1e-6))
        )

        return cls_targets, reg_targets, ctr_targets

    def forward(
        self,
        cls_logits: List[torch.Tensor],
        bbox_pred: List[torch.Tensor],
        centerness: List[torch.Tensor],
        locations: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> dict:
        """
        Args:
            cls_logits: List[(B, C, H, W)] per level.
            bbox_pred:  List[(B, 4, H, W)] per level.
            centerness: List[(B, 1, H, W)] per level.
            locations:  List[(H*W, 2)] per level.
            gt_bboxes:  List of (N_i, 4) per image in batch.
            gt_labels:  List of (N_i,) per image in batch.

        Returns:
            dict: 'loss_cls', 'loss_reg', 'loss_ctr', 'loss_total'
        """
        B = cls_logits[0].shape[0]

        # Flatten predictions across all levels
        all_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.n_classes) for c in cls_logits], dim=1)
        all_reg = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in bbox_pred], dim=1)
        all_ctr = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1) for c in centerness], dim=1)

        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_ctr_loss = 0.0
        total_pos = 0

        for b in range(B):
            cls_tgt, reg_tgt, ctr_tgt = self._compute_targets(
                locations, gt_bboxes[b], gt_labels[b]
            )

            pos_mask = cls_tgt < self.n_classes
            n_pos = pos_mask.sum().item()
            total_pos += n_pos

            # Classification: focal loss on all points
            total_cls_loss += focal_loss(
                all_cls[b], cls_tgt,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction='sum',
            )

            if n_pos > 0:
                # Regression: GIoU on positive points only
                # Decode ltrb → x1y1x2y2 for GIoU
                pos_locs = torch.cat(locations, dim=0)[pos_mask]  # (P_pos, 2)
                pos_reg_pred = all_reg[b][pos_mask]  # (P_pos, 4) — l,t,r,b
                pos_reg_tgt = reg_tgt[pos_mask]      # (P_pos, 4)

                # ltrb → x1y1x2y2
                pred_x1 = pos_locs[:, 0] - pos_reg_pred[:, 0]
                pred_y1 = pos_locs[:, 1] - pos_reg_pred[:, 1]
                pred_x2 = pos_locs[:, 0] + pos_reg_pred[:, 2]
                pred_y2 = pos_locs[:, 1] + pos_reg_pred[:, 3]
                pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

                tgt_x1 = pos_locs[:, 0] - pos_reg_tgt[:, 0]
                tgt_y1 = pos_locs[:, 1] - pos_reg_tgt[:, 1]
                tgt_x2 = pos_locs[:, 0] + pos_reg_tgt[:, 2]
                tgt_y2 = pos_locs[:, 1] + pos_reg_tgt[:, 3]
                tgt_boxes = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=1)

                total_reg_loss += giou_loss(pred_boxes, tgt_boxes, reduction='sum')

                # Centerness: BCE
                total_ctr_loss += F.binary_cross_entropy_with_logits(
                    all_ctr[b][pos_mask], ctr_tgt[pos_mask], reduction='sum'
                )

        # Normalize by total positive samples
        n_pos_safe = max(total_pos, 1)
        loss_cls = self.cls_weight * total_cls_loss / n_pos_safe
        loss_reg = self.reg_weight * total_reg_loss / n_pos_safe
        loss_ctr = self.ctr_weight * total_ctr_loss / n_pos_safe

        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_ctr': loss_ctr,
            'loss_total': loss_cls + loss_reg + loss_ctr,
            'n_pos': total_pos,
        }
