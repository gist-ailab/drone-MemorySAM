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

    # One-hot encode targets. FCOS uses `targets == n_classes` to mark background
    # (negatives), which has no foreground column → that row stays all-zeros so
    # focal loss pushes every class logit toward 0. Only scatter foreground rows
    # to avoid an out-of-bounds index for the background label.
    target_onehot = torch.zeros_like(logits)
    fg = targets < n_classes
    if fg.any():
        target_onehot[fg] = target_onehot[fg].scatter(
            1, targets[fg].unsqueeze(1), 1.0
        )

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
    giou = torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0)  # inf/nan guard (clamp(min=) only guards 0-denom)

    loss = 1 - giou

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of xyxy boxes → (N, M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


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
        assigner: str = 'fcos',
        atss_topk: int = 9,
        atss_scale: float = 8.0,
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
        # 'fcos' (regress-range + center, smallest-area tie-break) or 'atss'
        # (adaptive IoU-threshold positive selection per GT). ATSS usually lifts AP.
        self.assigner = assigner
        self.atss_topk = atss_topk
        self.atss_scale = atss_scale

    def _assign_targets(self, locations, gt_bboxes, gt_labels):
        if self.assigner == 'atss':
            return self._compute_targets_atss(locations, gt_bboxes, gt_labels)
        return self._compute_targets(locations, gt_bboxes, gt_labels)

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

    def _compute_targets_atss(
        self,
        locations: List[torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ATSS assignment for the anchor-free point head. Each location gets a virtual
        anchor box of side = atss_scale*stride. Per GT: take top-k nearest-center points
        per level → IoU threshold = mean+std over those candidates → positives = candidates
        with IoU≥thr whose center is inside the GT; multi-GT resolved by highest IoU.
        reg/centerness targets identical to FCOS."""
        device = locations[0].device
        all_locs = torch.cat(locations, dim=0)
        n_points = all_locs.shape[0]
        n_gts = gt_bboxes.shape[0]
        level_sizes = [loc.shape[0] for loc in locations]

        cls_targets = torch.full((n_points,), self.n_classes, dtype=torch.int64, device=device)
        reg_targets = torch.zeros((n_points, 4), dtype=torch.float32, device=device)
        ctr_targets = torch.zeros((n_points,), dtype=torch.float32, device=device)
        if n_gts == 0:
            return cls_targets, reg_targets, ctr_targets

        # virtual anchor box per point (side = atss_scale * level stride)
        pt_stride = torch.zeros(n_points, device=device)
        off = 0
        for lvl, n in enumerate(level_sizes):
            pt_stride[off:off + n] = self.fpn_strides[lvl]
            off += n
        half = self.atss_scale * pt_stride / 2.0
        anchors = torch.stack([all_locs[:, 0] - half, all_locs[:, 1] - half,
                               all_locs[:, 0] + half, all_locs[:, 1] + half], dim=1)  # (P,4)
        iou = _box_iou(anchors, gt_bboxes)  # (P, M)

        gcx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gcy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        dist = torch.sqrt((all_locs[:, 0:1] - gcx[None]) ** 2
                          + (all_locs[:, 1:2] - gcy[None]) ** 2 + 1e-9)  # (P, M)

        # per-level top-k nearest-center candidates per GT
        candidate = torch.zeros(n_points, n_gts, dtype=torch.bool, device=device)
        off = 0
        for lvl, n in enumerate(level_sizes):
            k = min(self.atss_topk, n)
            idx = dist[off:off + n].topk(k, dim=0, largest=False).indices + off  # (k, M)
            for m in range(n_gts):
                candidate[idx[:, m], m] = True
            off += n

        inside = ((all_locs[:, 0:1] >= gt_bboxes[None, :, 0]) & (all_locs[:, 0:1] <= gt_bboxes[None, :, 2])
                  & (all_locs[:, 1:2] >= gt_bboxes[None, :, 1]) & (all_locs[:, 1:2] <= gt_bboxes[None, :, 3]))

        pos = torch.zeros_like(candidate)
        for m in range(n_gts):
            cm = candidate[:, m]
            if cm.sum() == 0:
                continue
            ci = iou[cm, m]
            thr = ci.mean() + (ci.std(unbiased=False) if ci.numel() > 1 else ci.new_zeros(()))
            pos[:, m] = cm & (iou[:, m] >= thr) & inside[:, m]

        iou_masked = torch.where(pos, iou, torch.full_like(iou, -1.0))
        _, best_gt = iou_masked.max(dim=1)
        positive = pos.any(dim=1)

        sel = best_gt[positive]
        cls_targets[positive] = gt_labels[sel]
        px = all_locs[positive, 0]
        py = all_locs[positive, 1]
        l = px - gt_bboxes[sel, 0]
        t = py - gt_bboxes[sel, 1]
        r = gt_bboxes[sel, 2] - px
        b = gt_bboxes[sel, 3] - py
        reg_targets[positive] = torch.stack([l, t, r, b], dim=1)
        lr_min = torch.min(l, r); lr_max = torch.max(l, r)
        tb_min = torch.min(t, b); tb_max = torch.max(t, b)
        ctr_targets[positive] = torch.sqrt(
            (lr_min / lr_max.clamp(min=1e-6)) * (tb_min / tb_max.clamp(min=1e-6))
        ).clamp(min=0)
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
        # fp32 for numerically-stable loss under AMP (avoid fp16 overflow → inf/nan in GIoU/focal)
        all_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.n_classes).float() for c in cls_logits], dim=1)
        all_reg = torch.cat([r.permute(0, 2, 3, 1).reshape(B, -1, 4).float() for r in bbox_pred], dim=1)
        all_ctr = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1).float() for c in centerness], dim=1)

        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_ctr_loss = 0.0
        total_pos = 0

        for b in range(B):
            cls_tgt, reg_tgt, ctr_tgt = self._assign_targets(
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
