"""
NMS utilities for detection post-processing.
"""

import torch
from torchvision.ops import nms as tv_nms


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Class-aware NMS: apply NMS independently per class.

    Args:
        boxes: (N, 4) x1y1x2y2.
        scores: (N,).
        class_ids: (N,).
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: indices of kept detections.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Offset boxes by class_id to prevent cross-class suppression
    max_coord = boxes.max()
    offsets = class_ids.float() * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = tv_nms(boxes_for_nms, scores, iou_threshold)
    return keep
