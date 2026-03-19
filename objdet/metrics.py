"""
Detection evaluation metrics (mAP) using pycocotools.
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


def evaluate_coco(
    gt_annotation_path: str,
    predictions: List[dict],
    iou_type: str = 'bbox',
) -> Dict[str, float]:
    """
    Evaluate detections using COCO mAP metrics.

    Args:
        gt_annotation_path: Path to COCO GT annotation JSON.
        predictions: List of dicts, each with:
            'image_id': int
            'category_id': int (original COCO category ID)
            'bbox': [x, y, w, h] (COCO format)
            'score': float
        iou_type: 'bbox' for box detection.

    Returns:
        Dict with AP, AP50, AP75, etc.
    """
    if not HAS_PYCOCOTOOLS:
        raise ImportError("pycocotools is required for COCO evaluation. "
                          "Install with: pip install pycocotools")

    coco_gt = COCO(gt_annotation_path)

    if len(predictions) == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR_1': coco_eval.stats[6],
        'AR_10': coco_eval.stats[7],
        'AR_100': coco_eval.stats[8],
    }


def format_predictions_coco(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    image_id: int,
    idx_to_cat_id: Dict[int, int],
) -> List[dict]:
    """
    Convert model predictions to COCO evaluation format.

    Args:
        boxes: (N, 4) x1y1x2y2 in original image coordinates.
        scores: (N,).
        class_ids: (N,) 0-based contiguous class indices.
        image_id: COCO image ID.
        idx_to_cat_id: Mapping from contiguous idx → original COCO category ID.

    Returns:
        List of COCO prediction dicts.
    """
    preds = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        preds.append({
            'image_id': image_id,
            'category_id': idx_to_cat_id[int(class_ids[i].item())],
            'bbox': [x1, y1, x2 - x1, y2 - y1],  # → (x, y, w, h)
            'score': float(scores[i].item()),
        })
    return preds
