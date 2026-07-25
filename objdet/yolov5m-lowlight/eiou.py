"""EIoU box-regression loss for classic YOLOv5 — TRAIN-ONLY loss swap.

Drop-in replacement for ``utils.metrics.bbox_iou`` with the *same signature*, so
YOLOv5's ComputeLoss (``lbox += (1.0 - iou).mean()``) transparently minimises
1 - EIoU instead of 1 - CIoU. EIoU (Zhang et al., 2022) replaces CIoU's aspect
-ratio term with direct width/height penalties, which converges better and is
more stable on small / ambiguous boxes — common in low-light frames.

    EIoU = IoU - rho2(center)/c^2 - rho2(w)/cw^2 - rho2(h)/ch^2

where c is the diagonal of the smallest enclosing box and (cw, ch) its width and
height. No running state, no extra parameters -> the inference graph is
untouched, so the model stays i.MX-portable.

Applied by monkeypatch (see train_lowlight.py) — no YOLOv5 source edit.
"""
from __future__ import annotations

import torch


def bbox_iou_eiou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Signature-compatible with YOLOv5 bbox_iou; always returns EIoU.

    The GIoU/DIoU/CIoU flags are accepted (ComputeLoss passes CIoU=True) but
    ignored — we deliberately replace the regression criterion with EIoU.
    """
    # get the coordinates of bounding boxes (mirrors YOLOv5 bbox_iou)
    if xywh:  # (x,y,w,h) -> (x1,y1,x2,y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # (x1,y1,x2,y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # union area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # smallest enclosing box
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps                      # convex diagonal squared

    # squared center distance
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # EIoU: center + explicit width & height penalties
    rho_w2 = (w1 - w2) ** 2
    rho_h2 = (h1 - h2) ** 2
    cw2 = cw ** 2 + eps
    ch2 = ch ** 2 + eps
    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)


if __name__ == "__main__":
    # sanity: identical boxes -> EIoU == 1 ; disjoint -> < 0
    a = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    same = bbox_iou_eiou(a, a.clone(), xywh=False)
    far = bbox_iou_eiou(a, torch.tensor([[100.0, 100.0, 20.0, 20.0]]), xywh=False)
    shift = bbox_iou_eiou(a, torch.tensor([[12.0, 10.0, 20.0, 20.0]]), xywh=False)
    print(f"EIoU(identical)={same.item():.4f} (expect ~1.0)")
    print(f"EIoU(shifted)  ={shift.item():.4f} (expect <1, >far)")
    print(f"EIoU(disjoint) ={far.item():.4f} (expect <0)")
    assert abs(same.item() - 1.0) < 1e-4 and far.item() < shift.item() < same.item()
    print("EIoU sanity OK")
