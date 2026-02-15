#!/usr/bin/env python3
"""MULTIAQUA metrics fix 검증: num_classes=4, 모델 25채널 출력 시나리오."""
import torch
import sys
sys.path.insert(0, ".")
from semseg.metrics import Metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = Metrics(num_classes=4, ignore_label=255, device=device)

    # MULTIAQUA: B=1, H=64, W=64, 모델 25채널 출력
    preds = torch.randn(1, 25, 64, 64, device=device)  # 25채널 → argmax 시 0~24
    labels = torch.randint(0, 5, (1, 64, 64), device=device)  # 0,1,2,3 + 255
    labels[labels == 4] = 255  # ignore

    try:
        metrics.update(preds, labels)
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        print("OK: metrics.update 및 compute_iou/pixel_acc 성공")
        print(f"  mIoU={miou}, mAcc={macc}")
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
