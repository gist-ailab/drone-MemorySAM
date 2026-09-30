"""DGFusion 공식 가중치의 MUSES val 예측(sem_seg_predictions.json, 클래스별 RLE)을 우리 공식 재채점과 같은 축으로 채점한다.

출력 형식은 tools/eval_muses_official.py 의 report.json 과 같게 맞춘다:
  official_native_1080x1920.per_class · per_condition_official[<날씨>/<주야>].{n_images, mIoU_present_classes, per_class}
검증: 전체 mIoU 가 DGFusion 자체 평가기의 79.7183 과 같아야 한다.
"""
import collections
import json
import os
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

PRED = sys.argv[1]
GT_ROOT = sys.argv[2]          # .../muses/gt_semantic
OUT = sys.argv[3]

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]
LABEL_ID_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
N = 19

preds = json.load(open(PRED))
by_img = collections.defaultdict(list)
for e in preds:
    by_img[e["file_name"]].append(e)

hist_all = np.zeros((N, N), dtype=np.int64)
hist_cond = collections.OrderedDict()
n_cond = collections.Counter()
for fn in sorted(by_img):
    parts = fn.split("/")
    weather, tod = parts[-3], parts[-2]
    cond = f"{weather}/{tod}"
    stem = os.path.basename(fn)
    stem = stem[: stem.rindex("_frame_camera")] if "_frame_camera" in stem else os.path.splitext(stem)[0]
    gt_path = os.path.join(GT_ROOT, "val", weather, tod, stem + "_gt_labelTrainIds.png")
    gt = np.array(Image.open(gt_path)).astype(np.int64)
    pred = np.full(gt.shape, 255, dtype=np.int64)
    for e in by_img[fn]:
        tid = LABEL_ID_TO_TRAIN[int(e["category_id"])]
        rle = e["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode()}
        m = mask_utils.decode(rle).astype(bool)
        pred[m] = tid
    valid = gt != 255
    # 예측이 비어 있는 화소(있다면)는 오답으로 센다: 255 는 혼동행렬 밖이라 gt 쪽 FN 으로 들어가게 N 칸을 쓴다.
    p = np.where(pred == 255, N, pred)
    h = np.bincount(gt[valid] * (N + 1) + p[valid], minlength=N * (N + 1)).reshape(N, N + 1)
    hist_all += h[:, :N]
    hc = hist_cond.setdefault(cond, np.zeros((N, N), dtype=np.int64))
    hc += h[:, :N]
    n_cond[cond] += 1


def per_class(h):
    tp = np.diag(h).astype(np.float64)
    gt_px = h.sum(1).astype(np.float64)
    pr_px = h.sum(0).astype(np.float64)
    union = gt_px + pr_px - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    present = gt_px > 0
    return iou, present


iou, present = per_class(hist_all)
report = {
    "source": "DGFusion official MUSES weights (dgfusion_swin_tiny_bs8_180k_muses_clre.pth), val 250, native 1080x1920",
    "n_images": int(sum(n_cond.values())),
    "official_native_1080x1920": {
        "mIoU_present_classes": round(float(np.nanmean(np.where(present, iou, np.nan)) * 100), 4),
        "per_class": {c: (None if not present[i] else round(float(iou[i] * 100), 4)) for i, c in enumerate(CLASSES)},
        "classes_absent_in_gt": [c for i, c in enumerate(CLASSES) if not present[i]],
    },
    "per_condition_official": {},
}
for cond, h in sorted(hist_cond.items()):
    i2, p2 = per_class(h)
    report["per_condition_official"][cond] = {
        "n_images": int(n_cond[cond]),
        "mIoU_present_classes": round(float(np.nanmean(np.where(p2, i2, np.nan)) * 100), 4),
        "n_present_classes": int(p2.sum()),
        "per_class": {c: (None if not p2[i] else round(float(i2[i] * 100), 4)) for i, c in enumerate(CLASSES)},
    }
os.makedirs(OUT, exist_ok=True)
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), ensure_ascii=False, indent=1)
np.save(os.path.join(OUT, "hist_full.npy"), hist_all)
for cond, h in hist_cond.items():
    np.save(os.path.join(OUT, f"hist_{cond.replace('/', '_')}.npy"), h)
print("overall mIoU", report["official_native_1080x1920"]["mIoU_present_classes"], "n", report["n_images"])
for cond, d in report["per_condition_official"].items():
    print(f"  {cond:12s} n={d['n_images']:3d} mIoU={d['mIoU_present_classes']:.2f}")
