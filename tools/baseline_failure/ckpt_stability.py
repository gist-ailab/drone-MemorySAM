#!/usr/bin/env python3
"""
A5 — 체크포인트 간 안정성 분류. 여러 체크포인트의 예측 덤프를 공통 GT 로 이미지별로
채점해, 각 이미지가 체크포인트를 바꿔도 일관되게 나쁜지(always_fail) / 좋은지
(always_pass) / 흔들리는지(flip) 분류한다.

규칙(설계서 A5):
- 이미지별 mIoU 는 GT 에 없는 클래스를 NaN 제외한 평균(common 규약, D2 와 동일).
- 이미지별 mIoU 가 **그 체크포인트의 데이터셋 중앙값 미만**인 체크포인트 수를 센다.
  그 비율이 --fail_ratio(기본 9/11 ≈ 0.818) 이상이면 always_fail,
  반대로 중앙값 이상인 비율이 같은 기준을 넘으면 always_pass, 그 사이는 flip.

출력:
- `<out>/ckpt_stability_<split>.csv` — image_id, condition, case, [mIoU per tag],
  mean, std, min, max, n_below_median, label.
- `<out>/ckpt_stability_<split>.json` — 라벨별 장수, 조건별·케이스별 라벨 분포,
  클래스별 체크포인트 간 IoU 표준편차(+ 체크포인트별 중앙값·클래스별 IoU).

image_id 는 common.py 규약(데이터셋 루트 기준 상대 경로, 중첩)을 그대로 쓰고,
GT 와 예측·예측 간 장수(image_id 집합)가 어긋나면 명확한 에러로 멈춘다.

예:
  python tools/baseline_failure/ckpt_stability.py \
    --gt .../ours/test/gt --split test --out .../analysis/test \
    --preds ep40=.../ep40/test/pred ep55=.../ep55/test/pred ep70=.../ep70/test/pred
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES

# always_fail/always_pass 판정 기준비율 — 설계서 기본값 9/11(≈0.818).
DEFAULT_FAIL_RATIO = 9.0 / 11.0

LABELS = ("always_fail", "always_pass", "flip")


def _resolve_pred_dir(path):
    """모델 인자가 pred 디렉터리 자체이거나 split 루트일 수 있으니 정규화한다."""
    p = Path(path)
    if (p / "pred").is_dir():
        return p / "pred"
    return p


def per_image_miou(tag, pred_dir, gt_index):
    """한 체크포인트의 {image_id: mIoU_img(%)} 와 전역 혼동행렬·클래스별 IoU 반환.

    image_id 집합이 GT 와 어긋나면(장수 불일치) 명확한 에러로 멈춘다.
    """
    pred_dir = _resolve_pred_dir(pred_dir)
    pred_index = common.index_label_pngs(pred_dir, require_nested=True)
    gt_ids, pred_ids = set(gt_index), set(pred_index)
    if pred_ids != gt_ids:
        missing = sorted(gt_ids - pred_ids)[:3]
        extra = sorted(pred_ids - gt_ids)[:3]
        raise RuntimeError(
            f"[{tag}] image_id 집합이 GT 와 어긋난다(gt={len(gt_ids)} "
            f"pred={len(pred_ids)}; GT 에 없는 예측 예: {extra}, 예측 없는 GT 예: {missing}). "
            f"장수 불일치 — 덤프가 같은 상대 경로 규약·스플릿인지 확인하라.")

    per_img, global_cm = {}, np.zeros((N, N), dtype=np.int64)
    for image_id in sorted(gt_ids):
        gt = common.load_label_png(gt_index[image_id])
        pred = common.load_label_png(pred_index[image_id])
        if pred.shape != gt.shape:
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
        cm = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
        global_cm += cm
        miou = common.nanmean(common.per_image_iou(cm)) * 100.0
        if not np.isfinite(miou):
            raise RuntimeError(
                f"[{tag}] {image_id} 의 이미지별 mIoU 가 NaN 이다(GT 전 클래스 부재?) — "
                f"채점할 수 없는 이미지다.")
        per_img[image_id] = float(miou)
    per_class_iou, _ = common.global_miou_from_cm(global_cm)
    return per_img, per_class_iou


def classify(n_ckpts, n_below, fail_ratio):
    """n_below(중앙값 미만 체크포인트 수)로 라벨을 정한다(규칙 = 모듈 docstring 참조)."""
    if n_ckpts == 0:
        raise ValueError("체크포인트가 0개다")
    if n_below / n_ckpts >= fail_ratio:
        return "always_fail"
    if (n_ckpts - n_below) / n_ckpts >= fail_ratio:
        return "always_pass"
    return "flip"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="공통 GT trainID PNG 디렉터리")
    ap.add_argument("--preds", required=True, nargs="+", metavar="TAG=DIR",
                    help="체크포인트별 예측 디렉터리(체크포인트마다 하나). 예: ep40=.../pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail_ratio", type=float, default=DEFAULT_FAIL_RATIO,
                    help=f"always_fail/always_pass 판정 비율(기본 9/11={DEFAULT_FAIL_RATIO:.4f}). "
                         f"중앙값 미만 비율·중앙값 이상 비율이 이 값 이상이면 각각 판정한다.")
    args = ap.parse_args()

    if not 0.0 < args.fail_ratio <= 1.0:
        ap.error(f"--fail_ratio 는 (0, 1] 범위여야 한다: {args.fail_ratio}")

    preds = []
    for spec in args.preds:
        if "=" not in spec:
            ap.error(f"--preds 는 tag=dir 형식이어야 함: {spec}")
        tag, d = spec.split("=", 1)
        preds.append((tag, d))
    if len(preds) < 2:
        ap.error("안정성 분류에는 체크포인트 2개 이상이 필요하다(--preds 를 여러 개)")

    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    tags = [t for t, _ in preds]
    n_ckpts = len(preds)

    per_img_by_tag, medians, per_class_by_tag = {}, {}, {}
    for tag, d in preds:
        per_img, per_class_iou = per_image_miou(tag, d, gt_index)
        per_img_by_tag[tag] = per_img
        per_class_by_tag[tag] = per_class_iou
        medians[tag] = float(np.median(list(per_img.values())))
        print(f"[{tag}] {len(per_img)} imgs  이미지별 mIoU 중앙값={medians[tag]:.2f}")

    image_ids = sorted(gt_index)
    rows = []
    label_counts = {lab: 0 for lab in LABELS}
    cond_dist, case_dist = {}, {}
    for image_id in image_ids:
        vals = np.array([per_img_by_tag[t][image_id] for t in tags], dtype=np.float64)
        n_below = int(sum(per_img_by_tag[t][image_id] < medians[t] for t in tags))
        label = classify(n_ckpts, n_below, args.fail_ratio)
        cond, case = common.parse_condition_case(image_id)
        label_counts[label] += 1
        cond_dist.setdefault(cond, {lab: 0 for lab in LABELS})[label] += 1
        case_dist.setdefault(case, {lab: 0 for lab in LABELS})[label] += 1
        rows.append({
            "image_id": image_id, "condition": cond, "case": case,
            **{f"mIoU_{t}": round(float(v), 6) for t, v in zip(tags, vals)},
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
            "n_below_median": n_below,
            "label": label,
        })

    # 클래스별 체크포인트 간 IoU 표준편차(전역 혼동행렬 기반, 부재 클래스=0 포함 규약).
    class_iou_mat = np.array([per_class_by_tag[t] for t in tags], dtype=np.float64)
    class_iou_std = class_iou_mat.std(axis=0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ckpt_stability_{args.split}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "split": args.split,
        "n_checkpoints": n_ckpts,
        "tags": tags,
        "fail_ratio": args.fail_ratio,
        "label_counts": label_counts,
        "label_dist_by_condition": cond_dist,
        "label_dist_by_case": case_dist,
        "median_miou_per_ckpt": medians,
        "per_class_iou_per_ckpt": {t: per_class_by_tag[t] for t in tags},
        "per_class_iou_std_across_ckpts": {
            c: round(float(class_iou_std[i]), 6) for i, c in enumerate(common.CLASSES)},
        "rule": ("이미지별 mIoU 가 체크포인트 중앙값 미만인 비율 ≥ fail_ratio → always_fail, "
                 "중앙값 이상 비율 ≥ fail_ratio → always_pass, 그 사이 → flip. "
                 "이미지별 mIoU 는 GT 부재 클래스 NaN 제외 평균(common 규약)."),
    }
    json_path = out_dir / f"ckpt_stability_{args.split}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"[ckpt_stability] {len(rows)} imgs x {n_ckpts} ckpts  "
          f"라벨 장수={label_counts}  -> {csv_path.name}, {json_path.name}")


if __name__ == "__main__":
    main()
