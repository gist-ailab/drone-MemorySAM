#!/usr/bin/env python3
"""
C — 기준선 라벨 규약 대조 검증.

기준선(detectron2)이 쓰는 category_id 인덱스가 우리 규약(DELIVER 원본 라벨 255→0,
그 뒤 −1 → 0~24, ignore 255)과 같은지 확정한다. 두 방법을 모두 제공한다:

(i) IoU 이동량 탐색 (--pred_dir --gt_dir)
    변환된 PNG(rle_json_to_png, 기본 offset 0)와 우리 GT 로더로 읽은 GT 를 같은
    이미지에서 대조해, 예측 인덱스를 −1/0/+1 이동시킨 세 가설 중 전역 mIoU 가 최대인
    이동량을 보고한다(정상이면 0). GT 는 기본적으로 우리 GT 덤프(trainID)로 읽고,
    원본 semantic PNG 를 직접 주면 --gt-raw 로 deliver.py 규약 변환을 적용한다.

(ii) category_id 히스토그램 (--json)
    기준선 예측 JSON 에 나타난 category_id 의 최솟값·최댓값·빈도표를 출력한다.
    (0~24 면 우리와 같음, 1~25 면 1-based → offset −1 필요 신호.)

판정:
- 최적 이동량이 0 이 아니면 변환기(rle_json_to_png)에 `--label_offset <이동량>` 을
  주어 보정해야 한다. **기본은 보정 없이 에러(exit 3)로 멈춘다** — 추측 보정 금지.

예:
  # (ii) JSON 범위만
  python tools/baseline_failure/check_label_convention.py \
    --json output/.../inference/sem_seg_predictions.json
  # (i) 변환 PNG vs 우리 GT 덤프로 이동량 확정
  python tools/baseline_failure/check_label_convention.py \
    --pred_dir <ROOT>/dgffinal/test/pred --gt_dir <ROOT>/ours/test/gt
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES
SHIFTS = [-1, 0, 1]


def shift_labels(pred, shift):
    """trainID 예측을 shift 만큼 이동. 범위(0~N-1) 밖이 되면 255(ignore)로."""
    sp = pred.astype(np.int32)
    valid = sp != 255
    sp[valid] += shift
    out_of_range = valid & ((sp < 0) | (sp >= N))
    sp[out_of_range] = 255
    sp[~valid] = 255
    return sp


def method_i_iou_shift(pred_dir, gt_dir, gt_raw, limit):
    """(i) 이동량 −1/0/+1 각각의 전역 mIoU 를 재고 최댓값 이동량을 반환."""
    pred_index = common.index_label_pngs(pred_dir)
    gt_index = common.index_label_pngs(gt_dir)
    ids = sorted(set(pred_index) & set(gt_index))
    if not ids:
        raise RuntimeError(
            f"공통 image_id 가 없다(pred={len(pred_index)} gt={len(gt_index)}) — "
            f"상대 경로 규약이 어긋났을 수 있다.")
    if limit:
        ids = ids[:limit]

    hists = {s: np.zeros((N, N), dtype=np.int64) for s in SHIFTS}
    for image_id in ids:
        pred = common.load_label_png(pred_index[image_id])
        gt = (common.load_gt_deliver(gt_index[image_id]) if gt_raw
              else common.load_label_png(gt_index[image_id]))
        if pred.shape != gt.shape:
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
        for s in SHIFTS:
            hists[s] += common.confusion_matrix(shift_labels(pred, s), gt, N,
                                                common.IGNORE_LABEL)

    mious = {}
    for s in SHIFTS:
        _, miou = common.global_miou_from_cm(hists[s])
        mious[s] = miou
    best = max(SHIFTS, key=lambda s: mious[s])
    print(f"[check (i)] 이미지 {len(ids)} 장 · 이동량별 전역 mIoU:")
    for s in SHIFTS:
        mark = "  ← 최대" if s == best else ""
        print(f"    shift={s:+d}: mIoU={mious[s]:.4f}{mark}")
    if best == 0:
        print("[check (i)] ✅ 최적 이동량 0 — 기준선 category_id 가 우리 trainID 와 일치.")
    else:
        print(f"[check (i)] ⚠️ 최적 이동량 {best:+d} — rle_json_to_png 에 "
              f"--label_offset {best} 를 주어 보정하라(기본은 에러로 멈춤).")
    return best, mious


def method_ii_json_hist(json_path, sample):
    """(ii) JSON category_id 의 min/max/히스토그램."""
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("predictions", records.get("annotations", []))
    hist = Counter(int(r["category_id"]) for r in records)
    if not hist:
        raise RuntimeError(f"category_id 를 찾지 못함: {json_path}")
    lo, hi = min(hist), max(hist)
    print(f"[check (ii)] JSON category_id: min={lo} max={hi} "
          f"(고유 {len(hist)}종, 레코드 {sum(hist.values())}건)")
    for cid in sorted(hist):
        print(f"    id={cid:>3}: {hist[cid]}")
    if lo >= 0 and hi <= N - 1:
        print("[check (ii)] category_id 범위가 0~24 — 우리 trainID 와 정합(offset 0).")
    elif lo >= 1 and hi <= N:
        print("[check (ii)] ⚠️ 범위가 1~25 로 보임 → 1-based, --label_offset -1 신호.")
    else:
        print("[check (ii)] ⚠️ 예상 밖 범위 — 수동 확인 필요.")
    return {"min": lo, "max": hi, "histogram": dict(sorted(hist.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default=None, help="(i) 변환된 예측 PNG 디렉터리")
    ap.add_argument("--gt_dir", default=None, help="(i) GT 디렉터리(우리 GT 덤프 또는 원본)")
    ap.add_argument("--gt-raw", action="store_true",
                    help="(i) gt_dir 이 원본 semantic PNG 면 deliver.py 규약 변환 적용")
    ap.add_argument("--json", default=None, help="(ii) sem_seg_predictions.json")
    ap.add_argument("--limit", type=int, default=0, help="(i) 앞 N 장만(0=전체)")
    ap.add_argument("--sample", type=int, default=0, help="(ii) 미사용(호환용)")
    ap.add_argument("--strict", action="store_true",
                    help="이동량이 0 이 아니면 exit 3 으로 멈춘다(기본 True 동작)")
    args = ap.parse_args()

    if not args.pred_dir and not args.json:
        ap.error("--pred_dir/--gt_dir (방법 i) 또는 --json (방법 ii) 중 하나는 필요")

    best = 0
    if args.json:
        method_ii_json_hist(args.json, args.sample)
    if args.pred_dir:
        if not args.gt_dir:
            ap.error("--pred_dir 를 주면 --gt_dir 도 필요(방법 i)")
        best, _ = method_i_iou_shift(args.pred_dir, args.gt_dir, args.gt_raw, args.limit)

    # 기본 동작: 이동량이 0 이 아니면 에러로 멈춘다(추측 보정 금지).
    if args.pred_dir and best != 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
