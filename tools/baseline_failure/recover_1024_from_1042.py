#!/usr/bin/env python3
"""
A9-2 — 이미 만들어진 1042² 예측 PNG 를 1024² 격자로 되돌리는 근사 복원 유틸.

배경: 우리 덤프(dump_preds_ours.py)는 val.py 의 `_unpad_resize_to_orig`
(F.interpolate mode='nearest' = **floor 정렬**)로 원본 해상도 1042² 에 되돌려
저장했다. floor 정렬은 반 픽셀 계통 편차를 만들어(실측: 같은 예측을 floor 로 채점하면
52.34, 중심 정렬로 채점하면 53.68 — 얇은 Pole 은 51.3 vs 57.1), 이미지별 분석의
정본 채점 격자를 1024² 로 정한 지금 기준선(원래 1024²)과 축이 어긋난다.

- 새 덤프는 `dump_preds_ours.py --save_1024` 로 1024² 를 **정확**히 저장한다(권장).
- 이 도구는 **이미 만들어진** 1042² 덤프를 **중심 정렬 최근접**
  (`PIL.Image.resize((1024,1024), Image.NEAREST)`)으로 축소해 근사 복원한다.
  torch 의 `mode='nearest'`(floor 정렬)는 쓰지 않는다.

규칙:
- image_id·저장 경로 = 킷 규약 그대로(중첩 상대 경로, `common.index_label_pngs`
  require_nested + `common.save_label_png`). 저장이 끝나면 장수를 split 기대값
  (val 2005 / test 1897) 또는 --expected_count 와 assert 로 맞는다.
- 입력 PNG 가 1042² 가 아니면(예: 이미 1024²) 보정하지 않고 명확한 에러로 멈춘다.
- 라벨 값은 0~24 와 255(ignore) 만 허용 — 그 밖의 값이면 에러로 멈춘다.
- summary.json 에 `approx_from_1042: true`·`resize_method`·`src_resolution`·
  `dst_resolution`·장수를 기록해 근사 산출임이 항상 남게 한다.

예:
  python tools/baseline_failure/recover_1024_from_1042.py \
    --pred_dir <ROOT>/ours/test/pred --split test \
    --out <ROOT>/ours_rec1024 --expected_count 1897
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

SRC_SIZE = 1042   # 우리 덤프가 _unpad_resize_to_orig 로 되돌린 DELIVER native 해상도
DST_SIZE = 1024   # 정본 채점 격자
# 중심 정렬 최근접(floor 정렬 금지). 어느 쪽 구현인지 summary 에 그대로 남긴다.
RESIZE_METHOD = "PIL.Image.resize((1024,1024), Image.NEAREST)"

_VALID = set(range(common.N_CLASSES)) | {common.IGNORE_LABEL}


def resize_center_nearest(arr):
    """(H,W) trainID 배열을 중심 정렬 최근접(PIL NEAREST)으로 1024² 로 축소한다."""
    from PIL import Image
    return np.array(Image.fromarray(arr.astype(np.uint8))
                    .resize((DST_SIZE, DST_SIZE), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True,
                    help="1042² 예측 PNG 디렉터리(중첩 상대 경로 덤프)")
    ap.add_argument("--out", required=True, help="출력 루트(<out>/<split>/pred/... 생성)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--expected_count", type=int, default=None,
                    help="기대 장수(미지정 시 split 기본값: val 2005 / test 1897)")
    args = ap.parse_args()

    out_root = Path(args.out) / args.split
    pred_out = out_root / "pred"
    pred_out.mkdir(parents=True, exist_ok=True)

    index = common.index_label_pngs(args.pred_dir, require_nested=True)
    print(f"[recover1024] 입력 {len(index)} 장: {args.pred_dir}")

    n_saved = 0
    for image_id in sorted(index):
        arr = common.load_label_png(index[image_id])
        if arr.shape != (SRC_SIZE, SRC_SIZE):
            raise ValueError(
                f"입력 해상도가 {arr.shape[1]}x{arr.shape[0]} (기대 {SRC_SIZE}x{SRC_SIZE}, "
                f"image_id={image_id}) — 이미 1024² 라면 이 도구가 아니라 그대로 쓰고, "
                f"다른 해상도라면 되돌릴 수 없다. 보정하지 않고 멈춘다(정확 경로는 "
                f"dump_preds_ours.py --save_1024).")
        invalid = set(np.unique(arr).tolist()) - _VALID
        if invalid:
            raise ValueError(
                f"{image_id}: 허용 밖 라벨 값 {sorted(invalid)} — trainID 0~24 와 "
                f"255(ignore) 만 허용한다. 규약이 어긋난 덤프는 되돌리지 않는다.")
        recovered = resize_center_nearest(arr)
        common.save_label_png(pred_out / f"{image_id}.png", recovered)
        n_saved += 1

    # 장수 검증 — 덮어쓰기·누락 없이 전부 되돌렸는지 확인.
    expected = (args.expected_count if args.expected_count is not None
                else common.EXPECTED_COUNTS.get(args.split))
    if args.expected_count is not None:
        assert n_saved == args.expected_count, (
            f"장수 불일치: split='{args.split}' 에 {n_saved}장 복원(기대 "
            f"{args.expected_count}). 입력 덤프의 누락·경로 규약 오류를 의심하라.")
    else:
        common.assert_expected_count(n_saved, args.split,
                                     extra=f"(recover_1024_from_1042, out={out_root})")

    summary = {
        "split": args.split,
        "pred_dir": str(args.pred_dir),
        "num_images": n_saved,
        "expected_count": expected,
        "approx_from_1042": True,
        "resize_method": RESIZE_METHOD,
        "src_resolution": SRC_SIZE,
        "dst_resolution": DST_SIZE,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[recover1024] {n_saved} 장 복원 -> {pred_out}")
    print(f"[recover1024] summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
