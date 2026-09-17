#!/usr/bin/env python3
"""
D1 보조 — DELIVER GT 만 trainID PNG 로 덤프(GPU·모델 불필요, CPU 전용).

목적: 기준선 JSON→PNG 변환의 재현 검산(`rle_json_to_png.py --gt_dir`)은 우리 GT 덤프를
필요로 하는데, `dump_preds_ours.py` 는 모델 추론과 함께 GT 를 쓰므로 빈 GPU 를 기다려야
한다. 이 스크립트는 DELIVER `semantic/` 원본을 deliver.py 규약(255→0, −1 → 0~24 + 255
ignore)으로 읽어 같은 image_id 규약(`img/<cond>/<split>/<scene>/<frame>_rgb_front`)으로
`<out>/<split>/gt/<image_id>.png` 에 저장한다. dump_preds_ours.py 의 gt 덤프와 바이트
동일해야 하며(같은 함수·같은 규약), 장수가 기대값(val 2005 / test 1897)과 다르면 멈춘다.

예:
  python tools/baseline_failure/dump_gt_only.py --dataset_root /SSDb/jemo_maeng/dset/DELIVER \
      --split test --out /SSDe/jemo_maeng/baseline_failure_20260917/ours_gt
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True, help="DELIVER 루트(img/ semantic/ 포함)")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True, help="덤프 루트(<out>/<split>/gt/... 생성)")
    ap.add_argument("--expected_count", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    sem_root = root / "semantic"
    if not sem_root.is_dir():
        raise FileNotFoundError(f"semantic/ 가 없다: {sem_root}")
    gt_dir = Path(args.out) / args.split / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sem_root.glob(f"*/{args.split}/*/*_semantic_front.png"))
    n_saved = 0
    for f in files:
        # semantic/<cond>/<split>/<scene>/<frame>_semantic_front.png → RGB 경로 규약의 image_id
        rel = f.relative_to(root).as_posix()
        rgb_rel = rel.replace("semantic/", "img/", 1).replace("_semantic_front", "_rgb_front")
        image_id = common.image_id_from_rel(rgb_rel)
        common.save_label_png(gt_dir / f"{image_id}.png", common.load_gt_deliver(f))
        n_saved += 1

    expected = args.expected_count or common.EXPECTED_COUNTS.get(args.split)
    if expected is not None:
        assert n_saved == expected, (
            f"장수 불일치: split='{args.split}' GT {n_saved}장(기대 {expected}) — "
            f"dataset_root/스플릿 경로 규약 확인.")
    print(f"[dump_gt] {args.split}: {n_saved} 장 -> {gt_dir}")


if __name__ == "__main__":
    main()
