#!/usr/bin/env python3
"""가짜 기둥(Pole) 시각 확인 패널 — GT는 Building/Other인데 Pole로 예측한 픽셀.

배경: thin_error_anatomy 의 fp_sources.md 에서 Pole FP 상위 두 정답 클래스가
일관되게 Building > Other 로 나왔다(2026-10-11). 그 픽셀이 실제로 어떤 장면에서
나오는지(건물 모서리·창틀·간판 기둥 등) 눈으로 확인하기 위한 패널.

사용법::

  python tools/baseline_failure/false_pole_viz.py \
    --gt GT_DIR --target Q2s821=PRED_DIR --compare DGF=PRED_DIR \
    --deliver-root /SSDb/jemo_maeng/dset/DELIVER --split test \
    --out OUT_DIR --topn 20 --random_n 10 --random_min_px 1000

- 기준 모델(--target)의 예측에서 가짜 기둥 픽셀 수를 이미지마다 센다:
  (GT==Building 또는 GT==Other) AND (pred==Pole).
- 비교 모델(--compare, 1개)도 같은 방식으로 세어 제목에 같이 적는다.
- 전수 결과를 <out>/false_pole_counts.csv 에 저장(image_id, condition, 기준모델 px,
  비교모델 px), 기준모델 px 내림차순.
- 상위 topn 장 + (기준모델 px >= random_min_px 인 이미지 중, 상위 topn 제외) 무작위
  random_n 장을 패널로 그린다. 패널 한 행 = RGB · Depth · GT · pred(기준) ·
  pred(비교) · 가짜기둥 마스크(빨강) 오버레이(기준 모델 기준). 제목에 image_id·
  조건(경로의 두 번째 폴더)·가짜기둥 px(기준/비교)를 적는다.
- PNG는 <out>/panels/{top,random}/<image_id 평탄화>.png 로 저장(파일명에 '/'는 '__'
  로 치환 — basename 반복 방지).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.baseline_failure import common  # noqa: E402
from tools.baseline_failure.viz_panels import (  # noqa: E402
    rgb_path_from_id, modality_paths, load_rgb, colorize,
)

BUILDING = common.CLASSES.index("Building")
OTHER = common.CLASSES.index("Other")
POLE = common.CLASSES.index("Pole")


def false_pole_mask(gt, pred):
    return ((gt == BUILDING) | (gt == OTHER)) & (pred == POLE)


def condition_from_id(image_id):
    # image_id = img/<condition>/<split>/<scene>/<stem>_rgb_front
    parts = image_id.split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def count_all(gt_dir, target_dir, compare_dir):
    gt_idx = common.index_label_pngs(gt_dir, require_nested=True)
    tgt_idx = common.index_label_pngs(target_dir, require_nested=True)
    cmp_idx = common.index_label_pngs(compare_dir, require_nested=True)
    ids = sorted(set(gt_idx) & set(tgt_idx) & set(cmp_idx))
    if not ids:
        raise RuntimeError("GT·기준·비교 모델 공통 image_id 가 없다 — 경로 확인 필요")
    rows = []
    for image_id in ids:
        gt = common.load_label_png(gt_idx[image_id])
        tp = common.load_label_png(tgt_idx[image_id])
        cp = common.load_label_png(cmp_idx[image_id])
        if tp.shape != gt.shape:
            tp = common.resize_nearest(tp, *gt.shape)
        if cp.shape != gt.shape:
            cp = common.resize_nearest(cp, *gt.shape)
        tgt_px = int(false_pole_mask(gt, tp).sum())
        cmp_px = int(false_pole_mask(gt, cp).sum())
        rows.append((image_id, condition_from_id(image_id), tgt_px, cmp_px))
    rows.sort(key=lambda r: -r[2])
    return rows


def make_panel(image_id, cond, tgt_px, cmp_px, target_name, compare_name,
                deliver_root, split, gt_dir, target_dir, compare_dir, out_path):
    rgb_p = rgb_path_from_id(deliver_root, split, image_id)
    mpaths = modality_paths(rgb_p)
    gt = common.load_label_png(Path(gt_dir) / f"{image_id}.png")
    H, W = gt.shape
    tp = common.load_label_png(Path(target_dir) / f"{image_id}.png")
    cp = common.load_label_png(Path(compare_dir) / f"{image_id}.png")
    if tp.shape != gt.shape:
        tp = common.resize_nearest(tp, H, W)
    if cp.shape != gt.shape:
        cp = common.resize_nearest(cp, H, W)

    mask = false_pole_mask(gt, tp)
    overlay = load_rgb(mpaths["img"], (H, W)).copy()
    overlay[mask] = (230, 20, 20)

    cols = [
        ("RGB", load_rgb(mpaths["img"], (H, W))),
        ("Depth", load_rgb(mpaths["depth"], (H, W))),
        ("GT", colorize(gt)),
        (f"pred:{target_name}", colorize(tp)),
        (f"pred:{compare_name}", colorize(cp)),
        ("false-pole(red)", overlay),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3.0))
    for ax, (title, im) in zip(axes, cols):
        ax.imshow(im)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{image_id}  [{cond}]  false-pole px: {target_name}={tgt_px} / {compare_name}={cmp_px}",
        fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--target", required=True, metavar="NAME=DIR",
                    help="가짜 기둥 픽셀 수의 기준이 되는 모델")
    ap.add_argument("--compare", required=True, metavar="NAME=DIR",
                    help="제목·패널에 같이 보여줄 비교 모델(1개)")
    ap.add_argument("--deliver-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--random_n", type=int, default=10)
    ap.add_argument("--random_min_px", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_name, t_dir = args.target.split("=", 1)
    c_name, c_dir = args.compare.split("=", 1)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = count_all(args.gt, t_dir, c_dir)

    csv_path = out_root / "false_pole_counts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "condition", f"false_pole_px_{t_name}", f"false_pole_px_{c_name}"])
        w.writerows(rows)
    print(f"[false_pole_viz] 전수 {len(rows)}장 -> {csv_path}")

    top = rows[:args.topn]
    top_ids = {r[0] for r in top}
    pool = [r for r in rows if r[0] not in top_ids and r[2] >= args.random_min_px]
    rng = random.Random(args.seed)
    rand_sel = rng.sample(pool, min(args.random_n, len(pool)))
    print(f"[false_pole_viz] top={len(top)} random={len(rand_sel)}(후보 {len(pool)}장 중, "
          f"min_px={args.random_min_px})")

    made = 0
    for sub, sel in (("top", top), ("random", rand_sel)):
        for image_id, cond, tgt_px, cmp_px in sel:
            flat_name = image_id.replace("/", "__")
            out_path = out_root / "panels" / sub / f"{flat_name}.png"
            try:
                make_panel(image_id, cond, tgt_px, cmp_px, t_name, c_name,
                           args.deliver_root, args.split, args.gt, t_dir, c_dir, out_path)
                made += 1
            except Exception as e:  # noqa: BLE001 — 한 장 실패가 전체를 막지 않게.
                print(f"[false_pole_viz] {image_id} 실패: {e}")
    print(f"[false_pole_viz] 패널 {made}/{len(top) + len(rand_sel)} -> {out_root / 'panels'}")


if __name__ == "__main__":
    main()
