#!/usr/bin/env python3
"""한 클래스의 오탐(FP) 픽셀이 많은 장면을 모델별로 골라, 그 FP가 어떤 정답
클래스 위에서 생겼는지 분해한다.

사용법::

    python tools/baseline_failure/class_fp_scenes.py --gt GT_DIR \
      --models ours=PRED_DIR baseline=PRED_DIR --cls TrafficLight \
      --topk 20 --rank_by ours --out OUT_DIR

정의 (입력 규약은 thin_error_anatomy.py 와 동일)
==================================================
- 모든 모델은 공통 image_id 의 중첩 trainID PNG 만 사용한다
  (common.index_label_pngs, require_nested=True). 예측 크기가 GT 와 다르면
  GT native 크기로 최근접 보간한다(common.resize_nearest). GT=255 와 범위 밖
  예측/GT 는 common.confusion_matrix 와 똑같이 채점에서 제외한다.
- 조건·케이스는 common.parse_condition_case 로 image_id 에서 뽑는다.
- 클래스 c(=--cls)에 대해 이미지별로 TP=hist[c,c], FP=열 c 합−TP,
  FN=행 c 합−TP, 이미지 IoU=TP/(TP+FP+FN)을 센다. 합이 0이면 IoU 는 빈 칸이다.
- FP 의 정답 클래스 분포: 예측=c, 유효 GT≠c 인 픽셀을 정답 클래스별로 세고
  상위 3개(픽셀 수 내림차순, 같으면 클래스 id 오름차순)만 기록한다.
- 장면 순위는 rank_by 모델의 FP 내림차순(같으면 image_id 오름차순)이며,
  rank_by 의 기본값은 첫 번째 모델이다.

산출물
======
- <cls>_per_image.csv: 행 = 이미지 × 모델. 조건·케이스, TP/FP/FN 픽셀 수,
  이미지 IoU, FP 정답 클래스 상위 3(이름·픽셀 수).
- <cls>_top<k>.csv: rank_by 모델 FP 상위 k 장(파일명의 k 는 요청한 값 그대로).
  각 장의 모든 모델 FP·IoU 와 rank_by 모델의 FP 출처 상위 3을 열로 놓는다.
- <cls>_summary.md: 모델별 전체 FP 합, 모델별 정답 클래스 FP 합 상위 5,
  조건별 FP 합, rank_by 상위 k 장이 각 모델 전체 FP 에서 차지하는 비율.
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES


def _fmt(value):
    return "" if not np.isfinite(value) else f"{value:.6f}"


def _top_sources(counts, k=3):
    """정답 클래스별 FP 픽셀 수 배열에서 (클래스 이름, 픽셀 수) 상위 k(0 제외)."""
    order = [c for c in np.argsort(-counts, kind="stable") if counts[c] > 0][:k]
    return [(common.CLASSES[c], int(counts[c])) for c in order]


def _one_image(task):
    """이미지 하나의 클래스별 TP/FP/FN·IoU·FP 정답 클래스 분포."""
    image_id, gt_path, pred_paths, cls_id = task
    gt = common.load_label_png(gt_path)
    valid_gt = gt < N
    gt_mask = gt == cls_id
    tp, fp, fn, iou, sources = {}, {}, {}, {}, {}
    for name, path in pred_paths.items():
        pred = common.load_label_png(path)
        if pred.shape != gt.shape:
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
        cm = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
        t = int(cm[cls_id, cls_id])
        f = int(cm[:, cls_id].sum()) - t
        n = int(cm[cls_id, :].sum()) - t
        tp[name], fp[name], fn[name] = t, f, n
        union = t + f + n
        iou[name] = t / union if union else float("nan")
        fp_mask = (pred == cls_id) & valid_gt & ~gt_mask
        counts = np.bincount(gt[fp_mask].astype(np.int64), minlength=N)
        counts[cls_id] = 0
        sources[name] = counts
    return image_id, tp, fp, fn, iou, sources


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(out, cls, names, ids, rank_by, topk, stats, src_total):
    """stats[model][image_id] = (tp, fp, fn, iou, [(정답 클래스, 픽셀 수), ...])."""
    out.mkdir(parents=True, exist_ok=True)
    cond_case = {i: common.parse_condition_case(i) for i in ids}
    src_cols = [f"fp_gt{j}_{s}" for j in (1, 2, 3) for s in ("name", "pixels")]
    per_rows = []
    for image_id in ids:
        condition, case = cond_case[image_id]
        for name in names:
            t, f, n, iou, src = stats[name][image_id]
            row = [image_id, condition, case, name, t, f, n, _fmt(iou)]
            for j in range(3):
                row += list(src[j]) if j < len(src) else ["", ""]
            per_rows.append(row)
    _write_csv(out / f"{cls}_per_image.csv",
               ["image_id", "condition", "case", "model", "tp_pixels", "fp_pixels",
                "fn_pixels", "iou", *src_cols], per_rows)
    ranked = sorted(ids, key=lambda i: (-stats[rank_by][i][1], i))
    top = ranked[:topk]
    top_header = ["rank", "image_id", "condition", "case"]
    for name in names:
        top_header += [f"{name}_fp_pixels", f"{name}_iou"]
    top_rows = []
    for r, image_id in enumerate(top, start=1):
        condition, case = cond_case[image_id]
        row = [r, image_id, condition, case]
        for name in names:
            row += [stats[name][image_id][1], _fmt(stats[name][image_id][3])]
        src = stats[rank_by][image_id][4]
        for j in range(3):
            row += list(src[j]) if j < len(src) else ["", ""]
        top_rows.append(row)
    _write_csv(out / f"{cls}_top{topk}.csv", top_header + src_cols, top_rows)
    total_fp = {name: sum(stats[name][i][1] for i in ids) for name in names}
    lines = [f"# {cls} FP 장면 순위 ({len(ids)}장, rank_by={rank_by}, topk={topk})",
             "",
             "입력 규약·지표 정의는 도구 상단 README 를 참조한다. GT=255 는 제외한다.",
             "", "## 모델별 전체 FP 합", "",
             "| 모델 | FP 픽셀 합 | FP>0 장면 수 |", "|---|---:|---:|"]
    for name in names:
        scenes = sum(1 for i in ids if stats[name][i][1] > 0)
        lines.append(f"| {name} | {total_fp[name]} | {scenes} |")
    lines += ["", "## 정답 클래스별 FP 합 상위 5", ""]
    for name in names:
        lines += [f"### {name}", "", "| 정답 클래스 | FP 픽셀 |", "|---|---:|"]
        order = [c for c in np.argsort(-src_total[name], kind="stable")
                 if src_total[name][c] > 0][:5]
        if order:
            lines += [f"| {common.CLASSES[c]} | {int(src_total[name][c])} |" for c in order]
        else:
            lines.append("| (FP 없음) | 0 |")
        lines.append("")
    cond_fp = {name: {} for name in names}
    for image_id in ids:
        condition = cond_case[image_id][0]
        for name in names:
            cond_fp[name][condition] = cond_fp[name].get(condition, 0) \
                + stats[name][image_id][1]
    lines += ["## 조건별 FP 합", "",
              "| 조건 | " + " | ".join(names) + " |",
              "|---|" + "---:|" * len(names)]
    for condition in sorted({cond_case[i][0] for i in ids}):
        cells = [str(cond_fp[name].get(condition, 0)) for name in names]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines += ["", f"## rank_by({rank_by}) FP 상위 {len(top)}장이 각 모델 전체 FP 에서 차지하는 비율", "",
              "| 모델 | 상위 장면 FP 합 | 전체 FP 합 | 비율 |", "|---|---:|---:|---:|"]
    for name in names:
        in_top = sum(stats[name][i][1] for i in top)
        ratio = _fmt(in_top / total_fp[name]) if total_fp[name] else ""
        lines.append(f"| {name} | {in_top} | {total_fp[name]} | {ratio} |")
    (out / f"{cls}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--cls", required=True, metavar="NAME")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--rank_by", default=None,
                    help="장면 순위를 매길 모델 이름(기본=첫 번째 모델)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cls not in common.CLASSES:
        ap.error(f"--cls 는 common.CLASSES 안의 이름이어야 한다: {args.cls}")
    if args.topk <= 0 or args.workers <= 0:
        ap.error("topk 와 workers 는 양수여야 한다")
    model_dirs = {}
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"--models는 NAME=DIR 형식이어야 한다: {spec}")
        name, directory = spec.split("=", 1)
        if (not name or name in (".", "..") or not directory or
                "/" in name or "\\" in name or name in model_dirs):
            ap.error(f"모델 이름/경로 오류 또는 중복: {spec}")
        path = Path(directory)
        model_dirs[name] = path / "pred" if (path / "pred").is_dir() else path
    names = list(model_dirs)
    rank_by = args.rank_by or names[0]
    if rank_by not in model_dirs:
        ap.error(f"--rank_by 는 등록된 모델 이름이어야 한다: {rank_by}")
    cls_id = common.CLASSES.index(args.cls)
    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    model_index = {name: common.index_label_pngs(path, require_nested=True)
                   for name, path in model_dirs.items()}
    ids = set(gt_index)
    for name in names:
        ids &= set(model_index[name])
    ids = sorted(ids)
    if not ids:
        raise RuntimeError("GT와 모든 모델이 공유하는 image_id가 없다")
    if len(ids) < len(gt_index):
        print(f"[class_fp_scenes] 경고: GT {len(gt_index)}장 중 공통 {len(ids)}장만 집계")
    tasks = ((image_id, gt_index[image_id],
              {name: model_index[name][image_id] for name in names},
              cls_id) for image_id in ids)
    stats = {name: {} for name in names}
    src_total = {name: np.zeros(N, dtype=np.int64) for name in names}
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = executor.map(_one_image, tasks, chunksize=1) if executor \
            else map(_one_image, tasks)
        for i, (image_id, tp, fp, fn, iou, sources) in enumerate(results, start=1):
            for name in names:
                stats[name][image_id] = (tp[name], fp[name], fn[name], iou[name],
                                         _top_sources(sources[name]))
                src_total[name] += sources[name]
            if i % 100 == 0 or i == len(ids):
                print(f"[class_fp_scenes] {i}/{len(ids)}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    write_outputs(Path(args.out), args.cls, names, ids, rank_by, args.topk,
                  stats, src_total)
    print(f"[class_fp_scenes] 완료: {args.out}")


if __name__ == "__main__":
    main()
