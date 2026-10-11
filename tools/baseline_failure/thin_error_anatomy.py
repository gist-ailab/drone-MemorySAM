#!/usr/bin/env python3
"""저장된 DELIVER trainID 예측·GT PNG의 얇은 객체 오류를 CPU로 분석한다.

사용법::

    python tools/baseline_failure/thin_error_anatomy.py --gt GT_DIR \
      --models ours=PRED_DIR baseline=PRED_DIR --split test --out OUT_DIR

README (지표 정의)
==================
- 산출물: raw_confusion/<model>.npy, raw_confusion/conditions/<model>/<조건>.npy,
  precision_recall.csv, fp_sources.md, boundary_distance.csv,
  thickness_patch.csv, summary.md.
- 모든 모델은 공통 image_id의 중첩 PNG만 사용한다. 예측 크기가 다르면 GT native
  크기로 최근접 보간한다. GT=255 또는 범위 밖의 예측/GT는 common.confusion_matrix와
  똑같이 채점에서 제외한다. hist[gt, pred] 원시 int64 픽셀 수를 합산한다.
- 클래스 c의 TP=hist[c,c], FP=열 c의 합−TP, FN=행 c의 합−TP,
  IoU=TP/(TP+FP+FN), precision=TP/(TP+FP), recall=TP/(TP+FN).
  1−IoU의 FP/FN 몫은 각각 FP/(TP+FP+FN), FN/(TP+FP+FN)이다.
  분모가 0이면 CSV를 비운다. 전역 mIoU는 common.global_miou_from_cm 규약이다.
- FP 거리=예측 c이고 유효 GT가 c가 아닌 픽셀에서 최근접 GT c까지의 유클리드
  거리; FN 거리=GT c이고 예측이 c가 아닌 픽셀에서 최근접 예측 c까지의 거리.
  EDT로 측정하며 대상 마스크가 없으면 각각 '정답 없음'/'예측 없음'으로 집계한다.
  거리 구간은 native GT 픽셀 기준 (0,1], (1,3], (3,8], (8,16],
  (16,32], (32,∞)이다. 0은 FP/FN 정의상 생기지 않는다.
  모델 입력 픽셀 환산=거리×model_input_px/max(GT 높이, GT 너비),
  패치 환산=그 값/patch_px. CSV의 환산 열은 해당 구간 오류 픽셀의 평균 거리다.
  크기가 다른 GT도 이미지별로 환산한다. all_25_macro 비율은 오류가 있는 클래스별
  구간 비율의 산술평균이며, pixel_count는 25클래스 합계다.
- GT 클래스별 8연결 성분 M의 두께=2×max(성분 내부 EDT), GT 픽셀.
  패치 두께=두께×model_input_px/max(GT 높이, GT 너비)/patch_px.
  구간은 [0,0.25), [0.25,0.5), [0.5,1), [1,2), [2,4), [4,∞).
  찾음=|M∩P|/|M|≥0.5. P는 예측 c 마스크. 성분 bbox를 사방 3px
  팽창한 영역 B에서 성분 IoU=|M∩P|/|M∪(P∩B)|.
  '찾았지만 못 그림'=찾았고 성분 IoU<0.5. 그 비율의 분모는 찾은 성분 수,
  찾음률의 분모는 전체 성분 수다. 성분 픽셀 재현율 평균은 각 성분의
  |M∩P|/|M|를 동일 가중 평균한다. 성분이 없거나 찾은 성분이 없으면 비율을 비운다.
- 이 패치 두께는 GT 형상의 기술 통계다. 이 표만으로 백본의 인과 원인을 증명할
  수 없다. native 크기가 다른 예측은 보간 후 평가하므로 원래 격자와 차이가 날 수 있다.
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES
THIN = set(common.THIN_CLASS_IDS)
DIST_LABELS = ("<=1", "1-3", "3-8", "8-16", "16-32", ">32", "absent")
THICK_LABELS = ("<0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", ">=4")
STRUCTURE8 = np.ones((3, 3), dtype=np.uint8)


def class_metrics(cm):
    """행=GT·열=예측인 누적 원시 행렬의 클래스별 값."""
    tp = np.diag(cm).astype(np.int64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    def ratio(num, den):
        out = np.full(N, np.nan, dtype=np.float64)
        np.divide(num, den, out=out, where=den != 0)
        return out
    return {"tp": tp, "fp": fp, "fn": fn, "iou": ratio(tp, union),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "fp_loss": ratio(fp, union), "fn_loss": ratio(fn, union)}


def distance_bin_counts(error_mask, target_mask):
    """오류 픽셀의 native 거리 구간 수와 거리 합(native·모델·패치 전)."""
    counts = np.zeros(7, dtype=np.int64)
    sums = np.zeros(7, dtype=np.float64)
    if not error_mask.any():
        return counts, sums
    if not target_mask.any():
        counts[6] = int(error_mask.sum())
        return counts, sums
    distances = ndimage.distance_transform_edt(~target_mask)[error_mask]
    bins = np.searchsorted([1, 3, 8, 16, 32], distances, side="left")
    counts[:6] = np.bincount(bins, minlength=6)
    sums[:6] = np.bincount(bins, weights=distances, minlength=6)
    return counts, sums


def thickness_patch_bin(mask, model_input_px, patch_px, gt_side_px):
    """성분 bbox 마스크의 EDT 두께와 패치 구간."""
    padded = np.pad(mask, 1, constant_values=False)
    thick_gt = 2.0 * float(ndimage.distance_transform_edt(padded)[1:-1, 1:-1].max())
    thick_patch = thick_gt * model_input_px / gt_side_px / patch_px
    return thick_gt, thick_patch, int(np.searchsorted([0.25, 0.5, 1, 2, 4],
                                                       thick_patch, side="right"))


def components_for_class(gt_mask, model_input_px, patch_px, gt_side_px):
    """GT 라벨링과 두께를 모델 수와 관계없이 한 번만 계산한다."""
    labels, nlab = ndimage.label(gt_mask, structure=STRUCTURE8)
    comps = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        local = labels[sl] == k
        area = int(local.sum())
        _, _, bin_idx = thickness_patch_bin(local, model_input_px, patch_px, gt_side_px)
        comps.append((sl, local, area, bin_idx))
    return comps


def component_scores(pred_mask, component, shape):
    """bbox를 3px 늘린 국소 IoU와 성분 재현율."""
    sl, local, area, _ = component
    y0, y1 = sl[0].start, sl[0].stop
    x0, x1 = sl[1].start, sl[1].stop
    by0, by1 = max(0, y0 - 3), min(shape[0], y1 + 3)
    bx0, bx1 = max(0, x0 - 3), min(shape[1], x1 + 3)
    pred_box = pred_mask[by0:by1, bx0:bx1]
    inter = int((local & pred_mask[sl]).sum())
    union = area + int(pred_box.sum()) - inter
    return inter / area, inter / union


def _one_image(task):
    image_id, gt_path, pred_paths, model_input_px, patch_px = task
    gt = common.load_label_png(gt_path)
    scale = model_input_px / max(gt.shape)
    preds = {}
    cms = {}
    for name, path in pred_paths.items():
        pred = common.load_label_png(path)
        if pred.shape != gt.shape:
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
        preds[name] = pred
        cms[name] = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
    # 거리 배열 축: 클래스, FP/FN, 구간. 두께 배열: 군, 구간, [수, 찾음, 미묘사, 재현율 합].
    dist = {name: np.zeros((N, 2, 7), dtype=np.int64) for name in preds}
    dist_native = {name: np.zeros((N, 2, 7), dtype=np.float64) for name in preds}
    thick = {name: np.zeros((2, 6, 4), dtype=np.float64) for name in preds}
    valid_gt = (gt < N)
    for c in range(N):
        gt_mask = gt == c
        components = components_for_class(gt_mask, model_input_px, patch_px, max(gt.shape)) \
            if gt_mask.any() else []
        for name, pred in preds.items():
            pred_mask = pred == c
            fn = gt_mask & (pred < N) & ~pred_mask
            if fn.any():
                counts, sums = distance_bin_counts(fn, pred_mask)
                dist[name][c, 1] = counts
                dist_native[name][c, 1] = sums
            for comp in components:
                recall, iou = component_scores(pred_mask, comp, gt.shape)
                group = 0 if c in THIN else 1
                row = thick[name][group, comp[3]]
                row += (1, float(recall >= 0.5),
                        float(recall >= 0.5 and iou < 0.5), recall)
        fp_names = [name for name, pred in preds.items()
                    if np.any((pred == c) & valid_gt & ~gt_mask)]
        if fp_names:
            if gt_mask.any():
                gt_distance = ndimage.distance_transform_edt(~gt_mask)
                for name in fp_names:
                    fp = (preds[name] == c) & valid_gt & ~gt_mask
                    values = gt_distance[fp]
                    bins = np.searchsorted([1, 3, 8, 16, 32], values, side="left")
                    dist[name][c, 0, :6] = np.bincount(bins, minlength=6)
                    dist_native[name][c, 0, :6] = np.bincount(
                        bins, weights=values, minlength=6)
            else:
                for name in fp_names:
                    dist[name][c, 0, 6] = int(((preds[name] == c) & valid_gt).sum())
    return image_id, cms, dist, dist_native, thick, scale


def _fmt(value):
    return "" if not np.isfinite(value) else f"{value:.6f}"


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def scan(gt_index, model_index, model_input_px=1024, patch_px=16, limit=0, workers=8):
    names = list(model_index)
    ids = set(gt_index)
    for name in names:
        ids &= set(model_index[name])
    ids = sorted(ids)
    if limit > 0:
        ids = ids[:limit]
    if not ids:
        raise RuntimeError("GT와 모든 모델이 공유하는 image_id가 없다")
    if len(ids) < len(gt_index):
        print(f"[thin_error_anatomy] 경고: GT {len(gt_index)}장 중 공통 {len(ids)}장만 집계")
    tasks = ((image_id, gt_index[image_id],
              {name: model_index[name][image_id] for name in names},
              model_input_px, patch_px) for image_id in ids)
    cm_all = {n: np.zeros((N, N), dtype=np.int64) for n in names}
    cm_cond = {n: {cond: np.zeros((N, N), dtype=np.int64)
                   for cond in common.KNOWN_CONDITIONS} for n in names}
    dist = {n: np.zeros((N, 2, 7), dtype=np.int64) for n in names}
    sums_gt = {n: np.zeros((N, 2, 7), dtype=np.float64) for n in names}
    sums_model = {n: np.zeros((N, 2, 7), dtype=np.float64) for n in names}
    thick = {n: np.zeros((2, 6, 4), dtype=np.float64) for n in names}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(_one_image, tasks, chunksize=1) if executor else map(_one_image, tasks)
        for i, (image_id, cms, ds, ss, ts, scale) in enumerate(results, start=1):
            condition, _ = common.parse_condition_case(image_id)
            for name in names:
                cm_all[name] += cms[name]
                if condition in cm_cond[name]:
                    cm_cond[name][condition] += cms[name]
                dist[name] += ds[name]
                sums_gt[name] += ss[name]
                sums_model[name] += ss[name] * scale
                thick[name] += ts[name]
            if i % 100 == 0 or i == len(ids):
                print(f"[thin_error_anatomy] {i}/{len(ids)}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return ids, cm_all, cm_cond, dist, sums_gt, sums_model, thick


def write_outputs(out, names, ids, cm_all, cm_cond, dist, sums_gt, sums_model,
                  thick, split, patch_px):
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw_confusion"
    raw.mkdir(exist_ok=True)
    for name in names:
        np.save(raw / f"{name}.npy", cm_all[name])
        condition_dir = raw / "conditions" / name
        condition_dir.mkdir(parents=True, exist_ok=True)
        for cond in common.KNOWN_CONDITIONS:
            np.save(condition_dir / f"{cond}.npy", cm_cond[name][cond])
    metrics_rows = []
    for name in names:
        for cond, cm in [("all", cm_all[name]), *cm_cond[name].items()]:
            m = class_metrics(cm)
            for c in range(N):
                metrics_rows.append([name, cond, c, common.CLASSES[c],
                    int(m["tp"][c]), int(m["fp"][c]), int(m["fn"][c]),
                    *(_fmt(m[key][c]) for key in ("iou", "precision", "recall",
                                                 "fp_loss", "fn_loss"))])
    _write_csv(out / "precision_recall.csv",
               ["model", "condition", "class_id", "class_name", "tp_pixels",
                "fp_pixels", "fn_pixels", "iou", "precision", "recall",
                "fp_loss_share", "fn_loss_share"], metrics_rows)
    source_lines = ["# 얇은 클래스의 FP 출처와 FN 목적지", "",
                    "전체 이미지의 원시 픽셀 수. GT=255는 제외한다.", ""]
    for name in names:
        cm = cm_all[name]
        source_lines += [f"## {name}", ""]
        for c in common.THIN_CLASS_IDS:
            fp = sorted(((int(cm[g, c]), common.CLASSES[g]) for g in range(N) if g != c),
                        reverse=True)[:5]
            fn = sorted(((int(cm[c, p]), common.CLASSES[p]) for p in range(N) if p != c),
                        reverse=True)[:5]
            source_lines += [f"### {common.CLASSES[c]}", "",
                             "| 방향 | 클래스 | 픽셀 수 |", "|---|---|---:|"]
            source_lines += [f"| FP: GT | {label} | {count} |" for count, label in fp if count]
            source_lines += [f"| FN: 예측 | {label} | {count} |" for count, label in fn if count]
            source_lines += [""]
    (out / "fp_sources.md").write_text("\n".join(source_lines), encoding="utf-8")
    distance_rows = []
    for name in names:
        for kind_idx, kind in enumerate(("FP", "FN")):
            for c in [*common.THIN_CLASS_IDS, "all_25_macro"]:
                if isinstance(c, int):
                    counts = dist[name][c, kind_idx]
                    gt_sum = sums_gt[name][c, kind_idx]
                    model_sum = sums_model[name][c, kind_idx]
                    fraction = counts / counts.sum() if counts.sum() else np.full(7, np.nan)
                    label = common.CLASSES[c]
                else:
                    per_class = dist[name][:, kind_idx]
                    counts = per_class.sum(axis=0)
                    gt_sum = sums_gt[name][:, kind_idx].sum(axis=0)
                    model_sum = sums_model[name][:, kind_idx].sum(axis=0)
                    active = per_class.sum(axis=1) > 0
                    fractions = per_class[active] / per_class[active].sum(axis=1, keepdims=True)
                    fraction = fractions.mean(axis=0) if active.any() else np.full(7, np.nan)
                    label = "all_25_macro"
                for b, bin_label in enumerate(DIST_LABELS):
                    n = int(counts[b])
                    distance_rows.append([name, label, kind, bin_label, n,
                        _fmt(fraction[b]), _fmt(gt_sum[b] / n) if n and b < 6 else "",
                        _fmt(model_sum[b] / n) if n and b < 6 else "",
                        _fmt(model_sum[b] / n / patch_px) if n and b < 6 else ""])
    _write_csv(out / "boundary_distance.csv",
               ["model", "class_name", "error_type", "distance_bin_gt_px",
                "pixel_count", "fraction", "mean_distance_gt_px",
                "mean_distance_model_input_px", "mean_distance_patch"], distance_rows)
    thick_rows = []
    for name in names:
        for group_idx, group in enumerate(("thin_4", "other_21")):
            for b, bin_label in enumerate(THICK_LABELS):
                n, detected, poor, recall_sum = thick[name][group_idx, b]
                thick_rows.append([name, group, bin_label, int(n), int(detected), int(poor),
                    _fmt(detected / n) if n else "", _fmt(poor / detected) if detected else "",
                    _fmt(recall_sum / n) if n else ""])
    _write_csv(out / "thickness_patch.csv",
               ["model", "class_group", "thickness_bin_patch", "n_components",
                "n_detected", "n_found_but_poor", "detection_rate",
                "found_but_poor_rate", "mean_component_pixel_recall"], thick_rows)
    _write_summary(out / "summary.md", names, ids, cm_all, dist, thick, split)


def _write_summary(path, names, ids, cm_all, dist, thick, split):
    lines = [f"# 얇은 객체 오류 요약 ({split}, {len(ids)}장)", "",
             "같은 image_id의 저장 PNG를 GT native 크기에서 비교했다. 상세 정의는 도구 상단 README를 참조한다.", "",
             "## 전체 지표", "",
             "| 지표 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    metric = {n: class_metrics(cm_all[n]) for n in names}
    rows = [("mIoU (%)", lambda n: f"{common.global_miou_from_cm(cm_all[n])[1]:.4f}")]
    for c in common.THIN_CLASS_IDS:
        for key in ("iou", "precision", "recall", "fp_loss", "fn_loss"):
            rows.append((f"{common.CLASSES[c]} {key}", lambda n, c=c, key=key: _fmt(metric[n][key][c])))
    for label, func in rows:
        lines.append("| " + label + " | " + " | ".join(func(n) for n in names) + " |")
    lines += ["", "## 거리 분해", "",
              "각 칸은 해당 클래스 오류 중 구간 비율. all_25_macro는 오류가 있는 클래스별 비율의 평균이다.", "",
              "| 오류·클래스·구간 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for kind_idx, kind in enumerate(("FP", "FN")):
        for c in [*common.THIN_CLASS_IDS, "all_25_macro"]:
            label = common.CLASSES[c] if isinstance(c, int) else c
            for b in range(7):
                def cell(n):
                    arr = dist[n][c, kind_idx] if isinstance(c, int) else dist[n][:, kind_idx]
                    if arr.ndim == 1:
                        return _fmt(arr[b] / arr.sum()) if arr.sum() else ""
                    active = arr.sum(axis=1) > 0
                    return _fmt((arr[active, b] / arr[active].sum(axis=1)).mean()) if active.any() else ""
                lines.append(f"| {kind} {label} {DIST_LABELS[b]} | " +
                             " | ".join(cell(n) for n in names) + " |")
    lines += ["", "## 두께별 성분", "",
              "칸: 성분 수 / 찾음률 / 찾았지만 못 그림 비율 / 성분 픽셀 재현율 평균.", "",
              "| 군·패치 구간 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for group_idx, group in enumerate(("thin_4", "other_21")):
        for b, bin_label in enumerate(THICK_LABELS):
            cells = []
            for n in names:
                count, found, poor, rec_sum = thick[n][group_idx, b]
                cells.append(f"{int(count)} / {_fmt(found/count) if count else '—'} / "
                             f"{_fmt(poor/found) if found else '—'} / "
                             f"{_fmt(rec_sum/count) if count else '—'}")
            lines.append(f"| {group} {bin_label} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_input_px", type=int, default=1024)
    ap.add_argument("--patch_px", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.model_input_px <= 0 or args.patch_px <= 0 or args.workers <= 0 or args.limit < 0:
        ap.error("픽셀 크기와 workers는 양수, limit은 0 이상이어야 한다")
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
    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    model_index = {name: common.index_label_pngs(path, require_nested=True)
                   for name, path in model_dirs.items()}
    results = scan(gt_index, model_index, args.model_input_px, args.patch_px,
                   args.limit, args.workers)
    write_outputs(Path(args.out), list(model_dirs), *results, args.split, args.patch_px)
    print(f"[thin_error_anatomy] 완료: {args.out}")


if __name__ == "__main__":
    main()
