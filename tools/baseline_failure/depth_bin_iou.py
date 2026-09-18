#!/usr/bin/env python3
"""
A6 — 원본 depth 구간별 IoU. 각 픽셀을 DELIVER `depth/` 원본 depth 값의 **로그 스케일
5분위 구간**(기본)으로 나눠, 구간별로 모델별 IoU(25클래스 평균)·픽셀 정확도를 계산하고
모델 간 차이 행렬(예: dgf − caf)도 낸다.

- 구간 경계는 해당 split 전체 유효 depth 픽셀(depth > 0)에서 계산해 요약 JSON 에 남긴다
  (로그 값의 가중 분위수, np.quantile 'linear' 보간과 동일).
- depth 값이 0 이거나 GT 가 255 인 픽셀은 채점에서 제외한다.
- per-class IoU·mIoU 는 common 규약(전역 혼동행렬 기반, 부재 클래스 IoU=0 포함 25클래스
  평균)을 따른다.

출력:
- `<out>/depth_bin_iou_<split>.csv` — 구간별 행: bin, depth_lo, depth_hi, log_depth_lo,
  log_depth_hi, n_pixels, [miou_<model>, pacc_<model> ...], [diff_<a>_minus_<b> ...].
- `<out>/depth_bin_iou_<split>.json` — 구간 경계(로그·원본 단위), 구간별 픽셀 수,
  모델별 구간별 지표, 모델 간 차이.

예:
  python tools/baseline_failure/depth_bin_iou.py \
    --gt .../ours/test/gt --depth_root /ailab_mat2/dataset/DELIVER \
    --models ours=.../ours/test/pred dgf=.../dgffinal/test/pred caf=.../caf/test/pred \
    --split test --out .../analysis/test --bins 5
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES


def _resolve_pred_dir(path):
    """모델 인자가 pred 디렉터리 자체이거나 split 루트일 수 있으니 정규화한다."""
    p = Path(path)
    if (p / "pred").is_dir():
        return p / "pred"
    return p


def resolve_depth_dir(depth_root):
    """--depth_root 는 DELIVER depth 디렉터리 자체 또는 데이터셋 루트를 받는다."""
    p = Path(depth_root)
    if p.name == "depth":
        return p
    return p / "depth"


def depth_path_for(depth_dir, image_id):
    """image_id(`img/<cond>/<split>/<scene>/<stem>_rgb_front`) → 원본 depth PNG 경로.

    deliver.py 의 경로 치환 규약(img→depth, _rgb→_depth)을 그대로 적용한다.
    """
    if not image_id.startswith("img/"):
        raise ValueError(f"image_id 가 img/ 접두어로 시작하지 않는다: {image_id}")
    rel = image_id[len("img/"):].replace("_rgb", "_depth")
    return Path(depth_dir) / f"{rel}.png"


def load_depth(path):
    """원본 depth PNG → (H, W) float64(0 = 무효)."""
    from PIL import Image
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"depth 파일이 없다: {p} — --depth_root 를 확인하라")
    return np.array(Image.open(p)).astype(np.float64)


def resize_nearest_float(arr, out_h, out_w):
    """float 맵(depth)용 최근접 리사이즈(common.resize_nearest 는 uint8 로 캐스팅한다)."""
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr
    from PIL import Image
    return np.array(Image.fromarray(arr.astype(np.float32))
                    .resize((out_w, out_h), Image.NEAREST), dtype=np.float64)


def weighted_quantile(sorted_vals, counts, q):
    """가중 다변량 {sorted_vals[i]: counts[i]} 의 선형 보간 분위수(np.quantile 'linear')."""
    total = int(counts.sum())
    if total == 0:
        raise ValueError("분위수를 계산할 값이 없다")
    cw = np.cumsum(counts)                    # cw[j] = 값 순위 상 (j+1)번째까지의 개수

    def value_at(rank):                        # 0-based 순위 rank 의 값
        j = int(np.searchsorted(cw, rank, side="right"))
        return sorted_vals[min(j, len(sorted_vals) - 1)]

    pos = q * (total - 1)
    r0 = int(np.floor(pos))
    r1 = min(r0 + 1, total - 1)
    frac = pos - r0
    return float(value_at(r0) + frac * (value_at(r1) - value_at(r0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="공통 GT trainID PNG 디렉터리")
    ap.add_argument("--depth_root", required=True,
                    help="DELIVER depth 디렉터리 또는 데이터셋 루트(img/ 와 같은 층)")
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR",
                    help="모델별 예측 디렉터리. 예: ours=.../pred dgf=.../pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    if args.bins < 1:
        ap.error(f"--bins 는 1 이상이어야 한다: {args.bins}")

    models = []
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"--models 는 name=dir 형식이어야 함: {spec}")
        name, d = spec.split("=", 1)
        models.append((name, d))
    names = [n for n, _ in models]

    depth_dir = resolve_depth_dir(args.depth_root)
    if not depth_dir.is_dir():
        ap.error(f"depth 디렉터리가 없다: {depth_dir} — --depth_root 를 확인하라")

    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    pred_indices = {}
    for name, d in models:
        idx = common.index_label_pngs(_resolve_pred_dir(d), require_nested=True)
        missing = set(gt_index) - set(idx)
        if missing:
            print(f"[{name}] ⚠️ GT 에 있는데 예측 없는 이미지 {len(missing)} 장 "
                  f"(예: {sorted(missing)[:3]})")
        common_ids = set(gt_index) & set(idx)
        if not common_ids:
            raise RuntimeError(f"[{name}] GT 와 예측의 공통 image_id 가 없다 — 규약 확인.")
        pred_indices[name] = idx
    image_ids = sorted(set(gt_index) & set.intersection(*[set(pred_indices[n]) for n in names]))
    print(f"[depth_bin_iou] {len(image_ids)} imgs x {len(names)} models x {args.bins} bins")

    # ---- 패스 1: split 전체 유효 depth(depth>0) 픽셀로 로그 5분위 경계 계산 ----
    # PNG depth 는 양자화 값이므로 (값, 개수) 히스토그램으로 전체 분포를 정확히 갖는다
    # (전 픽셀 배열을 메모리에 올리지 않는다).
    counts = defaultdict(int)
    for image_id in image_ids:
        depth = load_depth(depth_path_for(depth_dir, image_id))
        gt = common.load_label_png(gt_index[image_id])
        if depth.shape != gt.shape:
            depth = resize_nearest_float(depth, gt.shape[0], gt.shape[1])
        v, c = np.unique(depth[depth > 0], return_counts=True)
        for val, cnt in zip(v.tolist(), c.tolist()):
            counts[float(val)] += int(cnt)
    if not counts:
        raise RuntimeError("유효한 depth(depth>0) 픽셀이 하나도 없다 — depth_root 확인.")
    sorted_vals = np.array(sorted(counts), dtype=np.float64)
    sorted_cnts = np.array([counts[v] for v in sorted_vals.tolist()], dtype=np.int64)
    log_vals = np.log(sorted_vals)
    qs = [k / args.bins for k in range(1, args.bins)]
    log_edges = np.array([weighted_quantile(log_vals, sorted_cnts, q) for q in qs])
    depth_edges = np.exp(log_edges)
    depth_min, depth_max = float(sorted_vals[0]), float(sorted_vals[-1])
    n_valid = int(sorted_cnts.sum())
    print(f"[depth_bin_iou] 유효 depth 픽셀 {n_valid} 개  로그 구간 경계="
          f"{np.round(log_edges, 4).tolist()}")

    # ---- 패스 2: 구간별 혼동행렬 누적 ----
    hists = {m: np.zeros((args.bins, N, N), dtype=np.int64) for m in names}
    bin_pixels = np.zeros(args.bins, dtype=np.int64)
    resized_depth = resized_pred = False
    for image_id in image_ids:
        depth = load_depth(depth_path_for(depth_dir, image_id))
        gt = common.load_label_png(gt_index[image_id])
        if depth.shape != gt.shape:
            depth = resize_nearest_float(depth, gt.shape[0], gt.shape[1])
            resized_depth = True
        valid = (depth > 0) & (gt != common.IGNORE_LABEL)
        bidx = np.searchsorted(log_edges, np.log(depth[valid]), side="right")
        for m, _d in models:
            pred = common.load_label_png(pred_indices[m][image_id])
            if pred.shape != gt.shape:
                pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
                resized_pred = True
            g, p, b = gt[valid].astype(np.int64), pred[valid].astype(np.int64), bidx
            keep = (g != common.IGNORE_LABEL) & (g >= 0) & (g < N) & (p >= 0) & (p < N)
            flat = np.bincount(b[keep] * (N * N) + g[keep] * N + p[keep],
                               minlength=args.bins * N * N).reshape(args.bins, N, N)
            hists[m] += flat
        bin_pixels += np.bincount(bidx, minlength=args.bins)

    # ---- 지표·출력 ----
    stats = {m: {"miou": [], "pacc": []} for m in names}
    for m in names:
        for b in range(args.bins):
            _, miou = common.global_miou_from_cm(hists[m][b])
            stats[m]["miou"].append(miou)
            stats[m]["pacc"].append(round(common.pixel_acc_from_cm(hists[m][b]), 6))
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"depth_bin_iou_{args.split}.csv"
    header = (["bin", "depth_lo", "depth_hi", "log_depth_lo", "log_depth_hi", "n_pixels"]
              + [f"miou_{m}" for m in names] + [f"pacc_{m}" for m in names]
              + [f"diff_{a}_minus_{b}" for a, b in pairs])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for b in range(args.bins):
            lo = 0.0 if b == 0 else float(depth_edges[b - 1])
            hi = float("inf") if b == args.bins - 1 else float(depth_edges[b])
            llo = -float("inf") if b == 0 else float(log_edges[b - 1])
            lhi = float("inf") if b == args.bins - 1 else float(log_edges[b])
            row = [b, round(lo, 4), ("" if np.isinf(hi) else round(hi, 4)),
                   ("" if np.isinf(llo) else round(llo, 6)),
                   ("" if np.isinf(lhi) else round(lhi, 6)), int(bin_pixels[b])]
            row += [stats[m]["miou"][b] for m in names]
            row += [stats[m]["pacc"][b] for m in names]
            row += [round(stats[a]["miou"][b] - stats[b_]["miou"][b], 4) for a, b_ in pairs]
            w.writerow(row)

    notes = ["구간 경계는 split 전체 유효 depth(depth>0) 픽셀의 로그 값 분위수(선형 보간).",
             "채점에서 depth==0 또는 GT==255 픽셀 제외. mIoU 는 부재 클래스 IoU=0 포함 25클래스 평균(common 규약)."]
    if resized_depth:
        notes.append("일부 이미지에서 depth 를 GT 해상도에 최근접 리사이즈했다(해상도 불일치).")
    if resized_pred:
        notes.append("일부 이미지에서 예측을 GT 해상도에 최근접 리사이즈했다(해상도 불일치).")
    summary = {
        "split": args.split, "bins": args.bins, "models": names,
        "edge_quantiles": qs, "log_edges": [float(e) for e in log_edges],
        "depth_edges": [float(e) for e in depth_edges],
        "depth_min": depth_min, "depth_max": depth_max,
        "n_valid_depth_pixels_edges": n_valid,
        "bin_pixels": bin_pixels.tolist(),
        "per_model": {m: {"miou25_per_bin": stats[m]["miou"],
                          "pixel_acc_per_bin": stats[m]["pacc"]} for m in names},
        "diff": {f"{a}_minus_{b}": [round(stats[a]["miou"][k] - stats[b]["miou"][k], 4)
                                     for k in range(args.bins)] for a, b in pairs},
        "notes": notes,
    }
    json_path = out_dir / f"depth_bin_iou_{args.split}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[depth_bin_iou] -> {csv_path.name}, {json_path.name}")


if __name__ == "__main__":
    main()
