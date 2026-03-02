#!/usr/bin/env python3
"""
MULTIAQUA thermal 이미지 전수 분석.

- Per-image: ROI range, mean, std, unique levels, histogram
- 시퀀스별(adr1_1, adr1_7, lj4_0 등) 통계
- Train/Val/Test 분할별 통계
- Global: 전체 히스토그램, outlier 탐지
- SAM 입력 전처리 방안 도출

Usage:
  python MISC/MULTIAQUA_utils/analyze_thermal.py
"""
import cv2
import numpy as np
import os
import glob
import json
from collections import defaultdict
from pathlib import Path

THERMAL_DIR = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera"
PROCESSED_DIR = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_processed"
DATASET_ROOT = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "thermal_analysis_result"


def load_split(name):
    path = os.path.join(DATASET_ROOT, f"{name}.txt")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def get_sequence(stem):
    """adr1_10_000700 → adr1_10"""
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def analyze_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    total_px = h * w
    padding_px = int(np.sum(img == 0))
    roi = img[img > 0]
    if roi.size == 0:
        return {
            "h": h, "w": w, "total_px": total_px,
            "padding_px": padding_px, "padding_ratio": 1.0,
            "roi_px": 0, "roi_min": 0, "roi_max": 0,
            "roi_mean": 0, "roi_std": 0, "roi_median": 0,
            "n_unique": 0, "unique_vals": [],
            "has_saturated": False,
        }
    unique = np.unique(roi)
    return {
        "h": h, "w": w, "total_px": total_px,
        "padding_px": padding_px,
        "padding_ratio": padding_px / total_px,
        "roi_px": int(roi.size),
        "roi_min": int(roi.min()),
        "roi_max": int(roi.max()),
        "roi_mean": float(roi.mean()),
        "roi_std": float(roi.std()),
        "roi_median": float(np.median(roi)),
        "n_unique": len(unique),
        "unique_vals": unique.tolist(),
        "has_saturated": int(roi.max()) >= 250,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_stems = load_split("train")
    val_stems = load_split("val")
    test_stems = load_split("test")

    # Discover files
    files = sorted(glob.glob(os.path.join(THERMAL_DIR, "*.png")))
    print(f"총 {len(files)}장 thermal_camera 이미지")
    print(f"Splits: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")

    # Per-image analysis
    per_image = {}
    seq_stats = defaultdict(list)
    split_stats = {"train": [], "val": [], "test": [], "unknown": []}
    global_hist = np.zeros(256, dtype=np.int64)
    roi_global_hist = np.zeros(256, dtype=np.int64)

    n_saturated = 0
    outlier_images = []

    for i, fpath in enumerate(files):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        info = analyze_image(fpath)
        if info is None:
            continue
        per_image[stem] = info

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        global_hist += h
        if info["roi_px"] > 0:
            roi = img[img > 0]
            rh, _ = np.histogram(roi, bins=256, range=(0, 256))
            roi_global_hist += rh

        seq = get_sequence(stem)
        seq_stats[seq].append(info)

        if stem in train_stems:
            split_stats["train"].append(info)
        elif stem in val_stems:
            split_stats["val"].append(info)
        elif stem in test_stems:
            split_stats["test"].append(info)
        else:
            split_stats["unknown"].append(info)

        if info["has_saturated"]:
            n_saturated += 1
            outlier_images.append((stem, info["roi_min"], info["roi_max"], info["roi_mean"], info["roi_std"]))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} processed...")

    print(f"분석 완료: {len(per_image)}장")

    # ============================
    # Report
    # ============================
    report = []
    report.append("=" * 80)
    report.append("MULTIAQUA Thermal Image Analysis Report")
    report.append("=" * 80)
    report.append(f"Source: {THERMAL_DIR}")
    report.append(f"Total images: {len(per_image)}")
    report.append("")

    # 1. Global statistics
    all_means = [v["roi_mean"] for v in per_image.values() if v["roi_px"] > 0]
    all_stds = [v["roi_std"] for v in per_image.values() if v["roi_px"] > 0]
    all_mins = [v["roi_min"] for v in per_image.values() if v["roi_px"] > 0]
    all_maxs = [v["roi_max"] for v in per_image.values() if v["roi_px"] > 0]
    all_nuniq = [v["n_unique"] for v in per_image.values() if v["roi_px"] > 0]
    all_pad = [v["padding_ratio"] for v in per_image.values()]

    report.append("1. GLOBAL STATISTICS (per-image ROI, padding excluded)")
    report.append("-" * 60)
    report.append(f"  ROI mean:   min={min(all_means):.1f}, max={max(all_means):.1f}, avg={np.mean(all_means):.1f}")
    report.append(f"  ROI std:    min={min(all_stds):.2f}, max={max(all_stds):.2f}, avg={np.mean(all_stds):.2f}")
    report.append(f"  ROI min:    min={min(all_mins)}, max={max(all_mins)}")
    report.append(f"  ROI max:    min={min(all_maxs)}, max={max(all_maxs)}")
    report.append(f"  Unique levels: min={min(all_nuniq)}, max={max(all_nuniq)}, avg={np.mean(all_nuniq):.1f}")
    report.append(f"  Padding ratio: min={min(all_pad):.2%}, max={max(all_pad):.2%}, avg={np.mean(all_pad):.2%}")
    report.append(f"  Saturated images (max>=250): {n_saturated} / {len(per_image)} ({100*n_saturated/len(per_image):.1f}%)")
    report.append("")

    # 2. Value range distribution
    report.append("2. VALUE RANGE DISTRIBUTION")
    report.append("-" * 60)
    range_buckets = defaultdict(int)
    for v in per_image.values():
        if v["roi_px"] == 0:
            range_buckets["empty"] += 1
            continue
        spread = v["roi_max"] - v["roi_min"]
        if spread <= 10:
            range_buckets["narrow (≤10)"] += 1
        elif spread <= 30:
            range_buckets["medium (11-30)"] += 1
        elif spread <= 100:
            range_buckets["wide (31-100)"] += 1
        else:
            range_buckets["very wide (>100)"] += 1
    for k, cnt in sorted(range_buckets.items()):
        report.append(f"  {k}: {cnt} images ({100*cnt/len(per_image):.1f}%)")
    report.append("")

    # 3. Per-sequence statistics
    report.append("3. PER-SEQUENCE STATISTICS")
    report.append("-" * 60)
    report.append(f"{'Sequence':<15} {'Count':>6} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>6} {'Uniq':>5} {'Saturated':>10}")
    for seq in sorted(seq_stats.keys()):
        infos = seq_stats[seq]
        valid = [i for i in infos if i["roi_px"] > 0]
        if not valid:
            continue
        s_min = min(i["roi_min"] for i in valid)
        s_max = max(i["roi_max"] for i in valid)
        s_mean = np.mean([i["roi_mean"] for i in valid])
        s_std = np.mean([i["roi_std"] for i in valid])
        s_uniq = np.mean([i["n_unique"] for i in valid])
        n_sat = sum(1 for i in valid if i["has_saturated"])
        report.append(f"  {seq:<13} {len(valid):>6} {s_min:>5} {s_max:>5} {s_mean:>7.1f} {s_std:>6.2f} {s_uniq:>5.1f} {n_sat:>10}")
    report.append("")

    # 4. Per-split statistics
    report.append("4. PER-SPLIT STATISTICS")
    report.append("-" * 60)
    for split_name in ["train", "val", "test"]:
        infos = split_stats[split_name]
        valid = [i for i in infos if i["roi_px"] > 0]
        if not valid:
            report.append(f"  {split_name}: 0 images")
            continue
        s_min = min(i["roi_min"] for i in valid)
        s_max = max(i["roi_max"] for i in valid)
        s_mean_avg = np.mean([i["roi_mean"] for i in valid])
        s_std_avg = np.mean([i["roi_std"] for i in valid])
        s_uniq_avg = np.mean([i["n_unique"] for i in valid])
        n_sat = sum(1 for i in valid if i["has_saturated"])
        report.append(f"  {split_name}: {len(valid)} images")
        report.append(f"    ROI range: [{s_min} - {s_max}]")
        report.append(f"    Mean of means: {s_mean_avg:.1f}")
        report.append(f"    Mean of stds:  {s_std_avg:.2f}")
        report.append(f"    Mean unique levels: {s_uniq_avg:.1f}")
        report.append(f"    Saturated: {n_sat}")
    report.append("")

    # 5. Saturated (outlier) images
    report.append("5. SATURATED IMAGES (max >= 250)")
    report.append("-" * 60)
    if outlier_images:
        for stem, mn, mx, m, s in outlier_images:
            report.append(f"  {stem}: [{mn}-{mx}], mean={m:.1f}, std={s:.1f}")
    else:
        report.append("  None")
    report.append("")

    # 6. thermal_processed vs thermal_camera 비교
    report.append("6. thermal_processed vs thermal_camera COMPARISON (sample)")
    report.append("-" * 60)
    proc_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.png")))
    n_identical = 0
    n_checked = 0
    diffs = []
    for pf in proc_files[:200]:
        pname = os.path.basename(pf)
        stem = pname.replace("_thermal.png", "")
        raw_path = os.path.join(THERMAL_DIR, stem + ".png")
        if not os.path.exists(raw_path):
            continue
        proc_img = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if proc_img is None or raw_img is None:
            continue
        n_checked += 1
        if proc_img.shape != raw_img.shape:
            diffs.append((stem, "shape_mismatch", proc_img.shape, raw_img.shape))
            continue
        if np.array_equal(proc_img, raw_img):
            n_identical += 1
        else:
            diff = np.abs(proc_img.astype(int) - raw_img.astype(int))
            diffs.append((stem, "pixel_diff", float(diff.max()), float(diff.mean())))
    report.append(f"  Checked: {n_checked} images")
    report.append(f"  Identical (pixel-for-pixel): {n_identical}")
    report.append(f"  Different: {len(diffs)}")
    for d in diffs[:10]:
        report.append(f"    {d}")
    report.append("")

    # 7. Preprocessing recommendations
    report.append("7. PREPROCESSING ANALYSIS FOR SAM INPUT")
    report.append("=" * 60)

    typical_range = max(all_maxs) - min(all_mins)
    typical_intra_std = np.mean(all_stds)
    report.append(f"  Dataset-wide ROI range: [{min(all_mins)} - {max(all_maxs)}] (span={typical_range})")
    report.append(f"  Per-image intra-ROI std (avg): {typical_intra_std:.2f}")
    report.append(f"  Quantization: 대부분 {min(all_nuniq)}~{int(np.median(all_nuniq))} unique levels")
    report.append("")

    pct_narrow = 100 * range_buckets.get("narrow (≤10)", 0) / len(per_image)
    report.append(f"  Narrow range (≤10 levels) images: {pct_narrow:.1f}%")
    report.append("")

    report.append("  OPTIONS:")
    report.append("  A) Per-image linear stretch (min-max → 0-255):")
    report.append("     + 각 이미지의 전체 dynamic range 활용")
    report.append("     + SAM pretrained features와 호환 (0-255)")
    report.append("     - 9단계 밴딩 아티팩트")
    report.append("     - 이미지 간 절대 온도 관계 사라짐")
    report.append("")
    report.append("  B) Per-image linear stretch (min-max → 100-200):")
    report.append("     + 밴딩 덜함 (100단계 중 9단계)")
    report.append("     + 절대 스케일이 이미지 간 비슷")
    report.append("     - SAM features가 중간 밝기만 보게 됨 (contrast 부족)")
    report.append("")
    report.append("  C) Global normalization (z-score with dataset mean/std):")
    report.append("     + 이미지 간 절대 온도 관계 보존")
    report.append("     - 이미지 내 std=2~3이 dataset std=12로 나눠져 0.2 수준으로 압축")
    report.append("     - SAM의 ImageNet pretrained encoder에 부적합한 분포")
    report.append("")
    report.append("  D) Bilateral(float) → Stretch → Mild CLAHE:")
    report.append("     + 밴딩 제거, 부드러운 그래디언트")
    report.append("     - 경계(해안선 등) 약화 가능")
    report.append("     - 원본 정보에 없는 보간 정보 추가")
    report.append("")
    report.append("  E) Stretch only + 모델 학습에 맡김:")
    report.append("     + 원본 구조 최대 보존")
    report.append("     + 밴딩은 모델이 학습으로 무시 가능")
    report.append("     - pretrained SAM encoder에는 불리 (finetuning 필수)")

    # Write report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "thermal_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {report_path}")

    # Save per-image stats as JSON (lightweight)
    stats_path = OUTPUT_DIR / "per_image_stats.json"
    # Remove unique_vals for compactness
    compact = {}
    for stem, info in per_image.items():
        c = dict(info)
        del c["unique_vals"]
        compact[stem] = c
    with open(stats_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"Per-image stats saved to: {stats_path}")

    # Save histograms
    np.save(OUTPUT_DIR / "global_histogram.npy", global_hist)
    np.save(OUTPUT_DIR / "roi_histogram.npy", roi_global_hist)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1) Global ROI histogram
        ax = axes[0, 0]
        ax.bar(range(256), roi_global_hist, width=1, color="steelblue")
        ax.set_title("Global ROI Pixel Histogram (padding excluded)")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 260)

        # 2) Zoomed ROI histogram (85-105)
        ax = axes[0, 1]
        ax.bar(range(85, 110), roi_global_hist[85:110], width=1, color="darkorange")
        ax.set_title("ROI Histogram [85-110] (main range)")
        ax.set_xlabel("Pixel Value")

        # 3) Per-image mean distribution
        ax = axes[0, 2]
        ax.hist(all_means, bins=50, color="green", alpha=0.7)
        ax.set_title(f"Per-image ROI Mean Distribution (n={len(all_means)})")
        ax.set_xlabel("Mean pixel value")
        ax.axvline(np.mean(all_means), color="red", ls="--", label=f"avg={np.mean(all_means):.1f}")
        ax.legend()

        # 4) Per-image std distribution
        ax = axes[1, 0]
        ax.hist(all_stds, bins=50, color="purple", alpha=0.7)
        ax.set_title("Per-image ROI Std Distribution")
        ax.set_xlabel("Std")
        ax.axvline(np.mean(all_stds), color="red", ls="--", label=f"avg={np.mean(all_stds):.2f}")
        ax.legend()

        # 5) Per-image unique levels
        ax = axes[1, 1]
        ax.hist(all_nuniq, bins=range(0, max(all_nuniq)+2), color="teal", alpha=0.7)
        ax.set_title("Per-image Unique Levels")
        ax.set_xlabel("# Unique values in ROI")

        # 6) Per-image max value
        ax = axes[1, 2]
        ax.hist(all_maxs, bins=50, color="crimson", alpha=0.7)
        ax.set_title("Per-image ROI Max Value")
        ax.set_xlabel("Max pixel value")
        ax.axvline(100, color="blue", ls="--", label="100")
        ax.axvline(250, color="red", ls="--", label="250 (saturated)")
        ax.legend()

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "thermal_analysis_plots.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Plots saved to: {fig_path}")

    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
