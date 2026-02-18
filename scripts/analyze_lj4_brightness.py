#!/usr/bin/env python3
"""
MULTIAQUA lj4 (야간) 도메인 RGB 밝기 분포 분석.
NIGHT_AUG BRIGHTNESS_RANGE / DARK_RANGE / MODERATE_RANGE 설정에 반영용.
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def analyze_brightness(rgb_dir: Path, stems: list) -> dict:
    """각 이미지별 평균 밝기(0~1) 계산 후 통계 반환."""
    brightness = []
    for s in stems:
        p = rgb_dir / f"{s}.png"
        if not p.exists():
            continue
        img = np.array(Image.open(str(p)).convert("RGB"))
        # 0~1 정규화 후 평균 (grayscale weight: 0.299R + 0.587G + 0.114B)
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]) / 255.0
        brightness.append(float(gray.mean()))
    arr = np.array(brightness)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "values": arr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night",
                       help="MULTIAQUA dataset root (parent of MULTIAQUA_night/)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save recommended ranges to file")
    args = parser.parse_args()
    root = Path(args.root)

    # Paths: root/MULTIAQUA_night/data/zed, root/test.txt (or root/MULTIAQUA_night/test.txt)
    rgb_dir = root / "MULTIAQUA_night" / "data" / "zed"
    if not rgb_dir.exists():
        rgb_dir = root / "data" / "zed"
    test_file = root / "test.txt"
    if not test_file.exists():
        test_file = root / "MULTIAQUA_night" / "test.txt"
    if not rgb_dir.exists() or not test_file.exists():
        print(f"ERROR: rgb_dir={rgb_dir} exists={rgb_dir.exists()}, test={test_file} exists={test_file.exists()}")
        print("Cannot find dataset. Run with --root /path/to/dataset")
        return

    with open(test_file) as f:
        all_stems = [line.strip() for line in f if line.strip()]
    lj4_stems = [s for s in all_stems if s.startswith("lj4")]

    print(f"Found {len(lj4_stems)} lj4 (night) images in test split")
    stats = analyze_brightness(rgb_dir, lj4_stems)
    print(f"\n=== lj4 RGB Brightness (0~1 scale) ===")
    print(f"  n_samples: {stats['n']}")
    print(f"  mean: {stats['mean']:.4f}  std: {stats['std']:.4f}")
    print(f"  min: {stats['min']:.4f}  max: {stats['max']:.4f}")
    print(f"  p5:  {stats['p5']:.4f}  p10: {stats['p10']:.4f}")
    print(f"  p25: {stats['p25']:.4f}  p50: {stats['p50']:.4f}")
    print(f"  p75: {stats['p75']:.4f}  p90: {stats['p90']:.4f}  p95: {stats['p95']:.4f}")

    # 권장 NIGHT_AUG 범위 (분포 기반)
    dark_lo = max(0.02, stats['p5'] - 0.015)
    dark_hi = stats['p25'] * 1.02
    mod_lo = stats['p25'] * 0.98
    mod_hi = min(0.6, stats['p95'] + 0.05)
    bright_lo = max(0.02, stats['p5'] - 0.01)
    bright_hi = min(0.55, stats['p95'] + 0.08)
    print(f"\n=== Recommended NIGHT_AUG (lj4-aligned) ===")
    print(f"  DARK_RANGE:     [{dark_lo:.3f}, {dark_hi:.3f}]   # ~25% 하위 (극저조도)")
    print(f"  MODERATE_RANGE: [{mod_lo:.3f}, {mod_hi:.3f}]   # ~75% 상위 (달빛·가로등)")
    print(f"  BRIGHTNESS_RANGE: [{bright_lo:.3f}, {bright_hi:.3f}]")

    # YAML 스니펫 (config에 복붙용)
    print(f"\n# --- configs/*_hardaug3.yaml NIGHT_AUG에 복사 ---")
    print(f"    DARK_RANGE    : [{dark_lo:.3f}, {dark_hi:.3f}]")
    print(f"    MODERATE_RANGE: [{mod_lo:.3f}, {mod_hi:.3f}]")
    print(f"    BRIGHTNESS_RANGE : [{bright_lo:.3f}, {bright_hi:.3f}]")
    if args.output:
        out = Path(args.output)
        out.write_text(
            f"# lj4 brightness analysis - {stats['n']} samples\n"
            f"# mean={stats['mean']:.4f} std={stats['std']:.4f}\n"
            f"DARK_RANGE: [{dark_lo:.3f}, {dark_hi:.3f}]\n"
            f"MODERATE_RANGE: [{mod_lo:.3f}, {mod_hi:.3f}]\n"
            f"BRIGHTNESS_RANGE: [{bright_lo:.3f}, {bright_hi:.3f}]\n"
        )
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
