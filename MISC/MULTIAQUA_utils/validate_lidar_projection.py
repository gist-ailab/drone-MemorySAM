"""
MULTIAQUA LIDAR projection 검증.

1) 정합(alignment) 확인: RGB 위에 LIDAR를 겹쳐서 저장 → 물체/수평선과 맞는지 눈으로 확인
2) 밀도 비교: 프레임별 점 개수·points_per_1k_pixels 출력 → DELIVER와 수치로 비교
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MISC.MULTIAQUA_utils.lidar_proces import (
    lidar_frame_stats,
    overlay_lidar_on_rgb,
)


def image_fill_stats(img_path: str) -> dict:
    """DELIVER 등 기성 lidar 이미지에서 배경이 아닌 픽셀 비율·개수 추정."""
    import cv2
    import numpy as np
    img = cv2.imread(img_path)
    if img is None:
        return {}
    # color: 흰색(255) 배경 / grayscale: 검정(0) 배경
    if img.ndim == 3:
        is_white_bg = np.all(img >= 250, axis=2)
        total = img.shape[0] * img.shape[1]
        fill_pixels = int(np.sum(~is_white_bg))
    else:
        # grayscale: 0에 가까우면 배경
        is_white_bg = img >= 250
        total = img.size
        fill_pixels = int(np.sum(~is_white_bg))
    return {
        "fill_pixels": fill_pixels,
        "total_pixels": total,
        "fill_ratio": round(fill_pixels / total, 6) if total else 0,
        "H": img.shape[0],
        "W": img.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Validate MULTIAQUA LIDAR projection (overlay + stats)")
    parser.add_argument("--rgb", type=str, required=True, help="RGB image path (same frame as LIDAR)")
    parser.add_argument("--lidar-color", type=str, required=True, help="LIDAR color image path (e.g. ..._lidar_color.png)")
    parser.add_argument("--lidar-npy", type=str, default=None, help="LIDAR .npy path (for stats; optional)")
    parser.add_argument("--out-overlay", type=str, default=None, help="Output path for RGB+LIDAR overlay (default: same dir as lidar_color, suffix _overlay.png)")
    parser.add_argument("--alpha", type=float, default=0.55, help="LIDAR blend alpha (0=only RGB, 1=only LIDAR)")
    parser.add_argument("--compare-deliver", type=str, default=None, help="DELIVER lidar image path (e.g. 015950_lidar_front_color.png) to print fill ratio for comparison")
    args = parser.parse_args()

    rgb_path = Path(args.rgb)
    lidar_color_path = Path(args.lidar_color)
    if not rgb_path.exists():
        print(f"Error: RGB not found: {rgb_path}")
        return 1
    if not lidar_color_path.exists():
        print(f"Error: LIDAR color image not found: {lidar_color_path}")
        return 1

    # 1) Overlay: 정합 확인용
    out = args.out_overlay
    if not out:
        out = str(lidar_color_path.parent / (lidar_color_path.stem + "_overlay.png"))
    overlay_lidar_on_rgb(str(rgb_path), str(lidar_color_path), out, alpha=args.alpha)
    print(f"Overlay saved: {out}")
    print("  → 열어서 LIDAR 색이 나무/수면/보트 등 물체와 맞는지 확인하세요. 맞으면 projection은 올바릅니다.")

    # 2) MULTIAQUA 프레임 통계
    npy_path = args.lidar_npy
    if npy_path is None:
        # stem에서 추론: adr1_1_000400_lidar_color.png -> adr1_1_000400.npy
        stem = lidar_color_path.stem.replace("_lidar_color", "").replace("_lidar", "")
        for d in [lidar_color_path.parent, lidar_color_path.parent.parent]:
            cand = Path(d) / ".." / "lidar" / f"{stem}.npy"
            cand = cand.resolve()
            if cand.exists():
                npy_path = str(cand)
                break
            cand2 = Path(d) / f"{stem}.npy"
            if cand2.exists():
                npy_path = str(cand2)
                break
    if npy_path and Path(npy_path).exists():
        stats = lidar_frame_stats(npy_path, ref_image_path=str(rgb_path))
        if "error" in stats:
            print("MULTIAQUA stats:", stats)
        else:
            print("MULTIAQUA frame stats:")
            print(f"  num_points: {stats['num_points']}, in_fov: {stats['num_points_in_fov']}")
            print(f"  image: {stats['H']}x{stats['W']} = {stats['pixels']} px")
            print(f"  points_per_1k_pixels: {stats['points_per_1k_pixels']}")
    else:
        print("(LIDAR .npy not provided; skipping MULTIAQUA stats. Use --lidar-npy)")

    # 3) DELIVER 비교 (선택)
    if args.compare_deliver and Path(args.compare_deliver).exists():
        d = image_fill_stats(args.compare_deliver)
        if d:
            print("DELIVER reference (fill ratio from image):")
            print(f"  fill_pixels: {d['fill_pixels']}, total: {d['total_pixels']}, fill_ratio: {d['fill_ratio']}")
            print("  → MULTIAQUA points_per_1k_pixels와 이미지 fill 비율을 비교하면 데이터 밀도 차이를 알 수 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
