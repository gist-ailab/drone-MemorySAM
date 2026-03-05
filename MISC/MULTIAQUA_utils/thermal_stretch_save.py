#!/usr/bin/env python3
"""
Option E: Per-image percentile stretch (2%-98%) → 0-255.
태양 등 극단값 제외, ROI만 stretch 후 thermal_strech에 저장.

Usage:
  python MISC/MULTIAQUA_utils/thermal_stretch_save.py
"""
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm

SRC_DIR = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/thermal_camera"
DST_DIR = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/thermal_strech"
PADDING_THRESH = 10
P_LOW, P_HIGH = 2, 98


def stretch_image(img: np.ndarray) -> np.ndarray:
    """ROI(>padding_thresh)에 대해 percentile stretch → 0-255. 패딩은 0 유지."""
    roi = img > PADDING_THRESH
    if not roi.any():
        return np.zeros_like(img)
    rv = img[roi].astype(np.float32)
    p2, p98 = np.percentile(rv, [P_LOW, P_HIGH])
    if p98 - p2 < 1e-6:
        p2, p98 = float(rv.min()), float(rv.max())
    if p98 - p2 < 1e-6:
        out = np.zeros_like(img)
        out[roi] = 128
        return out
    stretched = np.clip((img.astype(np.float32) - p2) / (p98 - p2), 0, 1)
    out = (stretched * 255).astype(np.uint8)
    out[~roi] = 0
    return out


def main():
    src = Path(SRC_DIR)
    dst = Path(DST_DIR)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.png"))
    if not files:
        print(f"No PNG in {src}")
        return

    print(f"Option E (percentile {P_LOW}-{P_HIGH}% stretch) -> {dst}")
    print(f"Total: {len(files)} images")

    for fpath in tqdm(files, desc="Stretch"):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out = stretch_image(img)
        # 데이터셋 호환: multiaqua는 thermal_dir / f"{stem}_thermal.png" 사용
        stem = fpath.stem
        out_name = f"{stem}_thermal.png"
        cv2.imwrite(str(dst / out_name), out)

    print(f"Done. Saved to {dst}")


if __name__ == "__main__":
    main()
