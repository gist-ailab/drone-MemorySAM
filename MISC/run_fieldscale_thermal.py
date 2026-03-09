#!/usr/bin/env python3
"""
thermal_camera 폴더의 이미지를 Fieldscale로 처리해 thermal_processed_fieldscale 에 저장.
Ref: https://github.com/HyeonJaeGil/fieldscale
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fieldscale import Fieldscale

DATA_ROOT = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data")
THERMAL_DIR = DATA_ROOT / "thermal_camera"
OUT_DIR = DATA_ROOT / "thermal_processed_fieldscale3"

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Fieldscale 파라미터 (하얗게 나오면 아래 조정)
# - gamma: 높을수록 어두워짐 (기본 1.5 → 2.0~2.5 권장)
# - max_diff, min_diff: 낮을수록 극값 억제 강함 → 과노출 완화 (400 → 200~250)
# - clahe: False면 CLAHE 미적용 (밝기 과다 시 끄기)
FIELDSCALE_GAMMA = 2.2
FIELDSCALE_MAX_DIFF = 250
FIELDSCALE_MIN_DIFF = 250
FIELDSCALE_ITERATION = 7
FIELDSCALE_CLAHE = True
FIELDSCALE_CLAHE_CLIP = 1.5  # CLAHE clipLimit (낮을수록 대비 완만)


def ensure_grayscale(img):
    """Fieldscale은 2D 이미지를 기대. 3채널→그레이, 16비트→8비트 정규화.
    유효 범위가 좁을 때(예: 90~99) 0~255로 스트레칭해 객체 구분이 되도록 함."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float64), 0, 65535)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    # 8비트인데 범위가 좁으면 (90~99 등) 0~255로 스트레칭
    mn, mx = int(img.min()), int(img.max())
    if mx > mn and (mx - mn) < 100:
        img = np.clip((img.astype(np.float64) - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
    return img


def main():
    if not THERMAL_DIR.is_dir():
        print("thermal_camera 폴더가 없습니다:", THERMAL_DIR)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [f for f in THERMAL_DIR.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
    if not images:
        print("이미지가 없습니다:", THERMAL_DIR)
        return

    fieldscale = Fieldscale(
        max_diff=FIELDSCALE_MAX_DIFF,
        min_diff=FIELDSCALE_MIN_DIFF,
        iteration=FIELDSCALE_ITERATION,
        gamma=FIELDSCALE_GAMMA,
        clahe=FIELDSCALE_CLAHE,
        clahe_clip_limit=FIELDSCALE_CLAHE_CLIP,
        video=False,
    )

    for path in tqdm(sorted(images), desc="Fieldscale thermal"):
        img = cv2.imread(str(path), -1)
        if img is None:
            tqdm.write(f"Skip (read fail): {path.name}")
            continue
        gray = ensure_grayscale(img)
        try:
            out = fieldscale(gray)
        except Exception as e:
            tqdm.write(f"Skip ({e}): {path.name}")
            continue
        out_path = OUT_DIR / path.name
        cv2.imwrite(str(out_path), out)

    print("저장 경로:", OUT_DIR)
    print("처리 개수:", len(images))


if __name__ == "__main__":
    main()
