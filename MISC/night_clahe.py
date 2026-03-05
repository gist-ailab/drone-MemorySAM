#!/usr/bin/env python3
"""
zed 폴더에서 lj4_로 시작하는 이미지만 불러와, 원본 vs (gamma → CLAHE) 적용을 나란히 비교. gamma 높을수록 밝음.
방향키 ←/→ 및 하단 인덱스 바(슬라이더)로 이미지 전환. Q/ESC 종료.
"""
import cv2
import numpy as np
from pathlib import Path

# zed 폴더 경로 (서브폴더 없이 직접 경로)
ZED_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/zed")
# 파일명이 이 접두사로 시작하는 것만 사용 (예: lj4_1_085040.png)
FILE_PREFIX = "lj4_"
# gamma를 높일수록 밝아짐 (output = input^(1/gamma)). CLAHE 적용 전에 먼저 적용
GAMMA = 1.7

STRIP_H = 50
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
WINDOW_NAME = "night: original | gamma+CLAHE  [←/→]  Q/ESC: quit"


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """gamma를 높일수록 밝아짐 (output = input^(1/gamma))."""
    if bgr is None or bgr.size == 0:
        return bgr
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, table)


def apply_clahe_bgr(bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """BGR 이미지에 CLAHE 적용. LAB의 L 채널에만 적용 후 합성."""
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main():
    if not ZED_DIR.is_dir():
        print("zed 폴더가 없습니다:", ZED_DIR)
        return

    paths = []
    for f in sorted(ZED_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in EXTS and f.stem.startswith(FILE_PREFIX):
            paths.append(f)
    if not paths:
        print(f"zed 폴더에 '{FILE_PREFIX}'로 시작하는 이미지가 없습니다:", ZED_DIR)
        return

    n_total = len(paths)
    # 첫 프레임에서 원본 크기 확보 (dummy용)
    first_bgr = cv2.imread(str(paths[0]))
    ref_h, ref_w = (first_bgr.shape[0], first_bgr.shape[1]) if first_bgr is not None and first_bgr.size > 0 else (480, 640)
    canvas_w = ref_w * 2
    canvas_h = ref_h

    try:
        cv2.startWindowThread()
    except Exception:
        pass

    idx = [0]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    dummy = np.zeros((canvas_h + STRIP_H, canvas_w, 3), dtype=np.uint8)
    dummy[:] = (40, 40, 40)
    cv2.putText(dummy, "Loading...", (canvas_w // 2 - 50, (canvas_h + STRIP_H) // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WINDOW_NAME, dummy)
    cv2.waitKey(200)
    try:
        cv2.resizeWindow(WINDOW_NAME, canvas_w, canvas_h + STRIP_H)
    except cv2.error:
        pass
    try:
        cv2.createTrackbar("index", WINDOW_NAME, 0, max(0, n_total - 1), lambda _: None)
    except Exception:
        pass

    print(f"야간 이미지 {n_total}장. ←/→: 이전/다음. Q/ESC: 종료.")

    while True:
        idx[0] = min(max(0, idx[0]), n_total - 1)
        try:
            tb = cv2.getTrackbarPos("index", WINDOW_NAME)
            if 0 <= tb < n_total:
                idx[0] = tb
        except Exception:
            pass
        cur = idx[0]
        path = paths[cur]

        bgr = cv2.imread(str(path))
        if bgr is None:
            ref_h, ref_w = 480, 640
            left = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
            cv2.putText(left, "Load failed", (20, ref_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            right = left.copy()
        else:
            ref_h, ref_w = bgr.shape[0], bgr.shape[1]
            left = bgr.copy()
            # gamma 적용 후 CLAHE (gamma 높을수록 밝아짐 → 어두운 이미지 품질 개선)
            gamma_bgr = apply_gamma(bgr, GAMMA)
            right = apply_clahe_bgr(gamma_bgr)

        canvas = np.hstack([left, right])
        canvas_w = ref_w * 2
        canvas_h = ref_h
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "original", (12, ref_h - 10), font, 0.55, (0, 255, 200), 1)
        cv2.putText(canvas, f"gamma {GAMMA} + CLAHE", (ref_w + 12, ref_h - 10), font, 0.55, (0, 255, 255), 1)

        strip = np.zeros((STRIP_H, canvas_w, 3), dtype=np.uint8)
        strip[:] = (50, 50, 50)
        bar_w = canvas_w - 20
        bar_x0 = 10
        cv2.rectangle(strip, (bar_x0, 6), (bar_x0 + bar_w, 24), (80, 80, 80), 1)
        if n_total > 1:
            fill = int(bar_w * (cur + 1) / n_total)
            cv2.rectangle(strip, (bar_x0, 6), (bar_x0 + fill, 24), (0, 200, 255), -1)
        cv2.putText(strip, f" [{cur+1}/{n_total}] {path.name}  <- / -> keys", (10, 42), font, 0.5, (220, 220, 220), 1)
        canvas = np.vstack([canvas, strip])

        cv2.imshow(WINDOW_NAME, canvas)
        try:
            cv2.setTrackbarPos("index", WINDOW_NAME, cur)
        except Exception:
            pass

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (81, 2):
            idx[0] = max(0, idx[0] - 1)
            try:
                cv2.setTrackbarPos("index", WINDOW_NAME, idx[0])
            except Exception:
                pass
        elif key in (83, 3):
            idx[0] = min(n_total - 1, idx[0] + 1)
            try:
                cv2.setTrackbarPos("index", WINDOW_NAME, idx[0])
            except Exception:
                pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
