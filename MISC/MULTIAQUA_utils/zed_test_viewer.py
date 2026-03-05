#!/usr/bin/env python3
"""
MULTIAQUA_night zed RGB 이미지 뷰어.
- --split: train | val | test (어떤 스플릿 이미지를 볼지)
- 방향키: 이전/다음 이미지
- 슬라이더: 이미지 인덱스, gamma 보정 (0.1~3.0)

Usage:
  python MISC/MULTIAQUA_utils/zed_test_viewer.py --split test
  python MISC/MULTIAQUA_utils/zed_test_viewer.py --split train
  python MISC/MULTIAQUA_utils/zed_test_viewer.py --split val
"""
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night")
ZED_DIR = BASE_DIR / "MULTIAQUA_night" / "data" / "zed"
WIN_IMAGE = "zed | ←/→: prev/next"
WIN_CTRL = "Controls (index, gamma)"


def load_stems(split):
    """split: 'train' | 'val' | 'test'"""
    txt_path = BASE_DIR / f"{split}.txt"
    if not txt_path.exists():
        return []
    with open(txt_path) as f:
        stems = [line.strip() for line in f if line.strip()]
    return stems


def build_image_list(split):
    stems = load_stems(split)
    zed = Path(ZED_DIR)
    paths = []
    for s in stems:
        p = zed / f"{s}.png"
        if p.exists():
            paths.append((s, str(p)))
    return paths


def apply_gamma(img_bgr, gamma):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    inv_gamma = 1.0 / max(1e-3, float(gamma))
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def main():
    parser = argparse.ArgumentParser(description="MULTIAQUA_night zed 이미지 뷰어 (train/val/test)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="로드할 스플릿: train, val, test (default: test)")
    args = parser.parse_args()

    image_list = build_image_list(args.split)
    if not image_list:
        print(f"No images found. {args.split}.txt stems with existing {ZED_DIR}/*.png")
        return

    n = len(image_list)
    print(f"[{args.split}] {n} images. Arrow keys: prev/next. Sliders: index, gamma.")

    state = {"idx": 0, "gamma": 93}  # 93 → gamma≈1.0 (0~300 → 0.1~3.0)

    def on_index(val):
        state["idx"] = min(max(0, val), n - 1)

    def on_gamma(val):
        state["gamma"] = val

    # Qt 백엔드에서 트랙바가 NULL 창 핸들 에러를 일으킬 수 있음 → 트랙바 전용 창을 먼저 띄움
    use_trackbars = True
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    panel = np.zeros((90, 500, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    cv2.putText(panel, "index: image  |  gamma: 0.1~3", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow(WIN_CTRL, panel)
    cv2.waitKey(200)  # Qt가 창을 생성할 시간 확보

    try:
        cv2.createTrackbar("index", WIN_CTRL, 0, max(0, n - 1), on_index)
        cv2.createTrackbar("gamma (0.1~3)", WIN_CTRL, 93, 300, on_gamma)
    except cv2.error:
        use_trackbars = False
        print("Trackbars unavailable (Qt/display). Use keys: ←/→ index, -/= gamma, ESC quit.")

    cv2.namedWindow(WIN_IMAGE, cv2.WINDOW_NORMAL)

    while True:
        idx = state["idx"]
        gamma_val = 0.1 + (state["gamma"] / 300.0) * 2.9
        gamma_val = max(0.1, min(3.0, gamma_val))

        stem, path = image_list[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Failed to load", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            img = apply_gamma(img, gamma_val)
            info = f"[{args.split}] {idx+1}/{n}  {stem}  gamma={gamma_val:.2f}"
            cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WIN_IMAGE, img)
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            break
        if key == 81 or key == 2:  # Left
            state["idx"] = max(0, state["idx"] - 1)
            if use_trackbars:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
        if key == 83 or key == 3:  # Right
            state["idx"] = min(n - 1, state["idx"] + 1)
            if use_trackbars:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
        if not use_trackbars:
            if key == ord("-") or key == ord("_"):
                state["gamma"] = max(10, state["gamma"] - 10)
            if key == ord("=") or key == ord("+"):
                state["gamma"] = min(300, state["gamma"] + 10)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
