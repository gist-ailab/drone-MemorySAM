#!/usr/bin/env python3
"""
zed, zed_night, zed_night_to_day, zed_day 4개 폴더 이미지를 원본 해상도로 한 줄 concat 시각화.
- 기준 해상도: 첫 번째로 있는 이미지(zed 우선)의 원본 크기
- a: 이전 이미지, d: 다음 이미지
- 하단 슬라이드바: 이미지 인덱스 한꺼번에 넘기기

Usage:
  python MISC/MULTIAQUA_utils/zed_concat_viewer.py
  python MISC/MULTIAQUA_utils/zed_concat_viewer.py --split train
"""
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

DATA_ROOT = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data")
SUBFOLDERS = ["zed", "zed_night", "zed_night_to_day", "zed_day"]
EXT = ".png"
LABEL_H = 28
WIN_IMAGE = "zed | zed_night | zed_night_to_day | zed_day  [a/d: prev/next]"
WIN_CTRL = "Index (slider)"


def load_stems(split: str = None):
    """split이 있으면 {split}.txt에서 stem 로드, 없으면 zed 폴더에서 *.png stem 수집."""
    root = DATA_ROOT.parent.parent  # MULTIAQUA_night2
    if split:
        txt = root / f"{split}.txt"
        if txt.exists():
            with open(txt) as f:
                return [line.strip() for line in f if line.strip()]
    zed_dir = DATA_ROOT / "zed"
    if not zed_dir.exists():
        return []
    stems = []
    for f in sorted(zed_dir.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            stems.append(f.stem)
    return stems


def build_stem_list(split: str = None):
    """최소 한 폴더라도 존재하는 stem만 반환."""
    stems = load_stems(split)
    out = []
    for s in stems:
        for sub in SUBFOLDERS:
            p = DATA_ROOT / sub / f"{s}{EXT}"
            if p.exists():
                out.append(s)
                break
    return out


def load_four(stem: str):
    """(zed, zed_night, zed_night_to_day, zed_day) 순서로 BGR 이미지 리스트. 없으면 None."""
    imgs = []
    for sub in SUBFOLDERS:
        path = DATA_ROOT / sub / f"{stem}{EXT}"
        if path.exists():
            img = cv2.imread(str(path))
            imgs.append(img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            imgs.append(None)
    return imgs


def get_reference_size(imgs: list) -> tuple:
    """첫 번째로 존재하는 이미지의 (h, w) 반환. 없으면 (480, 640)."""
    for img in imgs:
        if img is not None and img.size > 0:
            return img.shape[:2]
    return (480, 640)


def resize_to_size(img: np.ndarray, ref_h: int, ref_w: int) -> np.ndarray:
    """이미지를 원본 해상도(ref_h, ref_w)로 맞춤. 없으면 검정 칸."""
    if img is None or img.size == 0:
        return np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if (h, w) == (ref_h, ref_w):
        return img.copy()
    return cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)


def make_row(imgs: list, labels: list, ref_h: int, ref_w: int):
    """4개 이미지를 원본 크기(ref_h, ref_w)로 맞춰 가로로 이어붙이고, 각 위에 라벨."""
    cells = []
    for img, label in zip(imgs, labels):
        cell = resize_to_size(img, ref_h, ref_w)
        label_bar = np.zeros((LABEL_H, ref_w, 3), dtype=np.uint8)
        label_bar[:] = (50, 50, 50)
        cv2.putText(
            label_bar, label, (8, LABEL_H - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        cell_with_label = np.vstack([label_bar, cell])
        cells.append(cell_with_label)
    row = np.hstack([np.array(c) for c in cells])
    # 구분선
    total_w = row.shape[1]
    for i in range(1, 4):
        x = ref_w * i
        row[:, x : x + 2] = (80, 80, 80)
    return row


def main():
    parser = argparse.ArgumentParser(description="zed / zed_night / zed_night_to_day / zed_day concat 뷰어")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"],
                        help="지정 시 train/val/test.txt stem만 사용. 없으면 zed 폴더 전체")
    args = parser.parse_args()

    stems = build_stem_list(args.split)
    if not stems:
        print(f"No stems found under {DATA_ROOT} (subfolders: {SUBFOLDERS})")
        return

    n = len(stems)
    print(f"Stems: {n}. Keys: a=prev, d=next. Slider: index. ESC=quit.")

    state = {"idx": 0}

    def on_trackbar(val):
        state["idx"] = min(max(0, val), n - 1)

    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    ctrl_panel = np.zeros((60, 600, 3), dtype=np.uint8)
    ctrl_panel[:] = (45, 45, 45)
    cv2.putText(ctrl_panel, "Slider: image index  |  a: prev  d: next  ESC: quit", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.imshow(WIN_CTRL, ctrl_panel)
    cv2.waitKey(100)

    try:
        cv2.createTrackbar("index", WIN_CTRL, 0, max(0, n - 1), on_trackbar)
    except cv2.error:
        pass

    cv2.namedWindow(WIN_IMAGE, cv2.WINDOW_NORMAL)

    while True:
        idx = state["idx"]
        stem = stems[idx]
        imgs = load_four(stem)
        ref_h, ref_w = get_reference_size(imgs)
        row = make_row(imgs, SUBFOLDERS, ref_h, ref_w)
        # 상단에 stem / 인덱스 표시
        info = np.zeros((LABEL_H, row.shape[1], 3), dtype=np.uint8)
        info[:] = (30, 30, 30)
        cv2.putText(info, f"  {stem}  [{idx+1} / {n}]", (10, LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        canvas = np.vstack([info, row])
        cv2.imshow(WIN_IMAGE, canvas)

        key = cv2.waitKey(80) & 0xFF
        if key == 27:
            break
        if key == ord("a") or key == ord("A"):
            state["idx"] = max(0, state["idx"] - 1)
            try:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
            except Exception:
                pass
        if key == ord("d") or key == ord("D"):
            state["idx"] = min(n - 1, state["idx"] + 1)
            try:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
            except Exception:
                pass

        # 슬라이더가 다른 창에서 움직였을 수 있음
        try:
            tb_val = cv2.getTrackbarPos("index", WIN_CTRL)
            if 0 <= tb_val < n and tb_val != state["idx"]:
                state["idx"] = tb_val
        except Exception:
            pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
