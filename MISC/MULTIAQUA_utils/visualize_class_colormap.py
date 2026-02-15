"""
RGB + SegMask concat 시각화 (클래스-컬러 매핑 확인용).

- RGB | Seg(colored) | Overlay | Legend 4패널
- 마스크 픽셀값 0, 1, 2, ... 각각 다른 색으로 표시
- 범례: idx: Class이름 (* = 현재 이미지에 존재)
- 시각적으로 확인 후 class_names 리스트를 multiaqua.py CLASSES에 반영

실행: python visualize_class_colormap.py
조작: 방향키(←→) 인덱스 이동, 트랙바, Q 종료
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night"
MAX_DISPLAY_H = 600
LEGEND_PATCH_SIZE = 20
LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX

# 인덱스별 구분 쉬운 색상 (BGR, 0~6+)
COLORMAP = [
    [70, 130, 180],   # 0: 하늘색
    [45, 60, 150],    # 1: 파랑
    [107, 142, 35],   # 2: 초록
    [220, 20, 60],    # 3: 빨강
    [255, 165, 0],    # 4: 주황
    [128, 0, 128],    # 5: 보라
    [0, 255, 255],    # 6: 시안
    [255, 255, 0],    # 7: 노랑
    [128, 128, 128],  # 8: 회색
]
# BGR로 변환 (위는 RGB 기준)
COLORMAP_BGR = [list(reversed(c)) for c in COLORMAP]


def get_stems(root: Path, rgb_dir: Path, ann_dir: Path) -> Tuple[List[str], List[int]]:
    """(all_stems, split_boundaries)."""
    splits = ["train", "val", "test"]
    all_stems = []
    boundaries = [0]
    for s in splits:
        split_file = root / f"{s}.txt"
        if not split_file.exists():
            continue
        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
        for st in stems:
            if (rgb_dir / f"{st}.png").exists() and (ann_dir / f"{st}.png").exists():
                all_stems.append(st)
        boundaries.append(len(all_stems))
    return all_stems, boundaries


def load_rgb(rgb_dir: Path, stem: str) -> np.ndarray:
    p = rgb_dir / f"{stem}.png"
    img = np.array(Image.open(str(p)).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_mask_raw(ann_dir: Path, stem: str) -> np.ndarray:
    """원본 마스크 픽셀 값 (0, 1, 2, ...) 그대로 반환."""
    p = ann_dir / f"{stem}.png"
    arr = np.array(Image.open(str(p)))
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def mask_to_colored(mask: np.ndarray) -> np.ndarray:
    """마스크를 컬러맵으로 변환."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    uniq = np.unique(mask)
    for v in uniq:
        vi = int(v)
        if vi < 0 or vi == 255:
            color = [40, 40, 40]
        else:
            color = COLORMAP_BGR[vi % len(COLORMAP_BGR)]
        out[mask == v] = color
    return out


def draw_legend(class_names: list, unique_in_mask: set) -> np.ndarray:
    """범례 패널: idx -> Class 이름, 현재 이미지에 있으면 표시."""
    pad = 20
    line_h = 28
    patch_sz = LEGEND_PATCH_SIZE
    n_lines = max(len(class_names), max(unique_in_mask) + 1) if unique_in_mask else len(class_names)
    n_lines = max(n_lines, 1)
    panel_w = 240
    panel_h = pad * 2 + n_lines * line_h
    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 30
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (80, 80, 80), 1)
    cv2.putText(panel, "idx: Class (appears in image)", (pad, pad + 20), LEGEND_FONT, 0.5, (200, 200, 200), 1)
    y = pad + 20 + line_h
    for i in range(n_lines):
        color = COLORMAP_BGR[i % len(COLORMAP_BGR)] if i < len(COLORMAP_BGR) else [150, 150, 150]
        cv2.rectangle(panel, (pad, y - patch_sz), (pad + patch_sz, y), color, -1)
        cv2.rectangle(panel, (pad, y - patch_sz), (pad + patch_sz, y), (100, 100, 100), 1)
        name = class_names[i] if i < len(class_names) else f"Class_{i}"
        in_mask = " *" if i in unique_in_mask else ""
        cv2.putText(panel, f"{i}: {name}{in_mask}", (pad + patch_sz + 8, y - 4), LEGEND_FONT, 0.4, (220, 220, 220), 1)
        y += line_h
    return panel


def resize_to_display(img: np.ndarray, max_h: int = MAX_DISPLAY_H) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / h
    return cv2.resize(img, (int(w * scale), max_h), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="MULTIAQUA RGB+Seg class colormap visualization")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="MULTIAQUA_night root")
    args = parser.parse_args()
    root = Path(args.root)
    data_root = root / "MULTIAQUA_night"
    rgb_dir = data_root / "data" / "zed"
    ann_dir = data_root / "annotations"
    if not ann_dir.exists():
        print(f"Annotations dir not found: {ann_dir}")
        return

    all_stems, split_boundaries = get_stems(root, rgb_dir, ann_dir)
    splits = ["train", "val", "test"]
    if not all_stems:
        print(f"No data under {root}")
        return
    # Pad boundaries for safe indexing
    while len(split_boundaries) < 4:
        split_boundaries.append(len(all_stems))
    print(f"Total {len(all_stems)} samples: train {split_boundaries[1]-split_boundaries[0]}, "
          f"val {split_boundaries[2]-split_boundaries[1]}, test {split_boundaries[3]-split_boundaries[2]}")

    # 확인된 매핑: 0=Recording Boat(ignore), 1=Static, 2=Dynamic, 3=Water, 4=Sky
    class_names = ["Recording Boat", "Static", "Dynamic", "Water", "Sky"]
    max_seen = 0
    for stem in all_stems[:100]:
        m = load_mask_raw(ann_dir, stem)
        if m.size:
            max_seen = max(max_seen, int(np.max(m)))
    for i in range(len(class_names), max_seen + 1):
        class_names.append(f"Class_{i}")

    idx = [0]

    def get_split_name():
        i = idx[0]
        for k, b in enumerate(split_boundaries[1:]):
            if i < b:
                return splits[k]
        return splits[-1]

    win = "MULTIAQUA Class Colormap: arrows index, Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("index", win, 0, max(0, len(all_stems) - 1), lambda v: idx.__setitem__(0, min(max(0, v), len(all_stems) - 1)))

    while True:
        i = min(max(0, idx[0]), len(all_stems) - 1)
        idx[0] = i
        stem = all_stems[i]
        rgb = load_rgb(rgb_dir, stem)
        mask = load_mask_raw(ann_dir, stem)
        unique_vals = set(np.unique(mask).tolist()) - {255}
        seg_colored = mask_to_colored(mask)
        # RGB | Seg | Overlay
        overlay = cv2.addWeighted(rgb, 0.6, seg_colored, 0.4, 0)
        # 같은 크기로
        h, w = rgb.shape[:2]
        seg_colored = cv2.resize(seg_colored, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
        row = np.hstack([rgb, seg_colored, overlay])
        # 범례 패널
        legend = draw_legend(class_names, unique_vals)
        lh, lw = legend.shape[:2]
        if row.shape[0] < lh:
            row = cv2.resize(row, (int(row.shape[1] * lh / row.shape[0]), lh), interpolation=cv2.INTER_LINEAR)
        elif row.shape[0] > lh:
            legend = cv2.resize(legend, (lw, row.shape[0]), interpolation=cv2.INTER_NEAREST)
        row = np.hstack([row, legend])
        row = resize_to_display(row)
        cv2.putText(row, "RGB | Seg(colored) | Overlay | Legend", (10, 24), LEGEND_FONT, 0.6, (0, 255, 0), 2)
        cv2.putText(row, f"{get_split_name()} [{i+1}/{len(all_stems)}] {stem}", (10, row.shape[0] - 10), LEGEND_FONT, 0.5, (255, 255, 255), 2)
        cv2.putText(row, f"unique in mask: {sorted(unique_vals)}", (10, row.shape[0] - 32), LEGEND_FONT, 0.45, (200, 200, 200), 1)
        cv2.imshow(win, row)
        cv2.setTrackbarPos("index", win, i)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == 81 or key == 2:
            idx[0] = max(0, i - 1)
        if key == 83 or key == 3:
            idx[0] = min(len(all_stems) - 1, i + 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
