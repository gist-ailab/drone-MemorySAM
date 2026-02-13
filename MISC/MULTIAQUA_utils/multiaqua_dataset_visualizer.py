"""
MULTIAQUA 데이터셋 시각화.
- train / val / test 선택 (트랙바)
- 인덱스 이동: 방향키(←→) 또는 트랙바
- 4패널: rgb | ir(thermal) | lidar | rgb+seg overlay
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 프로젝트 루트
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semseg.datasets.multiaqua import MULTIAQUA

ROOT = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night"
MAX_DISPLAY_W = 480  # 패널 하나당 최대 너비
OVERLAY_ALPHA = 0.5  # rgb+seg 블렌딩


def tensor_to_bgr(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) RGB uint8 -> (H,W,3) BGR."""
    if x.dim() == 3:
        x = x.permute(1, 2, 0).numpy()
    else:
        x = x.numpy()
    if x.shape[-1] == 3:
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def build_rgb_seg_overlay(rgb: np.ndarray, label: np.ndarray, palette, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """RGB 위에 segmentation 컬러맵 블렌딩. label은 0-based."""
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    seg_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        mask = label == cls_id
        seg_rgb[mask] = palette[cls_id].numpy() if isinstance(palette, torch.Tensor) else np.array(palette[cls_id])
    seg_rgb = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
    if seg_rgb.shape[:2] != rgb.shape[:2]:
        seg_rgb = cv2.resize(seg_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = rgb.astype(np.float32) * (1 - alpha) + seg_rgb.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_to_display(img: np.ndarray, max_w: int = MAX_DISPLAY_W) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def main():
    splits = ["train", "val", "test"]
    datasets = {}
    for s in splits:
        try:
            datasets[s] = MULTIAQUA(root=ROOT, split=s, transform=None, modals=["img", "lidar", "thermal"])
        except Exception as e:
            print(f"Skip {s}: {e}")
            datasets[s] = None
    if not any(datasets.values()):
        print("No split available.")
        return

    current_split_idx = [0]
    current_index = [0]

    def get_ds():
        s = splits[current_split_idx[0]]
        return datasets.get(s)

    def get_n():
        ds = get_ds()
        return len(ds) if ds else 0

    win = "MULTIAQUA: [1/2/3] or trackbar split, arrows index, Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("split (0=train 1=val 2=test)", win, 0, 2, lambda v: current_split_idx.__setitem__(0, v))
    cv2.createTrackbar("index", win, 0, max(0, get_n() - 1), lambda v: current_index.__setitem__(0, min(max(0, v), max(0, get_n() - 1))))

    while True:
        ds = get_ds()
        if ds is None:
            cv2.waitKey(1)
            continue
        n = len(ds)
        idx = min(max(0, current_index[0]), n - 1)
        current_index[0] = idx

        sample, label = ds[idx]
        img, lidar, thermal = sample[0], sample[1], sample[2]
        # (C,H,W) -> BGR (H,W,3)
        rgb = tensor_to_bgr(img)
        ir = tensor_to_bgr(thermal)
        lidar_bgr = tensor_to_bgr(lidar)
        label_np = label.numpy()
        rgb_seg = build_rgb_seg_overlay(rgb, label_np, MULTIAQUA.PALETTE)

        th, tw = rgb.shape[0], rgb.shape[1]
        ir_s = cv2.resize(ir, (tw, th), interpolation=cv2.INTER_LINEAR)
        lidar_s = cv2.resize(lidar_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgb_seg_s = cv2.resize(rgb_seg, (tw, th), interpolation=cv2.INTER_LINEAR)

        rgb_d = resize_to_display(rgb)
        ir_d = resize_to_display(ir_s)
        lidar_d = resize_to_display(lidar_s)
        seg_d = resize_to_display(rgb_seg_s)
        th_d, tw_d = rgb_d.shape[0], rgb_d.shape[1]
        ir_d = cv2.resize(ir_d, (tw_d, th_d))
        lidar_d = cv2.resize(lidar_d, (tw_d, th_d))
        seg_d = cv2.resize(seg_d, (tw_d, th_d))

        row1 = np.hstack([rgb_d, ir_d])
        row2 = np.hstack([lidar_d, seg_d])
        w_max = max(row1.shape[1], row2.shape[1])
        if row1.shape[1] < w_max:
            row1 = np.hstack([row1, np.zeros((row1.shape[0], w_max - row1.shape[1], 3), dtype=np.uint8)])
        if row2.shape[1] < w_max:
            row2 = np.hstack([row2, np.zeros((row2.shape[0], w_max - row2.shape[1], 3), dtype=np.uint8)])
        canvas = np.vstack([row1, row2])

        split_name = splits[current_split_idx[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "rgb", (10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "ir", (tw_d + 10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "lidar", (10, th_d + 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "rgb+seg", (tw_d + 10, th_d + 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"{split_name} [{idx+1}/{n}] {ds.stems[idx]}", (10, canvas.shape[0] - 10), font, 0.5, (255, 255, 255), 2)

        cv2.imshow(win, canvas)
        cv2.setTrackbarPos("split (0=train 1=val 2=test)", win, current_split_idx[0])
        cv2.setTrackbarPos("index", win, idx)

        key = cv2.waitKey(1) & 0xFF
        # 종료: ESC 또는 소문자 q만 (81은 일부 환경에서 왼쪽 방향키와 겹침)
        if key == 27 or key == ord("q"):
            break
        if key == ord("1"):
            current_split_idx[0] = 0
            current_index[0] = 0
        if key == ord("2"):
            current_split_idx[0] = 1
            current_index[0] = 0
        if key == ord("3"):
            current_split_idx[0] = 2
            current_index[0] = 0
        # 방향키: 81/2=왼쪽, 83/3=오른쪽 (플랫폼별 코드 다름)
        if key == 81 or key == 2:
            current_index[0] = max(0, idx - 1)
        if key == 83 or key == 3:
            current_index[0] = min(n - 1, idx + 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
