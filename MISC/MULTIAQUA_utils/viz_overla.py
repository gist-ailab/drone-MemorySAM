"""
RGB + LIDAR 페어 시각화.
zed(RGB), lidar_processed(lidar, lidar_color)에서 같은 stem으로 페어 잡아
rgb, lidar, lidar_color, overlay 2x2로 띄우고, 방향키로 이전/다음, 트랙바로 인덱스 이동.
"""

import cv2
import numpy as np
from pathlib import Path

RGB_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed")
LIDAR_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/lidar_processed2")

# 표시용 최대 크기 (넘치면 리사이즈)
MAX_DISPLAY_W = 640
OVERLAY_ALPHA = 0.55


def build_pairs():
    """lidar_processed에서 _lidar_color.png 목록 기준으로 stem 수집, RGB 경로와 매칭."""
    pairs = []
    for p in sorted(LIDAR_DIR.glob("*_lidar_color.png")):
        stem = p.stem.replace("_lidar_color", "")
        rgb_path = RGB_DIR / f"{stem}.png"
        lidar_path = LIDAR_DIR / f"{stem}_lidar.png"
        if rgb_path.exists() and lidar_path.exists():
            pairs.append({
                "stem": stem,
                "rgb": str(rgb_path),
                "lidar": str(lidar_path),
                "lidar_color": str(p),
            })
    return pairs


def make_overlay(rgb: np.ndarray, lidar_color: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """RGB 위에 lidar_color 블렌딩 (메모리만)."""
    if rgb.shape[:2] != lidar_color.shape[:2]:
        lidar_color = cv2.resize(
            lidar_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    white = np.all(lidar_color >= 250, axis=2)
    blend = rgb.astype(np.float32)
    blend[~white] = (1 - alpha) * rgb[~white] + alpha * lidar_color[~white]
    return np.clip(blend, 0, 255).astype(np.uint8)


def resize_to_display(img: np.ndarray, max_w: int = MAX_DISPLAY_W) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def load_and_build_frame(pair: dict) -> tuple:
    rgb = cv2.imread(pair["rgb"])
    lidar = cv2.imread(pair["lidar"])
    lidar_color = cv2.imread(pair["lidar_color"])
    if rgb is None or lidar is None or lidar_color is None:
        return None, None, None, None
    overlay = make_overlay(rgb, lidar_color)
    # grayscale lidar를 3채널로 (나란히 보기 위해)
    if lidar.ndim == 2:
        lidar = cv2.cvtColor(lidar, cv2.COLOR_GRAY2BGR)
    return rgb, lidar, lidar_color, overlay


def main():
    pairs = build_pairs()
    if not pairs:
        print("No pairs found. Check RGB_DIR and LIDAR_DIR.")
        return
    n = len(pairs)
    print(f"Found {n} pairs. Use left/right arrow or trackbar.")

    current_index = [0]  # list so callback can mutate

    def on_trackbar(val):
        current_index[0] = min(max(0, val), n - 1)

    win = "rgb | lidar | lidar_color | overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("index", win, 0, max(0, n - 1), on_trackbar)

    while True:
        idx = current_index[0]
        pair = pairs[idx]
        rgb, lidar, lidar_color, overlay = load_and_build_frame(pair)
        if rgb is None:
            continue

        rgb_s = resize_to_display(rgb)
        th, tw = rgb_s.shape[0], rgb_s.shape[1]
        # 네 칸 모두 (tw, th)로 맞춤 (RGB/LIDAR 해상도 달라도 동일 크기로 표시)
        lidar_s = cv2.resize(lidar, (tw, th), interpolation=cv2.INTER_LINEAR)
        if lidar_s.ndim == 2:
            lidar_s = cv2.cvtColor(lidar_s, cv2.COLOR_GRAY2BGR)
        lidar_color_s = cv2.resize(lidar_color, (tw, th), interpolation=cv2.INTER_NEAREST)
        overlay_s = cv2.resize(overlay, (tw, th), interpolation=cv2.INTER_LINEAR)

        # 2x2 그리드
        top = np.hstack([rgb_s, lidar_s])
        bottom = np.hstack([lidar_color_s, overlay_s])
        canvas = np.vstack([top, bottom])

        # 라벨
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, x, y in [
            ("rgb", 10, 30),
            ("lidar", tw + 10, 30),
            ("lidar_color", 10, th + 30),
            ("overlay", tw + 10, th + 30),
        ]:
            cv2.putText(canvas, label, (x, y), font, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"{pair['stem']} [{idx+1}/{n}]", (10, canvas.shape[0] - 15), font, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, canvas)
        cv2.setTrackbarPos("index", win, idx)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 81 or key == 2:  # left
            current_index[0] = max(0, idx - 1)
        if key == 83 or key == 3:  # right
            current_index[0] = min(n - 1, idx + 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
