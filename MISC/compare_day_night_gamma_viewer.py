#!/usr/bin/env python3
"""
OpenCV 뷰어: zed / zed_compare / zed_compare 순으로 vconcat.
0번째 행(zed)만 하단 gamma 슬라이더로 조절, 나머지 행은 원본 그대로.
이미지 전환: 하단 인덱스 바 클릭 또는 방향키 ← →.
"""
import cv2
import numpy as np
from pathlib import Path

BASE = Path("/home/jemo/drone-demo/MULTIAQUA_night/MULTIAQUA_night/data")
ZED_SUB = "zed"  # 고정 (0번째 행, gamma 적용)
# BASE 아래 비교할 서브폴더 이름 리스트 → 1번째 행, 2번째 행, ... (예: zed_compare 두 번이면 같은 폴더 두 행)
COMPARE_SUBFOLDERS = [
    "zed_compare",
    "zed_compare",
]

MAX_ROW_HEIGHT = 360
STRIP_H = 56
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
WINDOW_NAME = "zed (gamma) | compare...  [←/→ or click bar]  Q/ESC: quit"
GAMMA_MIN, GAMMA_MAX = 10, 300  # 0.1 ~ 3.0 (x10)


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """BGR 이미지에 gamma 보정. gamma 높을수록 밝아짐 (output = input^(1/gamma))."""
    if bgr is None or bgr.size == 0:
        return bgr
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, table)


def resize_to_fit(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """비율 유지하며 target_w x target_h 안에 맞춤, 여백은 검정."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out[:] = 0
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def build_layout(zed_bgr, compare_imgs: list, labels: list, ref_w: int, ref_h: int, gamma: float):
    """zed(0번째, gamma 적용) + compare 이미지들을 vconcat. ref_w x ref_h 기준으로 리사이즈."""
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = [(0, 255, 0), (0, 255, 255), (255, 200, 0)]
    # 0번째: zed (gamma 적용)
    if zed_bgr is not None and zed_bgr.size > 0:
        zed_g = apply_gamma(zed_bgr, gamma)
        r0 = resize_to_fit(zed_g, ref_w, ref_h)
    else:
        r0 = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    cv2.putText(r0, labels[0] + f" (gamma={gamma:.2f})", (8, ref_h - 8), font, 0.55, colors[0], 1)
    rows.append(r0)
    # 1번째~: compare (그대로)
    for i, bgr in enumerate(compare_imgs):
        if bgr is not None and bgr.size > 0:
            r = resize_to_fit(bgr, ref_w, ref_h)
        else:
            r = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
        name = labels[i + 1] if i + 1 < len(labels) else f"compare_{i}"
        cv2.putText(r, name, (8, ref_h - 8), font, 0.55, colors[(i + 1) % len(colors)], 1)
        rows.append(r)
    return np.vstack(rows)


def _stems_in_dir(path: Path) -> set:
    s = set()
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in EXTS:
            s.add(f.stem)
    return s


def _find_img(folder: Path, stem: str) -> Path | None:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    zed_dir = BASE / ZED_SUB
    if not zed_dir.is_dir():
        print("zed 폴더가 없습니다:", zed_dir)
        return
    compare_dirs = [(name, BASE / name) for name in COMPARE_SUBFOLDERS]
    for name, path in compare_dirs:
        if not path.is_dir():
            print(f"비교 폴더가 없습니다: {name} -> {path}")
            return

    common = _stems_in_dir(zed_dir)
    for _name, path in compare_dirs:
        common &= _stems_in_dir(path)
    stems = sorted(common)
    if not stems:
        print("zed와 모든 비교 폴더에 공통으로 있는 이미지가 없습니다.")
        return

    # ref 크기: 첫 번째 zed 이미지 기준 (최대 높이 제한)
    first_zed = _find_img(zed_dir, stems[0])
    ref_h, ref_w = MAX_ROW_HEIGHT, 640
    if first_zed:
        im0 = cv2.imread(str(first_zed))
        if im0 is not None and im0.size > 0:
            h, w = im0.shape[:2]
            scale = min(1.0, MAX_ROW_HEIGHT / h)
            ref_h = int(h * scale)
            ref_w = int(w * scale)
    n_rows = 1 + len(compare_dirs)
    canvas_h = ref_h * n_rows
    canvas_w = ref_w

    try:
        cv2.startWindowThread()
    except Exception:
        pass

    n_total = len(stems)
    labels = [ZED_SUB] + [name for name, _ in compare_dirs]
    idx = [0]
    gamma_x10 = [100]  # 1.0 (gamma=1.0)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y < canvas_h:
            return
        rel_x = max(0, min(1, x / canvas_w))
        new_idx = int(rel_x * n_total)
        if new_idx >= n_total:
            new_idx = n_total - 1
        idx[0] = new_idx

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    dummy = np.zeros((canvas_h + STRIP_H, canvas_w, 3), dtype=np.uint8)
    dummy[:] = (40, 40, 40)
    cv2.putText(dummy, "Loading...", (canvas_w // 2 - 50, (canvas_h + STRIP_H) // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WINDOW_NAME, dummy)
    cv2.waitKey(200)
    try:
        cv2.resizeWindow(WINDOW_NAME, canvas_w, canvas_h + STRIP_H)
    except cv2.error:
        pass
    try:
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    except cv2.error:
        print("Mouse callback skipped (use <- -> keys to change image)")
    try:
        cv2.createTrackbar("index", WINDOW_NAME, 0, max(0, n_total - 1), lambda _: None)
    except Exception:
        pass
    try:
        cv2.createTrackbar("gamma x10", WINDOW_NAME, GAMMA_MIN, GAMMA_MAX, lambda _: None)
        cv2.setTrackbarPos("gamma x10", WINDOW_NAME, 100)  # 1.0
    except Exception:
        pass

    while True:
        idx[0] = min(max(0, idx[0]), n_total - 1)
        try:
            track_pos = cv2.getTrackbarPos("index", WINDOW_NAME)
            if 0 <= track_pos < n_total:
                idx[0] = track_pos
        except Exception:
            pass
        try:
            gamma_x10[0] = cv2.getTrackbarPos("gamma x10", WINDOW_NAME)
            gamma_x10[0] = max(GAMMA_MIN, min(GAMMA_MAX, gamma_x10[0]))
        except Exception:
            pass
        gamma_val = gamma_x10[0] / 100.0

        cur_idx = idx[0]
        stem = stems[cur_idx]

        zed_path = _find_img(zed_dir, stem)
        zed_bgr = cv2.imread(str(zed_path)) if zed_path else None
        compare_imgs = []
        for _name, folder in compare_dirs:
            p = _find_img(folder, stem)
            compare_imgs.append(cv2.imread(str(p)) if p else None)

        if zed_bgr is None and all(im is None for im in compare_imgs):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            cv2.putText(canvas, "Load failed: " + stem, (20, canvas_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            canvas = build_layout(zed_bgr, compare_imgs, labels, ref_w, ref_h, gamma_val)

        strip = np.zeros((STRIP_H, canvas_w, 3), dtype=np.uint8)
        strip[:] = (50, 50, 50)
        bar_w = canvas_w - 20
        bar_x0 = 10
        cv2.rectangle(strip, (bar_x0, 8), (bar_x0 + bar_w, 28), (80, 80, 80), 1)
        if n_total > 1:
            fill_w = int(bar_w * (cur_idx + 1) / n_total)
            cv2.rectangle(strip, (bar_x0, 8), (bar_x0 + fill_w, 28), (0, 200, 255), -1)
        cv2.putText(strip, f" [{cur_idx+1}/{n_total}] {stem}  gamma={gamma_val:.2f}  <- bar / <- ->  Q: quit", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        canvas = np.vstack([canvas, strip])

        cv2.imshow(WINDOW_NAME, canvas)
        try:
            cv2.setTrackbarPos("index", WINDOW_NAME, cur_idx)
        except Exception:
            pass

        key = cv2.waitKey(30)
        if key == -1:
            continue
        key_plain = key & 0xFF
        if key_plain in (ord("q"), ord("Q"), 27):
            break
        if key_plain in (81, 2, ord("a")) or key == 65361:
            idx[0] = max(0, idx[0] - 1)
            try:
                cv2.setTrackbarPos("index", WINDOW_NAME, idx[0])
            except Exception:
                pass
        elif key_plain in (83, 3, ord("d")) or key == 65363:
            idx[0] = min(n_total - 1, idx[0] + 1)
            try:
                cv2.setTrackbarPos("index", WINDOW_NAME, idx[0])
            except Exception:
                pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
