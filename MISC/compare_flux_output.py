#!/usr/bin/env python3
"""
여러 비교 폴더를 리스트로 지정해 위→아래 vconcat 비교 뷰어.
- 이미지 창(WIN_MAIN) / 슬라이더 창(WIN_CTRL) 분리 — 슬라이더는 항상 별도 창에 표시.
- 첫 번째 폴더(zed) 이미지에만 brightness/contrast/gamma 적용.
- RGB가 나오다 안 나오는 경우: OpenCV Qt 백엔드에서 imshow 직후 화면 갱신이 한 프레임 밀릴 수 있음.
  원인: waitKey 대기 시간이 짧거나, 창이 가려졌다가 다시 보일 때 재그리기 지연.
  대응: waitKey(30) 유지, 컨트롤 창 분리로 이미지 창만 갱신, 필요 시 환경변수 OPENCV_IMSHOW_DELAY 없음.
"""
import cv2
import numpy as np
from pathlib import Path

BASE = Path("/home/jemo/drone-demo/MULTIAQUA_night/MULTIAQUA_night/data")
COMPARE_DIRS = [
    ("zed", BASE / "zed"),
    ("zed_daygamma2.0", BASE / "zed_daygamma2.0"),
    ("zed_day3gamma1.5clahe", BASE / "zed_day3gamma1.5clahe"),
    ("zed_day_quick", BASE / "zed_day_quick"),
]
LABEL_H = 26
STRIP_H = 52
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
LABEL_COLORS = [(0, 255, 200), (200, 200, 0), (200, 100, 255), (0, 200, 255), (255, 150, 0)]
WIN_MAIN = "Compare (images)"
WIN_CTRL = "Compare (sliders)"


def get_common_stems(dir_list: list) -> list:
    """dir_list 내 모든 폴더에 존재하는 이미지 stem 목록 (확장자 제외)."""
    if not dir_list:
        return []
    exts = EXTS
    common = None
    for _name, folder in dir_list:
        if not folder or not Path(folder).exists():
            continue
        stems = set()
        for f in Path(folder).iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                stems.add(f.stem)
        if common is None:
            common = stems
        else:
            common = common & stems
    return sorted(common) if common else []


def find_path(folder: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def apply_brightness_contrast_gamma(
    bgr: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    """brightness/contrast/gamma를 zed 이미지에만 적용. brightness,contrast 1.0=무변경, gamma 1.0=무변경."""
    if bgr is None or bgr.size == 0:
        return bgr
    out = bgr.astype(np.float32)
    # contrast: (x - 127.5) * contrast + 127.5
    out = (out - 127.5) * contrast + 127.5
    # brightness: * brightness
    out = out * brightness
    out = np.clip(out, 0, 255).astype(np.uint8)
    # gamma
    if abs(gamma - 1.0) > 1e-3:
        inv_g = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, table)
    return out


def resize_to_size(img: np.ndarray, ref_w: int, ref_h: int) -> np.ndarray:
    """이미지를 원본 기준 크기(ref_w x ref_h)로 리사이즈."""
    if img is None or img.size == 0:
        return np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
    return cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)


def add_label(img: np.ndarray, label: str, color: tuple) -> np.ndarray:
    bar = np.zeros((LABEL_H, img.shape[1], 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.putText(bar, label, (8, LABEL_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return np.vstack([bar, img])


def build_canvas(
    images: list,
    dir_names: list,
    ref_w: int,
    ref_h: int,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    """images[0]에만 brightness/contrast/gamma 적용 후 전부 위→아래 vconcat. ref_w x ref_h."""
    rows = []
    for i, (bgr, name) in enumerate(zip(images, dir_names)):
        if i == 0 and bgr is not None and bgr.size > 0:
            bgr = apply_brightness_contrast_gamma(bgr, brightness, contrast, gamma)
        cell = resize_to_size(bgr, ref_w, ref_h)
        if i == 0 and (images[0] is None or images[0].size == 0):
            cv2.putText(cell, "zed (no image)", (ref_w // 2 - 60, ref_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        label = name + (f" (b={brightness:.2f} g={gamma:.2f})" if i == 0 else "")
        rows.append(add_label(cell, label, color))
    return np.vstack(rows)


def main():
    dir_list = [(name, Path(path)) for name, path in COMPARE_DIRS]
    for name, folder in dir_list:
        if not folder.exists():
            print(f"폴더가 없습니다: {name} -> {folder}")
            return

    stems = get_common_stems(dir_list)
    if not stems:
        print("모든 비교 폴더에 공통으로 존재하는 이미지가 없습니다.")
        return

    n = len(stems)
    num_dirs = len(dir_list)
    dir_names = [d[0] for d in dir_list]

    # ref 크기: 첫 stem으로 어떤 폴더에서든 처음 로드되는 이미지 기준 (zed 실패 시 다른 폴더 사용)
    ref_h, ref_w = 480, 640
    for _name, folder in dir_list:
        p = find_path(folder, stems[0])
        if p:
            img = cv2.imread(str(p))
            if img is not None and img.size > 0:
                ref_h, ref_w = img.shape[0], img.shape[1]
                break

    # 캔버스 높이 = 상단바 + (라벨+이미지)*행수 + 하단 스트립
    row_cell_h = LABEL_H + ref_h
    canvas_content_h = LABEL_H + num_dirs * row_cell_h
    canvas_h = canvas_content_h + STRIP_H
    canvas_w = ref_w

    state = {"idx": 0, "brightness": 100, "contrast": 100, "gamma": 100}

    def on_index(v):
        state["idx"] = min(max(0, v), n - 1)

    def on_brightness(v):
        state["brightness"] = v

    def on_contrast(v):
        state["contrast"] = v

    def on_gamma(v):
        state["gamma"] = v

    try:
        cv2.startWindowThread()
    except Exception:
        pass

    # 이미지 창: RGB만 표시 (트랙바 없음)
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    dummy = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    dummy[:] = (40, 40, 40)
    cv2.putText(dummy, "Loading...", (canvas_w // 2 - 50, canvas_h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WIN_MAIN, dummy)
    cv2.waitKey(100)
    try:
        cv2.resizeWindow(WIN_MAIN, canvas_w, canvas_h)
    except cv2.error:
        pass

    # 슬라이더 전용 창 (별도 창이라 Qt에서 트랙바가 잘 보임)
    ctrl_w, ctrl_h = 420, 180
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    ctrl_panel = np.zeros((ctrl_h, ctrl_w, 3), dtype=np.uint8)
    ctrl_panel[:] = (50, 50, 50)
    cv2.putText(ctrl_panel, "index / brightness / contrast / gamma", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow(WIN_CTRL, ctrl_panel)
    cv2.waitKey(100)
    try:
        cv2.resizeWindow(WIN_CTRL, ctrl_w, ctrl_h)
    except cv2.error:
        pass
    try:
        cv2.createTrackbar("index", WIN_CTRL, 0, max(0, n - 1), on_index)
        cv2.createTrackbar("brightness", WIN_CTRL, 100, 200, on_brightness)
        cv2.createTrackbar("contrast", WIN_CTRL, 100, 200, on_contrast)
        cv2.createTrackbar("gamma", WIN_CTRL, 100, 300, on_gamma)
    except (cv2.error, Exception):
        pass

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or y < canvas_content_h:
            return
        rel = max(0, min(1, x / canvas_w))
        state["idx"] = min(int(rel * n), n - 1)

    try:
        cv2.setMouseCallback(WIN_MAIN, on_mouse)
    except cv2.error:
        pass

    print(f"비교 폴더 {num_dirs}개: {dir_names}. 이미지 창 + 슬라이더 창 분리. ←/→ g/G b/B c/C. Q/ESC 종료.")

    while True:
        idx = state["idx"]
        stem = stems[idx]
        images = []
        for _name, folder in dir_list:
            p = find_path(folder, stem)
            img = cv2.imread(str(p)) if p else None
            images.append(img)
        for i in range(len(images)):
            if images[i] is None:
                images[i] = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)

        b = state["brightness"] / 100.0
        c = state["contrast"] / 100.0
        g = 0.1 + (state["gamma"] / 300.0) * 2.9

        canvas = build_canvas(images, dir_names, ref_w, ref_h, b, c, g)
        info_bar = np.zeros((LABEL_H, canvas_w, 3), dtype=np.uint8)
        info_bar[:] = (30, 30, 30)
        cv2.putText(info_bar, f"  {stem}  [{idx+1}/{n}]  {ref_w}x{ref_h}  b={b:.2f} c={c:.2f} g={g:.2f}", (10, LABEL_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        canvas = np.vstack([info_bar, canvas])

        # 하단 스트립: 인덱스 바 + g/G, b/B, c/C 안내 (트랙바 미표시 시 대비)
        strip = np.zeros((STRIP_H, canvas_w, 3), dtype=np.uint8)
        strip[:] = (50, 50, 50)
        bar_w = canvas_w - 20
        bar_x0 = 10
        cv2.rectangle(strip, (bar_x0, 6), (bar_x0 + bar_w, 24), (80, 80, 80), 1)
        if n > 1:
            fill_w = int(bar_w * (idx + 1) / n)
            cv2.rectangle(strip, (bar_x0, 6), (bar_x0 + fill_w, 24), (0, 200, 255), -1)
        cv2.putText(strip, f" [{(idx+1)}/{n}]  g/G=gamma b/B=bright c/C=contrast  <- ->  Q:quit", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        canvas = np.vstack([canvas, strip])

        cv2.imshow(WIN_MAIN, canvas)
        cv2.waitKey(1)  # Qt 백엔드에서 이미지 창 갱신이 밀리지 않도록 한 프레임 처리
        try:
            cv2.setTrackbarPos("index", WIN_CTRL, idx)
            cv2.setTrackbarPos("brightness", WIN_CTRL, state["brightness"])
            cv2.setTrackbarPos("contrast", WIN_CTRL, state["contrast"])
            cv2.setTrackbarPos("gamma", WIN_CTRL, state["gamma"])
        except Exception:
            pass

        key = cv2.waitKey(30)
        if key == -1:
            continue
        key_plain = key & 0xFF
        if key_plain in (ord("q"), ord("Q"), 27):
            break
        if key_plain in (81, 2, ord("a")) or key == 65361:
            state["idx"] = max(0, state["idx"] - 1)
            try:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
            except Exception:
                pass
        elif key_plain in (83, 3, ord("d")) or key == 65363:
            state["idx"] = min(n - 1, state["idx"] + 1)
            try:
                cv2.setTrackbarPos("index", WIN_CTRL, state["idx"])
            except Exception:
                pass
        elif key_plain == ord("g"):
            state["gamma"] = max(10, state["gamma"] - 10)
        elif key_plain == ord("G"):
            state["gamma"] = min(300, state["gamma"] + 10)
        elif key_plain == ord("b"):
            state["brightness"] = max(10, state["brightness"] - 10)
        elif key_plain == ord("B"):
            state["brightness"] = min(200, state["brightness"] + 10)
        elif key_plain == ord("c"):
            state["contrast"] = max(10, state["contrast"] - 10)
        elif key_plain == ord("C"):
            state["contrast"] = min(200, state["contrast"] + 10)

        try:
            tb_idx = cv2.getTrackbarPos("index", WIN_CTRL)
            if 0 <= tb_idx < n:
                state["idx"] = tb_idx
            state["brightness"] = cv2.getTrackbarPos("brightness", WIN_CTRL)
            state["contrast"] = cv2.getTrackbarPos("contrast", WIN_CTRL)
            state["gamma"] = cv2.getTrackbarPos("gamma", WIN_CTRL)
        except Exception:
            pass

    cv2.destroyWindow(WIN_MAIN)
    cv2.destroyWindow(WIN_CTRL)


if __name__ == "__main__":
    main()
