#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# --- 경로 설정 ---
DATA_ROOT = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data")
ZED_DIR = DATA_ROOT / "zed"
LIDAR_DIR = DATA_ROOT / "lidar_processed"
THERMAL_PROCESSED_DIR = DATA_ROOT / "thermal_processed_fieldscale3"

EXTS = (".png", ".jpg", ".jpeg", ".bmp")
MAX_H = 480
PANEL_W = 640
STRIP_H = 60
WIN = "2x2 Viewer [Click Bar to Jump | Q: Quit]"

def get_stems(folder: Path) -> set:
    if not folder.exists(): return set()
    return {f.stem for f in folder.iterdir() if f.is_file() and f.suffix.lower() in EXTS}

def find_path(folder: Path, stem: str, suffix: str = "") -> Path | None:
    """suffix: lidar는 stem 뒤에 '_lidar' 붙음 (예: stem_lidar.png)."""
    for ext in EXTS:
        p = folder / f"{stem}{suffix}{ext}"
        if p.exists(): return p
    return None

def _thermal_original_to_uint8(img: np.ndarray) -> np.ndarray | None:
    if img is None or img.size == 0: return None
    if img.dtype == np.uint8: return img.copy()
    max_val = 65535.0 if img.dtype == np.uint16 else (float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 65535.0)
    return np.clip(img.astype(np.float64) / max_val * 255, 0, 255).astype(np.uint8)

def to_bgr_display(img, ref_h: int, panel_w: int):
    out = np.zeros((ref_h, panel_w, 3), dtype=np.uint8)
    # 패널 자체의 배경도 완전한 검정(0,0,0)으로 설정
    out[:] = (0, 0, 0)
    if img is None or img.size == 0:
        cv2.putText(out, "no image", (panel_w // 2 - 40, ref_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return out
    h, w = img.shape[:2]
    scale = min(ref_h / h, panel_w / w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    y0, x0 = (ref_h - nh) // 2, (panel_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out

def overlay_jet(zed_bgr: np.ndarray | None, proc_u8: np.ndarray | None, alpha_zed: float = 0.2, alpha_thermal: float = 0.8):
    """패딩 영역(0)을 물리적으로 완전히 도려내어 파란색을 제거한 오버레이"""
    if zed_bgr is None or proc_u8 is None: return None
    
    hz, wz = zed_bgr.shape[:2]
    th_resized = cv2.resize(proc_u8, (wz, hz), interpolation=cv2.INTER_NEAREST)
    
    # 1. 값이 0보다 큰 '진짜 데이터' 영역만 추출 (마스크)
    mask = th_resized > 0
    
    # 2. JET 컬러맵 적용 (이 시점엔 0이 파란색으로 바뀜)
    jet_bgr = cv2.applyColorMap(th_resized, cv2.COLORMAP_JET)
    
    # 3. 마스크가 없는(패딩) 부분의 JET 컬러를 강제로 검정(0)으로 밀어버림
    jet_bgr[~mask] = 0
    
    # 4. 블렌딩: ZED와 마스킹된 JET을 합성
    blended = cv2.addWeighted(zed_bgr, alpha_zed, jet_bgr, alpha_thermal, 0)
    
    # 5. [중요] 최종 출력물에서 마스크 영역만 합성본을 넣고, 나머지는 0(검정) 유지
    res = np.zeros_like(zed_bgr)
    res[mask] = blended[mask]
    
    return res

def main():
    dirs = [ZED_DIR, LIDAR_DIR, THERMAL_PROCESSED_DIR]
    if not all(d.is_dir() for d in dirs):
        print("경로를 확인하세요.")
        return

    # lidar 파일명: stem_lidar.png → 공통 stem은 lidar에서 _lidar 제거한 것과 매칭
    lidar_stems = get_stems(LIDAR_DIR)
    lidar_base = {s.removesuffix("_lidar") for s in lidar_stems if s.endswith("_lidar")}
    common = get_stems(ZED_DIR) & get_stems(THERMAL_PROCESSED_DIR) & lidar_base
    stems = sorted(list(common))
    n = len(stems)
    if n == 0: return

    state = {'idx': 0}
    canvas_w = PANEL_W * 2
    canvas_content_h = MAX_H * 2

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y >= canvas_content_h:
                bar_x_start, bar_x_end = 10, canvas_w - 10
                if bar_x_start <= x <= bar_x_end:
                    rel = (x - bar_x_start) / (bar_x_end - bar_x_start)
                    state['idx'] = max(0, min(int(rel * n), n - 1))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cur_idx = state['idx']
        stem = stems[cur_idx]

        # 이미지 로드
        zed = cv2.imread(str(find_path(ZED_DIR, stem)))
        th_raw = cv2.imread(str(find_path(LIDAR_DIR, stem, "_lidar")), -1)
        th_proc = cv2.imread(str(find_path(THERMAL_PROCESSED_DIR, stem)), -1)

        proc_u8 = _thermal_original_to_uint8(th_proc)
        overlay = overlay_jet(zed, proc_u8)

        # 패널별 BGR 변환
        p_zed = to_bgr_display(zed, MAX_H, PANEL_W)
        p_th = to_bgr_display(_thermal_original_to_uint8(th_raw), MAX_H, PANEL_W)
        p_fs = to_bgr_display(proc_u8, MAX_H, PANEL_W)
        p_ov = to_bgr_display(overlay, MAX_H, PANEL_W)

        # 텍스트
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(p_zed, "RGB", (20, 40), f, 0.8, (0, 255, 0), 2)
        cv2.putText(p_th, "LiDAR", (20, 40), f, 0.8, (0, 255, 255), 2)
        cv2.putText(p_fs, "Fieldscale (Gray)", (20, 40), f, 0.8, (255, 200, 0), 2)
        cv2.putText(p_ov, "Overlay (JET Masked)", (20, 40), f, 0.8, (255, 0, 255), 2)

        # 2x2 합성
        canvas = np.vstack([np.hstack([p_zed, p_th]), np.hstack([p_fs, p_ov])])

        # 바 생성
        bar = np.zeros((STRIP_H, canvas_w, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        bx, by, bw, bh = 10, 15, canvas_w - 20, 25
        cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (70, 70, 70), 1)
        if n > 0:
            fill_w = int(bw * (cur_idx + 1) / n)
            cv2.rectangle(bar, (bx, by), (bx + fill_w, by + bh), (0, 150, 255), -1)
        
        cv2.putText(bar, f"[{cur_idx+1}/{n}] {stem} | Click bar to jump", (bx, STRIP_H - 8), f, 0.5, (180, 180, 180), 1)

        final_view = np.vstack([canvas, bar])
        cv2.imshow(WIN, final_view)

        key = cv2.waitKey(20)
        if key == -1: continue
        k = key & 0xFF
        if k in (ord('q'), 27): break
        elif k in (ord('a'), 81, 65361): state['idx'] = max(0, state['idx'] - 1)
        elif k in (ord('d'), 83, 65363): state['idx'] = min(n - 1, state['idx'] + 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()