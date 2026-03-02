import numpy as np
import cv2
import os
import glob

ROOT = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/thermal_camera'
RGB_ROOT = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/zed'

def preprocess_thermal(thermal_raw, padding_thresh=10):
    """
    thermal_raw: uint8, shape (H, W), 값 범위 ~[91, 99] (9 levels), padding=0

    핵심: 이산 레벨 사이를 보간하되, 온도 경계(해안선 등)는 보존.
      ① Float-space Bilateral: 같은 온도 영역 내에서만 보간 (경계 보존)
      ② Percentile Stretch: 연속 float값을 0~255로 확장
      ③ Mild CLAHE: local contrast 보강
    """
    roi_mask = thermal_raw > padding_thresh

    if roi_mask.sum() == 0:
        return np.zeros_like(thermal_raw)

    # ① Float-space Bilateral Filter (Gaussian 대신)
    #    sigmaColor=1.2: 1.2 레벨 이내 차이만 스무딩 → 같은 영역 내 보간
    #                     2+ 레벨 차이(해안선, 수면-하늘 경계)는 보존
    #    sigmaSpace=5.0: 공간적 5px 반경
    thermal_f = thermal_raw.astype(np.float32)
    blurred_f = cv2.bilateralFilter(thermal_f, d=11, sigmaColor=1.2, sigmaSpace=5.0)
    blurred_f[~roi_mask] = 0

    # ② Percentile Stretch (연속 float → 0~255)
    roi_vals = blurred_f[roi_mask]
    p2, p98 = np.percentile(roi_vals, [2, 98])
    if p98 - p2 < 0.5:
        p2, p98 = float(roi_vals.min()), float(roi_vals.max())
    if p98 - p2 < 0.5:
        return np.zeros_like(thermal_raw)

    stretched = np.clip((blurred_f - p2) / (p98 - p2), 0, 1)
    stretched[~roi_mask] = 0
    stretched_u8 = (stretched * 255).astype(np.uint8)

    # ③ Mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    result = clahe.apply(stretched_u8)
    result[~roi_mask] = 0

    return result


def find_corresponding_rgb(thermal_path):
    """
    thermal 이미지 경로와 유사한 이름을 가진 RGB 파일 찾기 (간단 매칭)  
    예시: thermal/000014.png -> zed/000014.png
    """
    basename = os.path.basename(thermal_path)
    rgb_path = os.path.join(RGB_ROOT, basename)
    if os.path.exists(rgb_path):
        return rgb_path
    # 혹시 파일명이 다르면 glob 매칭
    candidates = sorted(glob.glob(os.path.join(RGB_ROOT, '*.png')))
    if not candidates:
        return None
    # 이름이 가장 비슷한 파일 반환 (번호 기준)
    thermal_num = ''.join(filter(str.isdigit, basename))
    for c in candidates:
        if thermal_num in c:
            return c
    return candidates[0]

if __name__ == "__main__":
    TEST_LIST = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/test.txt'

    if os.path.exists(TEST_LIST):
        with open(TEST_LIST) as f:
            names = [l.strip() for l in f if l.strip()]
        image_files = [os.path.join(ROOT, n + '.png') for n in names]
        image_files = [p for p in image_files if os.path.exists(p)]
        print(f"Test set: {len(image_files)} images loaded from {TEST_LIST}")
    else:
        image_files = sorted(glob.glob(os.path.join(ROOT, '*.png')))

    if not image_files:
        print("No PNG images found.")
        exit(1)

    idx = 0
    num_images = len(image_files)
    win_name = "Thermal+RGB Viewer (a/d or arrows: Prev/Next, ESC: Exit)"

    while True:
        img_path = image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        rgb_img = None
        rgb_path = find_corresponding_rgb(img_path)
        if rgb_path and os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is not None:
            roi = img > 10
            h_t, w_t = img.shape

            # --- A) Linear Stretch Only: 91-99 → 0-255, 필터 없음 ---
            stretch_only = np.zeros_like(img)
            if roi.any():
                rv = img[roi].astype(np.float32)
                lo, hi = float(rv.min()), float(rv.max())
                if hi > lo:
                    s = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
                    stretch_only = (s * 255).astype(np.uint8)
                stretch_only[~roi] = 0

            # --- B) Mid-range Stretch: 91-99 → 100-200 ---
            stretch_mid = np.zeros_like(img)
            if roi.any():
                rv = img[roi].astype(np.float32)
                lo, hi = float(rv.min()), float(rv.max())
                if hi > lo:
                    s = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
                    stretch_mid = (s * 100 + 100).astype(np.uint8)
                stretch_mid[~roi] = 0

            # --- C) Bilateral + Stretch + CLAHE (현재 방식) ---
            thermal_out = preprocess_thermal(img)

            # --- RGB ---
            if rgb_img is not None:
                h_r, w_r = rgb_img.shape[:2]
                if (h_r != h_t) or (w_r != w_t):
                    rgb_resized = cv2.resize(rgb_img, (w_t, h_t), interpolation=cv2.INTER_AREA)
                else:
                    rgb_resized = rgb_img
            else:
                rgb_resized = np.zeros((h_t, w_t, 3), dtype=np.uint8)

            # 3ch 변환
            raw_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            so_3ch = cv2.cvtColor(stretch_only, cv2.COLOR_GRAY2BGR)
            so_3ch[~roi] = 0
            sm_3ch = cv2.cvtColor(stretch_mid, cv2.COLOR_GRAY2BGR)
            sm_3ch[~roi] = 0
            bl_3ch = cv2.cvtColor(thermal_out, cv2.COLOR_GRAY2BGR)
            bl_3ch[~roi] = 0

            # 라벨
            cv2.putText(raw_3ch, 'Raw', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(so_3ch, 'Stretch 0-255', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(sm_3ch, 'Stretch 100-200', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(bl_3ch, 'Bilateral+CLAHE', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(rgb_resized, 'RGB', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            # 상단행: Raw | Stretch0-255 | Stretch100-200 | Bilateral
            # 하단행: RGB | Overlay(stretch) | Overlay(mid) | Overlay(bilateral)
            top_row = np.concatenate([raw_3ch, so_3ch, sm_3ch, bl_3ch], axis=1)

            def make_overlay(gray_3ch, rgb, mask):
                ov = rgb.copy()
                cm = cv2.applyColorMap(cv2.cvtColor(gray_3ch, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
                cm[~mask] = 0
                ov[mask] = (0.5 * cm[mask].astype(np.float32) + 0.5 * rgb[mask].astype(np.float32)).astype(np.uint8)
                return ov

            ov_so = make_overlay(so_3ch, rgb_resized, roi)
            ov_sm = make_overlay(sm_3ch, rgb_resized, roi)
            ov_bl = make_overlay(bl_3ch, rgb_resized, roi)
            cv2.putText(rgb_resized, 'RGB', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(ov_so, 'Ovl 0-255', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(ov_sm, 'Ovl 100-200', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(ov_bl, 'Ovl Bilateral', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            bot_row = np.concatenate([rgb_resized, ov_so, ov_sm, ov_bl], axis=1)
            concat_img = np.concatenate([top_row, bot_row], axis=0)

            # 축소
            scale = min(1.0, 1920.0 / concat_img.shape[1], 1000.0 / concat_img.shape[0])
            if scale < 1.0:
                concat_img = cv2.resize(concat_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            rv = img[roi] if roi.any() else img.ravel()
            label = f"[{idx+1}/{num_images}] {os.path.basename(img_path)} Raw({rv.min()}-{rv.max()})"
            cv2.putText(concat_img, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(win_name, concat_img)
            key = cv2.waitKeyEx(0)

            if key == 27:
                break
            elif key in (81, ord('a'), 65361):
                idx = (idx - 1) % num_images
            elif key in (83, ord('d'), 65363):
                idx = (idx + 1) % num_images
        else:
            print(f"Failed to load image: {img_path}")
            idx = (idx + 1) % num_images

    cv2.destroyAllWindows()