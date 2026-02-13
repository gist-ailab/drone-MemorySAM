import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import json

def calculate_roi_stats(image_paths):
    """
    이미지 리스트를 받아 검정색 패딩(0)을 제외한 영역의 Mean, Std를 계산합니다.
    Welford 온라인 알고리즘 사용 → 이미지 한 장씩만 메모리에 올리므로 메모리 사용량 O(1).
    """
    # Welford's online algorithm: n, mean, M2 (sum of squared differences from current mean)
    n = 0
    mean = 0.0
    M2 = 0.0
    min_val, max_val = 255.0, 0.0

    print(f"총 {len(image_paths)}장의 이미지에서 통계 추출 중 (한 장씩 처리, 메모리 절약)...")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        roi = img[img > 0]
        if roi.size == 0:
            continue
        # 배치로 Welford 업데이트 (이미지당 한 번만, 메모리=한 장 분량)
        batch = roi.astype(np.float64).ravel()
        n_batch = len(batch)
        n_new = n + n_batch
        delta = batch - mean
        mean = mean + np.sum(delta) / n_new
        M2 = M2 + np.sum(delta * (batch - mean))
        n = n_new
        min_val = min(min_val, float(batch.min()))
        max_val = max(max_val, float(batch.max()))

    if n == 0:
        print("유효한 데이터(0이 아닌 픽셀)가 하나도 없습니다.")
        return None, None, None, None

    var = M2 / n
    std_val = np.sqrt(var)
    mean_val = mean

    print("\n=== [Result] ROI Statistics (Padding 0 Excluded) ===")
    print(f"Calculated Mean: {mean_val:.4f}")
    print(f"Calculated Std:  {std_val:.4f}")
    print(f"Min Value: {min_val:.1f}")
    print(f"Max Value: {max_val:.1f}")
    print(f"Total pixels (ROI): {n}")

    return mean_val, std_val, min_val, max_val




image_folder = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_processed" 
png_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))

if png_files:
    result = calculate_roi_stats(png_files)
    if result[0] is not None:
        m, s, min_v, max_v = result
        stats = {"mean": float(m), "std": float(s), "min": float(min_v), "max": float(max_v)}
    else:
        m, s = None, None
        stats = {"mean": None, "std": None}
    print("\n[Saved to dictionary]")
    print(stats)
else:
    print("이미지 파일이 없습니다.")
    m, s = None, None
    stats = {}

# Save the stats to a json file
if stats:
    with open("thermal_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

single_img = png_files[0] if png_files else None
if single_img and m is not None and s is not None and s > 0:
    img = cv2.imread(single_img, cv2.IMREAD_GRAYSCALE)
    img_norm = (img.astype(np.float32) - m) / s
    disp = np.clip(((img_norm + 2) / 4) * 255, 0, 255).astype(np.uint8)
    cv2.imshow("Thermal Normalized (m,s)", disp)
elif not png_files or m is None or s is None or s <= 0:
    print("[경고] 유효한 mean/std가 없어 정규화 이미지를 시각화하지 않습니다.")
