"""
MUSES Radar PNG → RGB 평면 투영 (공식 SDK 동일 파이프라인).
참조: https://github.com/timbroed/MUSES processing/radar_processing.py
- range-azimuth raw PNG 파싱 → 포인트클라우드 → radar2rgb + K로 RGB에 투영.
"""
import math
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .utils_muses import (
    create_image_from_point_cloud,
    filter_and_project_pcd_to_image,
    motion_compensate_pcd,
    enlarge_points_in_image,
)

# SDK 기본 상수 (FMCW 레이더 스펙)
STEP = 400
GROUND_LEVEL = 1.62
MAX_DISTANCE = 330.0768
RESOLUTION = 0.0438


def load_raw_radar_data(radar_path):
    """
    레이더 PNG 로드. SDK: 첫 채널만 사용 (range-azimuth raw).
    Returns: (H, W) uint8, W=400이면 SDK 포맷.
    """
    radar_path = Path(radar_path)
    if not radar_path.exists():
        raise FileNotFoundError(f"Radar not found: {radar_path}")
    img = cv2.imread(str(radar_path)) if cv2 is not None else None
    if img is None:
        return None
    if img.ndim == 2:
        return np.asarray(img, dtype=np.uint8)
    return np.asarray(img[:, :, 0], dtype=np.uint8)


def radar_to_point_cloud(
    front_radar_polar_image,
    intensity_threshold=0,
    image_fov_only=False,
    step=STEP,
    ground_level=GROUND_LEVEL,
    max_distance=MAX_DISTANCE,
    resolution=RESOLUTION,
):
    """
    SDK 동일: range-azimuth 레이더 데이터 → (N, 5) 포인트클라우드 [x, y, z, intensity, timestamp_s].
    front_radar_polar_image: (num_rows, step) uint8. rows 0:4 ts_s, 4:8 ts_ns, 10 valid, 11+ fft.
    """
    if front_radar_polar_image is None or front_radar_polar_image.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    if front_radar_polar_image.shape[1] != step:
        return np.zeros((0, 5), dtype=np.float64)

    min_azimuths = 145 if image_fov_only else 0
    max_azimuths = step - 1

    azimuths = -np.radians(np.linspace(-180 + (360 / step), 180, step).astype(np.float32))
    range_in_bins = int(max_distance // resolution)
    front_radar_polar_image = front_radar_polar_image[: range_in_bins + 11]
    if front_radar_polar_image.shape[0] < 11:
        return np.zeros((0, 5), dtype=np.float64)
    end_distance = np.round(range_in_bins * resolution, decimals=4)
    distance = np.linspace(resolution, end_distance, range_in_bins)

    front_radar_polar_image = front_radar_polar_image[:, min_azimuths : max_azimuths + 1]
    azimuths = azimuths[min_azimuths : max_azimuths + 1]

    # timestamps: (4, num_azimuths) bytes → uint32 per column
    ts_rows_s = front_radar_polar_image[:4]
    ts_rows_ns = front_radar_polar_image[4:8]
    timestamps_s = np.zeros(ts_rows_s.shape[1], dtype=np.uint32)
    timestamps_ns = np.zeros(ts_rows_ns.shape[1], dtype=np.uint32)
    for i in range(ts_rows_s.shape[1]):
        timestamps_s[i] = np.frombuffer(ts_rows_s[:, i].tobytes(), dtype=np.uint32)[0]
        timestamps_ns[i] = np.frombuffer(ts_rows_ns[:, i].tobytes(), dtype=np.uint32)[0]
    timestamps_us = (timestamps_s.astype(np.float64) * 1e6) + (timestamps_ns.astype(np.float64) / 1e9 * 1e6)

    valid_flag = front_radar_polar_image[10].astype(bool)

    fft_data = front_radar_polar_image[11:].astype(np.float64)
    valid_mask = (fft_data >= intensity_threshold) & np.broadcast_to(valid_flag, fft_data.shape)

    x = np.outer(distance, np.cos(azimuths))
    y = np.outer(distance, np.sin(azimuths))

    num_points = np.sum(valid_mask, axis=0)
    total_points = int(np.sum(num_points))
    if total_points == 0:
        return np.zeros((0, 5), dtype=np.float64)

    xyzi = np.zeros((total_points, 5), dtype=np.float64)
    xyzi[:, 0] = x.T[valid_mask.T]
    xyzi[:, 1] = y.T[valid_mask.T]
    xyzi[:, 2] = -ground_level
    xyzi[:, 3] = fft_data.T[valid_mask.T]
    xyzi[:, 4] = np.repeat(timestamps_us * 1e-6, num_points)
    return xyzi


def load_radar_as_pcd(radar_path, intensity_threshold=0, image_fov_only=False):
    """레이더 PNG → (N, 5) 포인트클라우드. SDK load_radar_as_pcd 동일."""
    raw = load_raw_radar_data(radar_path)
    if raw is None:
        return np.zeros((0, 5), dtype=np.float64)
    return radar_to_point_cloud(
        raw,
        intensity_threshold=intensity_threshold,
        image_fov_only=image_fov_only,
    )


def load_points_in_image_radar(
    radar_path,
    calib_data,
    scene_meta_dict=None,
    motion_compensation=False,
    muses_root=None,
    target_shape=(1920, 1080),
    intensity_threshold=0,
):
    """
    SDK 동일: 레이더 로드 → (선택) motion comp → radar2rgb + K로 RGB 평면 투영.
    Returns: (uv_img_cords_filtered (2,N), pcd_filtered)
    """
    radar_path = Path(radar_path)
    if not radar_path.exists():
        raise FileNotFoundError(f"Radar data {radar_path} does not exist")

    pcd_points = load_radar_as_pcd(radar_path, intensity_threshold, image_fov_only=True)
    if len(pcd_points) == 0:
        w, h = target_shape
        return np.zeros((2, 0), dtype=np.float64), np.zeros((0, 5), dtype=np.float64)

    if motion_compensation and scene_meta_dict is not None and muses_root is not None:
        radar2gnss = np.array(calib_data["extrinsics"]["radar2gnss"], dtype=np.float64)
        pcd_points = motion_compensate_pcd(
            muses_root, scene_meta_dict, pcd_points, radar2gnss, ts_channel_num=4
        )

    K_rgb = np.array(calib_data["intrinsics"]["rgb"]["K"], dtype=np.float64)
    radar2rgb = np.array(calib_data["extrinsics"]["radar2rgb"], dtype=np.float64)
    uv_img_cords_filtered, pcd_filtered = filter_and_project_pcd_to_image(
        pcd_points, radar2rgb, K_rgb, target_shape, min_distance=0.0, max_distance=150.0
    )
    return uv_img_cords_filtered, pcd_filtered


def load_radar_projection(
    radar_path,
    calib_data,
    scene_meta_dict=None,
    motion_compensation=False,
    muses_root=None,
    target_shape=(1920, 1080),
    enlarge_radar_points=False,
):
    """
    SDK load_radar_projection 동일.
    레이더 PNG → RGB 평면 투영 이미지 (H, W, 3) float. 채널: range, intensity, height(0).
    """
    uv, pcd_f = load_points_in_image_radar(
        radar_path,
        calib_data,
        scene_meta_dict=scene_meta_dict,
        motion_compensation=motion_compensation,
        muses_root=muses_root,
        target_shape=target_shape,
        intensity_threshold=0,
    )
    image = create_image_from_point_cloud(
        uv, pcd_f, target_shape, height_channel=False, dtype=np.float32
    )
    if enlarge_radar_points and cv2 is not None:
        kernel = np.ones((9, 9), np.uint8)
        for c in range(3):
            ch = np.ascontiguousarray(image[:, :, c].astype(np.float32))
            image[:, :, c] = cv2.dilate(ch, kernel)
    return image
