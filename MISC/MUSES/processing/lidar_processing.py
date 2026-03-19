"""
MUSES LiDAR 처리: SDK와 동일하게 6열 float64 로드, 투영, 선택적 motion compensation.
참조: https://github.com/timbroed/MUSES processing/lidar_processing.py
"""
import os
from pathlib import Path

import numpy as np

from .utils_muses import (
    create_image_from_point_cloud,
    filter_and_project_pcd_to_image,
    motion_compensate_pcd,
    enlarge_points_in_image,
)


def load_lidar_data(lidar_path):
    """
    SDK 동일: .bin → (N, 6). 논문: (x, y, z, intensity, mirror #, timestamp).
    float64 우선, 실패 시 float32 시도 (배포 데이터가 float32인 경우 대비).
    """
    lidar_path = Path(lidar_path)
    if not lidar_path.exists():
        raise FileNotFoundError(f"Lidar not found: {lidar_path}")
    for dtype in (np.float64, np.float32):
        raw = np.fromfile(lidar_path, dtype=dtype)
        n = len(raw)
        if n % 6 == 0:
            return raw.reshape(-1, 6).astype(np.float64)
    raise ValueError(f"LiDAR size {n} not divisible by 6. Expected 6 floats per point.")


def load_points_in_image_lidar(
    lidar_path,
    calib_data,
    scene_meta_data=None,
    motion_compensation=False,
    muses_root=None,
    target_shape=(1920, 1080),
):
    """
    Load lidar, optionally motion-compensate, then project to RGB image.
    Returns: (uv_img_cords_filtered (2,N), pcd_filtered)
    """
    pcd_points = load_lidar_data(lidar_path)

    if motion_compensation and scene_meta_data is not None and muses_root is not None:
        extrinsics = calib_data.get("extrinsics", {})
        lidar2gnss = extrinsics.get("lidar2gnss")
        if lidar2gnss is not None:
            pcd_points = motion_compensate_pcd(
                muses_root, scene_meta_data, pcd_points, lidar2gnss, ts_channel_num=5
            )
        # else: no lidar2gnss in calib, skip motion comp

    K_rgb = calib_data["intrinsics"]["rgb"]["K"]
    lidar2rgb = calib_data["extrinsics"]["lidar2rgb"]
    uv_img_cords_filtered, pcd_filtered = filter_and_project_pcd_to_image(
        pcd_points, lidar2rgb, K_rgb, target_shape
    )
    return uv_img_cords_filtered, pcd_filtered


def load_lidar_projection(
    lidar_path,
    calib_data,
    scene_meta_dict=None,
    motion_compensation=False,
    muses_root=None,
    target_shape=(1920, 1080),
    enlarge_lidar_points=False,
):
    """
    SDK load_lidar_projection 동일.
    Returns: (H, W, 3) float image with channels (range, intensity, height).
    """
    uv, pcd_f = load_points_in_image_lidar(
        lidar_path,
        calib_data,
        scene_meta_data=scene_meta_dict,
        motion_compensation=motion_compensation,
        muses_root=muses_root,
        target_shape=target_shape,
    )
    image = create_image_from_point_cloud(uv, pcd_f, target_shape)

    if enlarge_lidar_points:
        height_pixel_mask = image[:, :, 2] != 0
        image[height_pixel_mask, 2] += 255
        image = enlarge_points_in_image(image, kernel_shape=(2, 2))
        height_pixel_mask_dilated = image[:, :, 2] != 0
        image[height_pixel_mask_dilated, 2] -= 255

    return image
