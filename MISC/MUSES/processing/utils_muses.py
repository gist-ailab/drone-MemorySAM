"""
MUSES SDK 스타일 유틸: calib 로드, 포인트 클라우드 투영/필터, 이미지 생성.
참조: https://github.com/timbroed/MUSES processing/utils.py
"""
import json
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def load_meta_data(muses_root):
    """Load meta.json; returns dict of entry_name -> entry_data (path_to_lidar, path_to_gnss, ...)."""
    path = Path(muses_root) / "meta.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_meta_entry_for_lidar(lidar_path, muses_root, meta_data):
    """
    Find meta entry whose path_to_lidar matches this lidar file.
    lidar_path: Path or str to lidar .bin file.
    meta_data: from load_meta_data (dict of entry_name -> {path_to_lidar, path_to_gnss, ...}).
    Returns entry dict or None.
    """
    if meta_data is None or not isinstance(meta_data, dict):
        return None
    lidar_path = Path(lidar_path).resolve()
    muses_root = Path(muses_root).resolve()
    try:
        rel = lidar_path.relative_to(muses_root)
    except ValueError:
        rel = lidar_path.name
    rel_str = str(rel).replace("\\", "/")
    name_lower = lidar_path.name.lower()
    for _key, entry in meta_data.items():
        if not isinstance(entry, dict):
            continue
        pl = entry.get("path_to_lidar") or entry.get("path_to_lidar ")
        if pl is None:
            continue
        pl = str(pl).replace("\\", "/")
        if pl.endswith(name_lower) or rel_str.endswith(pl) or pl in rel_str or name_lower in pl:
            return entry
    return None


def find_meta_entry_for_radar(radar_path, muses_root, meta_data):
    """Find meta entry whose path_to_radar matches this radar file. Returns entry dict or None."""
    if meta_data is None or not isinstance(meta_data, dict):
        return None
    radar_path = Path(radar_path).resolve()
    muses_root = Path(muses_root).resolve()
    try:
        rel = radar_path.relative_to(muses_root)
    except ValueError:
        rel = radar_path.name
    rel_str = str(rel).replace("\\", "/")
    name_lower = radar_path.name.lower()
    for _key, entry in meta_data.items():
        if not isinstance(entry, dict):
            continue
        pr = entry.get("path_to_radar") or entry.get("path_to_radar ")
        if pr is None:
            continue
        pr = str(pr).replace("\\", "/")
        if pr.endswith(name_lower) or rel_str.endswith(pr) or pr in rel_str or name_lower in pr:
            return entry
    return None


def load_muses_calibration_data(input_dir, file_name="calib.json", to_numpy=True):
    """
    Loads calibration data from a JSON file.
    input_dir: directory containing calib.json (e.g. radar_trainvaltest/muses or muses_root)
    """
    path = Path(input_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Calib not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if to_numpy:
        for camera in params.get("intrinsics", {}):
            for key in params["intrinsics"][camera]:
                params["intrinsics"][camera][key] = np.array(params["intrinsics"][camera][key])
        for key in params.get("extrinsics", {}):
            params["extrinsics"][key] = np.array(params["extrinsics"][key])
    return params


def filter_points_by_distance(point_cloud, min_distance=1.0, max_distance=None):
    """Filter point cloud by distance from origin (0,0,0)."""
    distances = np.sqrt(np.sum(point_cloud[:, :3] ** 2, axis=1))
    indices = distances >= min_distance
    filtered = point_cloud[indices]
    if max_distance is not None:
        indices = distances[indices] <= max_distance
        filtered = filtered[indices]
    return filtered


def project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb):
    """
    Project point cloud to image: uv = K @ (sensor2rgb @ P) then divide by Z.
    point_cloud_xyz: (N, 3). Returns uv_img_cords (3, N): [u, v, 1] style.
    """
    pcnp_ones = np.concatenate((point_cloud_xyz, np.ones((point_cloud_xyz.shape[0], 1))), axis=1)
    point_cloud_rgb = sensor2rgb @ pcnp_ones.T
    uv_img_cords = K_rgb @ (point_cloud_rgb[0:3, :] / np.maximum(point_cloud_rgb[2, :], 1e-6))
    return uv_img_cords


def filter_by_image_boundaries(pcd_lidar, uv_img_cords, h, w):
    """Keep points with 0 < u < w-1 and 0 < v < h-1."""
    valid = (
        (uv_img_cords[0, :] > 0)
        & (uv_img_cords[0, :] < (w - 1))
        & (uv_img_cords[1, :] > 0)
        & (uv_img_cords[1, :] < (h - 1))
    )
    return pcd_lidar[valid], uv_img_cords[:, valid]


def rescale_K(K_rgb, target_h, target_w, ref_h=1080, ref_w=1920):
    """Rescale intrinsics from reference size (ref_w x ref_h) to target (target_w x target_h)."""
    K = np.array(K_rgb, dtype=np.float64)
    scale_x = target_w / ref_w
    scale_y = target_h / ref_h
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def filter_and_project_pcd_to_image(
    pcd, sensor2rgb, K_rgb, target_shape=(1920, 1080), min_distance=1.0, max_distance=None
):
    """
    SDK 동일: 거리 필터 → 투영 → 이미지 경계 필터.
    target_shape: (width, height).
    Returns: (uv_img_cords_filtered (2,N), pcd_filtered)
    """
    pcd = filter_points_by_distance(pcd, min_distance=min_distance, max_distance=max_distance)
    if len(pcd) == 0:
        return np.zeros((2, 0), dtype=np.float64), np.zeros((0, pcd.shape[1]), dtype=pcd.dtype)
    w, h = target_shape
    K_rgb = rescale_K(K_rgb, h, w)
    point_cloud_xyz = pcd[:, :3]
    uv_img_cords = project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb)
    pcd_f, uv_f = filter_by_image_boundaries(pcd, uv_img_cords, h, w)
    return uv_f, pcd_f


def create_image_from_point_cloud(
    uv_img_cords_filtered,
    filtered_pcd_points,
    target_shape=(1920, 1080),
    height_channel=True,
    dtype=np.float32,
):
    """
    SDK 동일: 3채널 이미지 (range, intensity, height).
    uv_img_cords_filtered: (2, N) or (3, N) with u=row0, v=row1.
    height_channel: True면 z를 채널2에, False면 0 (레이더 등).
    """
    w, h = target_shape
    image = np.zeros((h, w, 3), dtype=dtype)
    if uv_img_cords_filtered.shape[1] == 0:
        return image
    u = uv_img_cords_filtered[0, :].astype(int)
    v = uv_img_cords_filtered[1, :].astype(int)
    x, y, z = filtered_pcd_points[:, 0], filtered_pcd_points[:, 1], filtered_pcd_points[:, 2]
    range_channel = np.sqrt(x**2 + y**2 + z**2)
    intensity = filtered_pcd_points[:, 3]
    height_entry = z if height_channel else np.zeros_like(z)
    image[v, u, 0] = range_channel
    image[v, u, 1] = intensity
    image[v, u, 2] = height_entry
    return image


# -------- Motion compensation (SDK utils.py) --------


def read_gnss_file(load_path):
    """Read GNSS text file; returns dict with keys like iTOW, timestamp, lon, lat, gSpeed, etc."""
    keys = [
        "iTOW", "timestamp", "tAcc", "lon", "lat", "height", "hMSL", "fixType", "numSV",
        "hAcc", "vAcc", "roll", "pitch", "heading",
        "accRoll", "accPitch", "accHeading",
        "angular_rate_roll", "angular_rate_pitch", "angular_rate_heading",
        "velN", "velE", "velD", "gSpeed", "sAcc", "pDOP", "magDec", "magAcc",
    ]
    with open(load_path, "r", encoding="utf-8") as g:
        values_str = g.readline().split()
    values = [int(x) for x in values_str]
    assert len(keys) == len(values), f"{load_path}: {len(keys)} keys vs {len(values)} values"
    return dict(zip(keys, values))


def load_gnss_data(muses_root, scene_meta_data):
    """scene_meta_data must have 'path_to_gnss'."""
    gnss_rel = scene_meta_data["path_to_gnss"]
    load_path = Path(muses_root) / gnss_rel
    if not load_path.exists():
        raise FileNotFoundError(f"GNSS file not found: {load_path}")
    return read_gnss_file(str(load_path))


def calculate_time_diff(pcd_points, scene_meta_data, ts_channel_num=5):
    """Time difference per point: pcd_timestamp_s - rgb_mid_exposure_s."""
    rgb_start = int(scene_meta_data["frame_camera_exposure_start_timestamp_us"])
    rgb_end = int(scene_meta_data["frame_camera_exposure_end_timestamp_us"])
    target_ts_s = (rgb_start + rgb_end) / 2.0 / 1e6
    return pcd_points[:, ts_channel_num] - target_ts_s


def apply_transformation(pcd_points, transformation_matrix):
    """Apply 4x4 to homogeneous xyz; overwrites pcd_points[:, :3] in place. Returns pcd_points."""
    homo = np.hstack((pcd_points[:, :3], np.ones((len(pcd_points), 1))))
    T = np.array(transformation_matrix)
    transformed = (T @ homo.T).T[:, :3]
    pcd_points[:, :3] = transformed
    return pcd_points


def apply_linear_correction(pcd_points, ublox_data, delta_ts_in_s):
    """gSpeed (mm/s?) * delta_ts -> x correction in m."""
    x_correction_m = ublox_data["gSpeed"] * delta_ts_in_s / 1e3
    pcd_points[:, 0] += x_correction_m
    return pcd_points


def apply_rotational_correction(pcd_points, ublox_data, delta_ts_in_s):
    """Angular rates (deg*1e-5/s?) * delta_ts -> small rotation applied to x,y,z."""
    roll_c = np.deg2rad(ublox_data["angular_rate_roll"] * delta_ts_in_s / 1e5)
    pitch_c = np.deg2rad(ublox_data["angular_rate_pitch"] * delta_ts_in_s / 1e5)
    yaw_c = np.deg2rad(ublox_data["angular_rate_heading"] * delta_ts_in_s / 1e5)
    sr, cr = np.sin(roll_c), np.cos(roll_c)
    sp, cp = np.sin(pitch_c), np.cos(pitch_c)
    sy, cy = np.sin(yaw_c), np.cos(yaw_c)
    R11 = cy * cp
    R12 = cy * sp * sr - sy * cr
    R13 = cy * sp * cr + sy * sr
    R21 = sy * cp
    R22 = sy * sp * sr + cy * cr
    R23 = sy * sp * cr - cy * sr
    R31 = -sp
    R32 = cp * sr
    R33 = cp * cr
    x, y, z = pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2]
    pcd_points[:, 0] = R11 * x + R12 * y + R13 * z
    pcd_points[:, 1] = R21 * x + R22 * y + R23 * z
    pcd_points[:, 2] = R31 * x + R32 * y + R33 * z
    return pcd_points


def apply_correction(pcd_points, ublox_data, delta_ts_in_s):
    apply_linear_correction(pcd_points, ublox_data, delta_ts_in_s)
    apply_rotational_correction(pcd_points, ublox_data, delta_ts_in_s)
    return pcd_points


def motion_compensate_pcd(muses_root, scene_meta_data, pcd_points, sensor2gnss, ts_channel_num=5):
    """
    LiDAR 포인트를 RGB 프레임 시점으로 ego-motion 보정.
    pcd_points: (N, 6) copy will be used; returns corrected (N, 6) in sensor frame.
    """
    pcd_points = np.array(pcd_points, dtype=np.float64, copy=True)
    ublox_data = load_gnss_data(muses_root, scene_meta_data)
    delta_ts_in_s = calculate_time_diff(pcd_points, scene_meta_data, ts_channel_num)
    apply_transformation(pcd_points, sensor2gnss)
    apply_correction(pcd_points, ublox_data, delta_ts_in_s)
    gnss2sensor = np.linalg.inv(sensor2gnss)
    apply_transformation(pcd_points, gnss2sensor)
    return pcd_points


def enlarge_points_in_image(image, kernel_shape=(2, 2)):
    """Dilate non-zero pixels (e.g. for lidar overlay visibility)."""
    if cv2 is None:
        return image
    kernel = np.ones(kernel_shape, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
