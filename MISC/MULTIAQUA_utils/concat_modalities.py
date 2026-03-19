#!/usr/bin/env python3
"""
MUSES 데이터셋: frame_camera(RGB), event_camera, lidar, gt_semantic(Mask)를 concat 해서 저장.
동영상 제작 시 --frames-dir 로 번호 순 저장 가능.

경로: frame_camera_trainvaltest, event_camera_trainvaltest, lidar_trainvaltest, gt_semantic_trainval
각 모달리티별 서브디렉터리/파일명 규칙이 다르면 frame_stem_to_modality_stem으로 후보 경로 시도.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import MUSES_ROOT, OUT_DIR, VAL_OUT_DIR, TEST_OUT_DIR, OUT_VIDEO_FRAMES

try:
    import h5py
except ImportError:
    h5py = None

# Cityscapes 19 classes (MUSES 동일) 시각화용 간이 팔레트 RGB
MASK_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107], [152, 251, 152],
    [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [32, 11, 119], [0, 0, 0], [111, 74, 0], [250, 170, 30], [230, 150, 140],
]


def load_stems(path: Path) -> list:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def get_stems_from_frame_camera(frame_dir: Path) -> list:
    """frame_camera 내 모든 PNG의 상대 경로 stem 수집 (하위 디렉터리 포함)."""
    stems = []
    for p in sorted(frame_dir.rglob("*.png")):
        try:
            rel = p.relative_to(frame_dir)
            stem = str(rel.with_suffix("")).replace("\\", "/")
            stems.append(stem)
        except ValueError:
            continue
    return sorted(stems)


def _to_uint8_display(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = arr.squeeze()
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    else:
        arr = np.zeros_like(arr) if np.issubdtype(arr.dtype, np.floating) else arr
    return arr.astype(np.uint8)


def _resolve_path(base: Path, stem: str, ext: str = ".png") -> Path:
    """stem이 'subdir/name' 형태일 수 있음. ext는 .png, .bin, .h5 등."""
    return base / f"{stem}{ext}"


def _path_part_to_muses_subdir(path_part: str, modality: str) -> str:
    """muses_frame_camera_test_clear_day -> muses/event_camera/... 또는 muses/frame_camera/... -> muses/event_camera/..."""
    # 이미 슬래시 경로인 경우 (val.txt 등): muses/frame_camera/test/clear/day -> muses/event_camera/test/clear/day
    if "/" in path_part and "frame_camera" in path_part:
        return path_part.replace("frame_camera", modality)
    parts = path_part.split("_")
    if len(parts) >= 4 and parts[1] == "frame" and parts[2] == "camera":
        mod_tokens = modality.split("_")
        new_parts = [parts[0]] + mod_tokens + parts[3:]
        return "/".join(new_parts)
    return path_part.replace("_", "/").replace("frame_camera", modality)


def frame_stem_to_modality_stem(stem: str, modality: str) -> list:
    """frame_camera 기준 stem을 lidar/event_camera/radar/gt_semantic 실제 경로 규칙으로 변환 후보 반환.
    실제 구조: muses/event_camera/test/clear/day/REC0006_frame_042692_event_camera.h5 """
    candidates = []
    if "/" in stem:
        path_part, file_part = stem.rsplit("/", 1)
        # 실제 MUSES 디렉터리: muses/event_camera/test/clear/day (언더스코어 -> 슬래시, 모달리티명)
        subdir = _path_part_to_muses_subdir(path_part, modality)
        file_id = file_part.replace("_frame_camera", "").strip("_")  # REC0006_frame_042692
        # event: REC0006_frame_042692_event_camera.h5
        if modality == "event_camera":
            candidates.append(f"{subdir}/{file_id}_event_camera")
        # lidar: REC0006_frame_042430_lidar.bin (frame 번호는 동기화에 따라 다를 수 있음)
        elif modality == "lidar":
            candidates.append(f"{subdir}/{file_id}_lidar")
        # radar: PNG 파일명에 UUID 있음, 나중에 glob으로 검색
        elif modality == "radar":
            candidates.append(f"{subdir}/{file_id}_radar")
        # gt_semantic
        elif modality == "gt_semantic":
            candidates.append(f"{subdir}/{file_id}")
        else:
            candidates.append(f"{subdir}/{file_part.replace('_frame_camera', f'_{modality}')}")
    # 기존 규칙도 시도
    s = stem.replace("frame_camera", modality)
    candidates.append(s)
    if "/" in stem:
        path_part, file_part = stem.rsplit("/", 1)
        path_part2 = path_part.replace("frame_camera", modality)
        file_part2 = file_part.replace("_frame_camera", f"_{modality}").replace("frame_camera", modality)
        candidates.append(f"{path_part2}/{file_part2}")
    if modality == "gt_semantic" and "_" in stem:
        tail = stem.split("/")[-1] if "/" in stem else stem
        base_id = tail.replace("_frame_camera", "").strip("_")
        candidates.append(base_id)
        candidates.append(f"muses_gt_semantic_trainval/{base_id}")
    return candidates


def load_rgb(frame_dir: Path, stem: str) -> np.ndarray | None:
    p = _resolve_path(frame_dir, stem)
    if not p.exists():
        return None
    return cv2.imread(str(p))


def _load_calib(calib_path: Path) -> dict | None:
    """calib.json 전체 로드. 없으면 None."""
    if not calib_path or not Path(calib_path).exists():
        return None
    try:
        with open(calib_path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_calib_lidar2rgb(calib_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """calib.json에서 lidar2rgb(4x4)와 rgb K(3x3) 반환. 없으면 None."""
    calib = _load_calib(calib_path)
    if not calib:
        return None
    try:
        K = np.array(calib["intrinsics"]["rgb"]["K"], dtype=np.float64)
        T = np.array(calib["extrinsics"]["lidar2rgb"], dtype=np.float64)
        return T, K
    except Exception:
        return None


def _load_calib_event(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """calib.json에서 event K(3x3), rgb K(3x3), event2rgb(4x4) 반환. Event 해상도·RGB 투영용."""
    calib = _load_calib(calib_path)
    if not calib:
        return None
    try:
        K_ev = np.array(calib["intrinsics"]["event"]["K"], dtype=np.float64)
        K_rgb = np.array(calib["intrinsics"]["rgb"]["K"], dtype=np.float64)
        T_ev2rgb = np.array(calib["extrinsics"]["event2rgb"], dtype=np.float64)
        return K_ev, K_rgb, T_ev2rgb
    except Exception:
        return None


def _load_lidar_bin_projected(
    path: Path, T_lidar2rgb: np.ndarray, K: np.ndarray, out_h: int, out_w: int
) -> np.ndarray | None:
    """LiDAR .bin을 lidar2rgb + K로 RGB 이미지 평면에 투영. 동일 프레임에서 RGB와 같은 시점."""
    try:
        data = np.fromfile(path, dtype=np.float32)
        n_total = len(data)
        for stride in (6, 4, 5):
            if n_total % stride != 0:
                continue
            n = n_total // stride
            if n == 0:
                continue
            pts = data[: n * stride].reshape(n, stride)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            intensity = pts[:, 3] if stride >= 4 else np.ones(n, dtype=np.float32)
            x_ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z, intensity = x[x_ok], y[x_ok], z[x_ok], intensity[x_ok]
            if x.size == 0:
                continue
            # LiDAR -> RGB 카메라 좌표 (4x4 @ [x,y,z,1])
            ones = np.ones_like(x)
            p_lidar = np.stack([x, y, z, ones], axis=1)
            p_rgb = (T_lidar2rgb @ p_lidar.T).T
            x_cam, y_cam, z_cam = p_rgb[:, 0], p_rgb[:, 1], p_rgb[:, 2]
            # 카메라 앞쪽만
            valid = z_cam > 1e-3
            if not np.any(valid):
                continue
            x_cam, y_cam, z_cam, intensity = x_cam[valid], y_cam[valid], z_cam[valid], intensity[valid]
            # 픽셀 좌표: u = fx*x_cam/z_cam + cx, v = fy*y_cam/z_cam + cy
            u = (K[0, 0] * x_cam / z_cam + K[0, 2])
            v = (K[1, 1] * y_cam / z_cam + K[1, 2])
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)
            in_frame = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
            u, v, intensity = u[in_frame], v[in_frame], intensity[in_frame]
            if u.size == 0:
                continue
            img = np.zeros((out_h, out_w), dtype=np.float64)
            np.add.at(img, (v, u), np.clip(intensity, 0, None))
            if img.max() > 0:
                img = np.log1p(img)
            u8 = _to_uint8_display(img)
            if u8 is not None and u8.max() >= 0:
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        return None
    except Exception:
        return None


def _load_lidar_bin(path: Path, out_h: int = 480, out_w: int = 640, verbose: bool = False) -> np.ndarray | None:
    """LiDAR .bin: 6 (x,y,z,intensity,...) 또는 4,5,3 per point. x-z 또는 x-y 뷰 (calib 미사용)."""
    for dtype in (np.float32, np.float64):
        try:
            data = np.fromfile(path, dtype=dtype)
            n_total = len(data)
            for stride in (6, 4, 5, 3):
                if n_total % stride != 0:
                    continue
                n = n_total // stride
                if n == 0:
                    continue
                pts = data[: n * stride].reshape(n, stride)
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                intensity = pts[:, 3] if stride >= 4 else np.ones(n, dtype=np.float32)
                x_ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x, y, z, intensity = x[x_ok], y[x_ok], z[x_ok], intensity[x_ok]
                if x.size == 0:
                    continue
                # 뷰: x-z (측면) 또는 x-y (조감). 범위가 0에 가까우면 x-y 사용
                x_min, x_max = x.min(), x.max()
                z_min, z_max = z.min(), z.max()
                y_min, y_max = y.min(), y.max()
                use_xy = (z_max - z_min) < 1e-6 or (abs(z_max) + abs(z_min)) < 1e-3
                if use_xy:
                    a_min, a_max = x_min, x_max
                    b_min, b_max = y_min, y_max
                    a, b = x, y
                else:
                    a_min, a_max = x_min, x_max
                    b_min, b_max = z_min, z_max
                    a, b = x, z
                if a_max <= a_min:
                    a_min, a_max = a_min - 1, a_max + 1
                if b_max <= b_min:
                    b_min, b_max = b_min - 1, b_max + 1
                a_n = np.clip((a - a_min) / (a_max - a_min) * (out_w - 1), 0, out_w - 1).astype(int)
                b_n = np.clip((b - b_min) / (b_max - b_min) * (out_h - 1), 0, out_h - 1).astype(int)
                img = np.zeros((out_h, out_w), dtype=np.float64)
                np.add.at(img, (b_n, a_n), np.clip(intensity, 0, None))
                if img.max() > 0:
                    img = np.log1p(img)
                u8 = _to_uint8_display(img)
                if u8 is not None:
                    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            if verbose and dtype == np.float64:
                print(f"  [LiDAR] .bin 파싱 실패: n_total={n_total}, stride 3,4,5,6 모두 불일치 (dtype={dtype})")
        except Exception as e:
            if verbose and dtype == np.float64:
                print(f"  [LiDAR] .bin 예외: {e}")
            continue
    if verbose:
        print(f"  [LiDAR] .bin float32/float64 모두 실패")
    return None


def load_lidar(
    lidar_dir: Path,
    stem: str,
    verbose: bool = False,
    calib_path: Path | None = None,
    ref_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if not lidar_dir.exists():
        if verbose:
            print(f"  [LiDAR] lidar_dir 없음: {lidar_dir}")
        return None
    p = None
    for cand in frame_stem_to_modality_stem(stem, "lidar"):
        for ext in (".bin", ".png"):
            q = _resolve_path(lidar_dir, cand, ext)
            if q.exists():
                p = q
                break
        if p is not None:
            break
    if p is None:
        p = _resolve_path(lidar_dir, stem)
    if not p or not p.exists():
        # frame 번호가 RGB와 다를 수 있음: REC 번호만 맞는 .bin 검색
        file_id = stem.split("/")[-1].replace("_frame_camera", "").strip("_") if "/" in stem else stem.replace("_frame_camera", "")
        rec = file_id.split("_")[0] if "_" in file_id else file_id
        for f in lidar_dir.rglob(f"*{rec}*_lidar.bin"):
            p = f
            break
        else:
            if verbose:
                print(f"  [LiDAR] 실패 (rec={rec}, lidar_dir={lidar_dir})")
            return None
    if verbose and p:
        print(f"  [LiDAR] 로드: {p.name}")
    if p.suffix.lower() == ".bin":
        if calib_path is not None and ref_shape is not None:
            cal = _load_calib_lidar2rgb(calib_path)
            if cal is not None:
                T, K = cal
                out_h, out_w = ref_shape[0], ref_shape[1]
                proj = _load_lidar_bin_projected(p, T, K, out_h, out_w)
                if proj is not None:
                    return proj
        return _load_lidar_bin(p, verbose=verbose)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        u8 = _to_uint8_display(img)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None
    if img.ndim == 3 and img.dtype == np.uint8:
        return img.copy()
    u8 = _to_uint8_display(img)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None


def _load_event_h5(
    path: Path,
    out_h: int = 720,
    out_w: int = 1280,
    verbose: bool = False,
    calib_event: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ref_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Event .h5: (x,y) 2D 히스토그램. calib_event=(K_ev,K_rgb,T_ev2rgb)+ref_shape 있으면 RGB 평면에 투영."""
    if h5py is None:
        return None
    import os
    os.environ.setdefault("HDF5_PLUGIN_PATH", os.path.abspath("."))
    f = None
    last_err = None
    for driver in (None, "sec2", "stdio"):
        try:
            if driver is None:
                f = h5py.File(path, "r")
            else:
                f = h5py.File(path, "r", driver=driver)
            break
        except (OSError, Exception) as e:
            last_err = e
            if "plugin" in str(e).lower() or "directory" in str(e).lower():
                if verbose and driver == "stdio":
                    print(f"  [Event] .h5 플러그인 오류. HDF5_PLUGIN_PATH 설정 또는 hdf5-plugin 설치 후 재시도.")
            continue
    if f is None:
        if verbose and last_err is not None:
            print(f"  [Event] .h5 열기 실패: {last_err}")
        return None
    try:
        with f:
            keys = list(f.keys())
            x, y = None, None

            def read_xy_from_dataset(ds):
                a = np.asarray(ds[:])
                if a.size == 0:
                    return None, None
                if a.ndim == 2 and a.shape[1] >= 2:
                    return a[:, 0].ravel(), a[:, 1].ravel()
                if a.ndim == 1 and a.dtype.names is not None:
                    for xname, yname in [("x", "y"), ("x_coord", "y_coord")]:
                        if xname in a.dtype.names and yname in a.dtype.names:
                            return a[xname].ravel(), a[yname].ravel()
                return None, None

            if "events" in keys:
                ev_node = f["events"]
                if isinstance(ev_node, h5py.Group):
                    sub = list(ev_node.keys())
                    if "x" in sub and "y" in sub:
                        x = np.asarray(ev_node["x"][:]).ravel()
                        y = np.asarray(ev_node["y"][:]).ravel()
                    elif "event_data" in sub:
                        x, y = read_xy_from_dataset(ev_node["event_data"])
                else:
                    ev = np.asarray(ev_node[:])
                    if ev.size > 0:
                        if ev.ndim == 2 and ev.shape[1] >= 2:
                            x, y = ev[:, 0].ravel(), ev[:, 1].ravel()
                        elif ev.ndim == 1 and ev.dtype.names is not None:
                            for xname, yname in [("x", "y"), ("x_coord", "y_coord")]:
                                if xname in ev.dtype.names and yname in ev.dtype.names:
                                    x, y = ev[xname].ravel(), ev[yname].ravel()
                                    break
            if x is None and "x" in keys and "y" in keys:
                x = np.asarray(f["x"][:]).ravel()
                y = np.asarray(f["y"][:]).ravel()
            if x is None and "event_data" in keys:
                x, y = read_xy_from_dataset(f["event_data"])
            if x is None or x.size == 0:
                if verbose:
                    print(f"  [Event] .h5 키: {keys}")
                return None
            x, y = x.astype(np.float64), y.astype(np.float64)

            # RGB 평면 투영: event (x,y) -> event 카메라 3D(깊이 1) -> event2rgb -> rgb K
            if calib_event is not None and ref_shape is not None:
                K_ev, K_rgb, T_ev2rgb = calib_event
                out_h, out_w = ref_shape[0], ref_shape[1]
                # event 픽셀 (x,y) -> 동차 [x,y,1] -> K_ev^{-1} @ [x,y,1] = 3D ray (depth 1)
                K_inv = np.linalg.inv(K_ev)
                uv1 = np.stack([x, y, np.ones_like(x)], axis=1)
                p_ev = (K_inv @ uv1.T).T
                p_ev_h = np.hstack([p_ev, np.ones((len(p_ev), 1))])
                p_rgb = (T_ev2rgb @ p_ev_h.T).T
                z_rgb = p_rgb[:, 2]
                valid = z_rgb > 1e-6
                if not np.any(valid):
                    return None
                p_rgb = p_rgb[valid]
                z_rgb = z_rgb[valid]
                u = (K_rgb[0, 0] * p_rgb[:, 0] / z_rgb + K_rgb[0, 2])
                v = (K_rgb[1, 1] * p_rgb[:, 1] / z_rgb + K_rgb[1, 2])
                u = np.round(u).astype(int)
                v = np.round(v).astype(int)
                in_frame = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
                u, v = u[in_frame], v[in_frame]
                if u.size == 0:
                    return None
                img = np.zeros((out_h, out_w), dtype=np.float64)
                np.add.at(img, (v, u), 1)
                if img.max() > 0:
                    img = np.log1p(img)
                u8 = _to_uint8_display(img)
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None

            # 기본: event 네이티브 해상도. calib에 event K만 있으면 2*cx, 2*cy 사용
            if calib_event is not None:
                K_ev = calib_event[0]
                out_w = max(int(2 * K_ev[0, 2]), 1)
                out_h = max(int(2 * K_ev[1, 2]), 1)
            elif out_w <= 0 or out_h <= 0:
                out_w = max(int(x.max()) + 1, 1)
                out_h = max(int(y.max()) + 1, 1)
            xi = np.clip(np.round(x).astype(np.int64), 0, out_w - 1)
            yi = np.clip(np.round(y).astype(np.int64), 0, out_h - 1)
            img = np.zeros((out_h, out_w), dtype=np.float64)
            np.add.at(img, (yi, xi), 1)
            if img.max() > 0:
                img = np.log1p(img)
            u8 = _to_uint8_display(img)
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None
    except Exception as e:
        if verbose:
            print(f"  [Event] .h5 예외: {e}")
        return None


def load_event(
    event_dir: Path,
    stem: str,
    verbose: bool = False,
    calib_path: Path | None = None,
    ref_shape: tuple[int, int] | None = None,
    event_proj: bool = False,
) -> np.ndarray | None:
    """Event: .h5 우선, 없으면 PNG. calib_path + event_proj + ref_shape 시 RGB 평면에 투영."""
    if not event_dir.exists():
        if verbose:
            print(f"  [Event] event_dir 없음: {event_dir}")
        return None
    p = None
    for cand in frame_stem_to_modality_stem(stem, "event_camera"):
        for ext in (".h5", ".png"):
            q = _resolve_path(event_dir, cand, ext)
            if q.exists():
                p = q
                break
        if p is not None:
            break
    if p is None:
        p = _resolve_path(event_dir, stem)
    if not p or not p.exists():
        # Event frame 번호가 RGB와 다를 수 있음: REC 번호로 .h5 검색 (동일 조건 경로 우선)
        file_id = stem.split("/")[-1].replace("_frame_camera", "").strip("_") if "/" in stem else stem.replace("_frame_camera", "")
        rec = file_id.split("_")[0] if "_" in file_id else file_id
        preferred_sub = ""
        if "/" in stem:
            path_part = stem.split("/")[0]
            preferred_sub = _path_part_to_muses_subdir(path_part, "event_camera")
        candidates = list(event_dir.rglob(f"*{rec}*_event_camera.h5"))
        for f in candidates:
            if preferred_sub and preferred_sub in str(f.as_posix()):
                p = f
                break
        else:
            p = candidates[0] if candidates else None
        if p is None or not p.exists():
            if verbose:
                print(f"  [Event] 실패 (rec={rec}, event_dir={event_dir})")
            return None
    if verbose:
        print(f"  [Event] 로드: {p.name}")
    if p.suffix.lower() in (".h5", ".hdf5"):
        if h5py is None and verbose:
            print("  [Event] h5py 미설치로 .h5 로드 불가")
        calib_ev = _load_calib_event(calib_path) if calib_path else None
        use_proj = event_proj and calib_ev is not None and ref_shape is not None
        return _load_event_h5(
            p,
            verbose=verbose,
            calib_event=calib_ev,
            ref_shape=ref_shape if use_proj else None,
        )
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        u8 = _to_uint8_display(img)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None
    if img.ndim == 3 and img.dtype == np.uint8:
        return img.copy()
    u8 = _to_uint8_display(img)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if u8 is not None else None


def load_radar(radar_dir: Path, stem: str, calib_path: Path | None = None) -> np.ndarray | None:
    """Radar: 파일명에 UUID가 있어 *REC*frame*radar*.png 로 검색. calib.json 참조용."""
    if not radar_dir.exists():
        return None
    file_id = stem.split("/")[-1].replace("_frame_camera", "").strip("_") if "/" in stem else stem.replace("_frame_camera", "")
    # REC0006_frame_042430 등에서 REC번호와 frame 번호 추출
    parts = file_id.split("_")
    if len(parts) >= 2:
        rec, frame = parts[0], "_".join(parts[1:])
    else:
        rec, frame = file_id, ""
    matches = list(radar_dir.rglob(f"*{rec}*{frame}*radar*.png")) + list(radar_dir.rglob(f"*{file_id}*radar*.png"))
    if not matches:
        return None
    p = matches[0]
    if calib_path is not None and calib_path.exists():
        try:
            with open(calib_path) as f:
                json.load(f)
        except Exception:
            pass
    try:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.ndim == 2:
        u8 = _to_uint8_display(img)
        if u8 is None or not isinstance(u8, np.ndarray) or u8.ndim != 2 or u8.size == 0:
            return None
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.dtype == np.uint8:
        return img.copy()
    u8 = _to_uint8_display(img)
    if u8 is None or not isinstance(u8, np.ndarray):
        return None
    if u8.ndim == 2 and u8.size > 0:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if u8.ndim == 3 and u8.size > 0:
        return np.ascontiguousarray(u8)
    return None


def load_mask(gt_dir: Path, stem: str) -> np.ndarray | None:
    if not gt_dir.exists():
        return None
    p = None
    for cand in frame_stem_to_modality_stem(stem, "gt_semantic"):
        q = _resolve_path(gt_dir, cand)
        if q.exists():
            p = q
            break
    if p is None:
        p = _resolve_path(gt_dir, stem)
    if not p.exists():
        # fallback: gt_semantic_trainval 하위에서 REC·frame id 포함 파일 검색
        file_id = stem.split("/")[-1].replace("_frame_camera", "").strip("_") if "/" in stem else stem.replace("_frame_camera", "")
        rec = file_id.split("_")[0] if "_" in file_id else file_id
        found = None
        for pattern in (f"*{file_id}*.png", f"*{rec}*_frame_*.png", f"*{rec}*.png"):
            for f in gt_dir.rglob(pattern):
                if f.suffix.lower() == ".png":
                    found = f
                    break
            if found is not None:
                p = found
                break
        if found is None:
            return None
    try:
        im = Image.open(str(p))
        arr = np.array(im)
    except Exception:
        return None
    if arr.ndim == 2:
        labels = arr.astype(np.int32)
    elif arr.ndim == 3:
        labels = arr[:, :, 0].astype(np.int32)
    else:
        return None
    h, w = labels.shape[:2]
    if h == 0 or w == 0:
        return None
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(min(len(MASK_PALETTE), 256)):
        out[labels == i] = np.array(MASK_PALETTE[i % len(MASK_PALETTE)], dtype=np.uint8)
    out[labels == 255] = np.array([0, 0, 0], dtype=np.uint8)
    out = np.ascontiguousarray(out).astype(np.uint8)
    if out.ndim != 3 or out.shape[2] != 3 or out.size == 0:
        return None
    # RGB -> BGR for OpenCV
    return np.ascontiguousarray(out[:, :, ::-1])


def resize_to_size(img: np.ndarray, ref_h: int, ref_w: int, use_nearest: bool = False) -> np.ndarray | None:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.ndim != 2 and img.ndim != 3:
        return None
    try:
        img = np.array(img, dtype=np.uint8, copy=True, order="C")
    except (ValueError, TypeError):
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    inter = cv2.INTER_NEAREST if use_nearest else cv2.INTER_LINEAR
    try:
        return cv2.resize(img, (ref_w, ref_h), interpolation=inter)
    except cv2.error:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        pil_img = pil_img.resize((ref_w, ref_h), PILImage.NEAREST if use_nearest else PILImage.BILINEAR)
        return np.array(pil_img)


def build_concat_panel(images: list, labels: list, ref_h: int, ref_w: int) -> np.ndarray:
    h, w = max(ref_h, 1), max(ref_w, 1)
    panels = []
    for img, label in zip(images, labels):
        out_panel = np.zeros((h, w, 3), dtype=np.uint8)
        out_panel[:] = (40, 40, 40)
        if img is not None and getattr(img, "size", 0) > 0:
            try:
                img = np.asarray(img, dtype=np.uint8)
            except (ValueError, TypeError):
                img = None
        if img is not None and isinstance(img, np.ndarray) and img.ndim >= 2 and img.size > 0:
            resized = resize_to_size(img, h, w, use_nearest=(label == "Mask"))
            if resized is not None and isinstance(resized, np.ndarray) and resized.shape[0] == h and resized.shape[1] == w:
                out_panel[:] = resized.astype(np.uint8)
        img_for_text = np.ascontiguousarray(out_panel, dtype=np.uint8)
        try:
            cv2.putText(img_for_text, str(label), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        except cv2.error:
            pass
        panels.append(img_for_text)
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser(description="MUSES modality concat (RGB, LiDAR, Mask)")
    ap.add_argument("--root", type=str, default=None, help="MUSES 루트 (기본: config)")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"), help="train/val/test.txt 사용")
    ap.add_argument("--list", type=str, default=None, help="stem 목록 txt 경로 (지정 시 --split 무시)")
    ap.add_argument("--out-dir", type=str, default=None, help="concat 이미지 저장 폴더")
    ap.add_argument("--frames-dir", type=str, default=None, help="영상용 프레임 폴더 (000000.png, ...)")
    ap.add_argument("--no-mask", action="store_true", help="마스크 패널 제외")
    ap.add_argument("--no-lidar", action="store_true", help="LiDAR 패널 제외")
    ap.add_argument("--no-event", action="store_true", help="Event 패널 제외")
    ap.add_argument("--no-radar", action="store_true", help="Radar 패널 제외")
    ap.add_argument("--lidar-proj", action="store_true", help="calib.json의 lidar2rgb+RGB K로 LiDAR를 RGB 이미지 평면에 투영 (동일 시점)")
    ap.add_argument("--event-proj", action="store_true", help="calib.json의 event2rgb+RGB K로 Event를 RGB 이미지 평면에 투영 (동일 시점). 미사용 시 event K로 해상도만 보정")
    ap.add_argument("--from-camera", action="store_true", help="split 파일 없을 때 frame_camera에서 stem 자동 수집")
    ap.add_argument("--verbose", "-v", action="store_true", help="첫 프레임만 모달리티별 로드 성공 여부 출력")
    args = ap.parse_args()

    root = Path(args.root) if args.root else MUSES_ROOT
    # 실제 MUSES 배포: frame_camera_trainvaltest, event_camera_trainvaltest, lidar_trainvaltest, radar_trainvaltest, gt_semantic_trainval
    frame_dir = root / "frame_camera_trainvaltest"
    event_dir = root / "event_camera_trainvaltest"
    lidar_dir = root / "lidar_trainvaltest"
    radar_dir = root / "radar_trainvaltest"
    gt_dir = root / "gt_semantic_trainval"
    radar_calib = radar_dir / "muses" / "calib.json"
    if not frame_dir.exists():
        frame_dir = root / "frame_camera"
        event_dir = root / "event_camera" if not event_dir.exists() else event_dir
        lidar_dir = root / "lidar" if not lidar_dir.exists() else lidar_dir
        radar_dir = root / "radar" if not radar_dir.exists() else radar_dir
        gt_dir = root / "gt_semantic" if not gt_dir.exists() else gt_dir
        radar_calib = radar_dir / "muses" / "calib.json"

    if args.list is not None:
        stem_list_path = Path(args.list)
        default_out = OUT_DIR
        stems = load_stems(stem_list_path) if stem_list_path.exists() else []
    else:
        if args.split == "val":
            stem_list_path = root / "val.txt"
            default_out = VAL_OUT_DIR
        elif args.split == "test":
            stem_list_path = root / "test.txt"
            default_out = TEST_OUT_DIR
        else:
            stem_list_path = root / "train.txt"
            default_out = OUT_DIR
        if stem_list_path.exists():
            stems = load_stems(stem_list_path)
        elif frame_dir.exists():
            stems = get_stems_from_frame_camera(frame_dir)
        else:
            stems = []
    if not stems:
        print("표시할 stem이 없습니다. frame_camera 폴더에 PNG가 있는지, 또는 --list 경로를 확인하세요.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    with_mask = not args.no_mask and gt_dir.exists()
    with_lidar = not args.no_lidar and lidar_dir.exists()
    with_event = not args.no_event and event_dir.exists()
    with_radar = not args.no_radar and radar_dir.exists()

    for idx, stem in enumerate(stems):
        images = []
        labels = []
        rgb = load_rgb(frame_dir, stem)
        images.append(rgb)
        labels.append("RGB")
        ref_h, ref_w = (rgb.shape[0], rgb.shape[1]) if rgb is not None and rgb.size > 0 else (None, None)
        ev_img = (
            load_event(
                event_dir,
                stem,
                verbose=(args.verbose and idx == 0),
                calib_path=radar_calib,
                ref_shape=(ref_h, ref_w) if ref_h and ref_w else None,
                event_proj=args.event_proj,
            )
            if with_event
            else None
        )
        if with_event:
            images.append(ev_img)
            labels.append("Event")
        lidar_img = (
            load_lidar(
                lidar_dir,
                stem,
                verbose=(args.verbose and idx == 0),
                calib_path=radar_calib if args.lidar_proj else None,
                ref_shape=(ref_h, ref_w) if (args.lidar_proj and ref_h and ref_w) else None,
            )
            if with_lidar
            else None
        )
        if with_lidar:
            images.append(lidar_img)
            labels.append("LiDAR")
        if with_radar:
            images.append(load_radar(radar_dir, stem, radar_calib))
            labels.append("Radar")
        mask_img = load_mask(gt_dir, stem) if with_mask else None
        if with_mask:
            images.append(mask_img)
            labels.append("Mask")
        if args.verbose and idx == 0:
            print(f"[verbose] stem={stem}")
            print(f"  RGB: {'OK' if rgb is not None else 'FAIL'}")
            if with_event:
                print(f"  Event: {'OK' if ev_img is not None else 'FAIL'}")
            if with_lidar:
                print(f"  LiDAR: {'OK' if lidar_img is not None else 'FAIL'}")
            if with_mask:
                print(f"  Mask: {'OK' if mask_img is not None else 'FAIL'}")
        ref_h, ref_w = None, None
        for im in images:
            if im is not None and im.size > 0:
                ref_h, ref_w = im.shape[0], im.shape[1]
                break
        if ref_h is None or ref_w is None:
            continue
        canvas = build_concat_panel(images, labels, ref_h, ref_w)
        canvas = np.array(canvas, dtype=np.uint8, copy=True, order="C")
        safe_stem = stem.replace("/", "_")
        out_path = out_dir / f"{safe_stem}_concat.png"
        try:
            cv2.imwrite(str(out_path), canvas)
        except cv2.error:
            Image.fromarray(canvas[:, :, ::-1].copy()).save(str(out_path))
        if frames_dir:
            try:
                cv2.imwrite(str(frames_dir / f"{idx:06d}.png"), canvas)
            except cv2.error:
                Image.fromarray(canvas[:, :, ::-1].copy()).save(str(frames_dir / f"{idx:06d}.png"))
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  {idx+1}/{len(stems)} {stem} -> {out_path.name}")

    print("저장 경로:", out_dir)
    if frames_dir:
        print("영상용 프레임:", frames_dir, f"({len(stems)}장)")
    print("완료.")


if __name__ == "__main__":
    main()
