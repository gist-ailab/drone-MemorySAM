"""
MUSES Event camera .h5 → RGB 평면 투영 (SDK 동일 파이프라인).
참조: https://github.com/timbroed/MUSES processing/event_camera_processing.py
- accumulate_events, stereo_rectify, render, load_event_camera_projection
"""
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import h5py
except ImportError:
    h5py = None
try:
    import hdf5plugin  # MUSES .h5 압축 필터; 있으면 등록 후 읽기 가능
except ImportError:
    hdf5plugin = None
try:
    import tables  # PyTables; h5py가 읽기 실패할 때(필터 문제) 폴백으로 사용
except ImportError:
    tables = None

from .utils_muses import rescale_K, enlarge_points_in_image


def _open_h5(event_path):
    """Open .h5; hdf5plugin 있으면 먼저 로드(압축 필터 등록), then try default/sec2/stdio."""
    event_path = Path(event_path)
    if not event_path.exists():
        raise FileNotFoundError(f"Event file not found: {event_path}")
    if h5py is None:
        raise ImportError("h5py required for event .h5 files")
    if hdf5plugin is not None:
        pass
    for driver in (None, "sec2", "stdio"):
        try:
            if driver is None:
                return h5py.File(event_path, "r")
            return h5py.File(event_path, "r", driver=driver)
        except Exception:
            continue
    raise RuntimeError("Could not open .h5 with any driver. Install hdf5plugin: pip install hdf5plugin")


def _read_events_xytep_pytables(event_path):
    """PyTables로 events/x,y,t,p 읽기 (h5py 필터 오류 시 폴백)."""
    if tables is None:
        return None, None, None, None
    event_path = Path(event_path)
    if not event_path.exists():
        return None, None, None, None
    try:
        with tables.open_file(str(event_path), "r") as f:
            if "/events" not in f:
                return None, None, None, None
            x = np.ravel(f.get_node("/events/x").read()).astype(np.float64)
            y = np.ravel(f.get_node("/events/y").read()).astype(np.float64)
            t = np.ravel(f.get_node("/events/t").read()).astype(np.float64)
            p = np.ravel(f.get_node("/events/p").read()).astype(np.int64)
            if len(x) == 0:
                return None, None, None, None
            return x, y, t, p
    except Exception:
        return None, None, None, None


def _read_events_xytep(f):
    """Read x, y, t, p (polarity). Supports events/x,y,t,p and events as group. Robust to h5 dtypes."""
    x, y, t, p = None, None, None, None

    def _read_ds(ds, dtype=np.float64):
        shape = ds.shape
        n = int(np.prod(shape))
        try:
            native = ds.dtype
            out = np.empty(shape, dtype=native)
            ds.read_direct(out)
            return np.ravel(out).astype(dtype)
        except Exception:
            pass
        for try_dtype in (dtype, np.float64, np.float32, np.uint32, np.uint16, np.uint8):
            out = np.empty(shape, dtype=try_dtype)
            try:
                ds.read_direct(out)
                return np.ravel(out).astype(dtype)
            except Exception:
                continue
        try:
            return np.ravel(np.asarray(ds).astype(dtype))
        except Exception:
            pass
        try:
            return np.ravel(ds[()].astype(dtype))
        except Exception:
            pass
        return np.zeros(n, dtype=dtype)

    if "events" in f:
        g = f["events"]
        if isinstance(g, h5py.Group):
            for key in ("x", "y", "t", "p"):
                if key not in g:
                    continue
                ds = g[key]
                target_dtype = np.int64 if key == "p" else np.float64
                try:
                    arr = _read_ds(ds, dtype=target_dtype)
                except Exception:
                    try:
                        arr = np.ravel(np.fromiter(ds, dtype=target_dtype, count=int(np.prod(ds.shape))))
                    except Exception:
                        if key == "p" and x is not None:
                            arr = np.ones(len(x), dtype=np.int64)
                        else:
                            continue
                if key == "x":
                    x = arr.astype(np.float64)
                elif key == "y":
                    y = arr.astype(np.float64)
                elif key == "t":
                    t = arr.astype(np.float64)
                elif key == "p":
                    p = arr.astype(np.int64)
            if x is not None and y is not None and t is None:
                t = np.zeros(len(x), dtype=np.float64)
            if x is not None and y is not None and p is None:
                p = np.ones(len(x), dtype=np.int64)
        else:
            try:
                arr = np.asarray(g, dtype=np.float64)
            except Exception:
                arr = np.array(g).astype(np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                x, y = arr[:, 0], arr[:, 1]
                t = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(x))
                p = arr[:, 3].astype(np.int64) if arr.shape[1] > 3 else np.ones(len(x), dtype=np.int64)
    if x is None or y is None or len(x) == 0:
        return None, None, None, None
    if t is None:
        t = np.zeros(len(x), dtype=np.float64)
    if p is None:
        p = np.ones(len(x), dtype=np.int64)
    return x, y, t, p


def accumulate_events(event_path, accumulate_over_last_us=0, verbose=False):
    """
    이벤트 로드. accumulate_over_last_us <= 0 이면 파일 전체 사용(시각화용, RGB 1장당 1 이벤트 이미지).
    > 0 이면 마지막 N us만 사용 (t 단위에 따라 ms/us/초 자동 추정).
    """
    x, y, t, p = None, None, None, None
    if tables is not None:
        x, y, t, p = _read_events_xytep_pytables(event_path)
        if x is not None and len(x) > 0:
            if verbose:
                print("[Event] read via PyTables")
        else:
            x, y, t, p = None, None, None, None
    if x is None or len(x) == 0:
        with _open_h5(event_path) as f:
            if verbose:
                print("[Event] .h5 keys:", list(f.keys()))
                if "events" in f:
                    g = f["events"]
                    print("[Event] events:", list(g.keys()) if isinstance(g, h5py.Group) else g.shape)
            x, y, t, p = _read_events_xytep(f)
        if x is not None and len(x) > 0 and np.max(np.abs(x)) == 0:
            x, y, t, p = _read_events_xytep_pytables(event_path)
    if x is None:
        if verbose:
            print("[Event] _read_events_xytep returned None (no x,y)")
        return np.array([]), np.array([]), np.array([]), np.array([])
    n_total = len(x)
    if accumulate_over_last_us <= 0:
        if verbose:
            print(f"[Event] full file: {n_total} events, t [{t.min():.3g}, {t.max():.3g}]")
        return x, y, t, p
    max_t = np.amax(t)
    if max_t > 0 and max_t < 1e3:
        window = accumulate_over_last_us / 1e6
    elif max_t >= 1e3 and max_t < 1e6:
        window = accumulate_over_last_us / 1e3
    else:
        window = accumulate_over_last_us
    min_t = max_t - window
    mask = t >= min_t
    n_keep = mask.sum()
    if verbose:
        print(f"[Event] raw events {n_total}, t range [{t.min():.3g}, {t.max():.3g}], after window {n_keep}")
    return x[mask], y[mask], t[mask], p[mask]


def get_event_calib(calib_data):
    """mtx1=rgb K, mtx2=event K, rotation_rgb2event, translation_rgb2event."""
    mtx1 = np.array(calib_data["intrinsics"]["rgb"]["K"], dtype=np.float64)
    mtx2 = np.array(calib_data["intrinsics"]["event"]["K"], dtype=np.float64)
    event2rgb = np.array(calib_data["extrinsics"]["event2rgb"], dtype=np.float64)
    rgb2event = np.linalg.inv(event2rgb)
    rotation_rgb2event = rgb2event[:3, :3]
    translation_rgb2event = rgb2event[:3, 3]
    return mtx1, mtx2, rotation_rgb2event, translation_rgb2event


def stereo_rectify(event_data, cm1, cm2, R, T, rgb_shape):
    """
    Rectify event data (event cam -> RGB view). cm1=rgb K, cm2=event K.
    Returns: event_pts_remap (2,N), R_rgb, P_rgb for later undistort.
    """
    if cv2 is None:
        raise ImportError("cv2 required for stereo_rectify")
    cm1 = np.ascontiguousarray(np.asarray(cm1, dtype=np.float64))
    cm2 = np.ascontiguousarray(np.asarray(cm2, dtype=np.float64))
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float64))
    T = np.ascontiguousarray(np.asarray(T, dtype=np.float64)).reshape(3)
    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        cm1, None, cm2, None, rgb_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )
    event_pts = np.stack([event_data[0], event_data[1]], axis=1).astype(np.float64)
    event_pts_remap = cv2.undistortPoints(event_pts, cm2, None, R=R2, P=P2)[:, 0, :].T
    return event_pts_remap, R1, P1


def undistord_events(P_rgb, R_rgb, event_pts_remap, mtx1):
    """Remap rectified event points to final RGB image coordinates."""
    if cv2 is None:
        raise ImportError("cv2 required")
    event_pts = np.stack([event_pts_remap[0], event_pts_remap[1]], axis=1).astype(np.float64)
    event_pts_final = cv2.undistortPoints(
        event_pts, P_rgb[:, :-1], None, R=np.linalg.inv(R_rgb), P=mtx1
    )[:, 0, :].T
    x = np.round(event_pts_final[0]).astype(int)
    y = np.round(event_pts_final[1]).astype(int)
    return x, y


def render(x, y, pol, H, W):
    """
    Accumulate events into image: channel 0 = positive, channel 1 = negative (SDK).
    Polarity: 1 or >0 → pos, 0 or <0 → neg.
    """
    if x.size == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pol = np.asarray(pol, dtype=np.int64).ravel()
    x = np.asarray(x, dtype=np.int64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, pol = x[valid], y[valid], pol[valid]
    idx_pos = (pol == 1) | (pol > 0)
    idx_neg = (pol == 0) | (pol < 0)
    data = x + 100000 * y
    if np.any(idx_pos):
        unique_pos, counts_pos = np.unique(data[idx_pos], return_counts=True)
        y_pos = unique_pos // 100000
        x_pos = unique_pos % 100000
        np.add.at(img[:, :, 0], (y_pos, x_pos), counts_pos)
    if np.any(idx_neg):
        unique_neg, counts_neg = np.unique(data[idx_neg], return_counts=True)
        y_neg = unique_neg // 100000
        x_neg = unique_neg % 100000
        np.add.at(img[:, :, 1], (y_neg, x_neg), counts_neg)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_points_in_image_event_camera(
    event_path, calib_data, rgb_width=1920, rgb_height=1080, accumulate_us=0, verbose=False
):
    """
    Load events, rectify to RGB frame. Returns x, y, p in RGB image coordinates.
    If rectification yields no in-frame points, returns (None, None, None) so caller can fallback to native.
    """
    x, y, t, p = accumulate_events(event_path, accumulate_over_last_us=accumulate_us, verbose=verbose)
    if verbose:
        print(f"[Event] accumulate: {len(x)} events, x=[{x.min():.0f},{x.max():.0f}] y=[{y.min():.0f},{y.max():.0f}]")
    if len(x) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    mtx1, mtx2, rotation_rgb2event, translation_rgb2event = get_event_calib(calib_data)
    mtx1 = rescale_K(mtx1, rgb_height, rgb_width)
    try:
        event_pts_remap, R_rgb, P_rgb = stereo_rectify(
            [x, y], mtx1, mtx2, rotation_rgb2event, translation_rgb2event, (rgb_width, rgb_height)
        )
        x, y = undistord_events(P_rgb, R_rgb, event_pts_remap, mtx1)
    except Exception as e:
        # 일부 OpenCV 빌드에서 stereoRectify 바인딩이 깨져있어 numpy를 인식 못하는 경우가 있음.
        # 그 경우엔 pinhole 모델로 간단 투영(깊이=1 가정)으로 대체한다.
        if verbose:
            print(f"[Event] stereoRectify failed ({e}); falling back to pinhole projection")
        event2rgb = np.array(calib_data["extrinsics"]["event2rgb"], dtype=np.float64)
        K_ev = np.array(calib_data["intrinsics"]["event"]["K"], dtype=np.float64)
        K_rgb = np.ascontiguousarray(mtx1, dtype=np.float64)
        K_inv = np.linalg.inv(np.ascontiguousarray(K_ev, dtype=np.float64))

        uv1 = np.stack([x.astype(np.float64), y.astype(np.float64), np.ones_like(x, dtype=np.float64)], axis=0)
        rays_ev = K_inv @ uv1  # (3, N)
        rays_ev_h = np.vstack([rays_ev, np.ones((1, rays_ev.shape[1]), dtype=np.float64)])  # (4, N)
        p_rgb = (event2rgb @ rays_ev_h)  # (4, N)
        z = p_rgb[2, :]
        valid = z > 1e-6
        if not np.any(valid):
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        p_rgb = p_rgb[:, valid]
        z = z[valid]
        u = (K_rgb[0, 0] * (p_rgb[0, :] / z) + K_rgb[0, 2])
        v = (K_rgb[1, 1] * (p_rgb[1, :] / z) + K_rgb[1, 2])
        x = np.round(u).astype(int)
        y = np.round(v).astype(int)
        p = p[valid]
    in_frame = (x >= 0) & (x < rgb_width) & (y >= 0) & (y < rgb_height)
    if verbose:
        print(f"[Event] after rectify: x=[{x.min()},{x.max()}] y=[{y.min()},{y.max()}], in_frame={in_frame.sum()}/{len(x)}")
    return x, y, p


def render_native(event_path, target_shape=(1920, 1080), accumulate_us=0, verbose=False):
    """
    Rectification 없이 이벤트 카메라 네이티브 좌표에 렌더 후 target_shape로 리사이즈.
    calib 불필요, 항상 뭔가 보이게 함.
    """
    x, y, t, p = accumulate_events(event_path, accumulate_over_last_us=accumulate_us, verbose=verbose)
    if len(x) == 0:
        return np.zeros((target_shape[1], target_shape[0], 3), dtype=np.uint8)
    x, y = np.round(x).astype(int), np.round(y).astype(int)
    x, y = x - x.min(), y - y.min()
    W_native = int(min(x.max() + 1, 4096))
    H_native = int(min(y.max() + 1, 4096))
    if verbose:
        print(f"[Event] native: {len(x)} events, x=[{x.min()},{x.max()}] y=[{y.min()},{y.max()}], size {W_native}x{H_native}")
    img = render(x, y, p, H_native, W_native)
    if cv2 and img.size > 0 and (img.shape[0] > 1 or img.shape[1] > 1):
        img = cv2.resize(img, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_NEAREST)
    elif img.shape[0] <= 1 or img.shape[1] <= 1:
        img = np.zeros((target_shape[1], target_shape[0], 3), dtype=np.uint8)
    else:
        try:
            from scipy.ndimage import zoom
            zoom_factors = (target_shape[1] / img.shape[0], target_shape[0] / img.shape[1], 1)
            img = zoom(img, zoom_factors, order=0)
        except ImportError:
            img = np.kron(img, np.ones((max(1, target_shape[1] // img.shape[0]), max(1, target_shape[0] // img.shape[1]), 1), dtype=img.dtype))[: target_shape[1], : target_shape[0]]
    img = _normalize_event_image_for_display(img)
    return img.astype(np.uint8)


def _normalize_event_image_for_display(img):
    """이벤트 카운트 이미지를 보이게 퍼센타일 스케일 (0~255). 전부 0이면 그대로."""
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        if ch.max() <= 0:
            continue
        p98 = np.percentile(ch[ch > 0], 98) if np.any(ch > 0) else ch.max()
        if p98 <= 0:
            p98 = 1
        out[:, :, c] = np.clip(ch / p98 * 255, 0, 255).astype(np.uint8)
    return out


def normalize_event_image_for_display(img):
    """공개: 이벤트 이미지 시각화용 정규화."""
    return _normalize_event_image_for_display(img)


def load_event_camera_projection(
    event_path,
    calib_data,
    target_shape=(1920, 1080),
    enlarge_event_camera_points=False,
    accumulate_us=0,
    use_native_fallback=True,
    verbose=False,
):
    """
    SDK 스타일: rectify → render. 실패 시 use_native_fallback이면 네이티브 렌더로 대체.
    Returns: (H, W, 3) uint8, channels 0/1 = pos/neg event counts.
    """
    rgb_width, rgb_height = target_shape
    x, y, p = load_points_in_image_event_camera(
        event_path, calib_data, rgb_width, rgb_height, accumulate_us=accumulate_us, verbose=verbose
    )
    image = render(x, y, p, rgb_height, rgb_width)
    n_nonzero = np.sum(np.any(image != 0, axis=2))
    if n_nonzero == 0 and use_native_fallback and len(x) > 0:
        if verbose:
            print("[Event] rectification produced no visible pixels, using native render")
        image = render_native(event_path, target_shape, accumulate_us=accumulate_us, verbose=verbose)
    elif n_nonzero == 0 and use_native_fallback:
        if verbose:
            print("[Event] no events in time window, trying longer window (3s) for native render")
        image = render_native(event_path, target_shape, accumulate_us=3000000, verbose=verbose)
    if enlarge_event_camera_points:
        image = enlarge_points_in_image(image, kernel_shape=(2, 2))
    return image
