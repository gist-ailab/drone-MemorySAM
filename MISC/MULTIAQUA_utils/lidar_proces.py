"""
MULTIAQUA LIDAR projection utilities.

LIDAR 입력 형식 (per point):
  - X, Y, Z: 3D 좌표
  - d: distance to camera (depth)
  - r: reflectivity
  - x, y: 2D projection onto the RGB image plane (픽셀 좌표)

하는 일:
  - 위치: (x, y) 그대로 사용 → RGB 이미지와 정렬된 2D 좌표이므로 그 자리에 점을 그림.
  - 색상: depth (d) 기준으로 컬러맵 적용.
  - 즉, "depth로 색을 정하고, (x,y)에 동그라미 하나 그리면 됨." RGB와 잘 오버레이되려면
    데이터셋이 준 (x,y)가 해당 RGB 프레임의 이미지 평면 투영이면 됨.
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, Union

from tqdm import tqdm

# .npy columns: [X, Y, Z, d, r, x, y]
IDX_X, IDX_Y, IDX_Z = 0, 1, 2
IDX_D, IDX_R = 3, 4
IDX_PX, IDX_PY = 5, 6  # 2D projection onto RGB image plane


def _make_circle_mask(radius: int) -> np.ndarray:
    """Boolean mask (2*radius+1, 2*radius+1) for a filled circle."""
    r = max(0, int(radius))
    if r == 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy <= r * r).astype(bool)


def _splat_points(
    H: int,
    W: int,
    xi: np.ndarray,
    yi: np.ndarray,
    values: np.ndarray,
    radius: int,
    reduce: str = "max",
    fill: float = np.nan,
) -> np.ndarray:
    """Draw each point (xi,yi) with value as a disk of given radius. reduce in ('max','min')."""
    out = np.full((H, W), fill, dtype=np.float64)
    if xi.size == 0:
        return out
    mask_full = _make_circle_mask(radius)
    r = mask_full.shape[0] // 2
    for i in range(len(xi)):
        yc, xc = int(yi[i]), int(xi[i])
        y0 = max(0, yc - r)
        y1 = min(H, yc + r + 1)
        x0 = max(0, xc - r)
        x1 = min(W, xc + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        my0 = y0 - (yc - r)
        mx0 = x0 - (xc - r)
        m = mask_full[my0 : my0 + (y1 - y0), mx0 : mx0 + (x1 - x0)]
        cur = out[y0:y1, x0:x1]
        if reduce == "max":
            cur_fill = np.nan_to_num(cur, nan=-np.inf)
            out[y0:y1, x0:x1] = np.where(m, np.maximum(cur_fill, values[i]), cur)
        else:
            cur_fill = np.nan_to_num(cur, nan=np.inf)
            out[y0:y1, x0:x1] = np.where(m, np.minimum(cur_fill, values[i]), cur)
    return out


def _get_canvas_size(
    x: np.ndarray,
    y: np.ndarray,
    ref_shape: Optional[Tuple[int, int]] = None,
    margin: int = 0,
) -> Tuple[int, int]:
    """Canvas (H,W) = ref_shape when given. Else infer from coords (not for RGB pairing)."""
    if ref_shape is not None:
        h, w = ref_shape[:2] if len(ref_shape) >= 2 else ref_shape
        return (int(h), int(w))
    x_valid = np.isfinite(x) & (x >= 0)
    y_valid = np.isfinite(y) & (y >= 0)
    if not (x_valid.any() and y_valid.any()):
        return (576, 1024)
    w = int(np.ceil(np.max(x[x_valid])) + 1 + margin)
    h = int(np.ceil(np.max(y[y_valid])) + 1 + margin)
    return (max(h, 1), max(w, 1))


def _find_ref_image_for_lidar(lidar_path: Path, ref_ext: str = ".png") -> Optional[Path]:
    """lidar .../lidar/stem.npy → .../zed/stem.png 등 동일 stem RGB 경로 탐색."""
    lidar_path = Path(lidar_path)
    stem = lidar_path.stem
    parent = lidar_path.parent
    for ref_dir_name in ("zed", "rgb", "img", "image"):
        ref_dir = parent.parent / ref_dir_name if parent.name == "lidar" else parent / ref_dir_name
        for cand in [ref_dir / f"{stem}{ref_ext}", ref_dir / f"{stem}_rgb{ref_ext}"]:
            if cand.exists():
                return cand
    return None


def project_lidar_to_image(
    lidar_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_stem: Optional[str] = None,
    ref_image_path: Optional[Union[str, Path]] = None,
    shape_hw: Optional[Tuple[int, int]] = None,
    colormap: str = "viridis",
    grayscale_channel: str = "reflectivity",
    overwrite: bool = True,
    point_radius: int = 3,
) -> Tuple[str, str]:
    """
    Load MULTIAQUA LIDAR .npy, project onto RGB plane, save grayscale and color PNGs.

    캔버스는 항상 RGB 이미지와 동일 해상도(H,W)로 생성합니다. ref_image_path 또는
    shape_hw로 크기를 정하며, 둘 다 없으면 lidar 경로 기준으로 zed/ 등 동일 stem RGB를
    찾아 씁니다. 따라서 출력 lidar/lidar_color 이미지는 대응 RGB와 픽셀 단위로 맞습니다.

    LIDAR .npy columns: [X, Y, Z, d, r, x, y]. (x,y)에 점을 그리고 depth(d)로 색 칠함.
    point_radius로 원 반지름 (0이면 1픽셀).

    Outputs (DELIVER-style):
        - {stem}_lidar.png       : grayscale, reflectivity (or distance) at each pixel
        - {stem}_lidar_color.png : RGB, depth (d) mapped to colormap

    Args:
        lidar_path: Path to .npy file.
        output_dir: Directory to save images. Default: same dir as lidar_path.
        output_stem: Basename without extension for outputs. Default: lidar_path.stem.
        ref_image_path: Reference RGB image to get (H, W). Optional.
        shape_hw: (H, W) for canvas. Used if ref_image_path is None.
        colormap: OpenCV colormap name for depth (e.g. 'viridis', 'COLORMAP_JET').
        grayscale_channel: 'reflectivity' (r) or 'distance' (d) for grayscale image.
        overwrite: Whether to overwrite existing files.
        point_radius: Radius in pixels for each LIDAR point (0 = single pixel).

    Returns:
        (path_grayscale, path_color)
    """
    lidar_path = Path(lidar_path)
    if not lidar_path.suffix.lower() == ".npy":
        raise ValueError(f"Expected .npy file, got {lidar_path.suffix}")

    data = np.load(lidar_path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            f"Expected lidar array shape (N, 7), got {getattr(data, 'shape', 'unknown')}"
        )

    x_proj = np.asarray(data[:, IDX_PX], dtype=np.float64)
    y_proj = np.asarray(data[:, IDX_PY], dtype=np.float64)
    d = np.asarray(data[:, IDX_D], dtype=np.float64)
    r = np.asarray(data[:, IDX_R], dtype=np.float64)

    # Canvas = RGB 해상도로 고정 (출력 lidar 이미지가 RGB와 동일 크기여야 함)
    ref_path_to_use = ref_image_path
    if ref_path_to_use is None and shape_hw is None:
        ref_path_to_use = _find_ref_image_for_lidar(lidar_path)
    if ref_path_to_use is not None:
        ref = cv2.imread(str(ref_path_to_use))
        ref_shape = (ref.shape[0], ref.shape[1]) if ref is not None else None
    else:
        ref_shape = None
    if ref_shape is None and shape_hw is None:
        raise ValueError(
            "ref_image_path or shape_hw required so canvas matches RGB. "
            f"Pass --ref-image or --H/--W, or place RGB in sibling dir (e.g. zed/{{stem}}.png)."
        )
    H, W = _get_canvas_size(x_proj, y_proj, ref_shape=ref_shape or shape_hw)

    # 2D (x,y) = projection onto RGB plane → 그 좌표에 점 그림
    xi = np.round(x_proj).astype(np.int32)
    yi = np.round(y_proj).astype(np.int32)

    # Bounds check
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi = xi[valid]
    yi = yi[valid]
    d = d[valid]
    r = r[valid]

    if xi.size == 0:
        # Empty canvas: black grayscale, white color background
        gray_img = np.zeros((H, W), dtype=np.uint8)
        color_img = np.ones((H, W, 3), dtype=np.uint8) * 255
    else:
        # Grayscale: splat with disk so points are visible (max reflectivity/distance per pixel)
        r_splat = max(0, int(point_radius))
        vals = r if grayscale_channel == "reflectivity" else d
        gray_img = _splat_points(H, W, xi, yi, vals, r_splat, reduce="max", fill=np.nan)
        mask_valid = np.isfinite(gray_img)
        if mask_valid.any():
            vmin, vmax = np.nanmin(gray_img), np.nanmax(gray_img)
            if vmax > vmin:
                gray_img = np.where(mask_valid, (gray_img - vmin) / (vmax - vmin) * 255, 0)
            else:
                gray_img = np.where(mask_valid, 255, 0)
        gray_img = np.nan_to_num(gray_img, nan=0.0).astype(np.uint8)

        # Color: depth (d) 기반 컬러맵. 같은 픽셀에 여러 점이 겹치면 가까운 점(min d) 사용
        d_norm = _splat_points(H, W, xi, yi, d, r_splat, reduce="min", fill=np.inf)
        mask_valid = np.isfinite(d_norm) & (d_norm != np.inf)
        if mask_valid.any():
            dmin, dmax = d_norm[mask_valid].min(), d_norm[mask_valid].max()
            if dmax > dmin:
                d_uint8 = np.where(
                    mask_valid,
                    (d_norm - dmin) / (dmax - dmin) * 255,
                    0,
                ).astype(np.uint8)
            else:
                d_uint8 = np.where(mask_valid, 255, 0).astype(np.uint8)
        else:
            d_uint8 = np.zeros((H, W), dtype=np.uint8)
        no_data = ~mask_valid

        cmap_id = getattr(cv2, colormap, cv2.COLORMAP_VIRIDIS)
        color_img = cv2.applyColorMap(d_uint8, cmap_id)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img[no_data] = [255, 255, 255]

    # Output paths
    output_dir = Path(output_dir) if output_dir is not None else lidar_path.parent
    output_stem = output_stem or lidar_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    path_grayscale = output_dir / f"{output_stem}_lidar.png"
    path_color = output_dir / f"{output_stem})_lidar_color.png"

    if overwrite or not path_grayscale.exists():
        cv2.imwrite(str(path_grayscale), gray_img)
    if overwrite or not path_color.exists():
        cv2.imwrite(str(path_color), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

    return str(path_grayscale), str(path_color)


def process_lidar_dir(
    lidar_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    ref_image_dir: Optional[Union[str, Path]] = None,
    ref_ext: str = ".png",
    shape_hw: Optional[Tuple[int, int]] = None,
    colormap: str = "COLORMAP_VIRIDIS",
    grayscale_channel: str = "reflectivity",
    overwrite: bool = True,
    point_radius: int = 3,
) -> list:
    """
    Process all .npy files in a directory and optionally align with reference RGBs.

    If ref_image_dir is given, for each lidar file `{stem}.npy` we look for
    a reference image `ref_image_dir / {stem}{ref_ext}` (or similar) to get (H,W).
    Otherwise shape_hw is used for all.

    Returns:
        List of (path_grayscale, path_color) for each processed file.
    """
    lidar_dir = Path(lidar_dir)
    output_dir = Path(output_dir) if output_dir is not None else lidar_dir
    ref_image_dir = Path(ref_image_dir) if ref_image_dir is not None else (lidar_dir.parent / "zed")
    npy_files = sorted(lidar_dir.glob("*.npy"))
    results = []

    for npy_path in tqdm(npy_files):
        stem = npy_path.stem
        ref_path = None
        if ref_image_dir.exists():
            ref_dir = ref_image_dir
            for cand in [ref_dir / f"{stem}{ref_ext}", ref_dir / f"{stem}_rgb{ref_ext}"]:
                if cand.exists():
                    ref_path = cand
                    break
        if ref_path is None:
            ref_path = _find_ref_image_for_lidar(npy_path, ref_ext)
        path_gray, path_color = project_lidar_to_image(
            npy_path,
            output_dir=output_dir,
            output_stem=stem,
            ref_image_path=ref_path,
            shape_hw=shape_hw,
            colormap=colormap,
            grayscale_channel=grayscale_channel,
            overwrite=overwrite,
            point_radius=point_radius,
        )
        results.append((path_gray, path_color))
    return results


def lidar_frame_stats(
    lidar_path: Union[str, Path],
    ref_image_path: Optional[Union[str, Path]] = None,
    shape_hw: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Compute per-frame stats for MULTIAQUA LIDAR (for comparison with DELIVER).
    Returns dict with num_points, H, W, points_per_1k_pixels.
    """
    data = np.load(lidar_path)
    if data.ndim != 2 or data.shape[1] < 7:
        return {"error": "invalid shape", "shape": getattr(data, "shape", None)}
    x_proj = np.asarray(data[:, IDX_PX], dtype=np.float64)
    y_proj = np.asarray(data[:, IDX_PY], dtype=np.float64)
    n_total = len(x_proj)
    if ref_image_path is not None:
        ref = cv2.imread(str(ref_image_path))
        H, W = (ref.shape[0], ref.shape[1]) if ref is not None else _get_canvas_size(x_proj, y_proj)
    elif shape_hw is not None:
        H, W = int(shape_hw[0]), int(shape_hw[1])
    else:
        H, W = _get_canvas_size(x_proj, y_proj)
    xi = np.round(x_proj).astype(np.int32)
    yi = np.round(y_proj).astype(np.int32)
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    n_valid = int(np.sum(valid))
    pixels = H * W
    points_per_1k = (n_valid / pixels * 1000) if pixels else 0
    return {
        "num_points": n_total,
        "num_points_in_fov": n_valid,
        "H": H,
        "W": W,
        "pixels": pixels,
        "points_per_1k_pixels": round(points_per_1k, 4),
        "points_per_image": n_valid,
    }


def overlay_lidar_on_rgb(
    rgb_path: Union[str, Path],
    lidar_color_path: Union[str, Path],
    output_path: Union[str, Path],
    alpha: float = 0.55,
) -> str:
    """
    Overlay LIDAR color image on RGB; save to output_path.
    Use this to visually verify that projected LIDAR aligns with the scene (objects, horizon).
    """
    rgb = cv2.imread(str(rgb_path))
    lidar = cv2.imread(str(lidar_color_path))
    if rgb is None or lidar is None:
        raise FileNotFoundError(f"RGB or LIDAR image not found: {rgb_path}, {lidar_color_path}")
    if rgb.shape[:2] != lidar.shape[:2]:
        lidar = cv2.resize(lidar, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Only blend where lidar has data (not white background)
    white = np.all(lidar >= 250, axis=2)
    blend = rgb.astype(np.float32)
    lidar_f = lidar.astype(np.float32)
    blend[~white] = (1 - alpha) * rgb[~white] + alpha * lidar[~white]
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blend)
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Project MULTIAQUA LIDAR .npy to DELIVER-style images")
    parser.add_argument("lidar_npy", type=str, help="Path to LIDAR .npy file (or directory)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--ref-image", type=str, default='/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed/adr1_4_004100.png', help="Reference RGB image for size")
    parser.add_argument("--H", type=int, default=None, help="Image height (if no ref)")
    parser.add_argument("--W", type=int, default=None, help="Image width (if no ref)")
    parser.add_argument("--colormap", type=str, default="COLORMAP_VIRIDIS")
    parser.add_argument("--grayscale", type=str, choices=("reflectivity", "distance"), default="reflectivity")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip existing outputs")
    parser.add_argument("--batch", action="store_true", help="lidar_npy is a directory; process all .npy")
    parser.add_argument("--point-radius", type=int, default=3, help="Radius in pixels for each LIDAR point (default 3, use 0 for single pixel)")
    args = parser.parse_args()

    shape_hw = None
    if args.H is not None and args.W is not None:
        shape_hw = (args.H, args.W)

    if args.batch:
        results = process_lidar_dir(
            args.lidar_npy,
            output_dir=args.output_dir,
            ref_image_dir=os.path.dirname(args.ref_image) if args.ref_image else None,
            shape_hw=shape_hw,
            colormap=args.colormap,
            grayscale_channel=args.grayscale,
            overwrite=not args.no_overwrite,
            point_radius=args.point_radius,
        )
        for pg, pc in results:
            print(pg, pc)
    else:
        path_gray, path_color = project_lidar_to_image(
            args.lidar_npy,
            output_dir=args.output_dir,
            ref_image_path=args.ref_image,
            shape_hw=shape_hw,
            colormap=args.colormap,
            grayscale_channel=args.grayscale,
            overwrite=not args.no_overwrite,
            point_radius=args.point_radius,
        )
        print("Grayscale:", path_gray)
        print("Color:", path_color)
