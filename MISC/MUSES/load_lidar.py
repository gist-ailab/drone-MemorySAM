"""
MUSES LiDAR .bin → RGB 카메라 평면 투영 (SDK 동일 파이프라인).
- project_sensors_to_rgb.py / motion_compensation (utils) 참조: https://github.com/timbroed/MUSES
- 6열 float64 (x,y,z,intensity,mirror,timestamp), 선택적 ego-motion 보정.
"""
import argparse
import sys
from pathlib import Path

# MISC/MUSES에서 실행 시 로컬 processing 패키지 로드
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from processing.utils_muses import load_muses_calibration_data, load_meta_data, find_meta_entry_for_lidar
from processing.lidar_processing import load_lidar_projection


def _find_rgb_for_lidar(lidar_path: Path, root: Path) -> Path | None:
    """lidar .../REC0006_frame_042875_lidar.bin → frame_camera.../REC0006_frame_042875_frame_camera.png"""
    name = lidar_path.stem.replace("_lidar", "_frame_camera")
    for base in ("frame_camera_trainvaltest", "frame_camera"):
        cand = root / base
        if not cand.exists():
            continue
        for p in cand.rglob(f"{name}.png"):
            return p
    return None


def _lidar_projection_to_vis(lidar_image):
    """
    (H,W,3) range, intensity, height → 0-255 시각화용.
    논문/SDK: range·intensity·height 채널을 rescale해서 보여줌.
    """
    out = np.zeros((*lidar_image.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        ch = lidar_image[:, :, c]
        valid = ch != 0
        if not np.any(valid):
            continue
        lo, hi = np.percentile(ch[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1
        out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description="MUSES LiDAR → RGB projection (SDK-style)")
    parser.add_argument("--muses_root", type=str, default="/media/jemo/새 볼륨/dset/drone/DATA/MUSES")
    parser.add_argument(
        "--lidar",
        type=str,
        default="lidar_trainvaltest/muses/lidar/test/clear/day/REC0006_frame_042875_lidar.bin",
    )
    parser.add_argument("--calib_dir", type=str, default=None, help="e.g. radar_trainvaltest/muses")
    parser.add_argument("--motion_compensation", action="store_true", help="Use meta.json + GNSS ego-motion comp")
    parser.add_argument("--enlarge_points", action="store_true", help="Dilate lidar points for visibility")
    parser.add_argument("--no_rgb", action="store_true", help="Show only lidar projection (no RGB underlay)")
    parser.add_argument("--lidar_only", action="store_true", help="RGB·overlay 없이 LiDAR 투영 이미지만 표시")
    args = parser.parse_args()

    root = Path(args.muses_root)
    lidar_path = root / args.lidar if not Path(args.lidar).is_absolute() else Path(args.lidar)
    if not lidar_path.exists():
        raise FileNotFoundError(f"Lidar not found: {lidar_path}")

    calib_dir = args.calib_dir or "radar_trainvaltest/muses"
    calib_dir = root / calib_dir if not Path(calib_dir).is_absolute() else Path(calib_dir)
    calib_data = load_muses_calibration_data(calib_dir)

    target_shape = (1920, 1080)
    scene_meta = None
    if args.motion_compensation:
        meta_data = load_meta_data(root)
        scene_meta = find_meta_entry_for_lidar(lidar_path, root, meta_data)
        if scene_meta is None:
            print("[WARN] meta.json not found or no entry for this lidar; skipping motion compensation.")

    lidar_image = load_lidar_projection(
        str(lidar_path),
        calib_data,
        scene_meta_dict=scene_meta,
        motion_compensation=args.motion_compensation and scene_meta is not None,
        muses_root=str(root),
        target_shape=target_shape,
        enlarge_lidar_points=args.enlarge_points,
    )
    # (H,W,3) float
    w, h = target_shape
    n_proj = np.sum(np.any(lidar_image != 0, axis=2))
    print(f"[INFO] LiDAR projection: {lidar_image.shape}, non-zero pixels {n_proj}")

    vis = _lidar_projection_to_vis(lidar_image)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if args.lidar_only:
        ax.imshow(vis)
        title = "LiDAR only (range / intensity / height)"
    else:
        rgb_path = None if args.no_rgb else _find_rgb_for_lidar(lidar_path, root)
        if rgb_path and rgb_path.exists():
            import cv2
            img = cv2.imread(str(rgb_path))
            if img is not None:
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
        else:
            ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        if np.any(vis != 0):
            ax.imshow(vis, alpha=0.6)
        title = "LiDAR projected to RGB (SDK-style)"

    if args.motion_compensation and scene_meta:
        title += " [motion comp ON]"
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
