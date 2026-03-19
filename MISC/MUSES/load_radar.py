"""
MUSES Radar PNG → RGB 평면 투영 시각화 (공식 SDK 스타일).
- SDK: https://github.com/timbroed/MUSES processing/radar_processing.py
- range-azimuth raw PNG → 포인트클라우드 → radar2rgb + K로 RGB에 투영.
- PNG가 SDK 포맷(폭 400 등)이 아니면 range-azimuth 원본을 그대로 표시.
"""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from processing.utils_muses import load_muses_calibration_data, load_meta_data, find_meta_entry_for_radar
from processing.radar_processing import (
    load_raw_radar_data,
    load_radar_projection,
)


def _find_radar_path(root: Path, stem_or_path: str) -> Path | None:
    """stem 또는 상대 경로로 레이더 PNG 검색. *REC*frame*radar*.png."""
    if (root / stem_or_path).exists():
        p = root / stem_or_path
        if p.suffix.lower() == ".png":
            return p
    file_id = stem_or_path.split("/")[-1].replace("_frame_camera", "").strip("_")
    parts = file_id.split("_")
    rec = parts[0] if parts else file_id
    frame = "_".join(parts[1:]) if len(parts) > 1 else ""
    for base in ("radar_trainvaltest", "radar"):
        cand = root / base
        if not cand.exists():
            continue
        matches = list(cand.rglob(f"*{rec}*{frame}*radar*.png")) + list(cand.rglob(f"*{file_id}*radar*.png"))
        if matches:
            return matches[0]
    return None


def _find_rgb_for_radar(radar_path: Path, root: Path) -> Path | None:
    """radar .../REC0008_frame_079714_xxx_radar.png → frame_camera.../REC0008_frame_079714_frame_camera.png"""
    stem = radar_path.stem
    for suffix in ("_radar", "_radar_camera"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)].rstrip("_")
            break
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("REC") and parts[1] == "frame":
        frame_camera_name = f"{parts[0]}_{parts[1]}_{parts[2]}_frame_camera"
    else:
        frame_camera_name = stem + "_frame_camera"
    for base in ("frame_camera_trainvaltest", "frame_camera"):
        cand = root / base
        if not cand.exists():
            continue
        for p in cand.rglob(f"{frame_camera_name}.png"):
            return p
    return None


def _radar_projection_to_vis(radar_image: np.ndarray) -> np.ndarray:
    """(H,W,3) float range/intensity/0 → 0–255 시각화. 퍼센타일 스케일."""
    out = np.zeros((*radar_image.shape[:2], 3), dtype=np.uint8)
    for c in range(3):
        ch = radar_image[:, :, c]
        valid = ch != 0
        if not np.any(valid):
            continue
        lo, hi = np.percentile(ch[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1
        out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description="MUSES Radar → RGB projection (SDK-style)")
    parser.add_argument("--muses_root", type=str, default="/media/jemo/새 볼륨/dset/drone/DATA/MUSES")
    parser.add_argument(
        "--radar",
        type=str,
        default="radar_trainvaltest/muses/radar/val/clear/day/REC0008_frame_079714_radar.png",
        help="레이더 PNG 경로(상대 또는 절대). 없으면 REC/frame으로 검색.",
    )
    parser.add_argument("--calib_dir", type=str, default=None, help="e.g. radar_trainvaltest/muses")
    parser.add_argument("--motion_compensation", action="store_true", help="meta.json + GNSS ego-motion 보정")
    parser.add_argument("--enlarge_points", action="store_true", help="레이더 포인트 확대 (9x9)")
    parser.add_argument("--no_rgb", action="store_true", help="RGB 배경 없이 레이더만")
    parser.add_argument("--radar_only", action="store_true", help="RGB·overlay 없이 레이더 투영 이미지만")
    parser.add_argument("--native", action="store_true", help="투영 생략, range-azimuth 원본 PNG만 표시")
    parser.add_argument("--save", type=str, default=None, help="저장 경로 (지정 시 창 대신 저장)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.muses_root)
    radar_path = _find_radar_path(root, args.radar)
    if radar_path is None:
        radar_path = root / args.radar if not Path(args.radar).is_absolute() else Path(args.radar)
    if not radar_path.exists():
        raise FileNotFoundError(f"Radar not found: {radar_path}")

    target_shape = (1920, 1080)
    w, h = target_shape

    if args.native:
        raw = load_raw_radar_data(radar_path)
        if raw is None:
            raise RuntimeError(f"Could not read radar image: {radar_path}")
        radar_vis = np.zeros((*raw.shape, 3), dtype=np.uint8)
        if raw.size > 0:
            p2, p98 = np.percentile(raw[raw > 0], [2, 98]) if np.any(raw > 0) else (0, 1)
            if p98 <= p2:
                p98 = p2 + 1
            radar_vis[:, :, 0] = radar_vis[:, :, 1] = radar_vis[:, :, 2] = np.clip(
                (raw.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255
            ).astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.imshow(radar_vis)
        ax.set_title("Radar range-azimuth (native)")
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved: {args.save}")
        else:
            plt.show()
        return

    calib_dir = args.calib_dir or "radar_trainvaltest/muses"
    calib_dir = root / calib_dir if not Path(calib_dir).is_absolute() else Path(calib_dir)
    calib_data = load_muses_calibration_data(calib_dir)

    scene_meta = None
    if args.motion_compensation:
        meta_data = load_meta_data(root)
        scene_meta = find_meta_entry_for_radar(radar_path, root, meta_data)
        if scene_meta is None and args.verbose:
            print("[WARN] meta.json에 이 레이더에 해당하는 항목 없음; motion comp 생략.")

    radar_image = load_radar_projection(
        str(radar_path),
        calib_data,
        scene_meta_dict=scene_meta,
        motion_compensation=args.motion_compensation and scene_meta is not None,
        muses_root=str(root),
        target_shape=target_shape,
        enlarge_radar_points=args.enlarge_points,
    )
    n_proj = np.sum(np.any(radar_image != 0, axis=2))
    if args.verbose:
        print(f"[INFO] Radar projection: {radar_image.shape}, non-zero pixels {n_proj}")

    vis = _radar_projection_to_vis(radar_image)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if args.radar_only:
        ax.imshow(vis)
        ax.set_title("Radar only (range / intensity)")
    else:
        rgb_path = None if args.no_rgb else _find_rgb_for_radar(radar_path, root)
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
        ax.set_title("Radar projected to RGB (SDK-style)")

    if args.motion_compensation and scene_meta:
        ax.set_title(ax.get_title() + " [motion comp ON]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
