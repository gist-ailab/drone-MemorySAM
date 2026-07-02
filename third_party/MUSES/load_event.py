"""
MUSES Event camera .h5 → RGB 평면 투영 시각화 (SDK 스타일).
- SDK: RGB 1장당 페어된 .h5 파일 하나 → 그 파일 전체 이벤트로 한 장 렌더 (예시처럼 촘촘함).
- 기본: --accumulate_us 0 → 파일 전체 사용. 짧은 구간은 --accumulate_us 30000(30ms) 등으로 지정.
- pip install tables 권장 (일부 .h5는 h5py만으로 읽기 실패).
"""
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from processing.utils_muses import load_muses_calibration_data
from processing.event_camera_processing import (
    load_event_camera_projection,
    render_native,
    normalize_event_image_for_display,
)


def _find_rgb_for_event(event_path: Path, root: Path) -> Path | None:
    """event .../REC0008_frame_079714_event_camera.h5 → frame_camera.../REC0008_frame_079714_frame_camera.png"""
    name = event_path.stem.replace("_event_camera", "_frame_camera")
    for base in ("frame_camera_trainvaltest", "frame_camera"):
        cand = root / base
        if not cand.exists():
            continue
        for p in cand.rglob(f"{name}.png"):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="MUSES Event camera → RGB projection (SDK-style)")
    parser.add_argument("--muses_root", type=str, default="/media/jemo/새 볼륨/dset/drone/DATA/MUSES")
    parser.add_argument(
        "--event",
        type=str,
        default="event_camera_trainvaltest/muses/event_camera/val/clear/day/REC0008_frame_079714_event_camera.h5",
    )
    parser.add_argument("--calib_dir", type=str, default=None, help="e.g. radar_trainvaltest/muses")
    parser.add_argument("--enlarge_points", action="store_true", help="Dilate event points")
    parser.add_argument("--no_rgb", action="store_true", help="RGB 배경 없이 이벤트만 (오버레이만)")
    parser.add_argument("--event_only", action="store_true", help="RGB·overlay 없이 Event 투영 이미지만 표시")
    parser.add_argument("--accumulate_us", type=int, default=0, help="누적 구간(us). 0=파일 전체(기본, RGB 1장당 1 이벤트 이미지). 30000=30ms, 3e6=3s")
    parser.add_argument("--save", type=str, default=None, help="저장 경로 지정 시 창 대신 파일로 저장 (headless)")
    parser.add_argument("--native", action="store_true", help="Rectification 생략, 이벤트 네이티브 좌표만 렌더 (calib 불필요)")
    parser.add_argument("--verbose", action="store_true", help="이벤트 개수·좌표 범위 등 디버그 출력")
    args = parser.parse_args()

    root = Path(args.muses_root)
    event_path = root / args.event if not Path(args.event).is_absolute() else Path(args.event)
    if not event_path.exists():
        raise FileNotFoundError(f"Event file not found: {event_path}")

    target_shape = (1920, 1080)
    if args.native:
        event_image = render_native(
            str(event_path),
            target_shape=target_shape,
            accumulate_us=args.accumulate_us,
            verbose=args.verbose,
        )
    else:
        calib_dir = args.calib_dir or "radar_trainvaltest/muses"
        calib_dir = root / calib_dir if not Path(calib_dir).is_absolute() else Path(calib_dir)
        calib_data = load_muses_calibration_data(calib_dir)
        event_image = load_event_camera_projection(
            str(event_path),
            calib_data,
            target_shape=target_shape,
            enlarge_event_camera_points=args.enlarge_points,
            accumulate_us=args.accumulate_us,
            verbose=args.verbose,
        )
    w, h = target_shape
    n_ev = np.sum(np.any(event_image != 0, axis=2))
    print(f"[INFO] Event projection: {event_image.shape}, non-zero pixels {n_ev}")

    if n_ev > 0:
        event_image = normalize_event_image_for_display(event_image)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if args.event_only:
        ax.imshow(event_image)
        ax.set_title("Event only (ch0=pos, ch1=neg)")
    else:
        rgb_path = _find_rgb_for_event(event_path, root)
        if rgb_path and rgb_path.exists() and not args.no_rgb:
            import cv2
            img = cv2.imread(str(rgb_path))
            if img is not None:
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
        else:
            ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        if np.any(event_image != 0):
            ax.imshow(event_image, alpha=0.6)
        ax.set_title("Event projected to RGB (SDK-style)")

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
