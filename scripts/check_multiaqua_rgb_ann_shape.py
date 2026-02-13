#!/usr/bin/env python3
"""
MULTIAQUA RGB vs annotation shape 진단 스크립트.
PIL로 헤더만 읽어 (W,H) 확인 → 빠름. 불일치 stem 목록 출력.
사용: python scripts/check_multiaqua_rgb_ann_shape.py [--root PATH]
"""
import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


def get_size(path: Path) -> tuple:
    """PIL로 헤더만 읽어 (W, H) 반환. 실패 시 None."""
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night",
        help="MULTIAQUA dataset root (parent of MULTIAQUA_night/)",
    )
    args = parser.parse_args()

    if Image is None:
        print("ERROR: PIL not installed. pip install Pillow")
        return

    root = Path(args.root)
    data_root = root / "MULTIAQUA_night"
    rgb_dir = data_root / "data" / "zed"
    ann_dir = data_root / "annotations"

    if not data_root.exists():
        print(f"ERROR: {data_root} not found")
        return

    mismatches = []
    load_errors = []
    checked = 0

    for split in ["train", "val", "test"]:
        split_file = root / f"{split}.txt"
        if not split_file.exists():
            print(f"Skip {split}: {split_file} not found")
            continue

        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]

        for stem in stems:
            rgb_path = rgb_dir / f"{stem}.png"
            ann_path = ann_dir / f"{stem}.png"

            if not rgb_path.exists() or not ann_path.exists():
                continue

            rgb_size = get_size(rgb_path)
            ann_size = get_size(ann_path)

            if rgb_size is None:
                load_errors.append((stem, "RGB load failed"))
                continue
            if ann_size is None:
                load_errors.append((stem, "Ann load failed"))
                continue

            # PIL Image.size = (width, height) -> (W, H)
            # tensor shape[1:] = (H, W)
            rW, rH = rgb_size
            aW, aH = ann_size

            if (rH, rW) != (aH, aW):
                mismatches.append((stem, (rH, rW), (aH, aW)))

            checked += 1

    print(f"Checked {checked} stems")
    if load_errors:
        print(f"\nLoad errors ({len(load_errors)}):")
        for stem, err in load_errors[:10]:
            print(f"  {stem}: {err}")
        if len(load_errors) > 10:
            print(f"  ... and {len(load_errors) - 10} more")
    if mismatches:
        print(f"\n>>> SHAPE MISMATCH ({len(mismatches)} stems):")
        for stem, rgb_shape, ann_shape in mismatches:
            print(f"  {stem}: rgb={rgb_shape} ann={ann_shape}")
    else:
        print("\nAll stems: RGB and annotation shapes MATCH.")
        print("Assertion failure must have another cause.")


if __name__ == "__main__":
    main()
