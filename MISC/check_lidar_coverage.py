#!/usr/bin/env python3
"""
train.txt / val.txt(또는 test.txt) 목록 기준으로 lidar_processed에 _lidar.png가
얼마나 있는지·누락된 개수 집계.

사용:
  python check_lidar_coverage.py
  # 또는
  python check_lidar_coverage.py --base /path/to/MULTIAQUA_night2
"""
import argparse
from pathlib import Path

BASE = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night")
DATA_BASE = BASE / "data"
LIDAR_DIR = DATA_BASE / "lidar_processed"
SPLIT_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2")  # train.txt, val.txt, test.txt 위치


def load_stems(path: Path) -> set:
    if not path.exists():
        return set()
    stems = set()
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 경로가 있으면 마지막 이름만 stem으로 (확장자 제거)
        name = line.split("/")[-1].strip()
        stem = Path(name).stem
        stems.add(stem)
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=None, help="MULTIAQUA_night2 루트 (기본: 서버 경로)")
    ap.add_argument("--split-dir", type=str, default=None, help="train.txt/val.txt 있는 디렉터리")
    args = ap.parse_args()

    if args.base:
        base = Path(args.base)
        lidar_dir = base / "data" / "lidar_processed"
    else:
        lidar_dir = LIDAR_DIR

    split_dir = Path(args.split_dir) if args.split_dir else SPLIT_DIR

    # lidar_processed 내 *_lidar.png → base stem
    if not lidar_dir.exists():
        print("lidar_processed 폴더 없음:", lidar_dir)
        lidar_stems = set()
    else:
        lidar_stems = set()
        for f in lidar_dir.iterdir():
            if not f.is_file() or not f.suffix.lower() == ".png":
                continue
            if f.stem.endswith("_lidar") and not f.stem.endswith("_lidar_color"):
                base_stem = f.stem.removesuffix("_lidar")
                lidar_stems.add(base_stem)
        print("lidar_processed: _lidar.png 파일 수 =", len(lidar_stems))

    # train / val (및 test) 목록
    for name in ("train.txt", "val.txt", "test.txt"):
        path = split_dir / name
        stems = load_stems(path)
        if not stems:
            if path.exists():
                print(f"{name}: 줄 수 0 (비어 있음)")
            else:
                print(f"{name}: 파일 없음 ({path})")
            continue
        have = stems & lidar_stems
        missing = stems - lidar_stems
        print(f"\n[{name}]")
        print(f"  목록 개수: {len(stems)}")
        print(f"  lidar 있음: {len(have)}")
        print(f"  lidar 누락: {len(missing)} ({100 * len(missing) / len(stems):.1f}%)")
        if missing and len(missing) <= 20:
            print("  누락 stem 예:", sorted(missing)[:20])
        elif missing:
            print("  누락 stem 예:", sorted(missing)[:10], "...")

    print()


if __name__ == "__main__":
    main()
