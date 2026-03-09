#!/usr/bin/env python3
"""lidar_processed2 폴더 내 파일명에서 ")_lidar_color" → "_lidar_color" 로 치환."""
from pathlib import Path

DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/lidar_processed2")
OLD = ")_lidar_color"
NEW = "_lidar_color"


def main():
    if not DIR.is_dir():
        print("폴더가 없습니다:", DIR)
        return
    renamed = 0
    for f in sorted(DIR.iterdir()):
        if not f.is_file():
            continue
        if OLD not in f.name:
            continue
        new_name = f.name.replace(OLD, NEW)
        dest = DIR / new_name
        if dest.exists() and dest != f:
            print("건너뜀 (대상 존재):", new_name)
            continue
        f.rename(dest)
        renamed += 1
        print(f.name, "->", new_name)
    print("총 이름 변경:", renamed)


if __name__ == "__main__":
    main()
