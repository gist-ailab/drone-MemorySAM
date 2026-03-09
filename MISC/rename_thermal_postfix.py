#!/usr/bin/env python3
"""thermal_processed_fieldscale3 폴더 내 모든 이미지를 stem_thermal.png 형태로 이름 변경."""
from pathlib import Path

DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2/MULTIAQUA_night/data/thermal_processed_fieldscale3")
EXTS = (".png", ".jpg", ".jpeg", ".bmp")
POSTFIX = "_thermal"


def main():
    if not DIR.is_dir():
        print("폴더가 없습니다:", DIR)
        return
    renamed = 0
    skipped = 0
    for f in sorted(DIR.iterdir()):
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        if f.stem.endswith(POSTFIX):
            skipped += 1
            continue
        new_name = f.stem + POSTFIX + f.suffix
        dest = DIR / new_name
        if dest.exists():
            print("건너뜀 (대상 존재):", new_name)
            skipped += 1
            continue
        f.rename(dest)
        renamed += 1
    print("이름 변경:", renamed, "건너뜀:", skipped)


if __name__ == "__main__":
    main()
