#!/usr/bin/env python3
"""
concat_test_modalities.py 로 저장한 frames/ 폴더의 이미지를 영상으로 인코딩.
ffmpeg 사용 (시스템에 설치 필요).
"""
import argparse
import subprocess
from pathlib import Path

from config import OUT_VIDEO_FRAMES


def main():
    ap = argparse.ArgumentParser(description="MCubeS concat frames -> video")
    ap.add_argument("--frames-dir", type=str, default=None, help="프레임 폴더 (기본: config.OUT_VIDEO_FRAMES)")
    ap.add_argument("--out", type=str, default="mcubes_test_modalities.mp4", help="출력 영상 파일명")
    ap.add_argument("--fps", type=int, default=10, help="초당 프레임")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir) if args.frames_dir else OUT_VIDEO_FRAMES
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = frames_dir.parent / out_path

    if not frames_dir.exists():
        print("프레임 폴더가 없습니다. 먼저 concat_test_modalities.py --frames-dir ... 로 저장하세요.")
        return
    n = len(list(frames_dir.glob("*.png")))
    if n == 0:
        print("프레임 이미지가 없습니다.")
        return

    # ffmpeg: %06d.png -> mp4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    print("실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("저장:", out_path)


if __name__ == "__main__":
    main()
