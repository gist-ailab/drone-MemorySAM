#!/usr/bin/env python3
"""
MUSES concat_modalities.py 로 저장한 이미지를 영상으로 인코딩.
- 폴더에 000000.png, 000001.png ... 있으면 %06d.png 패턴 사용.
- 그 외(*_concat.png 등)면 폴더 내 모든 PNG를 이름순 정렬해 concat demuxer로 인코딩.
ffmpeg 사용 (시스템에 설치 필요).
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from config import OUT_VIDEO_FRAMES


def main():
    ap = argparse.ArgumentParser(description="MUSES concat images -> video")
    ap.add_argument("--frames-dir", type=str, default=None, help="프레임 폴더 (000000.png 또는 *_concat.png 등)")
    ap.add_argument("--out", type=str, default="muses_modalities.mp4", help="출력 영상 파일명")
    ap.add_argument("--fps", type=int, default=10, help="초당 프레임")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir) if args.frames_dir else OUT_VIDEO_FRAMES
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = frames_dir.parent / out_path

    if not frames_dir.exists():
        print("프레임 폴더가 없습니다:", frames_dir)
        return
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        print("PNG 이미지가 없습니다.")
        return

    first = frames_dir / "000000.png"
    if first.exists():
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
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in pngs:
                path = p.resolve()
                f.write(f"file '{path}'\n")
            list_path = f.name
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-r", str(args.fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            print("실행 (이미지 {}장, 이름순):".format(len(pngs)), " ".join(cmd))
            subprocess.run(cmd, check=True)
        finally:
            Path(list_path).unlink(missing_ok=True)
    print("저장:", out_path)


if __name__ == "__main__":
    main()
