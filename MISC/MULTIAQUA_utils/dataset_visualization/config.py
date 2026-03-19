"""
MULTIAQUA 데이터셋 경로 설정 (동영상 프레임/시각화용).
- test.txt 기준으로 RGB, Thermal, LiDAR, Segmentation 마스크를 concat.
"""
from pathlib import Path

# 데이터셋 루트 (train/val/test.txt 있는 경로)
MULTIAQUA_ROOT = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2")
TEST_LIST = MULTIAQUA_ROOT / "test.txt"
VAL_LIST = MULTIAQUA_ROOT / "val.txt"

# MULTIAQUA_night 하위 데이터 경로
DATA_ROOT = MULTIAQUA_ROOT / "MULTIAQUA_night"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"   # 세그멘테이션 마스크 {stem}.png
DATA_DIR = DATA_ROOT / "data"

# 모달리티별 폴더 (data/ 아래)
RGB_DIR = DATA_DIR / "zed"                    # RGB: {stem}.png
THERMAL_DIR = DATA_DIR / "thermal_processed"  # Thermal: {stem}_thermal.png
LIDAR_DIR = DATA_DIR / "lidar_processed2"    # LiDAR: {stem}_lidar.png

# 출력 (기본: test_concat. val 시 스크립트에서 val_concat 사용)
OUT_DIR = Path(__file__).resolve().parent / "test_concat"
VAL_OUT_DIR = Path(__file__).resolve().parent / "val_concat"
OUT_VIDEO_FRAMES = OUT_DIR / "frames"
