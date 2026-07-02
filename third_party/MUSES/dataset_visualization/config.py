"""
MUSES 데이터셋 경로 설정 (동영상 프레임/시각화용).
Ref: MUSES - Multi-Sensor Semantic perception dataset (frame_camera, lidar, radar, gt_semantic/gt_panoptic)
"""
from pathlib import Path

# 데이터셋 루트
MUSES_ROOT = Path("/media/jemo/새 볼륨/dset/drone/DATA/MUSES")

# 모달리티 폴더 (실제 배포명)
FRAME_CAMERA_DIR = MUSES_ROOT / "frame_camera_trainvaltest"   # RGB
EVENT_CAMERA_DIR = MUSES_ROOT / "event_camera_trainvaltest"   # Event (PNG 있으면 사용)
LIDAR_DIR = MUSES_ROOT / "lidar_trainvaltest"                 # LiDAR
RADAR_DIR = MUSES_ROOT / "radar_trainvaltest"                 # Radar (선택)
GT_SEMANTIC_DIR = MUSES_ROOT / "gt_semantic_trainval"         # semantic segmentation

# split 목록 (MUSES 루트 또는 별도 폴더에 val.txt, test.txt, train.txt)
VAL_LIST = MUSES_ROOT / "val.txt"
TEST_LIST = MUSES_ROOT / "test.txt"
TRAIN_LIST = MUSES_ROOT / "train.txt"

# 출력
OUT_DIR = Path(__file__).resolve().parent / "concat_out"
VAL_OUT_DIR = Path(__file__).resolve().parent / "val_concat"
TEST_OUT_DIR = Path(__file__).resolve().parent / "test_concat"
OUT_VIDEO_FRAMES = OUT_DIR / "frames"
