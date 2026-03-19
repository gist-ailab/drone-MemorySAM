"""
MCubeS 데이터셋 경로 설정.
Ref: https://github.com/kyotovision-public/multimodal-material-segmentation
"""
from pathlib import Path

# 데이터셋 루트 (multimodal_dataset 폴더 바로 위 또는 그 자체)
MCUBES_ROOT = Path("/media/jemo/새 볼륨/dset/drone/DATA/MCubeS")
if not MCUBES_ROOT.exists():
    # 상대 경로 폴백
    MCUBES_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "dset" / "MCubeS"

# list_folder: train.txt, val.txt, test.txt
LIST_FOLDER = MCUBES_ROOT / "list_folder"
TEST_LIST = LIST_FOLDER / "test.txt"

# 모달리티별 폴더 (README 기준)
MODALITY_DIRS = {
    "RGB": MCUBES_ROOT / "polL_color",           # .png
    "DoLP": MCUBES_ROOT / "polL_dolp",          # .npy
    "AoLP_sin": MCUBES_ROOT / "polL_aolp_sin",   # .npy
    "AoLP_cos": MCUBES_ROOT / "polL_aolp_cos",   # .npy
    "NIR": MCUBES_ROOT / "NIR_warped",           # NIR image
}
# 세그멘테이션 마스크 (GT: material annotation, SS: semantic, SSGT4MS: condensed)
SEG_MASK_DIR = MCUBES_ROOT / "GT"  # material label per pixel
# 시각화용 선택 (전부 쓰거나 일부만). 마스크는 별도 옵션으로 추가됨.
VIS_MODALITIES = ["RGB", "DoLP", "AoLP_sin", "AoLP_cos", "NIR"]

# 출력
OUT_DIR = Path(__file__).resolve().parent / "test_concat"
OUT_VIDEO_FRAMES = OUT_DIR / "frames"  # 영상 제작 시 프레임 저장
