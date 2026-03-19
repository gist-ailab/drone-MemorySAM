# MULTIAQUA 데이터셋 시각화 / 동영상 프레임

Test 셋 기준으로 RGB, Thermal, LiDAR, Segmentation 마스크를 한 장으로 concat 하고, 동영상용 프레임을 생성합니다.

## 경로 (config.py)

- **루트**: `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2`
- **test.txt**: `{루트}/test.txt`
- **데이터**: `{루트}/MULTIAQUA_night/`
  - **annotations**: 세그멘테이션 마스크 `{stem}.png`
  - **data/zed**: RGB `{stem}.png`
- **data/thermal_processed**: Thermal `{stem}_thermal.png` (기본: 정규화 없이 원본/선형 스케일 표시)
- **data/lidar_processed2**: LiDAR **`{stem}_lidar.png`** (grayscale, reflectivity). 동일 폴더에 `{stem}_lidar_color.png`(depth colormap) 도 있음

## 사용법

### 1. Concat 이미지 저장

```bash
cd MISC/MULTIAQUA_utils/dataset_visualization
python concat_test_modalities.py
```

- 출력: `test_concat/{stem}_concat.png`
- 패널 순서: **RGB | Thermal | LiDAR | Mask** (고정 높이, 가로 concat)
- `--no-mask`: 마스크 패널 제외
- `--thermal-normalize`: Thermal에 percentile 정규화 적용 (기본은 미적용)
- 마스크 컬러: `semseg/datasets/multiaqua.py`의 `_BASE_PALETTE` 사용 (0=boat 회색, 1~4=Static/Dynamic/Water/Sky)

### 2. 동영상용 프레임까지 저장

```bash
python concat_test_modalities.py --frames-dir "$(pwd)/test_concat/frames"
```

- `test_concat/frames/000000.png`, `000001.png`, ... (test.txt 순서)

### 3. 동영상 제작

```bash
python make_video.py --frames-dir test_concat/frames --out multiaqua_test.mp4 --fps 10
```

- `ffmpeg` 필요

## 옵션

- `--root`: MULTIAQUA 루트 (기본: config)
- 패널 해상도: RGB 원본 해상도 기준으로 Thermal/LiDAR/Mask를 같은 크기로 맞춰 concat
