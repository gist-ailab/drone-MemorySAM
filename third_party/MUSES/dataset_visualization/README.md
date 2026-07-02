# MUSES 데이터셋 시각화 / 동영상 프레임

MUSES (Multi-Sensor Semantic perception dataset) 기준으로 RGB(frame_camera), LiDAR, 세그멘테이션(gt_semantic/gt_panoptic)을 한 장으로 concat 하고, 동영상용 프레임/영상 생성.

## 경로 (config.py)

- **루트**: `/media/jemo/새 볼륨/dset/drone/DATA/MUSES`
- **frame_camera_trainvaltest**: RGB (PNG)
- **event_camera_trainvaltest**: Event (**.h5** HDF5, 또는 PNG). h5py 필요
- **lidar_trainvaltest**: LiDAR (**.bin** 포인트클라우드, 또는 PNG). .bin은 6 float/point (x,y,z,intensity,mirror,time)
- **radar_trainvaltest**: Radar (PNG, 파일명에 UUID). **muses/calib.json** 참조
- **gt_semantic_trainval**: 세그멘테이션 마스크  
  경로 규칙: `muses/모달리티/test/clear/day/REC*_frame_*` 형태로 자동 변환
- **val.txt**, **test.txt**, **train.txt**: stem 목록 (루트 또는 SPLITS_DIR에 있으면 사용)

## 사용법

### 1. Concat 이미지 저장

```bash
cd MISC/MUSES/dataset_visualization
python concat_modalities.py --split val
```

- val.txt가 있으면 해당 stem만 처리, 없으면 `--from-camera` 로 frame_camera에서 stem 수집.
- 출력: `val_concat/{stem}_concat.png` (--split val 시)

### 2. split 파일 없을 때 (frame_camera에서 자동 수집)

```bash
python concat_modalities.py --split val --from-camera
```

### 3. 동영상용 프레임까지 저장

```bash
python concat_modalities.py --split val --frames-dir "$(pwd)/val_concat/frames"
```

### 4. 동영상 제작

```bash
python make_video.py --frames-dir val_concat --out muses_val.mp4 --fps 10
```

- 폴더에 `000000.png` 형식이 있으면 %06d 패턴, 아니면 모든 PNG 이름순 concat demuxer 사용.

## 옵션

- `--root`: MUSES 루트 (기본: config)
- `--split`: train / val / test (해당 txt 사용)
- `--list`: stem 목록 txt 경로 (지정 시 --split 무시)
- `--no-mask`, `--no-lidar`, `--no-event`, `--no-radar`: 해당 패널 제외
- `--lidar-proj`: **calib.json**의 `lidar2rgb` + RGB `K`로 LiDAR 포인트를 RGB 이미지 평면에 투영. LiDAR 패널이 RGB와 같은 시점(카메라 뷰)으로 표시됨. 미사용 시에는 LiDAR 좌표계의 x-z/x-y 2D 뷰.
- `--event-proj`: **calib.json**의 `event2rgb` + RGB `K`로 Event (x,y)를 RGB 이미지 평면에 투영. Event 패널이 RGB와 같은 시점으로 표시됨. **미사용 시에도** calib의 event `K`로 Event 해상도(2×cx, 2×cy)를 보정해 표시.
- `--from-camera`: split 파일 없을 때 frame_camera에서 stem 자동 수집
- `--verbose`, `-v`: 첫 프레임만 모달리티별 로드 성공 여부 및 경로 출력 (Event/LiDAR 실패 시 원인 확인용)

## Event / LiDAR가 안 나올 때

1. **Event**: `.h5` 파일은 **h5py** 필요. `pip install h5py` 후 재실행.  
   `Can't open directory (/usr/local/lib/plugin)` 오류 시: `export HDF5_PLUGIN_PATH=.` 또는 `export HDF5_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial` (경로는 환경에 맞게). 또는 `conda install hdf5-plugin` 후 재시도.
2. **경로**: `val.txt` 등 stem이 `muses/frame_camera/test/clear/day/REC*_frame_*_frame_camera` 형태면, Event/LiDAR는 `muses/event_camera/...`, `muses/lidar/...` 하위에서 같은 REC 번호로 검색. Event는 프레임 번호가 RGB와 다를 수 있어 `*REC*_event_camera.h5` rglob으로 매칭.
3. **진단**: `python concat_modalities.py --split val --verbose` 로 첫 프레임에서 Event/LiDAR 로드 성공 여부와 사용 경로를 확인.
4. **루트**: 데이터가 다른 경로에 있으면 `--root /path/to/MUSES` 지정.
