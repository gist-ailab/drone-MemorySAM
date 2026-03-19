# MCubeS 모달리티 시각화

[Multimodal Material Segmentation (CVPR 2022)](https://github.com/kyotovision-public/multimodal-material-segmentation) 데이터셋 구조 기준으로, Test 셋의 모달리티별 이미지를 한 장으로 concat 해서 저장하고, 필요 시 영상으로 만듭니다.

## 데이터 구조 (MCubeS)

- `list_folder/test.txt`: 테스트 이미지 세트 이름 목록 (한 줄에 하나, 예: `outscene1208_2_0000000150`)
- `polL_color/`: RGB 이미지 (`.png`)
- `polL_dolp/`: DoLP (`.npy`)
- `polL_aolp_sin/`, `polL_aolp_cos/`: AoLP sin/cos (`.npy`)
- `NIR_warped/`: NIR 이미지
- `GT/`: 세그멘테이션 마스크 (material annotation, `.png` 또는 `.npy`)

## 사용법

### 1. 경로 설정

`config.py`에서 `MCUBES_ROOT`를 실제 데이터 경로로 수정:

```python
MCUBES_ROOT = Path("/media/jemo/새 볼륨/dset/drone/DATA/MCubeS")
```

또는 실행 시 `--root`로 지정:

```bash
cd MISC/MCubeS
python concat_test_modalities.py --root "/media/jemo/새 볼륨/dset/drone/DATA/MCubeS"
```

### 2. Test 셋 concat 이미지 저장

```bash
python concat_test_modalities.py
```

- 기본 출력: `MISC/MCubeS/test_concat/` 아래에 `{stem}_concat.png` 저장
- 패널 순서: RGB | DoLP | AoLP_sin | AoLP_cos | NIR | **GT**(세그멘테이션 마스크, 클래스별 컬러). `--no-mask` 시 GT 제외

### 3. 영상용 프레임까지 저장

```bash
python concat_test_modalities.py --frames-dir "$(pwd)/test_concat/frames"
```

- `test_concat/frames/000000.png`, `000001.png`, ... 형태로 test.txt 순서대로 저장

### 4. 영상 제작

```bash
python make_video.py --frames-dir test_concat/frames --out mcubes_test.mp4 --fps 10
```

- `ffmpeg`가 설치되어 있어야 합니다.

## 옵션

- `--ref-h`, `--panel-w`: 패널 높이/너비 (기본 360, 400)
- `--modalities`: 사용할 모달리티 순서 (기본: RGB DoLP AoLP_sin AoLP_cos NIR)
- `--no-mask`: 세그멘테이션 마스크(GT) 패널을 붙이지 않음
