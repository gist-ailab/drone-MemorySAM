# D1 ViT-S+ — 멀티모달 실시간 검출 공인인증 패키지

이 브랜치(`chore/cert-prune`, 기반 `26-drone-certificate`)는 **공인인증(정량목표 검사) 전용
배포 패키지**다. 연구용 로그·논문·세그멘테이션 실험 경로는 모두 제거했고, 남은 것은
**① 인증 평가 ② detection 재학습 ③ YOLO 대조군** 세 갈래뿐이다.

---

## 1. 인증 대상 모델

| 항목 | 값 |
|---|---|
| 모델명 | **D1 ViT-S+** |
| 백본 | ReliaDINO — frozen **DINOv3 ViT-S+/16** (`vit_small_plus_patch16_dinov3`) + 모달리티별 LoRA(r=8) + ReliabilityGatedFusion + SimpleFPN |
| 검출 헤드 | **RF-DETR** NMS-free head (queries 300, group-DETR 13, decoder 4층, top-k 300) |
| 모달리티 | 3-modal — RGB(`rgb`) + LiDAR depth(`depth_map_lidar`) + Thermal(`thermal_aligned`) |
| 클래스 수 | 10 (클래스 이름은 annotation의 `categories`에서 읽으며, `cert_eval.py`가 시작 배너에 출력) |
| 입력 해상도 | 768 × 768 (stretch resize) |
| 파라미터 | 52,498,735 (52.5 M) — 본 패키지에서 CPU 빌드로 실측 |
| 인증 config | `configs/det/det_D1_vitsp_jarvis.yaml` |
| 인증 웨이트 | `det_D1_vitsp_20260723` / epoch 11 / md5 `b78f614cba1375bb54dfeacd5e58cef3` |

### 인증 수치 (poongsan_v2 final test, 3,239장, **predicted-scope**)

| 지표 | 값 |
|---|---|
| **mAP50 (전체)** | **0.9166** |
| mAP50 (night) | 0.8765 |
| mAP50 (normal) | 0.9418 |
| FPS | 7.38 (RTX 3090) / 약 10–12 (RTX 5080) |
| VRAM | 0.76 GB |

> 위 수치는 `certification/README.md`에 기록된 실측값이다. 이 README는 새 수치를 만들지 않는다.
> `cert_eval.py`는 `tools/_det_common.py`를 그대로 쓰므로 `tools/det_eval_breakdown.py`
> (mAP 분해) · `tools/det_fps_bench.py`(FPS)와 동일한 값을 낸다.

---

## 2. Dependency

### 2.1 검증된 환경 (이 패키지가 실제로 돌아가는 것을 확인한 조합)

```
python              3.10.19
torch               2.3.1+cu121
torchvision         0.18.1+cu121
timm                1.0.24      # 🔴 DINOv3 백본은 timm>=1.0 필수
pycocotools         2.0.11
opencv-python       4.13.0.92
numpy               2.2.6
PyYAML              6.0.3
pillow              11.3.0
einops              0.8.2
scipy               1.15.3
tqdm                4.67.3
hydra-core          1.3.2       # semseg/models/sam2/sam2/__init__.py가 import
iopath              0.1.10
tensorboard         2.20.0      # 학습 로깅(선택)
matplotlib          3.10.8      # 분석 도구(선택)
```

### 2.2 🔴 timm 버전 주의

**DINOv3 백본(`vit_small_plus_patch16_dinov3`)은 `timm>=1.0`에서만 존재한다.**
`timm<1.0`이면 백본 생성이 실패하고, config의 `BACKBONE_FALLBACK`
(`vit_small_patch14_reg4_dinov2`)으로 조용히 떨어져 **인증 웨이트와 구조가 달라진다.**
반드시 아래로 확인할 것:

```bash
python -c "import timm; print(timm.__version__)"     # >= 1.0 이어야 함
python -c "import timm; print('vit_small_plus_patch16_dinov3' in timm.list_models())"
```

> ⚠️ 리포에 함께 들어 있는 `requirements.txt` / `conda_environment.yml`(torch 2.9+cu128) /
> `environment.yaml`(`name: cmnext`, python 3.8 / torch 1.9)은 **이전 연구 환경의 잔존 스펙**이라
> 위 검증 조합과 일치하지 않는다. 참고용으로만 두었고, **인증 환경 구성은 2.3절을 따를 것.**
>
> 단, 세 파일에 있던 `timm==0.4.12` 핀만은 **위 fallback 사고를 그대로 유발하므로 `timm>=1.0`으로
> 정정했다**(2026-07-28). 나머지 항목은 손대지 않았다.

### 2.3 환경 만들기

```bash
conda create -n cert_d1 python=3.10 -y
conda activate cert_d1

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "timm>=1.0" pycocotools opencv-python numpy pyyaml pillow einops scipy tqdm \
            hydra-core iopath tensorboard matplotlib
```

기존 `MMSS_SAM` conda 환경이 있는 서버라면 그대로 써도 된다(위 검증 조합이 그 환경이다).

일부 서버는 시스템 timm이 낡아 `/SSDb/jemo_maeng/pylibs_p34`로 shadow 한다.
`certification/run_cert.sh`가 그 디렉터리가 있으면 자동으로 `PYTHONPATH` 앞에 붙인다.

---

## 3. 코드 실행 방법

모든 명령은 **리포 루트에서** 실행한다.

### (a) 인증 평가 — 엔터 한 번

```bash
bash certification/run_cert.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]

# 예)
bash certification/run_cert.sh \
    /ailab_mat2/personal/jemo_maeng/.../submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth \
    /SSDd/jemo_maeng/dset/poongsan_v2 \
    0
```

- `DATA_ROOT` 생략 시 기본값 `/SSDd/jemo_maeng/dset/poongsan_v2`, `GPU` 기본값 `0`.
- 파이썬 실행기를 바꾸려면 `PYTHON=/path/to/python bash certification/run_cert.sh ...`.
- 4번째 인자부터는 `cert_eval.py`로 그대로 전달된다.

동작 순서:
1. **시작 배너** — 모델/백본/헤드, 체크포인트 md5·epoch·load(missing/unexpected), 장치,
   입력 크기, 클래스 10종, 사용 모달리티, 이미지 수(night/normal), 데이터 루트, 출력 경로.
2. **ENTER 대기** (`--auto`로 생략).
3. **스트리밍 추론** — 이미지별 `추론 ms / 검출 수(클래스별) / GT 수 / TP·FP·FN`, 오버레이 PNG 저장.
4. **최종 리포트** — mAP(.50:.95)·mAP75, night/normal mAP50, per-class AP50,
   `►► mAP50 = … | FPS = … ◄◄` 한 줄, VRAM.

주요 옵션 (`certification/run_cert.sh <ckpt> <root> <gpu>` 뒤에 붙이거나 `cert_eval.py`에 직접):

| 옵션 | 의미 |
|---|---|
| `--auto` | ENTER 프롬프트 생략 (배치/무인 실행) |
| `--limit N` | 앞에서 N장만 (빠른 확인) |
| `--stride N` | N장마다 1장 (전 클립을 고르게 샘플링) |
| `--score-thresh` | 오버레이/화면 로그 표시 임계 (기본 0.3) |
| `--eval-thresh` | COCO 채점 임계 (기본 0.05) |
| `--mode {val,test}` | 사용할 annotation (기본 `val` = `ANNOTATION_VAL` = final test json) |
| `--show` | `$DISPLAY`가 있으면 cv2 창 표시 |
| `--lowlight-clips` | night 판정 클립 목록 (기본 `capture_20260618_114021,capture_20260618_115624`) |
| `--data-root` | 이 머신의 poongsan_v2 마운트로 `DATASET.ROOT` + `ANNOTATION_*` 재지정 |
| `--gpu` | CUDA device index |

`run_cert.sh` 없이 직접 부르는 형태:

```bash
python certification/cert_eval.py \
    --cfg  configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt <best_checkpoint.pth> \
    --data-root <poongsan_v2 mount> \
    --out  runs/cert_D1 --gpu 0 --auto
```

### (b) 학습/평가 분리 검증 (누수 없음 증명)

모델·GPU 불필요. annotation만 비교하며 exit code 0(PASS)/1(FAIL)을 낸다.

```bash
# config에서 ANNOTATION_TRAIN / ANNOTATION_VAL을 읽어 비교
python certification/check_split.py --cfg configs/det/det_D1_vitsp_jarvis.yaml

# 이 머신의 마운트로 경로만 갈아끼우기
python certification/check_split.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --data-root /SSDd/jemo_maeng/dset/poongsan_v2

# json 두 개를 직접 지정
python certification/check_split.py --train <train.json> --test <test.json>

# 리포트 저장 위치 지정 (기본은 현재 디렉터리의 split_check.json)
SPLIT_CHECK_OUT=runs/cert_D1/split_check.json \
python certification/check_split.py --cfg configs/det/det_D1_vitsp_jarvis.yaml
```

검사 수준 2가지:
- **frame-level** — 같은 이미지 파일이 양쪽 split에 있는가 (하드 누수). PASS 필수 조건.
- **clip-level** — 같은 캡처 세션(클립)을 공유하는가. 연속 프레임 near-duplicate 누수 여부.

PASS 조건 = frame-disjoint **AND** 카테고리 일치. (clip-disjoint까지 만족하면
"clean capture-holdout"으로 추가 출력.)

현재 poongsan final split(= `certification/README.md` 기록):
train 5클립 `112051 / 113007 / 113534 / 115206 / 120059` 12,681장,
eval 3클립 `114021 / 114808 / 115624` 3,239장 — **frame·clip 모두 겹침 0.**

### (c) detection 재학습

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=0 python train_det.py --cfg configs/det/det_D1_vitsp_jarvis.yaml

# 멀티 GPU (torchrun DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    train_det.py --cfg configs/det/det_D1_vitsp_jarvis.yaml

# 이어서 학습 / 평가만
python train_det.py --cfg <cfg> --resume <checkpoint.pth>
python train_det.py --cfg <cfg> --eval_only

# 빈 GPU 확인 (실행 전 필수)
bash scripts/pick_free_gpus.sh 4
```

학습 결과는 `<OUTPUT_DIR>/<config 파일명(확장자 제외)>/` 아래에 저장된다.
인증 config 기준: `outputs/det_D1_vitsp_jarvis/det_D1_vitsp_jarvis/`
→ `best_checkpoint.pth`, `epoch<N>_checkpoint.pth`, `config.yaml`.

학습 후 평가:

```bash
# 표준 평가 (predictions.json / metrics.json, --save_vis로 오버레이)
python val_det.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --det_checkpoint <best_checkpoint.pth> --mode val [--save_vis --save_dir <dir>]

# 인증과 동일 스택의 mAP 분해 (night/normal + per-class)
python tools/det_eval_breakdown.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt <best_checkpoint.pth> --out runs/breakdown_D1 --eval-scope predicted

# FPS 벤치
python tools/det_fps_bench.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt <best_checkpoint.pth> --out runs/fps_D1 --gpu 0
```

> `--eval-scope`: `predicted` = 데이터셋이 실제로 로드한 이미지만 채점(인증 수치가 이 기준),
> `annotation` = 전체 GT 이미지 대비 채점(학습 로그와 비교용). **두 값은 다르므로 섞지 말 것.**

남아 있는 det config (전부 ReliaDINO 계열):

| config | 헤드 | 비고 |
|---|---|---|
| `det_D1_vitsp_jarvis.yaml` | RF-DETR | **인증 대상** (ViT-S+) |
| `det_D1_vits_yeon.yaml` / `det_D1_vitb_yeon.yaml` | RF-DETR | 백본 크기 변형 |
| `det_D1_p37b_lowlr_yeon.yaml` / `det_D1_recovered_yeon.yaml` | RF-DETR | 학습 레시피 변형 |
| `det_P34/P35/P36_final_full.yaml` | FCOS | 이전 세대 |
| `det_P37_rfdetr_full.yaml` / `det_P37a_cefr_yeon.yaml` / `det_P37b_classtoken_yeon.yaml` | RF-DETR | 이전 세대 |
| `det_P38_m2f_yeon.yaml` / `det_P39_dpc_yeon.yaml` | M2F | 이전 세대 |

### (d) YOLO 대조군 (인증 모델과 독립)

세 갈래 모두 **별도 baseline**이며 인증 수치와 무관하다. 실행법은 각 디렉터리 README 참조.

| 경로 | 내용 |
|---|---|
| `objdet/yolov5m-lowlight/` | classic YOLOv5m(RGB-only) 저조도 학습레시피 ablation(b0→b3). 추론 그래프 불변(i.MX NPU portable). `train_ladder.sh` / `eval_lowlight.sh` |
| `objdet/yolo11m-rgb/` | YOLO11-m RGB-only 기준점. `convert_final_yolo.py` 변환 → `train_hinton.sh` 학습 → `viz_test_set.py` 시각화 |
| `objdet/gistolo/` | D1(3-modal)을 teacher로 RGB student에 distill. `teacher_dump.py`(인증 추론 스택 재사용) → `gistolo_labels.py` |

```bash
# YOLOv5m 저조도 ladder (외부 ultralytics/yolov5 clone 필요)
git clone https://github.com/ultralytics/yolov5 && export YOLOV5_DIR=$PWD/yolov5
DATA_YAML=<...>/poongsan_v2_rgb.yaml bash objdet/yolov5m-lowlight/train_ladder.sh b0 <gpu> 100
bash objdet/yolov5m-lowlight/eval_lowlight.sh <best.pt> $DATA_YAML <out_dir> <gpu>

# GISTOLO teacher dump (D1 웨이트 필요)
python objdet/gistolo/teacher_dump.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt <best_checkpoint.pth> --data-root <poongsan_v2 mount> \
    --out runs_gistolo/teacher_train_preds.json
```

---

## 4. 체크포인트

| 항목 | 값 |
|---|---|
| 런 이름 | `det_D1_vitsp_20260723` |
| 파일 | `best_checkpoint.pth` (epoch 11) |
| md5 | `b78f614cba1375bb54dfeacd5e58cef3` |
| 알려진 위치 | `/ailab_mat2/.../submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth` |

**웨이트는 이 리포에 포함되어 있지 않다.** 별도로 전달받아 임의 경로에 두고
`run_cert.sh`의 첫 인자로 넘기면 된다. 무결성 확인:

```bash
md5sum <best_checkpoint.pth>       # b78f614cba1375bb54dfeacd5e58cef3
```

`cert_eval.py`는 시작 배너에 md5 앞 12자리와 `load: missing=… unexpected=…`를 출력한다.
**정상 로드면 missing/unexpected가 0**이어야 한다.

### `.pth` 포맷 주의

`torch.load(..., weights_only=False)`로 읽는 **dict 체크포인트**다.

| 키 | 내용 |
|---|---|
| `model_state_dict` | 백본+헤드 전체 (인증 로드 경로) |
| `detector_state_dict` | 헤드만 (경량 재사용용) |
| `optimizer_state_dict`, `epoch`, `best_ap`, `metrics`, `config` | 학습 메타 |

`model_state_dict`가 있으면 그것을 `strict=False`로 로드하고, 없으면
`detector_state_dict`(헤드만)로 폴백한다 — **헤드만 든 파일을 넘기면 백본이
사전학습 초기값이 되어 수치가 무너지므로**, 배너의 missing/unexpected를 반드시 확인할 것.

---

## 5. 산출물·로그 경로

`run_cert.sh`는 `runs/cert_D1/`(고정)에 아래를 만든다.

| 경로 | 생성 주체 | 내용 |
|---|---|---|
| `runs/cert_D1/console_<YYYYmmdd_HHMMSS>.log` | `run_cert.sh` (tee) | 배너부터 최종 리포트까지 콘솔 전체(스트리밍 라인 포함) |
| `runs/cert_D1/inference_log.txt` | `cert_eval.py` | 이미지별 TSV — `idx ⇥ file_name ⇥ ms ⇥ det=N ⇥ GT=N ⇥ TP=N ⇥ FP=N ⇥ FN=N` (viz 실패 시 `[viz warn]` 라인) |
| `runs/cert_D1/cert_report.json` | `cert_eval.py` | `overall` / `night` / `normal` (AP, AP50, AP75 …), `per_class`, `fps`, `mean_latency_ms`, `vram_gb`, `checkpoint`(epoch·load·metrics), `cfg`, `ckpt` |
| `runs/cert_D1/viz/<NNNN>_<파일stem>.png` | `cert_eval.py` | 오버레이 — 예측=클래스색 박스+score, GT=녹색 얇은 박스 |

`check_split.py`의 리포트는 **기본적으로 현재 디렉터리의 `split_check.json`**에 저장된다
(`runs/cert_D1/` 아래가 아니다). 옮기려면 `SPLIT_CHECK_OUT` 환경변수를 쓴다.

기타:
- `val_det.py` → `--save_dir` (미지정 시 `<ckpt 디렉터리>/eval_<mode>/`) 에 `predictions.json`, `metrics.json`, `--save_vis` 시 `<stem>_det.png`.
- `tools/det_eval_breakdown.py --out <prefix>` → `<prefix>.json` + `<prefix>.md`.
- `tools/det_fps_bench.py --out <prefix>` → `<prefix>.json`.
- `train_det.py` → `<OUTPUT_DIR>/<cfg stem>/{best_checkpoint.pth, epoch<N>_checkpoint.pth, config.yaml}`.

---

## 6. 데이터셋 (poongsan_v2)

단일 루트(`DATASET.ROOT`) 아래에 캡처 클립별 이미지와 COCO annotation이 있다.

```
poongsan_v2/                          # = DATA_ROOT
├── capture_20260618_112051/          # 캡처 클립 단위 (train 5 + eval 3 = 총 8클립)
│   ├── rgb/                 *.png    #   img 모달리티 (file_name의 경로가 이 형태)
│   └── …                             #   lidar/thermal 파일 위치는 annotation의
│                                     #   per-image `modalities` dict가 지정
├── capture_20260618_113007/
│   …
└── _final_ann/
    ├── instances_train_egofill.json  # ANNOTATION_TRAIN  (train 5클립, 12,681장)
    └── instances_test_common.json    # ANNOTATION_VAL    (eval  3클립,  3,239장)
```

- 경로 해석은 **modalities-map 모드**다. COCO 이미지 엔트리마다 `modalities` dict가 있고
  거기에 담긴 **DATA_ROOT 기준 상대경로**로 각 모달리티 파일을 찾는다. config의
  `MODALITY_KEYS`(`img→rgb`, `lidar→depth_map_lidar`, `thermal→thermal_aligned`)는
  모델의 모달리티 이름을 그 dict의 키로 매핑한다. 즉 **디렉터리 이름이 아니라 annotation이
  실제 경로의 단일 출처**이므로, 데이터를 옮길 때 `_final_ann/`과 이미지의 상대구조를 함께 유지해야 한다.
- `REQUIRE_ALL_MODALITIES: true` — 3모달이 **모두** 디스크에 있는 프레임만 사용한다
  (LiDAR 커버리지가 부분적이라 교집합만 남음). 인증 수치의 3,239장은 이 필터 적용 후 수치다.
- `file_name`은 `capture_YYYYMMDD_hhmmss/rgb/xxx.png` 형태이며, 첫 경로 조각이 **클립 ID**다.
  `check_split.py`의 clip-level 검사와 night/normal 분류가 이 규칙을 쓴다.
- night 클립 기본값: `capture_20260618_114021`, `capture_20260618_115624` (1,768장).
  `capture_20260618_114808`(1,471장)은 normal.

### 머신별 DATA_ROOT 지정

config(`configs/det/det_D1_vitsp_jarvis.yaml`)에는 jarvis 경로가 하드코딩돼 있다.
**config를 고치지 말고** 아래 인자로 갈아끼운다 — `DATASET.ROOT`와
`ANNOTATION_TRAIN/VAL`(`<root>/_final_ann/<원래 파일명>`)이 함께 재지정된다.

```bash
bash certification/run_cert.sh <ckpt> /my/mount/poongsan_v2 0     # run_cert.sh 2번째 인자
python certification/cert_eval.py   ... --data-root /my/mount/poongsan_v2
python certification/check_split.py ... --data-root /my/mount/poongsan_v2
```

`train_det.py` / `val_det.py` / `tools/*`에는 `--data-root`가 없다.
다른 머신에서 재학습할 때는 config를 복사해 `DATASET.ROOT`와 `ANNOTATION_*`만 바꿔 쓴다.

---

## 7. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| 백본 생성 실패 또는 `BACKBONE_FALLBACK`(DINOv2)으로 떨어짐 | **timm < 1.0.** 일부 서버는 낡은 timm이 shadow 한다. `/SSDb/jemo_maeng/pylibs_p34`가 있으면 `run_cert.sh`가 자동으로 `PYTHONPATH`에 prepend 한다. 수동: `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$PYTHONPATH` |
| `torch.cuda.is_available()` = False → CPU로 떨어져 FPS/VRAM 무의미 | **GPU driver/library mismatch.** drone-demo 박스에서 확인된 문제. 드라이버 복구(재부팅) 또는 venv cuDNN 경로를 `LD_LIBRARY_PATH`에 프리픽스. `cert_eval.py` 배너의 `Device` 줄로 확인 |
| protobuf / tensorboard 관련 에러 | `run_cert.sh`가 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`을 설정한다. 직접 실행할 때도 동일하게 export |
| 배너의 `load: missing=…` 이 0이 아님 | 체크포인트/config 불일치. md5와 config(`--cfg`)가 인증 조합인지 확인 |
| `annotation not found` (check_split) | `--data-root`가 잘못됐거나 `_final_ann/`이 없음 |
| mAP가 문서값과 다름 | `--eval-scope`(predicted vs annotation)와 `--limit/--stride` 사용 여부를 확인. 인증 수치는 **predicted-scope · 전수(3,239장)** 기준 |

---

## 8. 리포 구조 (가지치기 후)

```
.
├── README.md                       # 이 문서 (인증 담당자용 단일 진입점)
├── certification/                  # 🔵 인증 평가 경로
│   ├── run_cert.sh                 #   엔터 한 번 러너
│   ├── cert_eval.py                #   스트리밍 추론 + mAP/FPS/VRAM 리포트
│   ├── check_split.py              #   학습·평가 분리(누수) 검증
│   └── README.md                   #   세부 참조 (본 README의 보조)
├── configs/det/                    # det 학습/평가 config (ReliaDINO 계열 13개)
│   └── det_D1_vitsp_jarvis.yaml    #   ★ 인증 config
├── train_det.py                    # det 학습 (torchrun DDP 지원)
├── val_det.py                      # det 평가
├── tools/
│   ├── _det_common.py              #   빌드·로딩·추론·COCO 채점 공통 (인증과 동일 스택)
│   ├── det_eval_breakdown.py       #   night/normal + per-class mAP 분해
│   └── det_fps_bench.py            #   FPS 벤치
├── objdet/                         # 검출 스택 + YOLO 대조군
│   ├── models/det_model.py         #   ReliaDINORFDETRDetector 등 검출기 정의
│   ├── datasets/multimodal_det.py  #   멀티모달 COCO 데이터셋
│   ├── losses.py metrics.py utils/nms.py augmentations_det.py
│   ├── tools/                      #   diag_det.py, merge_coco.py
│   ├── yolov5m-lowlight/           #   대조군 A — YOLOv5m 저조도 레시피 ablation
│   ├── yolo11m-rgb/                #   대조군 B — YOLO11m RGB-only 기준점
│   └── gistolo/                    #   대조군 C — D1 teacher → RGB student distill
├── semseg/                         # 백본·헤드 (det가 import 하는 것만 잔존)
│   ├── models/reliadino/           #   ★ ReliaDINO 백본 (DINOv3 + LoRA + 신뢰도 게이트 융합)
│   ├── models/rfdetr_head/         #   ★ RF-DETR 헤드 (+ _vendor/ 벤더링 원본, LICENSE 포함)
│   ├── models/sam2/sam2/modules/   #   ReliaDINO가 재사용하는 융합/신뢰도 모듈
│   ├── models/{backbones,heads,layers,modules}/, datasets/, utils/, augmentations_mm.py
│   │                               #   semseg 패키지 __init__ 이 끌어오는 전이 의존 (삭제 시 import 붕괴)
├── scripts/
│   ├── pick_free_gpus.sh           #   빈 GPU 선택
│   └── build_det_splits.py, build_det_v3.py   # det split 생성
├── requirements.txt, conda_environment.yml, environment.yaml   # ⚠️ 구 연구 환경 스펙 (2.2절 참고)
└── .gitignore
```

`semseg/` 아래에 남은 seg 계열 모듈(`cmnext`, `heads/*`, `datasets/*` 등)은 **연구용이 아니라**,
`semseg.models` / `semseg.datasets` 패키지의 `__init__.py`가 무조건 import 하기 때문에
남긴 것이다. 지우면 ReliaDINO import 자체가 깨진다.

---

## 부록: 파일 단위 참조

- 인증 세부(수치 출처·YOLO 슬롯 요약) — `certification/README.md`
- YOLOv5m 저조도 실험 설계와 측정 근거 — `objdet/yolov5m-lowlight/README.md`
- YOLO11m 기준점 및 split 공정성 — `objdet/yolo11m-rgb/README.md`
