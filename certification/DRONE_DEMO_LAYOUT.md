# drone-demo — 공인인증 평가 & KD 실험 (D1 ViT-S+ / GISTOLO)

이 머신(drone-demo, RTX 5090)에서 국책 R&D 공인인증 정량평가와 크로스모달 KD 실험을
수행한다. **모든 자산이 이 폴더 하나 아래에 있다.** (2026-08-07 홈 직하에 흩어져 있던
것을 여기로 정리)

```
~/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM-cert/
├── repo/          코드 (git, branch 26-drone-certificate)
├── datasets/      데이터셋 74G
├── outputs/       실험 산출물 3.3G
├── logs/          과거 실행 로그 67개
└── third_party/   yolov5 clone
```

## 바로 실행 — 공인인증 정량평가

```bash
cd ~/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM-cert/repo
bash certification/RUN_ON_DRONE_DEMO.sh
```
인자 없이 실행하면 된다(경로·conda 환경이 스크립트에 박혀 있음).
데이터/GPU를 바꿀 때만 `bash certification/RUN_ON_DRONE_DEMO.sh <DATA_ROOT> <GPU>`.

| 항목 | 지표 | 반복 |
|---|---|---|
| ① 표적위치 정확도 | mAP50 | 5회 |
| ② 영상 열화시 표적인식 성공률 | mAP50(야간) | 5회 |
| ③ 피아식별 정확도 | 분류 accuracy | 2회 (가중치 고정, 평가만) |

산출물은 `repo/runs/cert_all_<타임스탬프>/` — `summary.txt`(한눈 요약) ·
`console.log` · `det_trial1~5/`(회차별 리포트·추론로그·GT|Pred 오버레이) ·
`iff/`(리포트 + 정답 파랑/오답 빨강 시각화). 소요 약 20분.

평가 중 GT vs 예측 비교 창이 뜬다(디스플레이 자동 감지, `SHOW=0` 으로 끔).

## 모델

| 용도 | 모델 | 가중치 |
|---|---|---|
| 검출 (①②) | ReliaDINO ViT-S+ (3-modal) + RF-DETR NMS-free | `repo/weights/det_D1_vitsp_20260723/best_checkpoint.pth` |
| 피아식별 (③) | MobileNetV3-small (128px 크롭, Allies vs Enemies) | `repo/weights/iff_mobilenetv3.pt` |
| KD teacher | ReliaDINO ViT-L (계보 최고 정확도, 배포 불가) | `repo/weights/det_D1_recovered_20260723/best_checkpoint.pth` |

## 데이터셋

### 평가용 (이 머신에 있음)
| 경로 | 내용 |
|---|---|
| `datasets/poongsan_v2_test3modal/` | **공인인증 평가 데이터** 3,239장 · 3클립 · rgb/lidar/thermal |
| `datasets/poongsan_v2_test3modal/_final_ann/instances_test_common.json` | 평가 어노테이션 (야간 1,768 / 주간 1,471, 10클래스) |
| `datasets/poongsan_iff_crops/` | 피아식별 크롭 — train 8,459 / test 2,141 |

어노테이션의 `depth_map_lidar` 키는 프레임별로 raw(2,066) 또는
`depth_map_lidar_egofill`(1,173) 를 가리킨다. 그래서 3모달 완비 = **3,239장 전부**이고,
`REQUIRE_ALL_MODALITIES: true` 로도 전량 평가된다.

### 학습용 (이 머신에는 3-modal 서브셋만)
| 경로 | 내용 |
|---|---|
| `datasets/poongsan_v2_train3modal/` | 학습 프레임 중 **3모달 완비 7,043장** (KD teacher 특징 추출용) |
| `datasets/poongsan_v2_yolo_rgb/` | YOLO 포맷 RGB — train 12,681 / test 3,239 |
| `datasets/poongsan_v2_yolo_rgb_modal/` | 위의 3모달 완비 서브셋 — train 7,043 / test 2,066 |

**전체 학습 데이터 원본(12,681장 3-modal)은 이 머신에 없다.** 정본 위치는 아래 참조.

### KD 실험 파생물
`datasets/gistolo_teacher_feats{,_ms,_80}/` teacher 특징 · `poongsan_v2_yolo_rgb_gistolo_*`
증류 라벨 데이터셋 · `poongsan_rfdetr/` RF-DETR 학습 포맷.

## 원본 데이터 정본 (다른 서버/NAS)

| 위치 | 내용 |
|---|---|
| `jarvis:/SSDd/jemo_maeng/dset/poongsan_v2/` | **학습·평가 원본 전체** (train 12,681 + test 3,239, 전 모달) |
| `/ailab_mat2/personal/jemo_maeng/dset/poongsan_v2_yolo_rgb{,_modal}/` | YOLO 포맷 공유본 (모든 서버에서 접근) |
| `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/gistolo/` | KD·GISTOLO 산출물 정본 (가중치·분석·시각화·README) |

## 환경

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate drone_cert   # 인증 검출 (ReliaDINO, torch cu128)
conda activate drone_yolo   # YOLO 학습·평가
conda activate rfdetr       # RF-DETR 실험
```
RTX 5090(Blackwell sm_120)이라 **torch 는 cu128 빌드여야 한다.** `+cpu` 로 깔리면
`torch.cuda.is_available()` 이 False 가 된다 — 드라이버 문제로 오인하기 쉬우니
`nvidia-smi` 가 GPU 를 잡으면 torch 빌드부터 의심할 것.

## 코드

GitHub `gist-ailab/drone-MemorySAM` branch `26-drone-certificate`
- `certification/` 인증 평가 (cert_eval · iff_eval · run_cert_all · RUN_ON_DRONE_DEMO)
- `objdet/gistolo/` 크로스모달 KD (teacher 특징 추출 · feature KD 학습 · 비교 시각화)
