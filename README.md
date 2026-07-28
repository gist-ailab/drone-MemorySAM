# drone-MemorySAM — 멀티모달 세그멘테이션 & 검출 연구 리포

SAM2/SAM3의 시간축 memory attention을 **모달리티 축 Cross-Modal Fusion**으로 전용하고, 그 위에
**RBMA (Reliability-Biased Memory Attention)** — training-free reliability를 memory-attention logit에
additive bias로 가산 — 를 얹는 멀티모달 인식 연구 코드베이스다.

> 이 리포는 upstream [MemorySAM (HKUST, arXiv 2503.06700)](https://arxiv.org/abs/2503.06700)에서 출발해
> 전면 개작한 연구 fork다. 원본 README는 [`_archive/upstream_MemorySAM_README.md`](_archive/upstream_MemorySAM_README.md) 참조.

## 연구 트랙

| 트랙 | 목표 | 벤치마크 | 현재 최선 |
|------|------|----------|-----------|
| **Seg — 챌린지** | MACVi MULTIAQUA Challenge (야간 수상 드론) | MULTIAQUA, M-score | **82.10** (P9 ep131 / P22 ep120 공동 1위) |
| **Seg — 논문** | RBMA / ReliaDINO 논문 publish | DELIVER (목표 val ≥66.51 / test ≥56.71), MUSES | DELIVER **68.19 / 56.62** (P34) · MUSES 공식 test **79.788** (P39.1-seed2) |
| **Det — 국가 R&D** | poongsan indoor 멀티모달 검출 | COCO mAP50 (목표 0.85) | **0.8501** (P29-Det egofill, ⚠️ ckpt 소실) |

수치 출처와 조건은 아래 [데이터셋별 실험 가이드](#-데이터셋별-실험-가이드)의 각 "대표 성적" 표에 문서 링크와 함께 적어 두었다.

---

## 빠른 시작

```bash
conda activate MMSS_SAM                    # 또는 /home/jemo/anaconda3/envs/MMSS_SAM/bin/python
# 환경 복원이 필요하면: conda env create -f conda_environment.yml

# 대표 체크포인트의 정량 지표 재현 (bench ∈ deliver|muses|muses-official|multiaqua|det)
bash scripts/reproduce_eval.sh deliver
```

- **정량 재현의 단일 출처 = [`REPRODUCE.md`](REPRODUCE.md)** — 전제조건, 실측 패키지 버전, 데이터/ckpt 경로,
  벤치별 기대 수치, 트러블슈팅, **구조적으로 재현 불가한 항목**까지 전부 거기 있다. 이 README는 중복하지 않는다.
- **세션 규칙·명령어 canonical = [`CLAUDE.md`](CLAUDE.md)** (에이전트 공통 지침은 [`AGENTS.md`](AGENTS.md)).
- **연구 로그 front door = [`.claude_logs/00_INDEX.md`](.claude_logs/00_INDEX.md)** → 현재 스냅샷은
  [`.claude_logs/status/current.md`](.claude_logs/status/current.md).
- **구조 유지 규칙(브랜치·문서·코드·config 배치) = [`.claude_logs/meta/conventions.md`](.claude_logs/meta/conventions.md)** —
  파일 생성·코드 추가 전 필독.

> 🔴 **어떤 학습이든 실행 전 빈 GPU 확인이 필수다.** 판정 기준(`memory.used ≤ 2000MiB && util ≤ 10%`)과
> 자동 선택 방법은 [원격 서버에서 돌리기](#원격-서버에서-돌리기) 절 참조.

---

## 🔴 데이터셋별 실험 가이드

이 리포는 **4개 벤치마크**를 돌리며, 벤치마크마다 **모델 계열과 진입 스크립트가 다르다**. 먼저 이 표로 방향을 잡아라.

| 데이터셋 | 모달 | 클래스 | 모델 계열 | 학습 진입점 | 평가 진입점 |
|---|---|---|---|---|---|
| **DELIVER** | 4모달 img/depth/event/lidar | 25 | ReliaDINO (P34~P45) · SAM2 LoRA (P27~P33) | `train_reliadino.py` / `train_sam2_lora_paper.py` | `tools/eval_reliadino_ckpt.py`, `tools/eval_per_domain.py`, `val.py` |
| **MUSES** | 3모달 img/lidar/event (+radar 변형) | 19 | ReliaDINO (P34~P45) | `train_reliadino.py` | `tools/eval_reliadino_ckpt.py`, `tools/eval_muses_official.py`, `tools/predict_muses_test.py` |
| **MULTIAQUA** | 3모달 img/lidar/thermal | 4 | SAM2 LoRA (P8~P28) | `train_sam2_lora_paper.py` | `val_multiaqua.py`, `val.py` |
| **poongsan (det)** | 3모달 rgb/lidar-depth/thermal | 10 | ReliaDINO 백본 + det head (D1/P37~P39) · SAM2 P29~P31-Det | `train_det.py` | `val_det.py`, `tools/det_analysis_pipeline.py` |

> ⚠️ **모달 수 표기 의무** ([conventions.md](.claude_logs/meta/conventions.md) §모달 수 표기): 같은 모델도 모달 수가 다르면
> 별개 실험이다. 예 — P34 MUSES 3모달 test 78.979 vs 4모달 78.256 (0.72 차이).

---

### 1. DELIVER — 4모달 도시 주행 세그멘테이션 (논문 트랙 주 벤치)

**무엇을 푸는가**: CARLA 기반 합성 멀티모달 주행 데이터셋. RGB + depth + event + LiDAR 4모달을 융합해
25클래스 semantic segmentation을 수행하고, **날씨/조도 5조건(cloud·fog·night·rain·sun)별 robustness**를 본다.
논문 트랙의 헤드라인 벤치이며 목표는 **val ≥66.51 / test ≥56.71**([status/current.md](.claude_logs/status/current.md) 공식 목표).

**모달리티와 디렉터리 이름** (로더 [`semseg/datasets/deliver.py`](semseg/datasets/deliver.py) 기준):

| config `MODALS` 이름 | 실제 디렉터리 | 비고 |
|---|---|---|
| `img` | `img/` | RGB, 파일 접미사 `_rgb` |
| `depth` | **`hha/`** | ⚠️ 로더는 `/img`→`/hha`, `_rgb`→`_depth`로 치환한다. `depth/` 디렉터리도 별도로 존재하지만 **로더가 읽는 것은 `hha/`** |
| `event` | `event/` | 해상도가 달라 RGB 크기로 NEAREST 리사이즈 |
| `lidar` | `lidar/` | |
| (GT) | `semantic/` | 원본 1~25 → 로더가 `-1`로 0~24 변환, 0은 underflow로 255(ignore) |

**클래스 25종**: Building, Fence, Other, Pedestrian, Pole, RoadLine, Road, SideWalk, Vegetation, Cars, Wall,
TrafficSign, Sky, Ground, Bridge, RailTrack, GroundRail, TrafficLight, Static, Dynamic, Water, Terrain,
TwoWheeler, Bus, Truck (ignore = 255).

**데이터 경로 (이 허브 박스, 2026-07-28 실측)**: `/ailab_mat2/dataset/DELIVER` — ✅ **마운트됨**.

```
/ailab_mat2/dataset/DELIVER/
├── img/     <condition>/<split>/<seq>/*.png      # condition ∈ {cloud, fog, night, rain, sun}
├── hha/     <condition>/<split>/<seq>/*.png      #   split    ∈ {train, val, test}
├── depth/   <condition>/<split>/<seq>/*.png      #   예: img/cloud/val/MAP_4_point12/157550_rgb_front.png
├── event/   <condition>/<split>/<seq>/*.png
├── lidar/   <condition>/<split>/<seq>/*.png
└── semantic/<condition>/<split>/<seq>/*.png
```

**split 장수 (실측 glob)**:

| split | 전체 | cloud | fog | night | rain | sun |
|---|---|---|---|---|---|---|
| train | 3,983 | — | — | — | — | — |
| val | 2,005 | 398 | 400 | 410 | 398 | 399 |
| test | 1,897 | 379 | 379 | 379 | 380 | 380 |

⚠️ 리포에 커밋된 config의 `DATASET.ROOT`는 **학습을 돌린 서버의 로컬 경로**(`/NHNHOME/.../DELIVER`,
`/SSDb/jemo_maeng/dset/DELIVER` 등)라 이 박스엔 없다. 평가 시 `--dataset-root` 또는 `DATA_ROOT=`로 덮어써라.

#### 학습

```bash
conda activate MMSS_SAM

# (A) 로컬 — 빈 GPU 4장 자동 선택 후 DDP
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh 4)" \
torchrun --nproc_per_node=4 train_reliadino.py \
  --cfg configs/hpca100-deliver_rgbdel_P43_pdual.yaml

# (B) 원격 — tmux 세션 'jemo' 새 창, 원격 빈 GPU 4장 자동 배정
bash scripts/remote_exp.sh status hpca100                     # 먼저 빈 GPU 확인
bash scripts/remote_exp.sh run hpca100 \
  configs/hpca100-deliver_rgbdel_P43_pdual.yaml auto:4 4 train_reliadino.py
bash scripts/remote_exp.sh log hpca100 hpca100-deliver_rgbdel_P43_pdual
```

> ℹ️ **진입점 auto-detect (2026-07-28 수정됨)**: `remote_exp.sh`는 이제 **파일명이 아니라 config 내용**으로
> 진입점을 고른다 — `MODEL.DET_MODEL`이 있으면 `train_det.py`, `MODEL.NAME: ReliaDINO`면 `train_reliadino.py`,
> 그 외는 파일명 폴백(SAM3/RBMA → `train_sam3_rbma.py`, 기본 `train_sam2_lora_paper.py`).
> 이전에는 `*P34*|*P35*|*P36*` 이름만 매칭해서 **P37 이후 ReliaDINO config와 det config 60개가
> `train_sam2_lora_paper.py`로 잘못 라우팅**됐다. 위 예시처럼 5번째 인자로 진입점을 직접 넘기는 것도 계속 유효하다
> (그러려면 gpus·nproc도 함께 줘야 한다). 실행 시 `[remote_exp] entry auto-detect -> …` 로 선택 결과가 찍히니 확인하라.

SAM2 계열(P27~P33) DELIVER 학습은 진입점이 다르다:

```bash
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh 4)" \
torchrun --nproc_per_node=4 train_sam2_lora_paper.py \
  --cfg configs/deliver/b200-deliver_rgbdel_P32_physaug.yaml
```

#### 평가

```bash
# val / test mIoU — 학습 eval 경로를 그대로 재사용(프로토콜 동일)
python tools/eval_reliadino_ckpt.py \
  --cfg  configs/b200-deliver_rgbdel_P34_reliadino.yaml \
  --ckpt <NAS>/ckpts/P34_final_20260713/epoch120_68.19_top1_checkpoint.pth \
  --split both --gpu 0 --dataset-root /ailab_mat2/dataset/DELIVER

# 5조건(per-condition) 분해 — val.py를 조건별로 호출
python tools/eval_per_domain.py \
  --cfg configs/b200-deliver_rgbdel_P34_reliadino.yaml \
  --ckpt ep120=<ckpt.pth> \
  --dataset-root /ailab_mat2/dataset/DELIVER \
  --split test --gpu 0 --out-dir <out>

# 모듈 토글 즉검 (no-op/미배선 조기 검출 — 기동 후 ep30 이내 의무)
python tools/module_ablation.py --help
```

DELIVER는 test GT가 공개돼 있어 **로컬에서 test mIoU까지 나온다**(제출 서버 없음).

#### config 위치와 이름 규칙

| 대상 | 위치 | 예 |
|---|---|---|
| ReliaDINO 계열 (P34~P45) 학습 | **`configs/` 루트** | `b200-deliver_rgbdel_P34_reliadino.yaml`, `hpca100-deliver_rgbdel_P43_pdual.yaml` |
| SAM2 계열 (P4~P33) 학습 | `configs/deliver/` | `b200-deliver_rgbdel_P32_physaug.yaml` |
| 평가 전용 (MODEL_PATH 포함) | `configs/eval/` | `levine-deliver_rgbdel_P26_physaug.yaml` |
| 데드 실험 | `configs/archive/` | |

이름 규칙은 `<dataset>_<modal>_<version>_<aug>.yaml`이고 **신규 config에는 서버 접두어를 붙이지 않는다**
([`configs/README.md`](configs/README.md), [conventions.md](.claude_logs/meta/conventions.md) §4).
`b200-`/`hpca100-`/`jarvis-` 접두어는 기존 파일에만 남은 레거시 관행이다.
⚠️ 실제로 ReliaDINO DELIVER/MUSES config는 `configs/deliver/`가 아니라 `configs/` 루트에 서버 접두어를 달고 쌓여 있다 —
규칙과 어긋난 상태이며, 기존 파일명은 output 디렉터리 매핑 보존 때문에 **바꾸지 않는다**.

#### 대표 성적 (legal = val-best 선택만)

| 모델 | val-best mIoU | 그 에폭 test mIoU | 출처 |
|---|---|---|---|
| **P34-ReliaDINO** (최선) | **68.19** @ep120 | **56.62** | [status/current.md](.claude_logs/status/current.md) legal 표, [REPRODUCE.md §4.1](REPRODUCE.md) |
| P36-router | 67.74 @ep52 | 55.62 | 위와 동일 |
| P35-paper | 67.61 @ep78 | 55.52 | 위와 동일 |
| P38-MaskQueryLite | 65.19 @ep28 | 55.05 @ep62(test-best) | [experiments/registry.md](.claude_logs/experiments/registry.md) |
| P39-DPC | 65.68 @ep64 | test 5-cond 평균 50.98 (val↔test 역전) | [analysis/2026-07-20-p39-deliver-3ckpt-compare.md](.claude_logs/experiments/analysis/2026-07-20-p39-deliver-3ckpt-compare.md) |
| P32-CoRB (SAM2) | 64.12 @ep98 | 55.00 | [status/current.md](.claude_logs/status/current.md) |
| P29 (SAM2 RBMA) | 63.20 @ep100 | 54.34 @ep146 | [registry.md](.claude_logs/experiments/registry.md) |

> 🔴 **test-best ckpt(`test_epoch*_*.pth`) 수치는 인용 금지** — test셋 훔쳐보기라 논문 불가(2026-07-15 판정).
> "P34 test 57.60"은 이 이유로 **철회된 수치**다.

#### 이 데이터셋 특유의 함정

- **ISSUE-026 (ColorAugSSD brightness)** — uint8 입력을 [0,1]로 클램프해 발화 샘플의 RGB가 백색 상수로 붕괴.
  **2026-07-16 이후 `DGFUSION_AUG: true`로 돌린 DELIVER 학습이 전부 오염**됐고(P37a/P37b/P38-DELIVER/P39-DPC resume),
  해당 게이트 판정은 보류 상태다. MUSES 계보는 무영향. 2026-07-21 수정.
  ([issues-and-fixes.md](.claude_logs/issues/issues-and-fixes.md))
- **ISSUE-027 (`GRADIENT_CHECKPOINT: true` 금지)** — timm non-reentrant 재계산이 stale `active_modality`로
  비최종 모달의 LoRA gradient를 **무경고로 오염**시킨다. encoder에 강제 off 가드가 들어갔고 체크인 config 9종을 false로 정정.
- **val↔test 순위 역전** — P39-DPC에서 val 최고(65.68@ep64)가 test 5조건 평균 최저(50.98)였다. val 단독 판정 금지.
- **thin-class 붕괴** — Wall/Water/RailTrack/Bridge/Other가 반복적으로 IoU 0 근처로 죽는다. 판정 게이트에
  thin-class 조건(Wall≥13 / Water≥9.5 / RailTrack≥62)이 사전 등록돼 있다([registry.md](.claude_logs/experiments/registry.md)).
- **PhysAug는 공정 비교에서 배제** — 2026-07-20 user 판정. 헤드라인 비교표는 physaug-off 계열(P35/P36 fair·P38·P39)로만 구성한다.
- **공개 리더보드와 직접 비교 불가** — 우리 eval은 1024² letterbox 프로토콜이고, DELIVER 공식 GT 리사이즈 규약과
  동일한지 **미검증**이다([REPRODUCE.md §6](REPRODUCE.md)).

---

### 2. MUSES — 3모달 악천후 주행 세그멘테이션 (현행 최고 성적 트랙)

**무엇을 푸는가**: ETH Zurich의 실촬영 멀티센서 벤치. 날씨 4종 × 주야 2종 = **8개 조건 셀**에서 Cityscapes
19클래스 semantic segmentation을 수행한다. test GT는 비공개이며 **Codabench(대회 14005) 서버가 채점**한다.

**모달리티와 디렉터리 이름** (로더 [`semseg/datasets/muses.py`](semseg/datasets/muses.py) 기준):

| config `MODALS` 이름 | 실제 경로 | 인코딩 |
|---|---|---|
| `img` | `frame_camera/<split>/<weather>/<tod>/<stem>_frame_camera.png` | RGB uint8, 1080×1920 |
| `lidar` | `projected_to_rgb/lidar/.../<stem>_lidar.png` | uint16 PNG, `value = encoded/150 − 100`, 채널 [range, intensity, height] |
| `event` | `projected_to_rgb/event_camera/.../<stem>_event_camera.png` | uint8 [pos, neg, 0], log1p 압축 |
| `radar` | `projected_to_rgb/radar/.../<stem>_radar.png` | uint16, 채널 [range/150, intensity/255, occupancy] |
| (GT) | `gt_semantic/<split>/<weather>/<tod>/<stem>_gt_labelTrainIds.png` | trainId 0~18, 255=ignore |

**클래스 19종 (Cityscapes trainId)**: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.
**MUSES에는 depth 모달이 없다** — 로더가 `depth` 요청 시 명시적으로 거부한다.

**데이터 경로 (이 허브 박스, 2026-07-28 실측)**: `/ailab_mat2/dataset/MUSES` — ✅ **마운트됨**.

```
/ailab_mat2/dataset/MUSES/
├── frame_camera/    <split>/<weather>/<tod>/*.png     # weather ∈ {clear, fog, rain, snow}
├── gt_semantic/     <split>/<weather>/<tod>/*.png     # tod     ∈ {day, night}
├── projected_to_rgb/{lidar, event_camera, radar}/<split>/<weather>/<tod>/*.png
├── projected_to_rgb_dgf/                              # DGFusion 투영본 (비교용)
├── lidar/  radar/  event_camera/  gnss/               # 원시 센서
├── gt_panoptic/  gt_detection/  gt_uncertainty/       # 타 태스크 GT
└── calib.json  meta.json  tags_{train,val,test}.csv
```

**split 장수 (실측 glob)**:

| split | frame_camera | gt_semantic (`*_gt_labelTrainIds.png`) |
|---|---|---|
| train | 1,500 | 1,500 |
| val | 250 | 250 |
| test | 750 | **0 — 벤치마크가 GT를 withhold** |

로더는 `split='test'`에서 **의도적으로 `FileNotFoundError`를 던진다**(트레이너가 이걸 잡아 `testset=None`으로 처리).
test 채점은 Codabench 제출로만 가능하다.

#### 학습

```bash
# (A) 로컬
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh 4)" \
torchrun --nproc_per_node=4 train_reliadino.py \
  --cfg configs/jarvis-muses_rgbel_P39_1_rank.yaml

# (B) 원격 — auto-detect가 config 내용(MODEL.NAME: ReliaDINO)으로 train_reliadino.py를 고른다. 명시해도 무방
bash scripts/remote_exp.sh status jarvis
bash scripts/remote_exp.sh run jarvis \
  configs/jarvis-muses_rgbel_P39_1_rank.yaml auto:4 4 train_reliadino.py
bash scripts/remote_exp.sh log jarvis jarvis-muses_rgbel_P39_1_rank
```

#### 평가 — 프로토콜이 **두 가지**다

```bash
# (1) 학습 내부 프로토콜: 1024² letterbox 해상도에서 confusion matrix 누적
python tools/eval_reliadino_ckpt.py \
  --cfg configs/jarvis-muses_rgbel_P39_1_rank.yaml --ckpt <ckpt.pth> \
  --split val --gpu 0 --dataset-root /ailab_mat2/dataset/MUSES

# (2) 공식 native 프로토콜: logit에서 letterbox 패딩 crop → 1080×1920 업샘플 → argmax
python tools/eval_muses_official.py \
  --cfg configs/jarvis-muses_rgbel_P39_1_rank.yaml --ckpt <ckpt.pth> \
  --gpu 0 --dataset-root /ailab_mat2/dataset/MUSES --out <out_dir>
#   → 두 히스토그램(hist_1024 / hist_full)을 한 번의 forward에서 동시 산출해 사과-대-사과 비교

# (3) Codabench test 제출 PNG 생성 + 하드 검증
python tools/predict_muses_test.py \
  --cfg <cfg.yaml> --ckpt <ckpt.pth> --gpu 0 \
  --dataset-root /ailab_mat2/dataset/MUSES --out <pred_dir>
python tools/verify_submission.py <pred_dir> /ailab_mat2/dataset/MUSES/frame_camera/test
```

`bash scripts/reproduce_eval.sh muses` / `muses-official`이 (1)/(2)를 대표 ckpt로 한 줄 실행한다.

#### config 위치와 이름 규칙

MUSES 학습 config는 **`configs/` 루트**에 서버 접두어를 달고 있다(`configs/muses/` 디렉터리는 존재하지 않는다):
`jarvis-muses_rgbel_P39_1_rank.yaml`(3모달), `hpca100-muses_rgbelr_P34_reliadino.yaml`(4모달 = +radar, `rgbelr`),
`hpca100-muses_rgbel_P43_pdual.yaml` 등. `<modal>` 축약은 `rgbel` = img+event+lidar, `rgbelr` = +radar.

#### 대표 성적

| 모델 | val mIoU (내부 1024²) | **공식 test mIoU (Codabench)** | 출처 |
|---|---|---|---|
| **P39.1-rank seed2** (최고) | **82.62** @ep208 | **79.788** — MUSES test-best | [experiments/log.md §2026-07-27](.claude_logs/experiments/log.md) |
| P38-m2f | 82.22 @ep156 | 79.025 | [status/current.md](.claude_logs/status/current.md), [REPRODUCE.md §4.2](REPRODUCE.md) |
| P34-ReliaDINO 3모달 | 81.02 @ep276 (공식 native 80.86) | 78.979 | [status/current.md](.claude_logs/status/current.md) |
| P34-ReliaDINO 4모달(+radar) | 기록 없음 | 78.256 | [conventions.md](.claude_logs/meta/conventions.md) §모달 수 표기 |
| P39-DPC 3모달 | 81.52 @ep146 | 78.881 | [registry.md](.claude_logs/experiments/registry.md) |
| P39.1-rank 5-seed variance | 81.70 / 81.89 / 81.92 / 82.03 / 82.62 (평균 **82.03**) | — | [status/current.md](.claude_logs/status/current.md) |

**P39.1-seed2 per-condition 공식 test** ([log.md §2026-07-27](.claude_logs/experiments/log.md)):
clear 79.300 · fog 78.705 · rain 79.063 · snow 79.042 · day 80.246 · night 76.818 ·
**fog_night 69.610 (전 조합 최악)** · snow_day 71.155 < snow_night 77.413 (**snow 역전, 3회째 재현**).

참고 SOTA: mIoU 1위 GtA 82.39(카메라 단독, 미발표) — 우리와 −2.60. PQ 1위 DGFusion 61.03
([decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](.claude_logs/decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md)).

#### 이 데이터셋 특유의 함정

- **ISSUE-025 (radar 디코딩 3중 버그)** — `_open_radar` 폴스루 + 디스패치 오배선 + `RADAR_RANGE_MAX` 미정의로
  100m 클립(실측 센서 캡은 150.0m, 유효픽셀 2.76% 포화) + height 채널이 0.25 상수로 오염됐다.
  **radar를 포함한 4모달 실험만 영향**, 3모달 전 계보는 무영향. 2026-07-21 수정.
- **uint16 PNG는 `cv2.IMREAD_UNCHANGED`로만 읽어야 한다** — `torchvision.io.read_image()`는 UInt16 텐서를 내놓는데
  연산이 미구현이고, 8비트 캐스트는 range/height 정밀도를 파괴한다(로더 docstring에 명시).
- **letterbox 필수** — 1080×1920(16:9)을 정사각으로 패딩하지 않으면 val Resize가 1024×1820을 내고
  1820이 ViT patch 16으로 나눠지지 않는다. 라벨은 ignore=255로 패딩된다.
- **test GT 비공개** — 로컬에서 test mIoU를 계산할 방법이 없다. 모든 test 수치는 Codabench 채점 결과다.
- **fog_night이 구조적 병목** — 전 계보에서 최저 조건. P39.1-seed2 기준 69.61.
- **`snow_day < snow_night` 역전이 3회 재현**됐다(P34-4모달, P38-m2f, P39.1-seed2). 원인 미규명.

---

### 3. MULTIAQUA — 3모달 야간 수상 드론 세그멘테이션 (MACVi Challenge)

**무엇을 푸는가**: 드론이 촬영한 **야간 수상 환경**에서 RGB + LiDAR + Thermal 4클래스 세그멘테이션.
**train/val은 주간, test는 야간만**이라 도메인 갭이 이 벤치의 본질이다.
평가지표 **M-score = 0.75 × val_mIoU + 0.25 × test_mIoU**, 채점은 MACVi 챌린지 서버.

**모달리티와 디렉터리 이름** (로더 [`semseg/datasets/multiaqua.py`](semseg/datasets/multiaqua.py) 기준):

| config `MODALS` 이름 | 실제 경로 | config 키로 교체 가능 |
|---|---|---|
| `img` | `MULTIAQUA_night/data/zed/<stem>.png` | `DATASET.RGB_SUBROOT` (기본 `zed`) |
| `lidar` | `MULTIAQUA_night/data/lidar_processed/<stem>.png` | `DATASET.LIDAR_SUBROOT` |
| `thermal` | `MULTIAQUA_night/data/thermal_processed/<stem>.png` | `DATASET.THERMAL_SUBROOT` |
| (GT) | `MULTIAQUA_night/annotations/<stem>.png` | |

**클래스**: annotation 원본은 `0=Recording Boat(학습 시 ignore)`, `1=Static`, `2=Dynamic`, `3=Water`, `4=Sky`.
학습/평가 클래스 수는 **4** (Static, Dynamic, Water, Sky).

**데이터 경로 (이 허브 박스, 2026-07-28 실측)** — **두 루트가 모두 마운트돼 있다**:

| 루트 | 상태 | 쓰는 곳 |
|---|---|---|
| `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night` | ✅ 마운트됨 | [`CLAUDE.md`](CLAUDE.md) 프로젝트 개요가 가리키는 경로, 로더 기본값 |
| `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2` | ✅ 마운트됨 | **현행 최선 P9 config의 `DATASET.ROOT`**, `reproduce_eval.sh multiaqua` 기본값 |

```
<ROOT>/
├── train.txt / val.txt / test.txt        # 한 줄에 stem 하나
└── MULTIAQUA_night/
    ├── annotations/<stem>.png            # 3,098개 (양 루트 동일)
    └── data/
        ├── zed/                          # RGB (기본)
        ├── lidar_processed/  lidar/      # LiDAR
        └── thermal_processed/  thermal_camera/
```

⚠️ `_night2`에는 야간 변환·증강 실험용 RGB/thermal 파생 서브루트가 더 많다
(`zed_day`, `zed_night_to_day`, `zed_aug_night`, `thermal_processed_fieldscale*` 등) —
`RGB_SUBROOT`/`THERMAL_SUBROOT`로 골라 쓴다.

**split 장수 (`*.txt` 실측, 양 루트 동일)**: train **2,952** / val **145** / test **200**.
val은 주간 145장, test는 야간 200장이며 **test GT는 없다**(챌린지 서버 채점).

#### 학습

```bash
# (A) 로컬 — run_sam.sh는 빈 GPU를 자동 선택한다. 돌릴 config는 스크립트 안 CFG= 를 고쳐 지정한다
NGPU=4 bash run_sam.sh

# (A') 로컬 — config를 직접 주고 싶으면 torchrun으로
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh 4)" \
torchrun --nproc_per_node=4 train_sam2_lora_paper.py \
  --cfg configs/multiaqua/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml

# (B) 원격 — auto-detect가 train_sam2_lora_paper.py를 고른다
bash scripts/remote_exp.sh status levine
bash scripts/remote_exp.sh run levine \
  configs/multiaqua/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml auto:4
bash scripts/remote_exp.sh log levine levine-multiaqua_rgbtl_P9_hardaug8_physaug

# 단일 GPU 폴백
python train_sam2_lora_paper_singlegpu.py --cfg <config>
```

#### 평가 / 챌린지 제출

```bash
# val (주간)
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode val --model_path <ckpt>

# test (야간) + MACVi 제출 마스크 생성 (--macvi = eval_macvi/ 에 1-indexed 마스크 저장)
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode test --model_path <ckpt> --macvi

# 통합 평가 스크립트(MULTIAQUA/DELIVER/MUSES 공용)도 같은 인자를 받는다
python val.py --cfg configs/eval/<config>.yaml --mode val --model_path <ckpt>

# 이미 뽑은 0-indexed 마스크를 사후 변환할 때
python scripts/convert_for_macvi_submission.py --input_dir <pred/seg> --output_dir <pred_macvi>
```

#### config 위치와 이름 규칙

| 대상 | 위치 |
|---|---|
| 학습 (활성: P8·P9·P22·P28·SAM3RBMA·LoRASam) | `configs/multiaqua/` |
| 학습 (데드: P10~P21, P23~P26) | `configs/archive/` |
| 평가 (`MODEL_PATH` 포함) | `configs/eval/` — 학습 config와 **같은 파일명** |

`<modal>` 축약 `rgbtl` = img + thermal + lidar. `<aug>`는 `hardaug4`/`hardaug8`/`physaug` 등 증강 프리셋.

#### 대표 성적 (챌린지 서버 채점)

| 순위 | 모델 | Config | Val mIoU | Test mIoU | **M-score** | 제출 ID |
|---|---|---|---|---|---|---|
| **1** | **P9** | hardaug8_physaug (ep131, 재제출) | 93.29 | 70.91 | **82.10** | 16710 |
| **1** | **P22** | hardaug8_physaug (ep120) | 93.42 | 70.77 | **82.10** | 16932 |
| — | P9 | hardaug8_physaug (ep131, 원본) | 93.54 | 70.41 | 81.98 | 16683 |
| — | P21 | hardaug8_physaug (ep94) | 93.17 | 70.36 | 81.77 | 16792 |
| 2 | P9 | hardaug4 | 93.32 | 69.62 | 81.47 | 15635 |
| 3 | P13 | hardaug4 | 92.45 | 69.98 | 81.21 | 15997 |
| 4 | P10 | hardaug4 | 93.23 | 65.30 | 79.27 | 15731 |
| 5 | P8 | hardaug (기본) | 92.96 | 63.93 | 78.45 | 15561 |
| 9 | P11 | hardaug4 | 93.17 | 61.01 | 77.09 | 15851 |

전체 순위표(23위까지 + TTA·day-translation 변형)는 [`experiments/log.md`](.claude_logs/experiments/log.md) "전체 결과 요약(M-score 순)".
최선 ckpt = `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth`
([registry.md](.claude_logs/experiments/registry.md)).

#### 이 데이터셋 특유의 함정

- **🔴 Val↔Test 도메인 갭이 전부다** — Val(주간) 93~94% vs Test(야간) 58~71%. **모든 모델이 이 갭을 보인다.**
  val 점수로 모델을 고르면 야간에서 무너진다.
- **MACVi 서버는 1-indexed 마스크를 기대한다** — 로컬 모델은 0-indexed(0=Static…)를 출력한다.
  변환을 빼먹으면 리더보드 mIoU가 7.9%로 찍힌다(실제 사고). `--macvi` 또는 `convert_for_macvi_submission.py` 사용.
- **체크포인트 포맷 차이** — `_checkpoint.pth` = `{'model_state_dict', 'optimizer_state_dict', ...}` dict를
  `val_multiaqua.py`가 기대하고, raw `.pth`(state_dict 그 자체)는 `val_multiaqua_P9.py`가 직접 로드한다.
  현행 진입점(`val.py`, `tools/eval_reliadino_ckpt.py`, `val_det.py`)은 `ckpt.get('model_state_dict', ckpt)`로 둘 다 받는다.
- **MoE Gate "uniform" 문제는 측정 artifact** — 공간 평균(`_gate_callback`)으로는 uniform으로 보이지만
  per-token 분석에서는 실제로 분화돼 있다(entropy_ratio 0.55, max_weight 0.72).
- **ISSUE-001 — val에 NIGHT_AUG가 적용되지 않는다** → 모델 선택 기준이 왜곡될 수 있다(부분 미해결).
- **대표 ckpt가 정규 웨이트 루트에 없다** — P9 ep131은 학습 서버(levine)의 `outputs/`에만 있었고
  `ckpts/`·`ckpts_backup/`에서 찾지 못했다. `reproduce_eval.sh multiaqua`는 `CKPT=` 없이 실행하면 에러로 종료한다
  ([REPRODUCE.md §4.3](REPRODUCE.md)).
- **SAM2 계열이라 `sam2.1_hiera_base_plus.pt`가 필요**하다. 이 워킹트리에는
  `semseg/models/sam2/sam2/checkpoints/` 디렉터리 자체가 없다(가중치 git 미추적) — 직접 받아 넣어야 한다.

---

### 4. poongsan indoor — 3모달 실내 멀티모달 객체검출 (국가 R&D)

**무엇을 푸는가**: 실내 촬영 RGB + LiDAR-depth + Thermal 3모달로 **10클래스 COCO-포맷 객체검출**.
국가연구개발과제 산출물이며 목표는 **mAP50 0.85**. 저조도 robustness가 서사의 핵심이다.

**모달리티와 디렉터리 이름** (config `DATASET.MODALITY_KEYS`가 매핑):

| config `MODALS` | `MODALITY_KEYS` 값 = 실제 디렉터리 | 비고 |
|---|---|---|
| `img` | `rgb` | |
| `lidar` | `depth_map_lidar` | egofill 파생셋은 `depth_map_lidar_egofill` |
| `thermal` | `thermal_aligned` | |

capture 디렉터리에는 이 밖에도 `intensity_map_lidar`, `intensity_map_lidar_egofill`, `event_aligned`,
`lidar_aligned`, `thermal_raw_aligned`, `annotations`가 함께 들어 있다(모달 ablation용).

**클래스 10종** (COCO `categories` 실측): Allies, Enemies, Casualties, Windows, Doors, Obstacles,
Lighting, Emergency Exits, Fire Extinguishers, Landing Markers.

**데이터 경로 (이 허브 박스, 2026-07-28 실측)**: `/ailab_mat2/Projects/Drone/DATA/260618_poongsan` — ✅ **마운트됨**.

```
/ailab_mat2/Projects/Drone/DATA/260618_poongsan/
├── capture_20260618_112051/ ... capture_20260618_120059/   # 캡처 세션 8개
│   ├── rgb/  depth_map_lidar/  depth_map_lidar_egofill/
│   ├── intensity_map_lidar/  intensity_map_lidar_egofill/
│   ├── thermal_aligned/  thermal_raw_aligned/  event_aligned/  lidar_aligned/
│   └── annotations/
├── final/
│   ├── annotations/instances_{train,train_egofill}.json
│   │                instances_test{,_common,_egofill,_lidar_clean,_lowlight,_normal}.json
│   └── split_info.json
└── _det_splits/  versions/  VERSIONS.md
```

COCO json의 `file_name`은 `<ROOT>` 기준 상대경로다(예 `capture_20260618_114021/rgb/1781782829216769.png`).

**split 장수 (json 실측)**:

| annotation | 이미지 | 박스 | 용도 |
|---|---|---|---|
| `instances_train.json` | 12,681 | 35,237 | 학습 |
| `instances_train_egofill.json` | 12,681 | 35,237 | 학습 (LiDAR egofill 파생셋) |
| `instances_test_common.json` | 3,239 | 9,385 | **표준 평가셋** |
| `instances_test.json` | 3,423 | 9,984 | 전체 test |
| `instances_test_lowlight.json` | 1,768 | 4,116 | 저조도 서브셋 |
| `instances_test_normal.json` | 1,471 | 5,269 | 정상조도 서브셋 |

⚠️ 리포 config는 서버 로컬 경로 `/SSDb/jemo_maeng/dset/poongsan_v2/_final_ann/...`를 가리킨다.
파일명은 같지만 **위 마운트본과의 동일성은 검증되지 않았다**([REPRODUCE.md §4.4](REPRODUCE.md)).

#### 학습

```bash
conda activate MMSS_SAM      # 서버에 따라 openmmlab 등 timm>=1.0 환경
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export DET_GRAD_CLIP=0.1                                   # RF-DETR 헤드 필수
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # BS1에서 ~23GB/24GB

# (A) 로컬 — 빈 GPU 4장 자동 선택
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh 4)" \
torchrun --standalone --nproc_per_node=4 train_det.py \
  --cfg configs/det/det_D1_recovered_yeon.yaml

# (B) 원격 — auto-detect가 config의 MODEL.DET_MODEL을 보고 train_det.py를 고른다. 명시해도 무방
bash scripts/remote_exp.sh status yeon
bash scripts/remote_exp.sh run yeon configs/det/det_D1_recovered_yeon.yaml auto:4 4 train_det.py
```

#### 평가

```bash
python val_det.py --cfg configs/det/det_D1_recovered_yeon.yaml \
  --det_checkpoint <ckpt.pth> --mode val --score_thresh 0.0 [--save_vis]

# 표준 분석 파이프라인 (클래스별/조도별 분해, 모듈 ablation, FPS)
python tools/det_analysis_pipeline.py --help     # 가이드: tools/README_det_analysis.md
```

⚠️ `val_det.py`의 `--score_thresh` 기본값 **0.3**은 낮은-score 박스를 버려 학습 중 eval(`train_det.py`, 임계값 없음)보다
AP가 낮게 나온다. 학습 eval과 같은 조건으로 재려면 **`--score_thresh 0.0`**을 줘라.

#### config 위치와 이름 규칙

전부 **`configs/det/`**에 있고 이름은 `det_<버전>_<변형>[_<서버>].yaml` 형태다
(`det_P29_egofill_bengio.yaml`, `det_D1_recovered_yeon.yaml`, `det_P38_m2f_yeon.yaml` …).
모델별 헤드·seg 기제 대응표와 서버 세팅 절차는 [`configs/det/README_det_training.md`](configs/det/README_det_training.md)에 있다.

#### 대표 성적

| 실험 | 모델 | 평가셋 | 핵심 수치 | 출처 |
|---|---|---|---|---|
| `det_P29_egofill_bengio` | P29-Det + LiDAR egofill | 공식 v2 test | **mAP50 0.8501** @ep9 — 🎯 목표 달성 | [registry.md](.claude_logs/experiments/registry.md), [datasets/lidar-egofill.md](.claude_logs/datasets/lidar-egofill.md) |
| `det_P29_event_bengio` | P29-Det, img/**event**/thermal | 공식 v2 test | mAP50 0.8427 @ep14 (event ≈ egofill-lidar, −0.008) | [registry.md](.claude_logs/experiments/registry.md) |
| `det_D1_recovered` | P37b-ClassToken 백본 + RF-DETR | `instances_test_common.json` | AP **0.6377** / AP50 **0.9321** / AP75 **0.7283** (small/medium/large 0.1755/0.5497/0.7430) | [monitor-log.md](.claude_logs/experiments/monitor-log.md), [REPRODUCE.md §4.4](REPRODUCE.md) |
| `det_P29_v2` | P29-Det (clean label) | v2 공식 | mAP50 0.446 @ep9 | [det/p29det-data-fix.md](.claude_logs/det/p29det-data-fix.md) |
| `det_P31_v3clip_jarvis` | P31.1-Det | v3clip(비공식) | mAP50 0.4724 | [det/diagnosis-plan.md](.claude_logs/det/diagnosis-plan.md) |
| `det_P30_v2` | P30-Det | poongsan_v2 | mAP50 0.256 (소물체 붕괴로 dead) | [det/diagnosis-plan.md](.claude_logs/det/diagnosis-plan.md) |
| YOLO11m RGB-only | 외부 기준점 | label-v3 | mAP50 0.864 — "데이터 무죄, 스택 유죄" 근거 | [det/diagnosis-plan.md](.claude_logs/det/diagnosis-plan.md) |

#### 이 데이터셋 특유의 함정

- **🔴 `MODEL.GRAD_CHECKPOINT: false`는 필수다** — encoder의 `active_modality` 상태가 checkpoint 재계산 사이에
  보존되지 않아 **마지막 모달을 제외한 전 모달이 노이즈로 학습된다**(det는 3모달이므로 2개가 오염).
  ISSUE-027과 같은 뿌리. config 주석에도 "REQUIRED"로 박혀 있다.
- **RF-DETR 헤드는 `TRAIN.GRAD_CLIP: 0.1`** — 원래 P37 런은 하드코딩된 10.0(≈클리핑 없음)으로 붕괴했다.
  `DET_GRAD_CLIP` 환경변수로도 덮어쓸 수 있다.
- **RF-DETR 헤드는 COCO 가중치가 필요**하다 — `MODEL.COCO_CKPT: weights/rf-detr-large-2026.pth`(130MB, git 미추적).
  체크아웃마다 한 번 넣어야 한다. P38/M2F는 불필요.
- **`val_det.py` 기본 `--score_thresh 0.3`** (위 평가 절 참조).
- **🔴 목표 달성 ckpt(0.8501)는 회수 불가** — 보유 서버 bengio가 2026-07-16 HW 고장으로 사망했다.
  재현하려면 재학습이 필요하다([status/current.md](.claude_logs/status/current.md)).
- **데이터 비공개** — 국가 R&D 과제 자체 수집분이라 외부 재현이 구조적으로 불가능하다.

---

## 모델 계보와 코드 배치

### 계보 요약

두 계열이 있고 **진입 스크립트가 다르다**. 상세는 [`.claude_logs/models/arch-evolution.md`](.claude_logs/models/arch-evolution.md)(canonical).

| 세대 | 버전 | 핵심 변경 | 백본 | 학습 진입점 | 코드 위치 |
|---|---|---|---|---|---|
| SAM2 LoRA | P1~P7 | 초기 LoRA/MoE 실험 | SAM2 Hiera-B+ | `train_sam2_lora_paper.py` | `lora_sam/legacy.py` |
| SAM2 LoRA | **P8** | ConfidenceHeadV2 + sigmoid UAMM | 〃 | 〃 | `lora_sam/p08.py` |
| SAM2 LoRA | **P9** ★ | CrossModalFusionHead + max-norm UAMM — **MULTIAQUA 공동 1위** | 〃 | 〃 | `lora_sam/p09.py` |
| SAM2 LoRA | P10~P21, P23~P26 | adaptive fusion 데드군 (gate 상수수렴 병목) | 〃 | 〃 | `lora_sam/legacy.py` |
| SAM2 LoRA | **P22** ★ | adaptive fusion — **MULTIAQUA 공동 1위** | 〃 | 〃 | `lora_sam/p22.py` |
| SAM2 LoRA | P27~P31 | RBMA 도입 → SDC 라우팅 → class-token decoder → calibrated dual-reliability | 〃 | 〃 | `lora_sam/p27..p31.py` |
| SAM2 LoRA | P32~P33 | CoRB(corroboration bias) / CG-MoD | 〃 | 〃 | `lora_sam/p32.py`, `p33.py` |
| SAM3 | SAM3-RBMA | SAM3 포팅 + decoder repurpose | SAM3 | `train_sam3_rbma.py` | `semseg/models/sam3/` |
| **ReliaDINO** | **P34** ★ | DINOv3 ViT-L/16 frozen + per-modal LoRA + RBMA — **DELIVER 최선** | DINOv3 ViT-L/16 | `train_reliadino.py` | `semseg/models/reliadino/` |
| ReliaDINO | P35 / P36 | paper-freeze(bias·consistency off) / + per-class router | 〃 | 〃 | 〃 |
| ReliaDINO | P37a / P37b | CEFR(2-pass blend) / ClassToken residual | 〃 | 〃 | `reliadino/classtoken.py` |
| ReliaDINO | P38 | MaskQueryLite (Mask2Former-lite query head) | 〃 | 〃 | `reliadino/m2f_head.py` |
| ReliaDINO | P39 / **P39.1** ★ | Dual-Path Compete / rank 수리(gated-MLP trunk + VICReg) — **MUSES 최고** | 〃 | 〃 | `reliadino/model.py`, `fusion.py` |
| ReliaDINO | P40~P42 | RCA / FCR / MaskImg | 〃 | 〃 | 〃 |
| ReliaDINO | P43 | PanopticDual — 독립 주손실 mask-classification 헤드(PQ 산출 확보) | 〃 | 〃 | `reliadino/panoptic_head.py` |
| ReliaDINO | P44 / P45 | BMR(MMPareto + 상호증류 + coverage 마스킹) / FogStyle | 〃 | 〃 | `reliadino/mmpareto.py`, `p44.py` |
| Det | P29~P31-Det | SAM2 백본 + FCOS/query 헤드 | SAM2 | `train_det.py` | `lora_sam/det.py` |
| Det | P34~P39-Det, D1 | ReliaDINO 백본 + FCOS / RF-DETR / M2F 헤드 | DINOv3 | `train_det.py` | `objdet/models/det_model.py` |

(`lora_sam/` = `semseg/models/sam2/sam2/lora_sam/`)

### 새 모델을 어디에 넣는가

**SAM2 계열 (P8~P33 계보)** — [`semseg/models/sam2/sam2/lora_sam/`](semseg/models/sam2/sam2/lora_sam/__init__.py):

1. `lora_sam/pNN.py`에 클래스를 새로 만든다. **메가파일 부활 금지** —
   `sam_lora_image_encoder_seg.py`는 하위호환 shim이므로 여기에 클래스를 추가하지 말 것.
2. `lora_sam/__init__.py`에서 import하고 **`MODEL_REGISTRY`에 `"LoRA_Sam_PNN": LoRA_Sam_PNN` 행을 추가**한다.
   config의 `MODEL.LORA_MODEL` 값이 이 키로 조회된다.
3. 조회는 항상 `get_model(name)` — `eval(name)` 금지. 미등록 이름이면 사용 가능한 목록과 함께 `KeyError`가 난다.
4. 공통 모듈(MoE/LoRA adapter, fusion head, reliability/confidence)은
   `semseg/models/sam2/sam2/modules/{moe,fusion,reliability,common}.py`에 두고 재사용한다.
5. 폐기 판정된 버전은 **삭제하지 말고** `lora_sam/legacy.py`로 옮긴다(config 재현성).

**ReliaDINO 계열 (P34~P45)** — [`semseg/models/reliadino/`](semseg/models/reliadino/__init__.py):

문자열 레지스트리가 아니라 **config 플래그 + 빌더** 방식이다. `build_reliadino(cfg, n_classes)` 하나가
`MODEL` 블록의 토글(`FUSION.ATTN_BIAS`, `GATE`, `CALIBRATION`, `ROUTER`, `CLASS_TOKEN`, `M2F`, `P43`, `P44` …)을 읽어
같은 클래스를 다른 구성으로 조립한다. 새 기제는 **새 모듈 파일 + `model.py` 배선 + config 토글**로 추가하고,
`NAME: ReliaDINO`는 유지한다. 이 패키지는 timm을 요구하므로 `semseg.models.__init__`에서 import하지 않는다 —
`from semseg.models.reliadino import ReliaDINO, build_reliadino`처럼 명시적으로 import해야 한다.

**Det** — `MODEL.SEG_MODEL`(백본)과 `MODEL.DET_MODEL`(헤드)을 config에서 각각 지정하면
`train_det.py`가 분기해 [`objdet/models/det_model.py`](objdet/models/det_model.py)의
`ReliaDINODetector` / `ReliaDINORFDETRDetector` / `ReliaDINOM2FDetector` / `MemorySAMDetector*` 중 하나를 만든다.

### 🔴 코드 단일출처 규칙

**운용(학습/평가 기동) 전에 코드는 반드시 `develop` 브랜치에 병합돼 있어야 한다.**
feature 브랜치·worktree·서버 로컬에만 있는 코드로 학습을 돌리지 마라. **config도 코드다.**
원격 서버는 GitHub이 아니라 **로컬 허브를 `local` remote로 pull**하므로, `develop` push + 허브 최신화가 있어야
다른 세션·서버가 그 코드를 받을 수 있다. 절차와 사고 사례는 [`CLAUDE.md`](CLAUDE.md) §1.7 /
[`conventions.md`](.claude_logs/meta/conventions.md) §8 참조.

### 🔴 코드 검수 파이프라인

**"코드 작성 완료" = 커밋이 아니라 검수 통과.** 구현 → 커밋/운용 사이 의무 4단계
([conventions.md](.claude_logs/meta/conventions.md) §코드 검수 파이프라인):

1. **Fresh-eyes 검수**(작성자 아닌 에이전트 1기) — 무gradient 결선 / 죽은 디스패치 / 데이터 상수 vs 실측 /
   부호 규약 / config 키↔builder 파싱 일치 / aux·loss 대차 / DDP 7개 렌즈
2. **합성 스모크 의무 assert** — 신설 파라미터 전부 gradient>0, init 등가성(off = 기존 경로 byte-일치), eval 결정론
3. **데이터 로더/디코더** — 실데이터 통계를 실측해 주석으로 남긴 후에만 커밋
4. **기동 후 ≤ep30** — `tools/module_ablation.py` 토글 즉검(no-op/미배선 조기 검출)

계기가 된 사고: ISSUE-025(MUSES radar 디코딩 버그를 8일간 미발견 → radar 실험 전부 오염),
ISSUE-024(P37b `mask_proj` 무gradient 출고).

---

## 원격 서버에서 돌리기

서버 메타데이터의 **단일 출처는 [`scripts/servers.conf`](scripts/servers.conf)**, 실행·추적은
[`scripts/remote_exp.sh`](scripts/remote_exp.sh)다. 운영 매뉴얼은
[`.claude_logs/infra/servers-and-launch.md`](.claude_logs/infra/servers-and-launch.md).

```bash
bash scripts/remote_exp.sh servers                 # 레지스트리 출력 (repo_path / env / default_gpus)
bash scripts/remote_exp.sh status <server>         # 빈 GPU + tmux 'jemo' 세션 창 목록
bash scripts/remote_exp.sh list   <server>         # 창 목록만
bash scripts/remote_exp.sh run    <server> <config.yaml> [gpus|auto:N] [nproc] [entry]
bash scripts/remote_exp.sh log    <server> <cfg_name> [follow]
```

- 학습은 `nohup`이 아니라 **tmux 세션 `jemo`의 새 window**에서 돈다 → 접속이 끊겨도 살아 있고
  `tmux attach -t jemo`로 직접 볼 수 있다.
- 로그는 원격의 `logs/<cfg_name>/<cfg_name>_<timestamp>.log`. `run`이 `LOG=` 경로를 출력한다.
- master_port는 21600~21899 랜덤(충돌 회피).

### 서버 레지스트리 (`scripts/servers.conf` 실제 내용, 2026-07-28)

| alias | repo_path | env | default_gpus | 비고 |
|---|---|---|---|---|
| `gyuri` | **FILL_ME** | MMSS_SAM | 0 | port 100. repo_path 미기입 → 런처가 실행 거부 |
| `lecun` | `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM` | MMSS_SAM | 0 | port 300, 7×24GB. **`sam2` editable 미설치** → `PYTHONPATH=<repo>/semseg/models/sam2` 필요 |
| `bengio` | `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM` | MMSS_SAM | 0~7 | port 400, 8-GPU. ⚠️ 2026-07-16 HW 고장으로 사망 |
| `levine` | `/SSDe/jemo_maeng/src/Project/Drone/drone-MemorySAM` | MMSS_SAM | 0,1,2,3 | port 500. 현행 최선 P9 config가 `levine-` 접두어 (경로가 `SSDe`) |
| `yeon` | `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM` | MMSS_SAM | 0 | port 600. 경로가 `Drone24`가 아니라 `Drone` |
| `B200` | `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM` | MMSS_SAM | **FILL_ME** | 8×B200 180GB, **공용**. default를 의도적으로 비워 `auto:N` 강제. 프로세스는 unix user `gm_huis`로 뜬다 |
| `hpca100` | `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM` | `/home/jovyan/SSDb/jemo_maeng/venv/p34` (**venv 절대경로**) | 0,1,2,3 | GIST SCENT HPC(K8s pod). A100-SXM4-40GB ×4, Slurm 없음. **conda 아님** — env 컬럼이 venv 경로이고 런처가 `source`한다. `~/`는 25G뿐이라 작업은 `~/SSDb/jemo_maeng`에서. 클라이언트 MTU 1200 필요, github ssh:22 차단 → https clone |
| ~~`hinton`~~ | — | — | — | servers.conf에서 **주석 처리됨**(port 200 UNREACHABLE, 2026-06-24 기준) |

> ⚠️ **`jarvis`는 `servers.conf`에 등록돼 있지 않다.** 그런데 `configs/jarvis-*.yaml` 11종이 존재하고
> 실험 로그에도 jarvis 런이 다수 기록돼 있다. `remote_exp.sh`로 jarvis를 띄우려면 먼저 레지스트리에 행을 추가해야 한다.

### 🔴 빈 GPU 확인 필수 규칙

**어떤 실험이든 실행 전에 해당 서버의 빈 GPU를 확인하고, 비어 있는 GPU에만 배치한다.**
사용 중 GPU에 얹으면 OOM + 타인 작업 방해다.

- **판정 기준**: `memory.used ≤ 2000MiB && util ≤ 10%`. 메모리 적은 순으로 고른다.
  임계값은 `GPU_MAXMEM` / `GPU_MAXUTIL` 환경변수로 조정.
- **로컬**: [`scripts/pick_free_gpus.sh`](scripts/pick_free_gpus.sh) `N` 이 빈 GPU 인덱스를 콤마로 출력한다(부족하면 실패).
  `CUDA_VISIBLE_DEVICES`가 이미 있으면 그대로 echo — 직접 지정을 존중한다.
  런처는 이미 연결돼 있다: `NGPU=4 bash run_sam.sh` · `NGPU=1 bash run_sam3_train.sh` · `NPROC=2 bash run_sam3_rbma.sh <cfg>`.
- **원격**: `status <server>`로 먼저 보고, `run <server> <cfg> auto:N`으로 원격 빈 GPU N장을 자동 배정한다(`auto` = 1장).
  빈 GPU가 부족하면 런처가 **실행을 거부**한다.

### 기동 "검증"의 기준

기동 직후 "떴다"만 보고 살아났다고 판단하지 마라(2026-07-16 NCCL 데드락 오보 사고). 판정 기준:

1. **iteration이 실제로 전진하는가** (예: `73/187` → 25초 뒤 `92/187`)
2. **rank0 GPU util > 0인가** (0%면 collective 이탈 = 데드락)
3. **메모리가 가중치 수준(3~4GiB)이 아니라 실제 활성화 수준인가**
4. **첫 eval 통과**

---

## 산출물이 어디로 가는가

### 🔴 웨이트·분석·학습로그의 정규 루트 (모든 세션 공유)

```
/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/
├── ckpts/           # 회수 완료된 대표 웨이트   <run>_<YYYYMMDD>/
├── ckpts_backup/    # 서버별 원본 트리 백업     <server>/<SAVE_DIR 구조>/
├── analysis_logs/   # 평가·분석·시각화          <model>_eval_<YYYYMMDD>/ = report/ + viz/ + perdomain/
└── train_logs/      # 학습 런 로그
```

**모든 세션은 새 학습/평가/분석 산출을 여기 저장**하고, 새 전략 전에 이 루트를 먼저 확인한다.
원격(hpca100/jarvis/yeon)에서는 `rsync`로 회수·누적한다. 이 박스에서 ✅ **마운트 확인됨**.

> ⚠️ **실측 주의**: 2026-07-28 현재 이 루트에 실제로 존재하는 디렉터리는 `ckpts/`, `ckpts_backup/`, `analysis_logs/` 셋이다.
> **`train_logs/`는 아직 만들어져 있지 않다** — [`CLAUDE.md`](CLAUDE.md)가 규정한 위치이므로 학습 로그를 옮길 때 생성하면 된다.

경로 변천: `/mnt/HDD2/src/logs/`(ISSUE-023 ENOSPC) → `/drone_nas/drone/analysis_logs/`(flat) → 위 nested 구조(2026-07-17 재확정).

### 체크포인트 파일명 규칙

| 패턴 | 뜻 |
|---|---|
| `epoch<N>_<val점수>_top<K>_checkpoint.pth` | **val-best**. 논문/보고에 쓸 수 있는 정당한 선택 |
| `test_epoch<N>_<test점수>_top<K>_checkpoint.pth` | **test-best**. test셋 훔쳐보기 → **논문 인용 불가** |
| `last_checkpoint.pth` | 마지막 epoch 스냅샷 |
| `best_checkpoint.pth` | det 트랙(`train_det.py`) 최고 AP 스냅샷 |

### 학습 중 로컬 산출

- 학습 출력: config의 `SAVE_DIR`(ReliaDINO 계열, 예 `./outputs/ReliaDINO/<run>/`) 또는
  `OUTPUT_DIR`(det, 예 `outputs/det_D1_recovered_yeon`). SAM2 계열은 `outputs/MMSamP*/<cfg명>/<DATASET>_<BACKBONE>_<modals>/`.
- 학습 로그: `logs/<cfg_name>/<cfg_name>_<timestamp>.log` (로컬 런처·원격 런처 공통).
- `outputs/`·`logs/`는 git 미추적이라 **이 워킹트리에는 아직 없다**(첫 학습 시 생성).

### 제출물

- **MUSES (Codabench 14005)**: `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/muses/` —
  ✅ 마운트 확인됨. 제출 zip 13종 + `MUSES_TEST_RESULTS_INDEX.md`(제출 인덱스)가 있다.
- **MULTIAQUA (MACVi)**: `val_multiaqua.py --macvi`가 체크포인트 디렉터리 아래
  `<ckpt명>_eval_macvi/`에 1-indexed 마스크만 쓴다(`--save_dir`로 변경 가능).

### Weights & Biases

`train_sam2_lora_paper.py`는 project=`MemorySAM`으로 스칼라 + per-class IoU + **고정 val 이미지 10장**의
`[RGB | GT | Pred]` 패널을 로깅한다. 인증 우선순위는 **이미 설정된 env `WANDB_API_KEY` > repo-local `.wandb_key` > 머신의 `wandb login`**
(repo-local 키는 그 프로세스 환경변수로만 쓰여 공용 서버의 `~/.netrc`를 건드리지 않는다).
config에 `WANDB.ENABLE: false` 또는 env `WANDB_DISABLED=1`로 끌 수 있고, 키가 없어도 학습은 그대로 진행된다.
상세는 [`servers-and-launch.md`](.claude_logs/infra/servers-and-launch.md) §4.

---

## 리포 구조 요약

```
drone-MemorySAM/
├── CLAUDE.md / AGENTS.md              # 세션·에이전트 지침 (canonical)
├── REPRODUCE.md                       # 정량 재현 가이드 (기대 수치 + 재현 한계)
├── .claude_logs/                      # 연구 로그 — front door = 00_INDEX.md
│   ├── status/                        #   current.md(스냅샷 단일 출처) + history-2026H1/H2
│   ├── models/                        #   arch-evolution.md(canonical) · figures-ascii · explain/
│   ├── experiments/                   #   registry.md · log.md · monitor-log.md · plan.md · analysis/
│   ├── det/ datasets/ research/       #   det 진단 · 데이터셋 구축 · 관련연구(vault/ 포함)
│   ├── decisions/ infra/ issues/      #   설계 제안(ADR) · 서버·환경 · 이슈
│   └── meta/ archive/                 #   conventions·bot-roles·taskboard · 동결 문서
├── train_sam2_lora_paper.py           # SAM2 LoRA (P1~P33) 학습 — DDP
├── train_sam2_lora_paper_singlegpu.py #   단일 GPU 폴백
├── train_reliadino.py                 # ReliaDINO (P34~P45) 학습 — DELIVER/MUSES
├── train_sam3_rbma.py                 # SAM3-RBMA 학습
├── train_det.py                       # detection 학습 (poongsan)
├── val.py                             # 통합 평가 (MULTIAQUA/DELIVER/MUSES)
├── val_multiaqua.py                   # MULTIAQUA 평가 + MACVi 제출 (--macvi)
├── val_multiaqua_detailed.py          #   per-image 상세 로그판 (eval.sh가 호출)
├── val_det.py                         # detection 평가 (COCO AP)
├── run_sam.sh / run_sam3.sh           # 로컬 런처 (빈 GPU 자동 선택)
│   └── run_sam3_train.sh, run_sam3_rbma.sh   # → run_sam3.sh 로 exec 하는 얇은 래퍼
├── eval.sh                            # val/test 평가 런처 (구 eval_val.sh + eval_test.sh 통합)
├── configs/                           # deliver/ multiaqua/ det/ eval/ archive/ profiles/ + 루트(ReliaDINO)
├── semseg/
│   ├── datasets/                      #   deliver.py · muses.py · multiaqua.py · mcubes · nyu · …
│   ├── augmentations_mm.py            #   멀티모달 증강 (PhysAug/NightAug 포함)
│   └── models/
│       ├── sam2/sam2/lora_sam/        #     LoRA_Sam_P1~P33 + MODEL_REGISTRY + det.py
│       ├── sam2/sam2/modules/         #     공통 모듈 (moe/fusion/reliability/common)
│       ├── reliadino/                 #     P34~P45 (encoder·fusion·model + m2f/panoptic/classtoken/p44)
│       ├── sam3/  rfdetr_head/        #     SAM3 포팅 · RF-DETR 헤드
│       └── backbones/ heads/ layers/  #     CMNeXt 등 upstream 자산
├── objdet/                            # detection 스택 (det_model.py · datasets · losses · yolo11m-rgb 기준점)
├── tools/                             # 재사용 분석 도구 — README_seg_analysis.md / README_det_analysis.md
│                                      #   eval_reliadino_ckpt · eval_muses_official · predict_muses_test
│                                      #   eval_per_domain · module_ablation · viz_features · smoke_p4*
├── scripts/                           # remote_exp.sh · servers.conf · pick_free_gpus.sh · reproduce_eval.sh
├── analysis/  _paper/  _challenge_report/   # 분석 메모 · 논문 초안 · 챌린지 기술보고서(tex/pdf)
├── MISC/  Figure/  _archive/          # 일회성 유틸 · 그림 · upstream 원본 + oneoff 보관
└── (런타임 생성) outputs/  logs/  weights/
```

---

## 더 읽을 것

| 문서 | 언제 보나 |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | 지표를 다시 뽑아야 할 때 (기대 수치·재현 한계 포함) |
| [`.claude_logs/status/current.md`](.claude_logs/status/current.md) | 지금 무엇이 돌고 있고 무엇이 최선인지 |
| [`.claude_logs/experiments/registry.md`](.claude_logs/experiments/registry.md) | 실험 ↔ config ↔ ckpt ↔ 수치 대응표 |
| [`.claude_logs/experiments/log.md`](.claude_logs/experiments/log.md) | 실험 결과 상세 서사 (canonical) |
| [`.claude_logs/models/arch-evolution.md`](.claude_logs/models/arch-evolution.md) | 모델 버전별 구조·동기·한계 |
| [`.claude_logs/issues/issues-and-fixes.md`](.claude_logs/issues/issues-and-fixes.md) | **코드 작성 전** — 상단 이슈 상태 인덱스 표 |
| [`.claude_logs/research/novelty-and-related-work.md`](.claude_logs/research/novelty-and-related-work.md) | 노벨티·선행연구 비교 (논문 포지셔닝 논의 전 필독) |
| [`.claude_logs/infra/environment.md`](.claude_logs/infra/environment.md) | 실행 환경·경로·DDP·B200 파이프라인 튜닝 |
| [`.claude_logs/det/diagnosis-plan.md`](.claude_logs/det/diagnosis-plan.md) | det 작업 전 필독 |
| [`.claude_logs/experiments/plan.md`](.claude_logs/experiments/plan.md) | 실험 대기열·배치 계획 |
| [`configs/README.md`](configs/README.md) · [`configs/det/README_det_training.md`](configs/det/README_det_training.md) | config 분류·명명·서버 세팅 |
| [`tools/README_seg_analysis.md`](tools/README_seg_analysis.md) · [`tools/README_det_analysis.md`](tools/README_det_analysis.md) | 표준 분석 파이프라인 사용법 |
