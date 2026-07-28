# REPRODUCE.md — 정량 지표 재현 가이드

이 문서 하나만 따라가면 **명령 한 줄로** 대표 체크포인트의 정량 지표(mIoU / COCO AP)를 다시 뽑을 수 있다.

```bash
bash scripts/reproduce_eval.sh <bench>     # bench ∈ deliver | muses | muses-official | multiaqua | det
```

> 🔴 **먼저 읽어야 할 정직한 전제**
> 이 리포는 **연구실 내부 인프라 위에서만 완전히 재현된다**. 데이터셋(DELIVER / MUSES / MULTIAQUA / poongsan)과
> 학습된 체크포인트는 전부 **연구실 NAS·마운트 볼륨에만** 있고 공개 배포본이 없다. 외부에서 클론한 사람은
> 코드·설정·평가 파이프라인은 그대로 돌릴 수 있지만, **데이터와 가중치는 직접 구해서 `DATA_ROOT=` / `CKPT=` 로 끼워야 한다.**
> 무엇이 구조적으로 재현 불가인지는 아래 [§6 재현 한계](#6-재현-한계--구조적으로-불가능한-것) 에 전부 적어 두었다.

---

## 1. 전제조건

### 1.1 하드웨어 · 드라이버

| 항목 | 요구 |
|---|---|
| GPU | NVIDIA CUDA GPU 1장 이상. 대표 seg 모델은 DINOv3 ViT-L/16을 1024×1024 입력으로 4모달 forward → **평가 VRAM 24GB 권장** |
| 드라이버 | CUDA **12.1** 런타임(`torch 2.3.1+cu121`)과 호환되는 NVIDIA 드라이버 |
| VRAM 부족 시 | config의 `EVAL.BATCH_SIZE` 를 낮춰라 (기본 4 → 1). 자세한 건 [§5 트러블슈팅](#5-트러블슈팅) |

평가는 **단일 GPU**로 충분하다(DDP 불필요). 스크립트는 실행 전에 `scripts/pick_free_gpus.sh` 로
**빈 GPU**(`memory.used ≤ 2000MiB && util ≤ 10%`)를 자동으로 고른다 — 남의 학습 위에 얹히지 않는다.

### 1.2 conda 환경

```bash
# (A) 실제 환경 그대로 복원 — 권장
conda env create -f conda_environment.yml     # name: MMSS_SAM
conda activate MMSS_SAM

# (B) 이미 환경이 있으면 그 python을 그대로 쓰면 된다
PY=$(which python) bash scripts/reproduce_eval.sh deliver
```

`conda_environment.yml` 이 실제 `MMSS_SAM` 환경 export 본이다. `requirements.txt` / `environment.yaml` 은
**upstream MemorySAM에서 물려받은 구버전 목록이라 현 환경과 맞지 않는다** — 아래 §1.3의 실측 표를 기준으로 삼아라.

### 1.3 실측 패키지 버전 (`MMSS_SAM` env, 2026-07-28 `pip list` 기준)

| 패키지 | **실측 설치 버전** | `requirements.txt` 기재 | 비고 |
|---|---|---|---|
| python | **3.10.19** | (없음) | |
| torch | **2.3.1+cu121** | (없음) | requirements.txt에 torch가 아예 없다 |
| torchvision | **0.18.1+cu121** | (없음) | |
| torchaudio | **2.3.1+cu121** | (없음) | |
| numpy | **2.2.6** | `1.26.0` | ⚠️ 불일치 (numpy 2.x) |
| timm | **1.0.24** | `0.4.12` | ⚠️ 불일치 — **DINOv3 백본(`vit_large_patch16_dinov3`)은 timm 1.x 필수** |
| opencv-python | **4.13.0.92** | `<4.9` | ⚠️ 불일치 |
| tabulate | **0.9.0** | `0.8.10` | ⚠️ 불일치 |
| tensorboard | **2.20.0** | `2.10.0` | ⚠️ 불일치 |
| fvcore | **0.1.5.post20221221** | `0.1.5.post20220512` | ⚠️ 불일치 |
| einops | **0.8.2** | `0.4.1` | ⚠️ 불일치 |
| PyYAML | **6.0.3** | `6.0` | |
| scipy | **1.15.3** | (버전 무지정) | |
| matplotlib | **3.10.8** | (주석 처리됨) | |
| mmcv | **2.1.0** | (없음) | |
| mmengine | **0.10.7** | (없음) | |
| pycocotools | **2.0.11** | (버전 무지정) | det 평가 필수 |
| hydra-core | **1.3.2** | (없음) | SAM2 빌드 경로 |
| iopath | **0.1.10** | `0.1.10` | ✅ 일치 |
| huggingface_hub | **1.4.1** | (없음) | DINOv3 가중치 로드 |
| safetensors | **0.7.0** | (버전 무지정) | |
| pandas | **2.3.3** | (없음) | 분석 도구 |
| tqdm | **4.67.3** | `4.62.3` | |
| SAM-2 | **1.0 (editable, `semseg/models/sam2`)** | (없음) | SAM2 계열(P8~P32) 실행 시 필요 |

> `requirements.txt` 는 **일부러 갈아엎지 않았다** — upstream 이력 보존 목적. 실제 재현은 위 표 또는
> `conda_environment.yml` 을 따르고, `requirements.txt` 는 참고용으로만 봐라.

### 1.4 사전학습 백본 가중치

| 계열 | 필요한 가중치 | 위치 / 받는 법 |
|---|---|---|
| **P34~P44 (ReliaDINO, 현행 대표 모델)** | DINOv3 ViT-L/16 (timm `vit_large_patch16_dinov3`) | timm/HuggingFace Hub에서 자동 로드. **평가에는 사실상 불필요** — 체크포인트가 백본 포함 전체 state를 담고 있다. HF 접근이 막히면 경고 후 random-init으로 만들고 ckpt가 덮어쓴다([§5](#5-트러블슈팅)) |
| **P8~P32 (SAM2/LoRA 계열)** | `sam2.1_hiera_base_plus.pt` | `semseg/models/sam2/sam2/checkpoints/` 에 둔다. 다운로드: <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt> (또는 `semseg/models/sam2/download_ckpts.sh`) |

🔴 **현재 이 워킹트리에는 `semseg/models/sam2/sam2/checkpoints/` 디렉터리 자체가 없다**(가중치는 git 미추적).
SAM2 계열(MULTIAQUA P9 등)을 돌리려면 위 URL로 직접 받아서 넣어야 한다.

---

## 2. 데이터셋

### 2.1 실측 경로 (이 허브 박스, 2026-07-28 마운트 확인)

| 벤치 | 스크립트 기본 `DATA_ROOT` | 마운트 상태 | 확인 내용 |
|---|---|---|---|
| DELIVER (4모달 img/depth/event/lidar, 25cls) | `/ailab_mat2/dataset/DELIVER` | ✅ 마운트됨 | val 2,005장 / test 1,897장 (로더 glob `img/*/<split>/*/*.png` 로 확인) |
| MUSES (3모달 img/lidar/event, 19cls) | `/ailab_mat2/dataset/MUSES` | ✅ 마운트됨 | train 1,500 / val 250 / test 750, `gt_semantic/` · `projected_to_rgb/{lidar,event_camera,radar}/` 존재 |
| MULTIAQUA (RGB+LiDAR+Thermal, 4cls) | `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2` | ✅ 마운트됨 | `MULTIAQUA_night`(test 경로)도 존재 |
| poongsan indoor det (COCO, 10cls) | `/ailab_mat2/Projects/Drone/DATA/260618_poongsan` | ✅ 마운트됨 | `final/annotations/instances_test_common.json` = 3,239장 / 9,385 박스 |
| 웨이트·분석 산출물 NAS | `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM` | ✅ 마운트됨 | `ckpts/`, `ckpts_backup/`, `analysis_logs/` |
| 원격 서버 로컬 SSD | `/SSDb/...` | ❌ **없음** | jarvis/yeon/hpca100 **서버 로컬** 경로 — 이 박스에는 없다 |

⚠️ **리포에 커밋된 config의 `DATASET.ROOT` 는 대부분 학습을 돌린 서버의 로컬 경로**다
(`/SSDb/jemo_maeng/dset/MUSES`, `/NHNHOME/.../DELIVER`, `/SSDb/.../poongsan_v2` 등). 이 박스에서는 존재하지 않는다.
그래서 `scripts/reproduce_eval.sh` 는 **config를 직접 고치지 않고**, `DATA_ROOT` 를 반영한 **임시 config를 만들어**
평가 진입점에 넘긴다(`configs/` 원본은 건드리지 않는다).

### 2.2 다른 경로로 바꿔 끼우기

```bash
DATA_ROOT=/my/path/DELIVER bash scripts/reproduce_eval.sh deliver
DATA_ROOT=/my/path/MUSES   bash scripts/reproduce_eval.sh muses
ANN_VAL=/my/path/instances_test_common.json DATA_ROOT=/my/poongsan bash scripts/reproduce_eval.sh det
```

기대하는 디렉터리 레이아웃(로더 코드 기준):

- **DELIVER** (`semseg/datasets/deliver.py:139-144`) — `<ROOT>/img/<condition>/<split>/<seq>/*_rgb.png`.
  나머지 모달은 **디렉터리와 파일 접미사를 함께** 치환한다: 🔴 **depth 모달이 읽는 디렉터리는 `depth/`가 아니라 `hha/`**
  (`/img`→`/hha`, `_rgb`→`_depth`) · lidar = `/lidar` + `_lidar` · event = `/event` + `_event` · GT = `/semantic` + `_semantic`.
  디스크에 `depth/`도 함께 있지만 로더는 쓰지 않는다 — `hha/`가 없으면 depth 모달 로드가 실패한다.
- **MUSES** (`semseg/datasets/muses.py:143`) — `<ROOT>/frame_camera/<split>/<weather>/<tod>/*.png`, GT는 `<ROOT>/gt_semantic/...`, 투영 모달은 `<ROOT>/projected_to_rgb/{lidar,event_camera,radar}/...`
- **MULTIAQUA** — `CLAUDE.md` 프로젝트 개요 절 참조 (`img`/`lidar`/`thermal`)
- **det** — COCO json의 `file_name` 이 `<ROOT>` 기준 상대경로(`capture_YYYY.../rgb/<ts>.png`), 모달 폴더는 config `MODALITY_KEYS`(`rgb`/`depth_map_lidar`/`thermal_aligned`)

### 2.3 외부 공개 여부

| 데이터셋 | 외부 취득 |
|---|---|
| DELIVER | 공개 벤치마크 (원저자 배포) |
| MUSES | 공개 벤치마크 (원저자 배포, test GT 비공개 — Codabench 채점) |
| MULTIAQUA | MACVi MULTIAQUA Challenge 참가자 배포, test GT 비공개 (챌린지 서버 채점) |
| poongsan indoor | **비공개** — 국가 R&D 과제 자체 수집 데이터. 외부 재현 불가 |

---

## 3. 체크포인트

### 3.1 정규 위치 (모든 세션 공유, `CLAUDE.md` 지정)

```
/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/
├── ckpts/           # 회수 완료된 대표 웨이트  (<run>_<YYYYMMDD>/)
├── ckpts_backup/    # 서버별 원본 트리 백업     (<server>/<SAVE_DIR 구조>/)
├── analysis_logs/   # 평가·분석·시각화          (<model>_eval_<YYYYMMDD>/)
```

### 3.2 파일명 규칙

| 패턴 | 뜻 |
|---|---|
| `epoch<N>_<val점수>_top<K>_checkpoint.pth` | **val-best** 계열. 논문/보고에 쓸 수 있는 정당한 선택 |
| `test_epoch<N>_<test점수>_top<K>_checkpoint.pth` | **test-best** 계열. test셋 훔쳐보기 → **논문 인용 불가** (2026-07-15 판정, `.claude_logs/status/current.md`) |
| `last_checkpoint.pth` | 마지막 epoch 스냅샷 |
| `best_checkpoint.pth` | det 트랙(`train_det.py`) 최고 AP 스냅샷 |

### 3.3 `.pth` vs `_checkpoint.pth` — 포맷 차이 (`CLAUDE.md` 주의사항 1번)

| 확장자 | 내용 | 로더 |
|---|---|---|
| `..._checkpoint.pth` | `{'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}` **dict** | `val_multiaqua.py` 가 이 포맷을 기대 |
| `....pth` (raw) | `state_dict` **그 자체** | `val_multiaqua_P9.py` 가 직접 로드 |

현행 진입점(`val.py`, `tools/eval_reliadino_ckpt.py`, `val_det.py`)은 **둘 다 받는다** —
`ckpt.get('model_state_dict', ckpt)` 로 분기하므로 포맷을 신경 쓸 필요가 없다.
다만 `tools/eval_reliadino_ckpt.py` 는 로드 후 `missing==0 and unexpected==0` 을 **assert** 한다 —
config와 ckpt의 아키텍처가 어긋나면 조용히 키메라 모델로 평가되는 대신 **즉시 죽는다**(의도된 안전장치).

---

## 4. 재현 명령과 기대 수치

> 📏 **수치 규약**: seg는 mIoU(%), det는 COCO AP(0~1). 아래 "기대 수치"는 전부 `.claude_logs/` 에 기록된
> **실측값**이며 출처를 명시했다. 확인하지 못한 값은 **TODO**로 비워 두었다 — 추정치를 채우지 않았다.

### 4.1 DELIVER — seg, 4모달 (논문 트랙)

```bash
bash scripts/reproduce_eval.sh deliver
```

| 항목 | 값 |
|---|---|
| 모델 | **P34-ReliaDINO** (DINOv3 ViT-L/16 frozen + LoRA + RBMA), 4모달 img/depth/event/lidar |
| config | `configs/b200-deliver_rgbdel_P34_reliadino.yaml` |
| ckpt | `ckpts/P34_final_20260713/epoch120_68.19_top1_checkpoint.pth` (val-best) |
| 진입점 | `tools/eval_reliadino_ckpt.py` (학습 시 eval 경로를 그대로 재사용) |

| 지표 | **기대 수치** | 출처 |
|---|---|---|
| val mIoU | **68.19** | [`status/current.md`](.claude_logs/status/current.md) legal(val-best) 표 · [`experiments/monitor-log.md:663`](.claude_logs/experiments/monitor-log.md) |
| test mIoU (같은 ep120) | **56.62** | 위와 동일 · [`decisions/2026-07-16-p36-novelty-critical-review.md:13`](.claude_logs/decisions/2026-07-16-p36-novelty-critical-review.md) |
| 조건별(cloud/fog/night/rain/sun) 분해 | TODO — 이 ckpt 기준 per-condition 수치는 별도 러너 `tools/eval_per_domain.py` 로 뽑아야 하며 기록된 표를 확정하지 못했다 |

> ⚠️ `test_epoch140_57.6_...pth` 로 나온 **57.60은 test-best 선택이라 철회된 수치**다(`.claude_logs/status/current.md`).
> 인용하지 마라.
> ⚠️ 우리 eval은 1024×1024 letterbox 프로토콜이다. DELIVER 공개표(CAFuser의 `CMNEXT_EQUIVALENT_EVAL`)와
> GT 리사이즈 규약이 같은지는 **미확인 리스크**로 남아 있다([`monitor-log.md:720`](.claude_logs/experiments/monitor-log.md)).

### 4.2 MUSES — seg, 3모달 (현행 최고 성적)

```bash
bash scripts/reproduce_eval.sh muses            # 학습 내부 프로토콜(1024² letterbox) val mIoU
bash scripts/reproduce_eval.sh muses-official   # 공식 native 1080×1920 프로토콜로 재채점
```

| 항목 | 값 |
|---|---|
| 모델 | **P39.1-rank (seed2)** — P39-DPC + R-1(gated-MLP trunk) + R-2(VICReg), 3모달 img/lidar/event |
| config | `configs/jarvis-muses_rgbel_P39_1_rank.yaml` |
| ckpt | `ckpts_backup/jarvis/ReliaDINO/jarvis_muses_rgbel_P39_1_rank_seed2/MUSES_ReliaDINO-ViTL16_ile/epoch208_82.62_top1_checkpoint.pth` |

| 지표 | **기대 수치** | 출처 |
|---|---|---|
| val mIoU (학습 내부 프로토콜) | **82.62** @ep208 | [`experiments/registry.md`](.claude_logs/experiments/registry.md) seed2 행 · [`status/current.md`](.claude_logs/status/current.md) |
| val mIoU (공식 native 해상도) | **TODO** — seed2에 대한 official-protocol val 수치는 기록에 없다. (참고: P34 ep276은 내부 81.02 → 공식 **80.86**, `status/current.md`) |
| **test mIoU** | **79.788** (MUSES 신기록) — 단 **로컬 재현 불가**, Codabench 서버 채점 결과 | [`experiments/log.md:1605`](.claude_logs/experiments/log.md) |
| 5-seed variance (참고) | 81.70 / 81.89 / 81.92 / 82.03 / 82.62 (평균 82.03) | [`status/current.md`](.claude_logs/status/current.md) |

> ⚠️ **config 주의**: seed2 전용 config(`configs/jarvis-muses_rgbel_P39_1_rank_seed2.yaml`)는 **이 브랜치에 없다**.
> 스크립트는 아키텍처가 동일한 base config(`jarvis-muses_rgbel_P39_1_rank.yaml`, 시드만 다름)를 쓴다.
> 만약 실제로 아키텍처가 어긋나면 `tools/eval_reliadino_ckpt.py` 의 state_dict assert가 즉시 잡아낸다.
> 대안으로 config-ckpt 짝이 확실한 **P38-MUSES**(`configs/jarvis-muses_rgbel_P38_m2f.yaml` +
> `ckpts/P38_MUSES_20260720/epoch156_82.22_top1_checkpoint.pth`, 공식 test 79.025)를 쓸 수 있다:
> ```bash
> CFG=configs/jarvis-muses_rgbel_P38_m2f.yaml \
> CKPT=/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/P38_MUSES_20260720/epoch156_82.22_top1_checkpoint.pth \
>   bash scripts/reproduce_eval.sh muses
> ```

### 4.3 MULTIAQUA — seg, 3모달 (MACVi Challenge)

```bash
CKPT=<P9 ep131 체크포인트 경로> bash scripts/reproduce_eval.sh multiaqua
```

| 항목 | 값 |
|---|---|
| 모델 | **P9** (CrossModalFusionHead + max-norm UAMM), hardaug8_physaug ep131 |
| config | `configs/eval/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml` |
| ckpt | `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth` |

| 지표 | **기대 수치** | 출처 |
|---|---|---|
| Val mIoU (챌린지 서버 채점) | **93.29** | [`experiments/log.md:130`](.claude_logs/experiments/log.md) 제출 #16710 |
| Test mIoU (야간, 챌린지 서버 채점) | **70.91** — **로컬 재현 불가**(test GT 비공개) | 위와 동일 |
| **M-score** = 0.75·val + 0.25·test | **82.10** (공동 1위) | 위와 동일 |

> 🔴 **이 벤치는 현재 기본 체크포인트가 없다.** P9 ep131 웨이트는 학습 서버(levine)의 `outputs/` 에만 있었고
> **정규 웨이트 루트(`ckpts/`, `ckpts_backup/`)에서 찾지 못했다.** 그래서 스크립트는 `CKPT=` 없이 실행하면
> 명확한 에러로 종료한다. 또한 SAM2 계열이므로 §1.4의 `sam2.1_hiera_base_plus.pt` 가 있어야 한다.
> 로컬에서 뽑히는 값은 **val mIoU 뿐**이며, 위 93.29는 챌린지 서버 채점 기준이라 로컬 수치와 소수점 차이가 날 수 있다.

### 4.4 det — poongsan indoor 멀티모달 검출 (국가 R&D)

```bash
bash scripts/reproduce_eval.sh det
```

| 항목 | 값 |
|---|---|
| 모델 | **D1-recovered** (P37b-ClassToken seg 백본 + RF-DETR NMS-free head) |
| config | `configs/det/det_D1_recovered_yeon.yaml` |
| ckpt | `ckpts/det_D1_recovered_20260723/best_checkpoint.pth` (epoch 6) |
| 평가셋 | `final/annotations/instances_test_common.json` (3,239장 / 9,385 박스 / 10 클래스) |

| 지표 | **기대 수치** | 출처 |
|---|---|---|
| AP | **0.6377** | [`experiments/monitor-log.md:2390`](.claude_logs/experiments/monitor-log.md) — ckpt 메타 직접 조회 |
| AP50 | **0.9321** | 위와 동일 |
| AP75 | **0.7283** | 위와 동일 |
| AP_small / AP_medium / AP_large | 0.1755 / 0.5497 / 0.7430 | 위와 동일 |

> ⚠️ 위 수치는 학습 서버(yeon)의 `/SSDb/jemo_maeng/dset/poongsan_v2/_final_ann/instances_test_common.json` 기준이다.
> 스크립트 기본값은 이 박스에 마운트된 `/ailab_mat2/.../260618_poongsan/final/annotations/instances_test_common.json`
> — **파일명은 같지만 두 annotation의 동일성은 검증하지 못했다.** 어긋나면 수치가 달라진다.
> ⚠️ 스크립트는 `--score_thresh 0.0` 을 넘긴다. `val_det.py` 기본값 0.3은 낮은-score 박스를 버려
> 학습 중 eval(`train_det.py`, 임계값 없음)보다 AP가 낮게 나온다.

> 🎯 **과제 목표 mAP50 0.85를 달성한 실험**은 `det_P29_egofill_bengio`(**0.8501** @ep9, 공식 v2 test —
> [`experiments/registry.md`](.claude_logs/experiments/registry.md))인데, 그 체크포인트는 **bengio 서버에만 있었고
> 해당 서버는 HW 고장으로 사망**했다(2026-07-16, `status/current.md`). **재현 불가.**

### 4.5 한눈 요약

| bench | 대표 모델 | 로컬에서 뽑히는 지표 | 기대 수치 | 상태 |
|---|---|---|---|---|
| `deliver` | P34-ReliaDINO ep120 | val / test mIoU | **68.19 / 56.62** | ✅ 기대 수치 확보 |
| `muses` | P39.1-rank seed2 ep208 | val mIoU (내부 프로토콜) | **82.62** | ✅ 기대 수치 확보 |
| `muses-official` | 동일 | val mIoU (공식 native) | **TODO** | ⚠️ 기록 없음 |
| `multiaqua` | P9 ep131 | val mIoU | **93.29** (서버 채점 기준) | ⚠️ ckpt 부재 → `CKPT=` 필수 |
| `det` | D1-recovered ep6 | COCO AP/AP50/AP75 | **0.6377 / 0.9321 / 0.7283** | ✅ 기대 수치 확보(annotation 동등성 미검증) |

---

## 5. 트러블슈팅

기록으로 남아 있는 실제 사례만 적었다.

### 5.1 `ModuleNotFoundError: No module named 'sam2'`

lecun/yeon 등 `sam2` editable 설치가 없는 환경에서 SAM2 계열 코드를 돌릴 때 발생(`CLAUDE.md` 산출물 절).

```bash
PYTHONPATH=<repo>/semseg/models/sam2 bash scripts/reproduce_eval.sh multiaqua
```

### 5.2 체크포인트 ↔ config 아키텍처 불일치

- 증상: `AssertionError: state_dict mismatch — 모델/ckpt config 불일치: [...]` (`tools/eval_reliadino_ckpt.py`)
- 원인: ckpt를 만든 config와 다른 config로 모델을 빌드했다. 세대가 다르면(P38↔P39 등) 모듈 구성이 달라진다.
- 조치: [`experiments/registry.md`](.claude_logs/experiments/registry.md) 의 해당 실험 행에서 **config 경로**를 확인하고 `CFG=` 로 지정하라.
- `val.py` 는 `strict=False` 라 죽지 않고 **missing key를 경고로만** 찍는다(`val.py:79`). 그 경고를 무시하면
  random-init 모듈이 섞인 키메라 모델을 평가하게 된다 — 로그의 `missing keys` 를 반드시 봐라.

### 5.3 `.pth` / `_checkpoint.pth` 혼동

레거시 스크립트 한정 문제다. `val_multiaqua.py` 는 dict 포맷을, `val_multiaqua_P9.py` 는 raw state_dict를 기대한다(§3.3).
현행 진입점은 둘 다 처리한다.

### 5.4 OOM / batch size

- 평가 batch는 config `EVAL.BATCH_SIZE`(대부분 4)를 쓴다. 24GB 미만 GPU에서 1024² 4모달은 넘칠 수 있다.
- `tools/eval_reliadino_ckpt.py` 는 `--batch` 인자를 받지만, **프로토콜 요동(~2pt)이 마진보다 클 수 있어**
  기본은 config 값 유지가 원칙이다(스크립트 docstring). 굳이 바꿔야 하면 바꿨다고 명시하고 보고하라.
- 학습 쪽 OOM 이력: A100 40GB에서 4모달 P44는 BS2 → BS1로 내려야 했다(`status/current.md`).

### 5.5 빈 GPU가 없다

```
[에러] 빈 GPU가 없다 (기준: memory.used<=2000MiB & util<=10%).
```
`nvidia-smi` 로 확인하고, 정말 써도 되는 GPU가 있으면 `GPU=<idx>` 로 직접 지정하라.
임계값은 `GPU_MAXMEM` / `GPU_MAXUTIL` 환경변수로 조정한다(`scripts/pick_free_gpus.sh`).

### 5.6 HuggingFace 접근 불가 / 오프라인

DINOv3 가중치는 HF Hub에서 온다. 오프라인이면 다음 경고가 뜬다:

```
[ReliaDINO] all pretrained loads failed — falling back to RANDOM INIT ...
```

**평가에서는 무해하다** — 체크포인트가 백본까지 포함한 전체 state를 덮어쓰기 때문이다(`semseg/models/reliadino/encoder.py:231`).
학습에서는 절대 그냥 넘기면 안 된다. 필요하면 `HF_HUB_OFFLINE=1` 로 명시적으로 오프라인 실행하라.

### 5.7 `nvidia-smi` 는 되는데 `torch.cuda.is_available()` 이 False

드라이버 ↔ CUDA 런타임 불일치(`Error 804: forward compatibility was attempted on non supported HW`).
`torch 2.3.1+cu121` 과 호환되는 드라이버로 맞춰라.

---

## 6. 재현 한계 — 구조적으로 불가능한 것

정직하게 적는다.

| # | 재현 불가 항목 | 이유 |
|---|---|---|
| 1 | **모든 데이터셋 · 체크포인트의 외부 취득** | 연구실 NAS(`/ailab_mat2`, `/drone_nas`) 마운트 전제. 리포에는 어떤 가중치·데이터도 커밋돼 있지 않다. 외부인은 DELIVER/MUSES를 원저자에게서 직접 받고, 체크포인트는 받을 수 없다 |
| 2 | **MUSES test mIoU 79.788** | test GT 비공개. Codabench(대회 14005) 서버 채점 결과이며 로컬에서 계산할 수 없다 |
| 3 | **MULTIAQUA test mIoU 70.91 / M-score 82.10** | 야간 test GT 비공개, MACVi 챌린지 서버 채점 |
| 4 | **MULTIAQUA P9 ep131 로컬 val 재현** | 대표 ckpt가 정규 웨이트 루트에 없다(학습 서버 levine의 `outputs/`에만 존재). `CKPT=`로 직접 넣어야 한다 |
| 5 | **det 목표치 mAP50 0.8501** | 그 ckpt를 보유한 bengio 서버가 HW 고장으로 사망(2026-07-16). 웨이트 회수 불가 |
| 6 | **poongsan indoor 데이터 전체** | 국가 R&D 과제 자체 수집 비공개 데이터 |
| 7 | **MUSES official-protocol val 수치 (seed2)** | 측정된 기록이 없어 기대값을 적을 수 없다(TODO). 명령은 동작하지만 대조군이 없다 |
| 8 | **DELIVER 공개 리더보드와의 직접 비교** | 우리 eval의 GT 리사이즈 규약이 CAFuser/DELIVER 공식 규약과 동일한지 미검증(`monitor-log.md:720`) |
| 9 | **학습 재현(from scratch)** | 이 문서는 **평가 재현** 범위다. 학습은 다중 GPU·수십 시간이 필요하고 서버별 config가 갈린다 — `CLAUDE.md` 와 `.claude_logs/infra/servers-and-launch.md` 참조 |

---

## 7. 더 읽을 것

- [`CLAUDE.md`](CLAUDE.md) — 명령어 canonical, 경로 규약, GPU 사용 규칙
- [`.claude_logs/00_INDEX.md`](.claude_logs/00_INDEX.md) — 연구 로그 front door
- [`.claude_logs/status/current.md`](.claude_logs/status/current.md) — 현재 최선 모델 스냅샷 (단일 출처)
- [`.claude_logs/experiments/registry.md`](.claude_logs/experiments/registry.md) — 실험 ↔ config ↔ ckpt ↔ 수치 대응표
- [`.claude_logs/infra/environment.md`](.claude_logs/infra/environment.md) — 실행 환경·경로·체크포인트 포맷
- [`.claude_logs/issues/issues-and-fixes.md`](.claude_logs/issues/issues-and-fixes.md) — 알려진 이슈
