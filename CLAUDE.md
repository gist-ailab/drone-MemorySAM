# MemorySAM: Multimodal Segmentation via SAM2 Memory Attention

## System Instructions

너는 이 프로젝트의 AI 연구 보조 및 엔지니어이다.
항상 세션 간의 문맥(Context)을 유지하기 위해 아래 규칙을 엄격하게 따른다.

### 1. 세션 시작 시 (Initialization)

새로운 대화나 작업 지시를 받으면, 코드 수정을 시작하기 전에 **반드시** 아래 순서대로 `.claude_logs` 폴더 내의 파일들을 읽어라:

#### Step 0 — 역할 판별 (최우선)
- **가장 먼저** `09_bot_roles_guide.md`를 읽어라.
- 사용자의 첫 메시지에 역할 키워드("코드분석봇", "코딩봇", "실험분석봇", "그림봇")가 포함되어 있으면, 해당 역할의 지침을 이번 세션 전체에 적용한다.
- 역할이 지정되지 않으면 기본 모드(AI 연구 보조 및 엔지니어)로 동작한다.

#### Step 1 — 프로젝트 상태 파악
- **`00_INDEX.md`를 먼저 읽어라** — 폴더 전체를 6개 카테고리(프로젝트·아키텍처·Related Works·실험·환경/인프라·이슈)로 묶은 front door. 어떤 문서를 볼지 여기서 결정한다.
- `01_project_status.md`: **상단 "📌 현재 상태 스냅샷"이 현재 상태의 단일 출처** (하단은 역시간순 history). 전체 진행 상황·현재 최선 모델·남은 과제.
- `02_model_arch.md`: P8~P28 + SAM3-RBMA 모델 아키텍처 상세, 변천 과정, 각 버전의 한계점
- `03_experiment_log.md`: 모든 실험 결과, 체크포인트 경로, 챌린지 제출 결과
- `04_issues_and_fixes.md`: 알려진 이슈, 해결 기록, 코딩 시 주의사항 — **상단 "이슈 상태 인덱스 표" 먼저** (**코드 작성 전 반드시 확인**)
- `12_novelty_and_related_work.md`: **RBMA 노벨티 & 관련연구(canonical)** — 우리 모델 한눈에, 선행연구 vs RBMA 구조 차별표, 리뷰 방어 포인트, lit-check TODO. **연구 방향·논문 포지셔닝 논의 전 반드시 확인.** (원시 deep-research 로그는 `10_related_work.md`)
- `13_servers_and_launch.md`: **서버 레지스트리 & 원격 실험 자동 실행** — "X 실험을 <서버>에서 돌려줘" 류 지시를 받으면 **반드시 먼저 읽어라.** 서버 메타데이터 단일 출처는 `scripts/servers.conf`, 실행/추적은 `scripts/remote_exp.sh`.
- `14_environment_and_infra.md`: 실행 환경/명령, 데이터·가중치 경로, 체크포인트 포맷, DDP, B200 파이프라인 튜닝.

> `.claude_logs` 인덱스: **00 인덱스(front door)** · 01 상태(상단 스냅샷) · 02 모델상세 · 03 실험 · 04 이슈(상단 상태표) · 10 관련연구(raw) · 11 SAM3 plan · **12 노벨티&관련연구(canonical)** · **13 서버&원격실행** · 14 환경·인프라. **먼저 00**, 관련연구/노벨티는 **12**, 원격 학습 지시는 **13**을 읽어라. (05~07·P13_design_guide = 🗄 ARCHIVED)

### 2. 실험 및 코드 변경 시 (Execution)

- 모델 아키텍처를 수정하거나 실험 Config를 생성하면, 작업 후 반드시 `02_model_arch.md` 또는 `03_experiment_log.md`를 업데이트하여 기록을 남겨라.
- 버전(P8, P9, P10 등)을 명시하고, 왜 변경했는지(이전 실험 결과 기반) 타당한 이유를 적어라.
- 실험 결과 파일 경로는 프로젝트 기준 상대 경로로 기록해라.
- 새 선행연구를 조사했거나 RBMA 노벨티/차별점 논의가 갱신되면 `12_novelty_and_related_work.md`(canonical 비교표·판정)를 업데이트하고, 원시 조사 로그는 `10_related_work.md`에 추가해라.

### 3. 구현/작업 완료 시 자동 업데이트 (Auto-update)

- 새 모델 버전 구현, config 생성, 학습/평가 스크립트 수정 등 **의미 있는 작업이 완료되면** 사용자 요청 없이도 자동으로 `.claude_logs/01_project_status.md`를 업데이트해라.
  - 상태 변경 (예: "설계 완료 (구현 대기)" → "구현 완료 (학습 대기)")
  - 변경 파일 목록 및 핵심 내용 기록
  - 디자인 가이드 대비 의도적 차이가 있으면 사유 기록
- 모델 아키텍처 변경이 있었으면 `02_model_arch.md`도 함께 업데이트해라.

### 4. 세션 종료 시 (Wrap-up)

- 사용자가 "작업 끝", "기록해줘" 등의 말을 하면, 이번 세션에서 변경된 사항을 `.claude_logs/` 내 파일들에 요약 추가해라.

---

## 프로젝트 개요

**목표**: MACVi MULTIAQUA Challenge — 드론 촬영 야간 수상 환경에서 RGB + LiDAR + Thermal 멀티모달 세그멘테이션

**핵심 아이디어**: SAM2의 시간축 메모리 어텐션을 모달리티 축으로 전용하여, 멀티모달 Cross-Modal Fusion을 수행. 각 모달리티를 별도 "프레임"으로 인코딩 후, SAM2의 memory attention으로 상호 참조.

**데이터셋**: MULTIAQUA
- 클래스: Static(0), Dynamic(1), Water(2), Sky(3), ignore(255)
- Val = 주간 145장, Test = 야간만 (challenge server 평가)
- 모달리티: RGB (`img`), LiDAR (`lidar`), Thermal (`thermal`)
- 경로: `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night`

**평가 지표**: M-score = 0.75 × val_mIoU + 0.25 × test_mIoU (MACVi Challenge)

---

## 환경 설정

```bash
# Conda 환경
conda activate MMSS_SAM
# 또는 직접 경로: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python

# 학습
python train_sam2_lora_paper.py --cfg configs/<config>.yaml

# 평가 (val)
python val_multiaqua.py --cfg configs/eval_config/<config>.yaml --mode val --model_path <checkpoint_path>

# 평가 (test + challenge 제출)
python val_multiaqua.py --cfg configs/eval_config/<config>.yaml --mode test --model_path <checkpoint_path> --macvi

# P9 전용 시각화 평가 (MoE routing 분석 포함)
python val_multiaqua_P9.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml --mode val
python val_multiaqua_P9.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml --mode test
```

### 원격 서버에서 실험 실행 (tmux 세션 `jemo`)

"X 실험을 <서버>에서 돌려줘" → 아래 런처 사용. 상세는 `.claude_logs/13_servers_and_launch.md`, 서버 목록은 `scripts/servers.conf`.

```bash
# 서버 레지스트리 확인 (repo_path / env / default_gpus)
bash scripts/remote_exp.sh servers
# 서버 상태(빈 GPU + jemo 세션 창)
bash scripts/remote_exp.sh status bengio
# 실행: ssh -> tmux 세션 'jemo' 새 window -> torchrun -> logs/<cfg>/<cfg>_<ts>.log
bash scripts/remote_exp.sh run bengio configs/bengio-multiaqua_rgbtl_P9_hardaug6.yaml 0,1,2,3
# 진행 로그 추적
bash scripts/remote_exp.sh log bengio bengio-multiaqua_rgbtl_P9_hardaug6
```

### 📊 평가/분석 산출물 저장 위치 (모든 세션 공유)

**eval·분석·시각화 로그와 statistics·리포트는 로컬 박스 `/mnt/HDD2/src/logs/` 에 모은다 (단일 정규 위치).**
원격(B200/lecun/yeon)에서 돌린 결과는 휘발/분산되므로 `rsync`로 여기에 `<model>_eval_<YYYYMMDD>/`(예: `P29_eval_20260630/` = `report/` + `viz/` + `perdomain/`) 형태로 회수해 **누적**한다. 새 세션은 다음 작업 전략을 세우기 위해 **이 경로를 먼저 확인**한다.
- 재사용 도구(repo `tools/`): `eval_per_domain.py`(per-condition 러너) · `analyze_per_domain.py`(per-class 분류) · `viz_features.py`(feature/RBMA 패널) · `module_diagnostics.py`(모듈 정량). 모델 무관(`--cfg`/`--model_path`만 교체).
- lecun/yeon에서 SAM2 코드 실행 시 `sam2` editable 미설치면 `PYTHONPATH=<repo>/semseg/models/sam2` 지정.

---

## 핵심 코드 구조

```
drone-MemorySAM/
├── CLAUDE.md                          # 이 파일
├── .claude_logs/                      # AI 세션 로그
│   ├── 01_project_status.md
│   ├── 02_model_arch.md
│   └── 03_experiment_log.md
├── train_sam2_lora_paper.py           # 메인 학습 스크립트
├── val_multiaqua.py                   # 범용 평가 스크립트 (P8~P12)
├── val_multiaqua_P9.py                # P9 전용 시각화 + MoE routing 분석
├── diagnose_moe_gate.py               # MoE gate 진단 스크립트
├── configs/
│   ├── levine-multiaqua_rgbtl_P{8-12}_hardaug{2-4}.yaml  # 학습 configs
│   └── eval_config/                   # 평가 configs (MODEL_PATH 포함)
├── semseg/
│   └── models/sam2/sam2/
│       ├── sam_lora_image_encoder_seg.py  # LoRA_Sam_P8~P12 모델 정의
│       ├── sam_lola_utils.py              # SoftMoE_LoRA_Layer 등 유틸리티
│       └── checkpoints/
│           └── sam2.1_hiera_base_plus.pt  # SAM2 pretrained weight
└── outputs/
    ├── MMSamP8/   # P8 실험 결과들
    ├── MMSamP9/   # P9 실험 결과 (현재 최선)
    ├── MMSamP10/  # P10 실험 결과 (취소됨)
    └── MMSamP11/  # P11 실험 결과 (취소됨)
```

---

## 모델 버전 요약

| 버전 | 핵심 변경 | 최선 M-score | 상태 |
|------|----------|-------------|------|
| P8 | ConfidenceHeadV2 + sigmoid UAMM | 78.45 | hardaug 기반실험 완료 |
| **P9** | CrossModalFusionHead + max-norm UAMM | **81.98** (hardaug8 ep131) | **현재 최선** |
| P10 | CrossModalFusionHeadV2 + ModalAuxHead + oracle KL | 79.27 | 취소 (test 성능 하락) |
| P11 | P10 + MI routing loss | 77.09 | 취소 (MoE gate 진단 우선) |
| P12 | Input-Conditioned Soft MoE LoRA (cond_dim) | - | 설계만 완료 |
| P24 | P9 + SpatialQualityGating (scalar UAMM/AMF + CE teacher) | - | 학습 중 |
| P25 | Unified Spatial Quality Fusion (spatial UAMM/AMF, no CrossModalFusionHead) | - | 구현 완료 (학습 대기) |

**현재 최선 모델: P9 hardaug8_physaug ep131** — `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth`

---

## 주의사항

1. **Checkpoint 포맷 차이**: `.pth` = raw state_dict, `_checkpoint.pth` = `{'model_state_dict': ..., 'optimizer_state_dict': ..., ...}` 형태. `val_multiaqua.py`는 `_checkpoint.pth`를 기대하고, `val_multiaqua_P9.py`는 `.pth`를 직접 로드.
2. **Val vs Test 갭**: Val mIoU ~93-94% (주간) vs Test mIoU 58-70% (야간). 모든 모델이 이 갭을 보임.
3. **MoE Gate "Uniform" 문제**: 공간 평균(`_gate_callback`) 결과 uniform으로 보이지만, per-token 분석 시 실제로는 분화되어 있음 (entropy_ratio=0.55, max_weight=0.72). 측정 artifact임.
4. **NIGHT_AUG**: 야간 시뮬레이션 증강. hardaug4가 최종 튜닝 버전. `BRIGHTNESS_SAMPLING: dark_biased`로 극저조도 편향.
5. **DDP 학습**: `TRAIN.DDP: True`로 멀티GPU 학습. 단일 GPU 시 `train_sam2_lora_paper_singlegpu.py` 사용.
6. **🔴 GPU 가용성 확인 (모든 학습 실행 전 필수)**: 어떤 실험이든 돌리기 **전에 반드시 해당 서버의 빈 GPU를 확인하고, 비어 있는 GPU에만** 배치한다(사용 중 GPU에 얹지 않는다 → OOM/타인 작업 방해).
   - **로컬 런처**(`run_sam.sh` / `run_sam3_train.sh` / `run_sam3_rbma.sh`): `CUDA_VISIBLE_DEVICES`를 직접 주지 않으면 **`scripts/pick_free_gpus.sh`로 빈 GPU를 자동 선택**한다. 개수는 `NGPU=` (SAM2/3 train) 또는 `NPROC=` (rbma)로 지정. 빈 GPU가 부족하면 실행을 거부한다.
     - 예: `NGPU=4 bash run_sam.sh` · `NGPU=1 bash run_sam3_train.sh` · `CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash run_sam3_rbma.sh <cfg>`(직접 지정은 그대로 존중).
   - **원격 런처**(`scripts/remote_exp.sh`): 먼저 `status <server>`로 확인하고, `run <server> <cfg> auto:N`으로 **원격의 빈 GPU N장을 자동 배정**한다(`auto`=1장). 빈 GPU가 없으면 거부.
   - 판정 기준: GPU가 `memory.used ≤ 2000MiB && util ≤ 10%`이면 "빈 GPU"(환경변수 `GPU_MAXMEM`/`GPU_MAXUTIL`로 조정). 헬퍼/`auto`는 메모리 적은 순으로 고른다.
