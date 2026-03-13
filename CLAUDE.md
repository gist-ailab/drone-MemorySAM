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
- `01_project_status.md`: 전체 진행 상황, 현재 최선 모델, 남은 과제
- `02_model_arch.md`: P8~P13 모델 아키텍처 상세, 변천 과정, 각 버전의 한계점
- `03_experiment_log.md`: 모든 실험 결과, 체크포인트 경로, 챌린지 제출 결과
- `04_issues_and_fixes.md`: 알려진 이슈, 해결 기록, 코딩 시 주의사항 (**코드 작성 전 반드시 확인**)

### 2. 실험 및 코드 변경 시 (Execution)

- 모델 아키텍처를 수정하거나 실험 Config를 생성하면, 작업 후 반드시 `02_model_arch.md` 또는 `03_experiment_log.md`를 업데이트하여 기록을 남겨라.
- 버전(P8, P9, P10 등)을 명시하고, 왜 변경했는지(이전 실험 결과 기반) 타당한 이유를 적어라.
- 실험 결과 파일 경로는 프로젝트 기준 상대 경로로 기록해라.

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
