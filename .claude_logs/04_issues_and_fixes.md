# 이슈 및 해결 기록 (Issues & Fixes)

> 최종 업데이트: 2026-02-25
> 코딩 세션은 이 파일을 읽고 동일한 실수를 반복하지 말 것

---

## 열린 이슈 (Open Issues)

### ISSUE-001: Val에 NIGHT_AUG 미적용 → 모델 선택 기준 부적합 [심각]

**상태**: ✅ 해결됨 (2026-02-25)
**영향**: 모든 P 버전 (P8~P13)

**문제**:
- `get_val_augmentation()` (`semseg/augmentations_mm.py` line 593)에 NIGHT_AUG 없음
- Val = 주간 이미지 그대로 평가 → val mIoU = 주간 성능만 반영
- Test = 야간 이미지 → val best checkpoint ≠ test best checkpoint
- 실제로 모든 모델의 val mIoU가 93~94%로 거의 동일하여 모델 구분 불가
- 하지만 test mIoU는 35~70%로 편차 매우 큼

**해결**:
- `get_nightval_augmentation()` 함수 신규 추가 (`semseg/augmentations_mm.py` line 608~)
  - 기존 `get_val_augmentation()`은 수정하지 않음 (호환성 유지)
  - NightSim p=1.0 (항상 적용, dice-roll 제거)
  - CRM / Zero-out은 config 확률 그대로 (더 realistic)
  - 기하학적 증강(Flip, Crop) 없음
- `train_sam2_lora_paper.py` 변경:
  - `get_nightval_augmentation` import 추가
  - `nightvalset` / `night_valloader` 생성 (NIGHT_AUG.ENABLE 시에만)
  - `best_night_mIoU` / `best_night_epoch` 상태 변수 추가
  - Night-Val 평가 블록 추가 (`val_night/mIoU` TensorBoard 로깅)
  - `night_epoch{N}_{mIoU}_checkpoint.pth` 별도 저장
  - resume checkpoint에서 `best_night_miou` / `best_night_epoch` 복원
  - 최종 summary 테이블에 "Best Night-Val mIoU" 행 추가

**체크포인트 구분**:
- `epoch{N}_{mIoU}_checkpoint.pth`       → Day-Val best (주간 성능)
- `night_epoch{N}_{mIoU}_checkpoint.pth` → Night-Val best (야간 시뮬 성능)

**미해결 (후속 과제)**:
- hardaug4의 brightness 분포 vs 실제 test(lj4) brightness 분포 비교 → Night-Val이 test를 얼마나 잘 proxy하는지 검증 필요
- P13 학습 후 day-best vs night-best의 test mIoU 비교 필요

**관련 파일**:
- `semseg/augmentations_mm.py`: `get_nightval_augmentation()` (line 608~)
- `train_sam2_lora_paper.py`: night-val 평가 블록 (line 602~632)

---

### ISSUE-002: MoE Expert Collapse — Block 6-20에서 E1 사망 [중요]

**상태**: P13에서 수정됨 (검증 필요)
**영향**: P8, P9, P10, P11 (experts_b zero-init 사용하는 모든 버전)

**문제**:
- `SoftMoE_LoRA_Layer.reset_parameters()` (`sam_lola_utils.py` line 562~575)
- `experts_b` zero-init → 모든 expert 출력 = 0 → gate gradient = 0
- Rich-get-richer 현상 → Block 6-20 (15개, stage 3)에서 E1 사용률 < 3%
- 3-expert MoE가 실질적으로 2-expert로 동작, 용량 1/3 낭비

**진단 데이터** (val_pred_P9/uamm_amf_moe_log.json):
```
Block9_Q argmax_fraction:
  img:     E0=43~51%, E1=0~10%, E2=40~54%
  lidar:   E0=16~17%, E1=0~0.5%, E2=83~84%
  thermal: E0=84%,    E1=0.3~0.9%, E2=15~16%
  → E1 거의 미사용
```

**P13 수정 방법**:
- `LoRA_Sam_P13.__init__`에서 experts_b를 `kaiming_uniform_ * 0.01`로 재초기화
- `sam_lola_utils.py`는 수정하지 않음 (P9 체크포인트 호환성 유지)

**검증 방법**:
- P13 학습 후 Block9_Q에서 E1 argmax_fraction > 10% 확인
- 학습 초기(epoch 1~5)에 expert usage 로깅

---

### ISSUE-003: CrossModalFusionHead 상수 출력 [중요]

**상태**: P13에서 수정됨 (검증 필요)
**영향**: P9 (P10/P11은 HeadV2 사용하지만 유사 문제)

**문제**:
- `CrossModalFusionHead` (P9)의 UAMM/AMF 출력이 모든 이미지에서 동일:
  - UAMM: img=0.745, lidar=0.961, thermal=1.0
  - AMF: img=0.275, lidar=0.355, thermal=0.370
- 원인: GAP(65536 토큰 평균) + LayerNorm 정규화 → 입력 무관하게 같은 벡터
- 결과: adaptive fusion이 아닌 fixed fusion. 밤에 RGB가 어두워도 27.5% 가중치

**P13 수정 방법**:
- CrossModalFusionHead 제거 → ConfidenceAuxHead + Energy Score로 교체
- Energy Score는 aux head의 raw logit에서 계산 (학습 파라미터 없음)
- 학습/추론 동일 메커니즘 (P10의 train≠test 문제 없음)

**검증 방법**:
- P13 학습 후 uamm_amf_moe_log.json에서 이미지별 UAMM/AMF 값이 다른지 확인
- 특히 밤(어두운) 이미지 vs 밝은 이미지에서 img 가중치 차이 확인

---

### ISSUE-004: Spatial-wise Energy Weighting 확장 가능성 [아이디어]

**상태**: 보류 (P13 image-level 결과 보고 결정)
**영향**: P13 이후

**아이디어**:
- 현재 P13: Energy Score를 spatial mean → 이미지당 스칼라 1개
- 확장: mean 없이 (B, H_feat, W_feat) 유지 → feature map 위치마다 다른 가중치
- 예: 가로등 근처 RGB 토큰 → 높은 가중치, 어두운 영역 RGB 토큰 → 낮은 가중치

**보류 이유**:
- image-level만으로도 P9 대비 큰 개선 예상 (상수→가변)
- 한번에 두 가지 바꾸면 효과 분리 불가
- P13 결과 확인 후 추가 개선으로 시도

---

### ISSUE-005: 야간 합성 데이터 생성 — Diffusion 기반 Day→Night [아이디어]

**상태**: 보류 (P13 + Night-Val 결과 확인 후 결정)
**영향**: 전체 학습 파이프라인
**우선순위**: P13 학습 → 결과 분석 → 이후 병렬 진행 가능

**배경**:
- Val(주간) 93% vs Test(야간) 70% 갭이 핵심 병목
- NIGHT_AUG(프로그래밍 방식)로 no-aug 대비 +33.7 개선 (35.93→69.62)
- 하지만 NIGHT_AUG는 global brightness/contrast 조절 수준 → 실제 야간과 괴리
  - 가로등 조명, 수면 반사, 불균일 조명 패턴 등 미반영
- 실제 야간 데이터 추가 수집 불가 (드론, 수상 환경)

**접근 방법**: Flux/SDXL img2img + ControlNet
```
입력: 주간 RGB (145장) + segmentation GT (ControlNet 조건)
       ↓ Flux img2img (prompt: "nighttime, dark, drone aerial view, water")
       ↓ ControlNet(segmentation map) → 구조 보존
출력: 야간 합성 RGB (145장)

LiDAR: 원본 그대로 사용 (능동 센서, 주야 무관)
Thermal: 원본 그대로 사용 (열 기반, 구조 유지)
Label: 원본 GT 그대로 사용 (ControlNet이 구조 보존)
```

**장점**:
- Flux pretrained 사용 → 별도 학습 불필요 (inference only)
- ControlNet(seg map)으로 구조 보존 → label 일관성 높음
- 다양한 야간 조건 생성 가능 (달빛, 가로등, 완전 암흑 등)
- 학습 데이터 2배 (주간 145 + 야간합성 145)

**리스크**:
- Diffusion이 수상 환경/드론 시점에 최적화되어 있지 않을 수 있음
- 생성된 야간 RGB와 원본 LiDAR/Thermal 간 consistency 검증 필요
- 생성 품질에 따라 오히려 학습에 노이즈가 될 수 있음

**실행 계획 (P13 결과 확인 후)**:
1. Flux img2img + ControlNet(seg) 파이프라인 구축
2. 10장 샘플 생성 → 품질/구조보존 검증
3. 검증 통과 시 전체 145장 생성
4. 주간+야간합성 혼합 학습 실험
5. Night-Val + Challenge 제출로 효과 측정

**관련 도구**:
- Flux.1-dev / SDXL (Hugging Face diffusers)
- ControlNet (segmentation condition)
- 별도 GPU 필요 (학습과 병렬 실행 가능)

---

## 해결된 이슈 (Resolved Issues)

### RESOLVED-001: MoE Gate "Uniform" 분포 — 측정 Artifact

**해결일**: 2026-02-25
**영향**: P9 분석/진단

**문제**: `_gate_callback` (`sam_lola_utils.py` line 546-548)이 gate_weights의 spatial mean을 계산 → 65536개 토큰 평균이 CLT에 의해 항상 ~1/3으로 수렴 → "gate가 uniform"으로 잘못 해석

**해결**: per-token 분석 (entropy_ratio, argmax_fraction) 수행 → gate는 실제로 결정적 routing 수행 중 (Block9: entropy_ratio=0.22~0.25, max_weight=0.87)

**교훈**: 공간 평균은 per-token 다양성을 숨김. 항상 per-token 통계로 분석할 것.

---

### RESOLVED-002: P10/P11 Test 성능 하락

**해결일**: 2026-02-25 (원인 규명, P10/P11 취소)

**문제**: P10 M=79.27, P11 M=77.09 → P9(81.47) 대비 하락

**원인**:
- P10: Oracle KL loss가 주간(val) GT에 과적합 → 학습 시 oracle 있음, test 시 없음 → 메커니즘 불일치
- P11: MI loss가 이미 정상 작동하는 gate에 불필요한 제약 → 학습 방해
- 공통: 복잡도 추가 → overfitting 가속

**교훈**:
1. 학습 시와 추론 시 동일한 메커니즘을 사용해야 함
2. 이미 작동하는 컴포넌트에 추가 loss를 넣지 말 것
3. 진단(분석) 없이 loss/모듈을 추가하지 말 것

---

### RESOLVED-003: val_multiaqua_P9.py SyntaxError

**해결일**: 2026-02-25

**문제**: `from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *`가 함수 내부에 위치 → `SyntaxError: import * only allowed at module level`

**해결**: wildcard import를 `raise ValueError(f"Unknown LORA_MODEL: {lora_model_name}")` 으로 교체

---

### RESOLVED-004: Title Bar 흰색 마진 (val_multiaqua_P9.py 시각화)

**해결일**: 2026-02-25

**문제**: `_add_title_to_image()`에서 `plt.subplots()` + `tight_layout(pad=0)` 사용 → 흰색 padding 잔류

**해결**: `fig.add_axes([0, 0, 1, 1])` + `fig.patch.set_facecolor('#1a1a2e')` 로 전체 figure를 채움

---

## 코딩 시 주의사항 (Common Pitfalls)

### 1. Checkpoint 포맷 차이
- `.pth` = raw state_dict (`torch.load()` → dict of tensors)
- `_checkpoint.pth` = `{'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}`
- `val_multiaqua.py`는 `_checkpoint.pth` 기대, `val_multiaqua_P9.py`는 `.pth` 직접 로드
- 새 스크립트 작성 시 양쪽 포맷 모두 처리할 것

### 2. LoRA 모델 import
- P8~P13이 모두 `sam_lora_image_encoder_seg.py`에 있음
- config의 `LORA_MODEL` 값으로 동적 선택: `LoRA_Sam_P8`, `LoRA_Sam_P9`, ..., `LoRA_Sam_P13`
- wildcard import (`from ... import *`)를 함수 내부에서 사용하면 SyntaxError

### 3. MULTIAQUA 데이터셋 특수사항
- 클래스: Static(0), Dynamic(1), Water(2), Sky(3), ignore(255)
- Val = 주간 145장 (정답 있음), Test = 야간 200장 (정답 없음, challenge server 평가)
- Recording Boat 영역 = ignore(255) → 평가에서 제외, 시각화 시 회색 처리
- 이미지 크기가 다양 → ResizeWidthPadToSquare로 전처리

### 4. DDP 학습 관련
- `TRAIN.DDP: True` 시 `torchrun` 또는 `torch.distributed.launch` 사용
- 단일 GPU: `train_sam2_lora_paper_singlegpu.py` 또는 DDP=False
- LoRA parameter만 학습 → backbone은 freeze

### 5. SAM2 Memory Attention 순서
- 모달리티 처리 순서: img → lidar → thermal (config의 MODALS 순서)
- 각 모달리티가 이전 모달리티의 memory를 참조
- 순서 변경 시 성능이 달라질 수 있음 (미실험)

### 6. experts_b init 수정 위치 (P13)
- `sam_lola_utils.py`의 `reset_parameters()`를 직접 수정하면 P9 등 기존 모델에 영향
- P13에서는 `__init__`에서 LoRA 설치 후 experts_b만 재초기화하는 방식으로 구현
- 기존 모델 호환성 유지
