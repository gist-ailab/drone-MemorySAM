# 이슈 및 해결 기록 (Issues & Fixes)

> 최종 업데이트: 2026-02-27
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

**상태**: ❌ P13에서 수정 시도했으나 **해결 실패** (2026-02-26 검증 완료)
**영향**: P8, P9, P10, P11, P12, P13 (전 버전)

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

**P13 수정 시도**:
- `LoRA_Sam_P13.__init__`에서 experts_b를 `kaiming_uniform_ * 0.01`로 재초기화
- `sam_lola_utils.py`는 수정하지 않음 (P9 체크포인트 호환성 유지)

**P13 검증 결과 (2026-02-26)**:
- Collapse rate: P13 val 17.4% vs P12 val 16.0% → **개선 없음 (오히려 소폭 악화)**
- LiDAR collapse: ~27% (P12와 동일)
- Q blocks: 23-25% collapse, V blocks: 10-11% (P12와 동일 패턴)
- Stage별: S1(44-55%) > S4(30%) > S2(20%) > S3(9-13%)
- 실패 원인:
  1. Resume 학습 → 이전 gate weights가 로드되면서 init 효과 무력화
  2. kaiming * 0.01 (~0.005 수준)은 zero-init과 실질적 차이 미미
  3. 근본 원인이 init이 아니라 soft-MoE softmax의 winner-take-all 특성

**미해결**: 근본적인 해결책 필요 (load balancing loss, top-k routing, expert dropout 등)

---

### ISSUE-003: CrossModalFusionHead 상수 출력 [중요]

**상태**: ⚠️ P13에서 수정됨 — **부분 성공** (2026-02-26 검증 완료)
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

**P13 검증 결과 (2026-02-26)**:
- UAMM CV (이미지별 변동성): img val 0.112 (P12: 0.005, **22x 증가**) → **상수 수렴 문제 해결**
- Test에서도 img CV=0.073 (P12: 0.014, 5x 증가)
- **단, test LiDAR UAMM = 1.0 고정 (CV=0.000)** — LiDAR aux head가 항상 가장 높은 energy 출력
- Dynamic IoU +5.55pp 개선 → energy fusion이 모달리티 가중치를 유의미하게 변경
- Val mIoU -0.87pp → adaptive weight의 정확도가 P9의 안정적 상수 비율보다 val에서 불리

**결론**: 상수 출력 문제 자체는 해결됨. 하지만 adaptive weight의 **정확도**가 새로운 병목.

---

### ISSUE-006: Aux Head Mask 시각화 — Energy Score 검증 [구현 필요]

**상태**: 🔲 미구현
**우선순위**: 높음 — P13 Energy Score fusion의 실제 동작 검증 및 P14 방향 설정에 필수
**영향**: P13 평가/분석

**목적**:
- Aux head가 모달리티별로 무엇을 보고 있는지 시각적 확인
- Energy Score가 "잘못된 confidence"를 주는 케이스 식별 (특히 test LiDAR UAMM=1.0 문제)
- Thermal aux mask가 야간에 RGB보다 Dynamic/Sky를 더 잘 잡는 프레임 확인
- P14 설계에 필요한 실증 데이터 수집

**구현 사양**:

1. **저장 대상** (각 이미지에 대해):
   - RGB / LiDAR / Thermal 입력 이미지 (3장)
   - Aux head prediction mask — 모달리티별 argmax 컬러맵 (3장)
     - `aux_logits_img`, `aux_logits_lidar`, `aux_logits_thermal` 각각에 argmax
     - 컬러맵: Static=빨강, Dynamic=초록, Water=파랑, Sky=노랑, ignore=회색
   - Final prediction mask (1장)
   - Ground truth (val에서만, test는 없음)
   - Energy confidence weights: img/lidar/thermal 각 스칼라 값 (이미지 파일명에 포함 또는 별도 JSON)
   - UAMM/AMF weights: 이미지당 값

2. **출력 형식**:
   - 한 이미지당 1개의 panorama 이미지 (가로 배치):
     ```
     [RGB] [LiDAR] [Thermal] | [Aux_RGB] [Aux_LiDAR] [Aux_Thermal] | [Pred] [GT]
     ```
   - 각 aux mask 위에 energy confidence 값 표시 (e.g., "img: 0.28, E=3.42")
   - 파일명: `{image_id}_auxmask.png`
   - 별도 JSON: `{image_id}_energy.json` (수치 데이터)

3. **실행 위치**:
   - `val_multiaqua.py`에 추가 (`val_multiaqua_P9.py`는 삭제된 상태)
   - `--save-auxmask` 플래그로 활성화
   - 출력 디렉토리: `{save_dir}/auxmask/`

4. **코드 변경 필요 사항**:
   - `LoRA_Sam_P13.forward()`에서 `aux_logits`를 반환하도록 수정 (현재는 loss 계산에만 사용)
   - 또는 eval 모드에서 별도로 aux head forward를 호출
   - `compute_energy_confidence()`의 중간 값 (per-modality energy)도 저장

5. **분석 포인트** (시각화 결과 확인 시):
   - [ ] LiDAR aux mask가 Water/Dynamic을 못 잡는데 UAMM=1.0인 프레임 확인
   - [ ] Thermal aux mask가 Dynamic을 RGB보다 잘 잡는 프레임 비율
   - [ ] RGB aux mask가 야간에 완전히 깨지는 프레임에서 Energy Score가 RGB weight를 낮추는지
   - [ ] Sky를 잘못 예측하는 모달리티가 어느 것인지 (Sky -1.42pp 하락 원인)

**관련 파일**:
- `val_multiaqua.py`: 평가 스크립트 (시각화 추가 대상)
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: LoRA_Sam_P13, ConfidenceAuxHead
- `val_pred_P13/`: 기존 P13 평가 결과 디렉토리

---

### ISSUE-007: CRM/ZERO Overfitting — Night-Val↑ Test↓ 역전 현상 [부분 해결]

**상태**: 🟡 hardaug5에서 CRM/ZERO 제거 완료 (2026-02-27). 하지만 Sky collapse 여전 → 부분 원인.
**영향**: P13 (epoch39에서 발견), 잠재적으로 모든 P 버전
**우선순위**: 최고 — P13 epoch39에서 test mIoU -19.5pp 폭락 유발

**문제**:

CRM (`RandomRGBComplementaryMasking`, p=0.35)과 ZERO (`RandomRGBZeroOut`, p=0.09)가 RGB에 **exact zero** 값을 삽입하여 train-test 분포 불일치를 유발.

1. **Exact zero는 실제 센서 데이터에 없음**: 야간 RGB는 noise가 있는 near-zero (0.001~0.01), 절대 exact 0이 아님
2. **Normalize 후 고유한 feature vector 생성**: `(0-mean)/std = (-2.118, -2.036, -1.804)` — 자연 이미지에서 나타나지 않는 극단값
3. **Shortcut 학습**: "exact zero 감지 → RGB 무시, thermal/LiDAR 의존" — train/night-val에서는 유효, test에서는 무효
4. **Night-val 오염**: `get_nightval_augmentation()`에도 CRM/ZERO가 동일 확률로 적용 → night-val이 shortcut을 보상 → checkpoint 선택 오염

**정량적 증거** (P13):

| 지표 | Epoch17 | Epoch39 | Δ |
| --- | --- | --- | --- |
| Night-val | 87.71 | **89.53** (+1.82) | ✅ shortcut 학습 강화 |
| Test mIoU | 69.98 | **50.48** (-19.50) | ❌ shortcut 무효 |
| Test Sky | 75.12 | **23.36** (-51.76) | Sky가 가장 취약 |
| Test Sky=0 프레임 | 5/200 | **80/200** | 16배 증가 |

Sky가 가장 심각한 이유: 야간 하늘은 near-zero RGB → CRM/ZERO의 exact zero와 가장 유사 → shortcut이 가장 활성화되는 영역

**권장 조치**:

1. **Night-val에서 CRM/ZERO 제거** (`get_nightval_augmentation()`에서 CRM/ZERO 비활성화)
   - NightSim만 적용 → 실제 test 조건에 더 가까운 proxy
2. **학습 시 CRM/ZERO 확률 축소**: CRM_P 0.35→0.10, ZERO_P 0.09→0.03
3. **Exact zero → Noisy near-zero 대체**: `img[mask] = torch.randn_like(...) * 0.01`
4. **Early stopping 기준 개선**: night-val (CRM/ZERO 제거 버전)을 checkpoint 선택 기준으로 사용

**관련 코드**:

- `semseg/augmentations_mm.py`: `RandomRGBComplementaryMasking` (line 142), `RandomRGBZeroOut` (line 168), `get_nightval_augmentation` (line 609)
- Train config의 `NIGHT_AUG.CRM_P`, `NIGHT_AUG.ZERO_P`

---

### ISSUE-010: 로깅 시스템 전면 개선 — 모듈별 동작 모니터링 부재 [부분 해결]

**상태**: 🟡 Training script 부분 해결 (2026-02-27). Eval script 미해결.
**우선순위**: 중간 — Training 로깅은 trackio로 대폭 개선됨
**영향**: train_sam2_lora_paper.py, val_multiaqua.py, 전 P 버전

**문제 요약**:

모델이 매 forward에서 `_last_uamm_scores`, `_last_amf_weights`, `_last_moe_gates`, `_last_aux_logits` 등을 내부 버퍼에 저장하지만, **학습 스크립트가 이 버퍼를 한 번도 읽지 않음**. 평가 스크립트도 일부만 사용. 결과적으로 fusion, MoE routing, expert collapse, aux head 품질을 학습 중에 전혀 모니터링할 수 없음.

---

#### A. Training Script (train_sam2_lora_paper.py) 빈틈

**TensorBoard에 기록 안 되는 것들:**

| 누락 항목 | 심각도 | 현재 상태 | 추가할 TB key |
|-----------|--------|-----------|--------------|
| Gate loss | HIGH | tqdm에만 표시, 학습 후 소실 | `train/gate_loss` |
| MI loss | HIGH | tqdm에만 표시, 학습 후 소실 | `train/mi_loss` |
| UAMM per modality | HIGH | 학습 중 미수집 | `train/uamm_img`, `_lidar`, `_thermal` |
| AMF per modality | HIGH | 학습 중 미수집 | `train/amf_img`, `_lidar`, `_thermal` |
| Aux loss per modality | HIGH | 3 모달리티 합산 후 기록 | `train/aux_loss_img`, `_lidar`, `_thermal` |
| Per-class IoU (매 eval) | MEDIUM | new best일 때만 텍스트 | `val/iou_static`, `_dynamic`, `_water`, `_sky` |
| Night per-class IoU | MEDIUM | new best일 때만 텍스트 | `val_night/iou_static`, `_dynamic`, `_water`, `_sky` |
| MoE routing entropy | MEDIUM | 미수집 | `train/moe_entropy_mean` |
| Expert collapse count | MEDIUM | 미수집 | `train/expert_collapse_count` |

**이전 TensorBoard 기록은 6개 스칼라만**: `train/loss`, `train/proto_loss`, `train/aux_loss`, `train/lr`, `val/mIoU`, `val_night/mIoU`

**2026-02-27 개선 (trackio 전환)**:

- TensorBoard → trackio 전환 (TensorBoard fallback 유지)
- Training: total_loss, seg_loss, proto_loss, aux_loss, gate_loss, mi_loss, lr, warmup_ramp
- Day-Val: mIoU, pixel_acc, mean_f1, per-class IoU/acc/f1, best_mIoU
- Night-Val: 동일한 포괄적 메트릭 세트 (`val_night/` prefix)
- tqdm: 0값 loss 숨김, P16 warmup 상태 표시

**구현 방법**:
1. 매 eval 주기에 모델 버퍼에서 `_last_uamm_scores`, `_last_amf_weights` 읽어 TB에 기록
2. Aux loss 계산 루프에서 per-modality loss를 리스트로 따로 저장 후 개별 기록
3. Gate loss / MI loss를 epoch 평균으로 TB에 기록 (현재 tqdm 표시 코드 바로 옆에 추가)
4. `print_iou()`의 per-class 결과를 매 eval마다 TB에 기록 (new best 조건 제거)

---

#### B. Evaluation Script (val_multiaqua.py) 빈틈

| 누락 항목 | 심각도 | 현재 상태 | 추가할 형식 |
|-----------|--------|-----------|------------|
| Per-block MoE gate weights | HIGH | 24블록 mean으로 축소 → 블록별 정보 소실 | JSON dict (block별) |
| Expert utilization / entropy per block | HIGH | 미수집 | JSON |
| Energy Score raw values per modality | HIGH | softmax 후 weight만 저장, 원시 energy 폐기 | JSON per-image |
| Aux head predictions (ISSUE-006) | HIGH | `_last_aux_logits` 저장되지만 미사용 | PNG + JSON |
| Confusion matrix | MEDIUM | `metrics.hist` 계산되지만 미저장 | PNG heatmap |
| Per-image IoU | MEDIUM | aggregate만 | CSV |

**Per-block MoE gate 수정 방법**:
현재 코드가 `np.stack(moe_gate_collector, axis=0).mean(axis=0)`로 즉시 축소.
→ mean 대신 블록별 dict로 저장: `{"block0_Q": [e0, e1, e2], "block0_V": [e0, e1, e2], ...}`

**Energy Score raw values 수정 방법**:
`compute_energy_confidence()` 내부에서 중간값(per-modality raw energy, softmax 전 값)을 반환하도록 수정.
→ return `(weights, raw_energies)` 형태로 변경, `_last_energy_raw` 버퍼 추가.

---

#### C. 삭제된 스크립트

`val_multiaqua_P9.py`가 삭제된 상태. CLAUDE.md에서 참조 중이나 실제 파일 없음.
- per-block MoE routing 분석 기능이 사라짐
- **권장**: `val_multiaqua.py`에 해당 기능을 통합하거나, 별도 진단 스크립트로 분리

---

#### D. 구현 우선순위

1. **즉시 (ISSUE-006과 함께)**: Aux mask 시각화 + Energy raw values 저장
2. **단기**: Per-block MoE gate를 JSON에 블록별로 기록 (mean 축소 제거)
3. **단기**: Aux loss per-modality 분리 기록 (TB)
4. **단기**: Gate loss / MI loss TB 기록 (tqdm 옆에 1줄 추가)
5. **중기**: Per-class IoU 매 eval TB 기록, Confusion matrix 저장
6. **중기**: Expert collapse 자동 감지 + 경고 (training 중)

**관련 파일**:
- `train_sam2_lora_paper.py`: Training TB logging 추가 대상
- `val_multiaqua.py`: Eval logging 확장 대상
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: 모델 버퍼 접근 (`_last_*`)
- `semseg/models/sam2/sam2/sam_lola_utils.py`: `SoftMoE_LoRA_Layer._gate_callback`

---

### ISSUE-004: Spatial-wise Confidence Weighting → P15 구현 예정

**상태**: **P15로 구현 예정** (설계 완료, 구현 대기)
**영향**: P15
**상세 설계**: `02_model_arch.md` P15 섹션 참조

**아이디어**:
- P13/P14: confidence를 spatial mean → 이미지당 스칼라 1개 `(B, m)`
- P15: mean 없이 `(B, m, H_feat, W_feat)` 유지 → **위치마다 다른 모달리티 가중치**
- 예: 가로등 근처 RGB 토큰 → 높은 가중치, 어두운 영역 RGB 토큰 → 낮은 가중치

**P15에서 ISSUE-004와 함께 수정되는 문제 (ISSUE-009 통합)**:

1. **Spatial-wise**: `(B, m)` → `(B, m, H, W)` (본 이슈)
2. **Energy → Calibrated Entropy**: "confident but wrong" 문제 해결 (ISSUE-009)
3. **Gradient 격리**: `.detach()` 적용 (ISSUE-008 gradient 경로 문제)
4. **Aux Warmup**: 초기 N epoch uniform weight → aux head 충분히 학습 후 활성화

**기대 효과**:

- Sky 영역: LiDAR 억제 (LiDAR는 상공 포인트 없음) → Sky IoU 하락 방지
- Water 영역: RGB 억제 (야간 수면 암전) → LiDAR/Thermal 활용
- Dynamic 영역: 위치별 최적 모달리티 선택 → Dynamic IoU 개선

**전제 조건 (P14 결과에서 확인된 위험)**:

- Spatial confidence map의 정확도는 aux mask 품질에 의존
- P14에서 aux mask가 여전히 GT 대비 부정확 → spatial confidence도 부정확할 가능성
- 하지만 entropy 기반은 energy 기반보다 "confident but wrong" 케이스에 강건 (ISSUE-009 참조)

---

### ISSUE-008: Aux Head 품질 한계 — Frozen Backbone Feature 정보량 부족 [구조적]

**상태**: 🔴 확인됨 (2026-02-27). P13/P14 공통 근본 문제.
**영향**: Energy Score fusion 방식 전체 (P13, P14, P15)
**우선순위**: 높음 — aux mask 품질이 Energy Score의 전제조건

**문제**:

P13(공유 aux head)과 P14(독립 aux head) 모두에서 aux mask 품질이 GT 대비 매우 부정확. 모달리티 간 "어느 것이 낫다" 비교 자체가 불가능한 수준.

**근본 원인**: Frozen SAM2 Hiera backbone feature의 정보량 한계

1. SAM2는 자연 이미지(SA-1B)로 pretrained → 야간 수상 환경, LiDAR 점군, Thermal gradient의 모달리티별 특성이 feature에 잘 인코딩되지 않음
2. Backbone이 frozen → 새로운 도메인에 적응 불가. LoRA만으로는 feature 자체의 품질을 근본적으로 바꿀 수 없음
3. Aux decoder(Conv 2-3 layer)가 아무리 커져도 입력 feature의 정보 부족을 보상할 수 없음

**Aux decoder 크기 실험 (P13 vs P14)**:

| | P13 (공유 1개) | P14 (독립 3개) | 차이 |
|---|---|---|---|
| Aux Head | ConfidenceAuxHead (1×1 conv) | ModalAuxDecoder (3×3 conv) | 독립화 + 확대 |
| Aux mask 품질 | GT와 큰 괴리 | **소폭 개선, 여전히 부족** | 유의미한 개선 없음 |
| LiDAR UAMM | 1.0 고정 (test) | 1.0 고정 (test) | **동일** |

**구조적 한계 분석**:

| 속성 | Main Decoder (SAM2 track_step) | Aux Decoder |
|---|---|---|
| 입력 | UAMM 이후 vision_feats + **cross-modal memory** | 단일 모달리티 backbone_fpn[0] |
| 구조 | Transformer decoder + memory attention + upsampling | Conv 2-3 layer |
| 정보 | **3개 모달리티 상호 참조** | 해당 모달리티만 |
| 목적 | 최종 segmentation | 모달리티별 품질 추정 |

Aux decoder는 구조적으로 main decoder와 같아질 수 없음:
- cross-modal 정보를 쓰면 "개별 모달리티 품질 측정"이라는 목적에 부합하지 않음
- 단일 모달리티 feature만으로는 정확한 segmentation이 어려움 (특히 야간)

**Energy Score 신뢰성 조건**:

```
Aux mask 부정확 → Energy Score 무의미 (현재 상태)
Aux mask 정확하되 overconfident → Energy Score 오도됨
Aux mask 정확하고 well-calibrated → Energy Score 유효 ✓
```

Energy Score가 올바르게 작동하려면 aux mask의 **정확도 + calibration** 모두 필요.

**검토된 대안들**:

1. **Aux decoder 확대** (4-5 layer + skip connection): 소폭 개선 가능하나 frozen feature 병목 해결 불가
2. **Prototype-based aux**: gradient 오염 없음 (`.data` EMA). 하지만 선형 분류기 수준 → aux mask 품질 더 떨어질 수 있음
3. **Backbone 일부 unfreeze**: 가장 직접적이나 overfitting 위험 + SAM2 pretrained knowledge 손실 가능
4. **Label smoothing / Focal loss**: calibration 개선에 도움. 하지만 mask 정확도 자체는 안 올림

**현재 결론**: frozen backbone feature 위에서 aux mask 품질을 근본적으로 올리는 것은 구조적으로 어려움. Energy Score fusion 방식 자체의 재검토가 필요할 수 있음.

**gradient 경로 주의사항 (2026-02-27 확인)**:

현재 P14에서 energy score 계산에 `.detach()` 없음 → main loss gradient가 aux heads + LoRA에 역전파. LoRA가 두 가지 목표를 동시에 최적화:
1. 좋은 segmentation feature (main loss)
2. "적절한" energy score를 만드는 feature (간접 gradient)

→ 두 목표 충돌 가능. `compute_energy_confidence([z.detach() for z in aux_logits_list])` 로 gradient 차단 권장.

---

### ISSUE-009: Energy Score "Confident but Wrong" — Calibrated Entropy로 교체

**상태**: **P15에서 수정 예정** (설계 완료)
**영향**: P13, P14 (Energy Score 사용하는 모든 버전)
**우선순위**: 높음 — ISSUE-008과 함께 Energy Score fusion 실패의 직접 원인

**문제**:

Energy Score `E(x) = -T * logsumexp(z/T)` 는 **logit magnitude** 기반 confidence.
모달리티가 "자신있게 틀리는" 경우 오히려 높은 점수를 부여:

```
LiDAR aux head → Sky 영역에서 Water로 확신있게 오예측
→ logit: [Static=1, Dynamic=0, Water=8, Sky=0]
→ Energy Score 높음 (logsumexp ≈ 8)
→ UAMM이 LiDAR에 높은 가중치 → Sky IoU 붕괴
```

**정량적 증거**:

| 버전 | Test LiDAR UAMM | Test Sky IoU | 비고 |
| --- | --- | --- | --- |
| P9 | 0.961 (상수) | 76.54 | 고정 비율로 안정 |
| P13 | **1.000 (고정)** | 75.12 | Energy가 LiDAR 맹신 |
| P14 | **1.000 (고정)** | **36.47** | 더 심각한 맹신 |

P13/P14 모두 test에서 LiDAR UAMM=1.000, stdev=0.0000 (200장 전부 동일).
Energy Score가 LiDAR를 항상 "가장 confident"로 판정.

**P15 해결책: Calibrated Entropy**

```python
# Energy (문제): logit 크기 → confident but wrong에 취약
energy = -T * logsumexp(z / T, dim=1)
conf = -energy  # 높은 logit = 높은 confidence (위험)

# Entropy (해결): 확률 분포 균등도 → 불확실성 직접 측정
probs = softmax(z / T, dim=1)
entropy = -(probs * log(probs)).sum(dim=1)
confidence = 1 - entropy / log(num_classes)  # 0~1 정규화
```

Entropy의 장점:

- 4클래스에 골고루 분산된 예측 = 높은 entropy = 낮은 confidence
- 단일 클래스에 집중된 예측 = 낮은 entropy = 높은 confidence
- **aux head가 부정확하면** (Sky에서 모든 클래스에 비슷한 확률) → entropy 높음 → 자동 억제
- Temperature `T`로 calibration 가능 (val에서 grid search)

**한계**: aux head가 "하나의 틀린 클래스에 확신"하면 entropy도 낮음 → 여전히 실패 가능.
하지만 Energy보다는 robust: Energy는 logit magnitude만 보지만, Entropy는 분포 형태를 봄.

**관련**: ISSUE-004 (spatial-wise), ISSUE-008 (aux head 품질), `02_model_arch.md` P15 섹션

---

### ISSUE-005: 야간 합성 데이터 생성 — Diffusion 기반 Day→Night [아이디어]

**상태**: **M=85 달성을 위한 최유력 접근** (Night Aug 포화 확인됨)
**영향**: 전체 학습 파이프라인
**우선순위**: 높음 — Night Aug만으로는 +1~2pp가 한계, +7.4pp 필요

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

### 7. 평가 출력 디렉토리 네이밍 (2026-02-28 변경)

- **변경 전**: `val_pred/`, `test_pred/`, `eval_macvi/` (체크포인트 구분 불가, 덮어쓰기 위험)
- **변경 후**: 체크포인트 이름이 prefix로 붙음
  - `val_multiaqua.py`: `{ckpt_prefix}_val_pred/`, `{ckpt_prefix}_test_pred/`, `{ckpt_prefix}_eval_macvi/`
  - `val_multiaqua_detailed.py`: `{ckpt_prefix}_val_pred_{P버전}/`, `{ckpt_prefix}_test_pred_{P버전}/`
  - 결과 txt: `eval_{split}_{ckpt_prefix}_{timestamp}.txt`
- `ckpt_prefix` = checkpoint 파일명에서 `_checkpoint` 제거 (예: `epoch28_93.77_top1`)
- `--save_dir` 직접 지정 시 prefix 미적용 (기존 동작 유지)

### 8. P16/P17 평가 시 `_current_epoch` 설정 (2026-02-28 변경)

- P16/P17은 warmup schedule 사용 (`_current_epoch < 10` → uniform weights)
- 체크포인트 로드 시 `_current_epoch`은 저장되지 않음 → 기본값 0
- **`_current_epoch=0`이면 entropy fusion이 비활성화** (uniform 1/m으로 동작)
- `val_multiaqua.py`, `val_multiaqua_detailed.py` 모두 로드 후 `model._current_epoch = 9999` 설정
- P15 이하 모델은 `_current_epoch` 속성 없음 → `hasattr` 체크로 호환성 유지
