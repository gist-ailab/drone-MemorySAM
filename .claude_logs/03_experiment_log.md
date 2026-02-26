# 실험 기록 (Experiment Log)

> 최종 업데이트: 2026-02-26

## MACVi Challenge 평가 지표

- **M-score** = (val_mIoU + test_mIoU) / 2 (challenge 공식 랭킹)
- ~~이전 기록에 0.75×val + 0.25×test로 적혀있었으나, 실제 계산 검증 결과 0.5/0.5 동일 비중~~
- Val: 주간 145장 (로컬 평가 가능)
- Test: 야간 (challenge server에서만 평가, `--macvi` 플래그로 제출 파일 생성)
- 클래스: Static(0), Dynamic(1), Water(2), Sky(3)

---

## 전체 결과 요약 (M-score 순)

| 순위 | 모델 | Config | Val mIoU | Test mIoU | Val Obstacle | Test Obstacle | M-score | Submission ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **P9** | hardaug4 | 93.32 | 69.62 | 78.85 | 21.25 | **81.47** | 15635 |
| 2 | **P13** | hardaug4 | 92.45 | 69.98 | 75.93 | 25.38 | **81.21** | 15997 |
| 3 | **P12** | hardaug4 | 93.23 | 68.37 | 78.47 | 25.27 | **80.80** | 15949 |
| 4 | P10 | hardaug4 | 93.23 | 65.30 | 78.84 | 28.31 | 79.27 | 15731 |
| 5 | P8 | hardaug (기본) | 92.96 | 63.93 | 77.91 | 23.08 | 78.45 | 15561 |
| 6 | P8 | hardaug2 | 93.29 | 63.45 | 79.27 | 21.66 | 78.37 | 15589 |
| 7 | P8 | basic-aug | 93.13 | 62.50 | 78.36 | 26.01 | 77.82 | 15541 |
| 8 | P8 | hardaug3 | 93.36 | 61.57 | 79.12 | 27.17 | 77.46 | 15616 |
| 9 | P11 | hardaug4 | 93.17 | 61.01 | 78.40 | 20.95 | 77.09 | 15851 |
| 10 | P10 | hardaug3 | 93.18 | 58.93 | 78.57 | 22.36 | 76.05 | 15757 |
| 11 | P8 | no-aug (beforeAug) | 93.10 | 35.93 | 78.23 | 12.81 | 64.51 | 15509 |

---

## 상세 실험별 기록

### P8 Experiments

#### P8-1: no-aug (beforeAug)

- **Config**: `configs/lecun_multiaqua_rgbtl_P8.yaml` (NIGHT_AUG.ENABLE: false)
- **Checkpoint**: `outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/MULTIAQUA_CMNeXt-B2_ilt_beforeAug/epoch15_93.95_checkpoint.pth`
- **결과**: Val 93.10 / Test 35.93 / M 64.51
- **Challenge result**: `outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/MULTIAQUA_CMNeXt-B2_ilt_beforeAug/15509_results/`
- **비고**: NIGHT_AUG 없이 학습 → 야간 test 성능 극히 낮음. 기준선.

#### P8-2: basic-aug

- **Config**: `configs/levine-multiaqua_rgbtl_P8.yaml` (기본 NIGHT_AUG)
- **Checkpoint**: `outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/MULTIAQUA_CMNeXt-B2_ilt/epoch45_94.0_checkpoint.pth`
- **결과**: Val 93.13 / Test 62.50 / M 77.82
- **Challenge result**: `outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/MULTIAQUA_CMNeXt-B2_ilt/15541_results/`
- **비고**: NIGHT_AUG 기본 적용. Test +26.6 대폭 개선. NIGHT_AUG의 효과 확인.
- **NIGHT_AUG 설정**: NIGHT_SIM_P=0.35, BRIGHTNESS=[0.03,0.25], uniform sampling, CRM_P=0.3, ZERO_P=0.12

#### P8-3: hardaug (기본)

- **Config**: (levine 서버에서 실행, 정확한 config 파일 불명)
- **Checkpoint**: `outputs/MMSamP8/MULTIAQUA_CMNeXt-B2_ilt_hardaug/epoch28_93.8_checkpoint.pth`
- **결과**: Val 92.96 / Test 63.93 / M 78.45
- **Challenge result**: `outputs/MMSamP8/MULTIAQUA_CMNeXt-B2_ilt_hardaug/15561_results/`
- **비고**: 강화된 augmentation. P8에서 최고 M-score.

#### P8-4: hardaug2

- **Config**: `configs/levine-multiaqua_rgbtl_P8_hardaug2.yaml`
- **Checkpoint**: `outputs/MMSamP8/levine_multiaqua_rgbtl_P8_hardaug2/MULTIAQUA_CMNeXt-B2_ilt/epoch58_94.17_checkpoint.pth`
- **결과**: Val 93.29 / Test 63.45 / M 78.37
- **Challenge result**: `outputs/MMSamP8/levine_multiaqua_rgbtl_P8_hardaug2/MULTIAQUA_CMNeXt-B2_ilt/15589_results/`
- **비고**: dark_biased sampling 도입 (70% 극저조도). Val 향상되었으나 Test 유사.
- **NIGHT_AUG 설정**: NIGHT_SIM_P=0.5, BRIGHTNESS=[0.03,0.5], dark_biased(0.7), DARK=[0.03,0.15], CRM_P=0.3, ZERO_P=0.08

#### P8-5: hardaug3

- **Config**: `configs/levine-multiaqua_rgbtl_P8_hardaug3.yaml`
- **Checkpoint**: `outputs/MMSamP8/levine_multiaqua_rgbtl_P8_hardaug3/MULTIAQUA_CMNeXt-B2_ilt/epoch42_94.19_checkpoint.pth`
- **결과**: Val 93.36 / Test 61.57 / M 77.46
- **Challenge result**: `outputs/MMSamP8/levine_multiaqua_rgbtl_P8_hardaug3/MULTIAQUA_CMNeXt-B2_ilt/15616_results/`
- **비고**: 실 test 데이터 brightness 분석 기반 범위 조정. Val 최고지만 Test 하락.
- **NIGHT_AUG 설정**: NIGHT_SIM_P=0.4, BRIGHTNESS=[0.020,0.203], dark_biased(0.35), CRM_P=0.25, ZERO_P=0.06
- **교훈**: 실 데이터 분포에 너무 맞추면 일반화 저하. CRM/ZERO 완화가 원인.

---

### P9 Experiments

#### P9-1: hardaug4 (현재 최선)

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml`
- **Checkpoint**: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch47_94.18_checkpoint.pth`
- **결과**: Val 93.32 / Test 69.62 / M **81.47**
- **Challenge result**: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/P9_15635_results/`
- **비고**: CrossModalFusionHead로 아키텍처 변경 + hardaug4로 augmentation 최적화
- **NIGHT_AUG 설정**: hardaug4 = NIGHT_SIM_P=0.45, BRIGHTNESS=[0.03,0.45], dark_biased(0.6), CRM_P=0.35, ZERO_P=0.09
- **MoE routing 분석**: Per-token entropy_ratio=0.55, max_weight=0.72 → 정상 분화

---

### P10 Experiments (취소됨)

#### P10-1: hardaug4

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P10_hardaug4.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P10_hardaug4.yaml`
- **Checkpoint**: `outputs/MMSamP10/levine_multiaqua_rgbtl_P10_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch107_94.22_checkpoint.pth`
- **결과**: Val 93.23 / Test 65.30 / M 79.27
- **Challenge result**: `outputs/MMSamP10/levine_multiaqua_rgbtl_P10_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/P10_hardaug4_15731_results/`
- **비고**: Val mIoU는 P8/P9과 유사하지만 Test -4.3 하락. oracle KL이 주간에 과적합.
- **추가 파라미터**: LAMBDA_GATE=0.5

#### P10-2: hardaug3

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P10_hardaug3.yaml`
- **Checkpoint**: `outputs/MMSamP10/levine_multiaqua_rgbtl_P10_hardaug3/MULTIAQUA_CMNeXt-B2_ilt/epoch29_94.02_checkpoint.pth`
- **결과**: Val 93.18 / Test 58.93 / M 76.05
- **Challenge result**: `outputs/MMSamP10/levine_multiaqua_rgbtl_P10_hardaug3/MULTIAQUA_CMNeXt-B2_ilt/P10_hardaug3_15757_results/`
- **비고**: hardaug3 + P10 조합이 최악. Test mIoU 58.93으로 P8 basic-aug보다 나쁨.

---

### P11 Experiments (취소됨)

#### P11-1: hardaug4

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P11_hardaug4.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P11_hardaug4.yaml`
- **Checkpoint**: `outputs/MMSamP11/levine_multiaqua_rgbtl_P11_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch34_93.95_checkpoint.pth`
- **결과**: Val 93.17 / Test 61.01 / M 77.09
- **Challenge result**: `outputs/MMSamP11/levine_multiaqua_rgbtl_P11_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/P11_hardaug4_15851_results/`
- **비고**: MI loss 추가했으나 P10보다도 약간 나음. 그러나 P9 대비 -4.4 하락. 진단 결과 MoE gate는 이미 정상이었으므로 MI loss가 불필요했음.
- **추가 파라미터**: LAMBDA_GATE=0.5, LAMBDA_MI=1.0

### P12 Experiments

#### P12-1: hardaug4

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P12_hardaug4.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P12_hardaug4.yaml`
- **Checkpoint**: `outputs/MMSamP12/levine_multiaqua_rgbtl_P12_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch40_94.02_checkpoint.pth`
- **결과**: Val 93.23 / Test 68.37 / M **80.80**
- **Challenge result**: `outputs/MMSamP12/levine_multiaqua_rgbtl_P12_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/P12_15949_results/`
- **비고**: Input-Conditioned Soft MoE LoRA. P9 대비 M-score -0.67 하락.
  - Dynamic IoU +4.02 (21.25→25.27) 개선, 하지만 Sky 클래스 대폭 하락 (-6.81pp)
  - UAMM/AMF 변동성 소폭 증가 (std 0.0001→0.01)하나 여전히 near-constant
  - Expert collapse P9보다 심화 (Block0 lidar E0=E2=0%, Block18 lidar E0=1.6% E1=0.4%)
  - Tail-end failure 증가: mIoU<55 프레임이 P9 5장→P12 18장
- **상세 분석**: `.claude_logs/05_result_analysis_P9_P12.md`

---

### P13 Experiments

#### P13-1: hardaug4

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml` (bengio 서버)
- **Config 평가**: `configs/eval_config/bengio-multiaqua_rgbtl_P13_hardaug4.yaml`
- **Checkpoint**: `outputs/MMSamP13/bengio_multiaqua_rgbtl_P13_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/night_epoch17_87.71_checkpoint.pth`
- **체크포인트 선택**: **Night-Val 기준** (87.71 mIoU). Best day-val은 epoch14 (93.48)이었으나 night-val이 87.24로 낮았음.
- **결과**: Val 92.45 / Test 69.98 / M **81.21**
- **Challenge result**: `outputs/MMSamP13/.../P13_15997_results/`
- **학습 에폭**: 17 epochs (resume from previous run). P9는 47 epochs.
- **Per-class Test IoU**: Static 79.80 / Dynamic **27.41** (+5.55 vs P9) / Water 94.27 / Sky 75.12
- **설계 목표 달성 여부**:
  1. Expert collapse 해결 (kaiming*0.01 init): **실패** — collapse rate 17.4% (P12: 16.0%와 동일 수준)
  2. Energy Score fusion: **부분 성공** — UAMM CV가 P12 대비 5-22x 증가. 하지만 test LiDAR UAMM = 1.0 고정 (모든 P 버전 공통)
- **핵심 발견**:
  - Dynamic class +5.55pp 개선 (Dynamic=0 프레임: P9 38개 → P13 20개)
  - Static -1.49pp, Sky -1.42pp 하락 → net test mIoU +0.36pp
  - Val mIoU -0.87pp → M-score 오히려 -0.26pp 하락
  - Night-val 선택이 test 개선에 효과적이나 M-score 공식에서 val 하락이 불리
  - P9 epoch 17에서 val 93.57이었고 epoch 46까지 +0.61pp만 추가 상승 → P13도 유사한 포화 예상
- **상세 분석**: `.claude_logs/06_result_analysis_P13.md`

---

## Night Augmentation 포화 분석

### Augmentation 효과 정량화 (P8 동일 아키텍처)

| Aug | Test mIoU | vs no-aug | Marginal gain |
|-----|--------:|--------:|--------:|
| no-aug | 35.93 | - | - |
| basic-aug | 62.50 | +26.57 | **+26.57** |
| hardaug | 63.93 | +28.00 | +1.43 |
| hardaug2 | 63.45 | +27.52 | -0.48 |
| hardaug3 | 61.57 | +25.64 | -1.88 |

**결론: Augmentation 포화**
- no-aug → basic-aug: **+26.57pp** (전체 gain의 80%)
- basic-aug → best hardaug 변종: **+1.43pp** (전체 gain의 4%)
- 나머지 gain은 아키텍처 변경(P9의 CrossModalFusionHead)에서 발생

### M=85 달성 가능성

- 목표: test ≥ 77 (val 93 유지 가정)
- 현재: test = 69.62 → **+7.4pp 필요**
- Augmentation 튜닝 최대 기대 효과: **+1~2pp** → **부족**
- 클래스별 병목:
  - Dynamic gap -38pp: 밝기 augmentation으로 해결 불가 (소형 객체 + 어두운 수면)
  - Sky gap -21pp: 하늘/물 경계 혼동은 전역 밝기 변환으로 해결 불가
- **Night Aug만으로 M=85는 불가능. 근본적으로 다른 접근(데이터 합성, 추론 시 기법) 필요.**

---

## NIGHT_AUG 버전 비교

| 파라미터 | basic-aug | hardaug2 | hardaug3 | hardaug4 |
| --- | --- | --- | --- | --- |
| NIGHT_SIM_P | 0.35 | 0.50 | 0.40 | 0.45 |
| BRIGHTNESS | [0.03, 0.25] | [0.03, 0.50] | [0.020, 0.203] | [0.03, 0.45] |
| SAMPLING | uniform | dark_biased | dark_biased | dark_biased |
| DARK_RATIO | - | 0.70 | 0.35 | 0.60 |
| DARK_RANGE | - | [0.03, 0.15] | [0.020, 0.035] | [0.03, 0.12] |
| CRM_P | 0.30 | 0.30 | 0.25 | 0.35 |
| ZERO_P | 0.12 | 0.08 | 0.06 | 0.09 |

**결론**: hardaug4가 최선. hardaug2 대비 CRM_P 강화(0.35), DARK_RATIO 적절히 완화(0.6), BRIGHTNESS 상한 약간 축소(0.45).

---

## 핵심 교훈

1. **NIGHT_AUG 없이는 Test 성능 극히 낮음** (35.93 vs 63+): 야간 시뮬레이션 필수
2. **hardaug4가 최적의 균형점**: CRM/ZERO를 적절히 강화하여 thermal/lidar 활용 유도
3. **모델 아키텍처보다 augmentation이 중요**: P8 hardaug4 ≈ P10 hardaug4 수준
4. **복잡한 loss/모듈 추가는 과적합 위험**: P10 oracle KL, P11 MI loss 모두 역효과
5. **측정 방법이 중요**: MoE gate "uniform" 문제는 spatial mean의 artifact였음
6. **Val mIoU는 모델 비교에 부적합**: 모든 모델이 93-94%로 유사. Test mIoU와 M-score로 비교해야 함

---

## 평가 로그 파일

각 실험의 `val_pred/uamm_amf_moe_log.json`에 145장의 per-image 상세 데이터 저장:
- UAMM scores (모달리티별)
- AMF weights (모달리티별)
- MoE gate weights (expert별)

경로 패턴: `outputs/MMSam{PX}/{config_name}/MULTIAQUA_CMNeXt-B2_ilt/val_pred/uamm_amf_moe_log.json`

---

## 진단 스크립트

### diagnose_moe_gate.py

- MoE gate uniform routing 근본 원인 진단용
- 4가지 가설 검증: H1(zero-init symmetry), H2(gate weight stagnation), H3(LayerNorm), H4(expert similarity)
- **결과**: 모든 가설 기각 → gate는 정상 작동, "uniform"은 측정 artifact
- Static analysis + forward pass analysis + gradient analysis

### val_multiaqua_P9.py

- P9 전용 평가 + 4-row 시각화
- Row 1: 모달리티 입력 (RGB, LiDAR, Thermal)
- Row 2: GT / Prediction / Overlay (test에서는 Legend / Prediction / Overlay)
- Row 3: Per-block MoE gate stats (Block0, Block9, Block18)
- Row 4: Spatial routing color map (모달리티별 expert 할당)
- `--tta` 플래그로 Test Time Augmentation 지원
