# 실험 기록 (Experiment Log)

> 최종 업데이트: 2026-03-05

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
| 11 | **P9** | hardaug4 **Day-Trans** (test night→day, img2img-turbo) | 93.30 | 64.50 | 78.87 | 16.60 | **78.90** | 16478 |
| 12 | **P9** | hardaug4 **Gamma TTA** [1.0,1.5,2.0,2.5] | 93.30 | 58.89 | — | 16.05 | **76.10** | 16412 |
| 13 | **P9** | **hardaug6** (ep20) | 92.00 | 59.91 | 74.63 | 17.69 | **75.95** | 16340 |
| 14 | P9 | hardaug6 (ep85) | 93.40 | 57.63 | 79.34 | 24.75 | 75.51 | 16339 |
| 15 | P14 | hardaug5 | 93.18 | 55.36 | 78.69 | 23.60 | 74.27 | 16062 |
| 16 | **P17** | hardaug5 (night_ep35) | 92.60 | 53.86 | 76.29 | 17.98 | **73.23** | 16107 |
| 17 | **P9** | hardaug4 **Night2** (day→night I2I 학습) | 92.91 | 53.18 | 77.59 | 19.10 | **73.04** | 16482 |
| 18 | P17 | hardaug5 (ep28) | 92.99 | 52.69 | 77.72 | 28.36 | 72.84 | 16108 |
| 19 | P13 | hardaug4 (ep39) | 92.86 | 50.48 | 77.08 | 14.53 | 71.67 | 16044 |
| 20 | P15 | hardaug5 | 93.17 | 48.94 | 78.31 | 24.96 | 71.05 | 16087 |
| 21 | **P19** | hardaug5 (ep36) | 93.44 | **45.82** | 79.24 | 23.50 | **69.63** | 16313 |
| 22 | **P16** | hardaug5 (night_ep31) | 93.14 | 43.70 | 78.68 | 20.56 | **68.42** | 16106 |
| 23 | P8 | no-aug (beforeAug) | 93.10 | 35.93 | 78.23 | 12.81 | 64.51 | 15509 |
| — | **P9** | hardaug4 **CV** (heuristic b+0.11 c×0.9, 56장 mIoU<65) | 93.30 | 64.86 | 78.87 | 17.16 | **79.08** | 16485 |
| — | **P9** | hardaug4 **CV2** (heuristic b+0.11 c×0.9, 5장 mIoU<55) | 93.30 | — | — | — | **—** | 제출 대기 |

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

#### P13-2: hardaug4 epoch39 (Test Crash)

- **Config**: 동일 (P13-1과 동일 학습, 더 긴 에폭)
- **Checkpoint**: `outputs/MMSamP13/bengio_multiaqua_rgbtl_P13_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/night_epoch39_89.53_checkpoint.pth`
- **결과**: Val 92.86 / Test **50.48** / M **71.67**
- **Challenge result**: `outputs/MMSamP13/.../P13_16044_results/`
- **Submission ID**: 16044
- **Night-Val**: 89.53 (epoch17: 87.71, +1.82 개선)
- **Per-class Test IoU**: Static 66.27 / Dynamic 15.44 / Water 94.42 / Sky **23.36**
- **Epoch17→39 변화**:
  - Val: 92.45→92.86 (+0.41) ✅
  - Night-val: 87.71→89.53 (+1.82) ✅
  - **Test: 69.98→50.48 (-19.50)** ❌ — **치명적 하락**
  - Sky: 75.12→23.36 (-51.76pp) — crash의 67%
  - 80/200 프레임에서 Sky IoU=0 (epoch17에서는 5개)
  - 192/200 프레임에서 성능 하락
- **원인 분석**: **CRM/ZERO Overfitting (ISSUE-007)**
  - CRM(p=0.35) + ZERO(p=0.09) → 학습 샘플 44%에 exact-zero RGB 패턴
  - Exact zero는 실제 센서 데이터에 없는 artifact → train-test 분포 불일치
  - Night-val에도 CRM/ZERO 동일 적용 → 오염된 proxy, shortcut 학습이 night-val도 개선
  - 더 많이 학습할수록 shortcut 강화 → test 성능 역행
- **교훈**: Night-val에서 CRM/ZERO를 제거해야 신뢰 가능한 test proxy. Epoch17이 P13 sweet spot.

---

### P14 Experiments

#### P14-1: hardaug5 (night_epoch47)

- **Config 학습**: `configs/bengio-multiaqua_rgbtl_P14_hardaug5.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml` (TODO: P14 전용 eval config 확인)
- **Checkpoint**: `outputs/MMSamP14/bengio_multiaqua_rgbtl_P14_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/night_epoch47_90.75_top1_checkpoint.pth`
- **체크포인트 선택**: Night-Val 기준 (90.75 mIoU). Day-val best epoch47 (94.06).
- **결과**: Val 93.18 / Test **55.36** / M **74.27**
- **Challenge result**: `outputs/MMSamP14/.../eval_macvi/` (Submission #16062)
- **Per-class Test IoU**: Static 62.57 / Dynamic 22.87 / Water 92.92 / Sky **36.47**
- **Obstacle IoU**: Val 78.69 / Test 23.60
- **hardaug5 변경사항**: CRM/ZERO 완전 제거 + BRIGHTNESS [0.02, 0.20] (실측 정렬) + NIGHT_SIM_P 0.60
- **P14 아키텍처 변경**: ConfidenceAuxHead×1(공유) → ModalAuxDecoder×3(독립)
- **핵심 발견**:
  1. **M-score 74.27 — P9 대비 -7.20 심각한 하락**
  2. **Sky IoU 36.47%**: 73/200 프레임 Sky<10%, 56/200 프레임 Sky<1%
  3. **LiDAR UAMM = 1.000 고정** (test 200장 전부, stdev=0.000) — P13과 동일 문제
  4. **RGB 억제**: test UAMM img=0.555 (val=0.752) → Sky 인식에 핵심인 RGB가 절반으로 감소
  5. **Aux mask 품질**: P13 대비 개선되었으나 여전히 GT 대비 매우 부정확. 모달리티 간 비교 불가 수준
  6. **Model uncertainty**: test mean_entropy 0.570 (val 0.178의 3.2배), high_uncertainty_ratio 63.4%
  7. **MoE routing**: val/test 간 거의 동일 (entropy_ratio stdev < 0.02) — routing 자체는 안정적이나 여전히 고정
- **CRM/ZERO 제거 효과**: hardaug5에서 CRM/ZERO 제거했으나 Sky collapse 여전히 발생 → ISSUE-007은 Sky 문제의 일부 원인이었으나 유일 원인은 아님
- **실패 원인 분석**:
  - Energy Score가 LiDAR를 항상 최고 confident로 판정 → Sky 영역에서 LiDAR (무의미) 기반 예측
  - Image-level scalar fusion의 근본 한계 — Sky/Water 영역별로 최적 모달리티가 다르지만 반영 불가
  - Aux decoder가 frozen backbone feature 기반 → 야간/주간 공통 feature 패턴에서 학습하므로 domain-specific quality 판별 불가

---

### P15 Experiments

#### P15-1: hardaug5 (epoch46, day-val best) — 역대 최악

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P15_hardaug5.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P15_hardaug5.yaml`
- **Checkpoint**: `outputs/MMSamP15/levine_multiaqua_rgbtl_P15_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/epoch46_93.92_top1_checkpoint.pth`
- **Night-val best**: `night_epoch45_90.39_top1_checkpoint.pth` (미평가)
- **체크포인트 선택**: Day-Val best (epoch46, 93.92 mIoU). ⚠️ Night-val best 미사용
- **결과**: Val **93.17** / Test **48.94** / M **71.05**
- **Challenge result**: Submission #16087
- **Per-class Test IoU**: Static 60.94 / Dynamic 26.58 / Water 92.27 / Sky **16.66**
- **Obstacle IoU**: Val 78.31 / Test 24.96

- **P15 아키텍처**: P14 + Spatial-wise Energy Weighting `(B, m, H, W)`
  - `compute_spatial_energy_confidence()`: Energy Score 유지 (NOT entropy)
  - `.detach()` 미적용, Warmup 미적용
  - 설계 4가지 Fix 중 Fix 3(spatial-wise)만 단독 적용

- **UAMM 분석 (val vs test)**:

| 모달리티 | Val UAMM (mean±std) | Test UAMM (mean±std) | Δ |
| --- | --- | --- | --- |
| img | 0.716±0.026 | 0.566±0.028 | **-0.150 (-21%)** |
| lidar | 0.834±0.047 | 0.956±0.010 | +0.122 (+15%) |
| thermal | 0.554±0.032 | 0.630±0.009 | +0.076 (+14%) |

- **P15 vs P9 UAMM 비교**: P15가 실제로 val/test 적응을 수행 (P9는 std≈0 고정). LiDAR도 1.0에 고정되지 않음 (mean 0.956).
- **하지만 test mIoU는 역대 최악 (48.94)**

- **핵심 발견 — Spatial-wise가 noise를 증폭**:
  1. **Sky IoU 16.66%**: 111/200 프레임 Sky=0%, 152/200 프레임 Sky<10%
  2. **P14(scalar, Sky 36.47%) → P15(spatial, Sky 16.66%)**: Spatial-wise 적용 후 오히려 **-19.81pp 악화**
  3. **"Spatial Amplification Effect"**: 부정확한 energy score가 pixel-level에서 그대로 전파. Scalar는 이미지 평균으로 error가 smooth되지만, spatial은 aux mask의 모든 local error가 증폭됨
  4. **Energy Score + Spatial + no-detach + no-warmup = 최악의 조합**
  5. **Checkpoint 선택 리스크**: day-val best(epoch46) 사용. Night-val best(epoch45)로 재평가 시 개선 가능성 있음

- **교훈**:
  1. Spatial-wise를 단독 적용하면 오히려 해로움 — aux mask 정확도 개선(Fix 1,2,4)이 선행되어야 함
  2. Energy Score의 "confident but wrong" 문제가 spatial에서 더 치명적
  3. Fix 3은 Fix 1+2+4와 함께 적용해야 효과 있음 (P16의 접근)

---

### P16 Experiments

#### P16-1: hardaug5 (night_epoch31, night-val best)

- **Config 학습**: `configs/bengio-multiaqua_rgbtl_P16_hardaug5.yaml`
- **Config 평가**: `configs/eval_config/bengio-multiaqua_rgbtl_P16_hardaug5.yaml`
- **Checkpoint**: `outputs/MMSamP16/bengio_multiaqua_rgbtl_P16_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/night_epoch31_90.56_top1_checkpoint.pth`
- **체크포인트 선택**: Night-Val 기준 (90.56 mIoU)
- **결과**: Val 93.14 / Test **43.70** / M **68.42** — **역대 최악**
- **Challenge result**: Submission #16106
- **Per-class Test IoU**: Static 58.19 / Dynamic 20.76 / Water 92.24 / **Sky 3.17**
- **Obstacle IoU**: Val 78.68 / Test 20.56
- **P16 아키텍처**: P14의 ModalAuxDecoder + 4 Fixes 통합
  1. `.detach()` gradient 격리
  2. Energy Score → Calibrated Entropy
  3. Spatial-wise `(B, m, H, W)` 가중치
  4. Aux Warmup Schedule (10ep uniform + 5ep linear ramp)
- **Sky 완전 붕괴**: 157/200 프레임 Sky=0, 191/200 프레임 Sky<10%
- **UAMM 분석 (test)**: img=0.758±0.012, lidar=0.819±0.014, **thermal=0.923±0.011** (thermal 지배)
- **CV < 0.02**: 거의 고정 비율 → adaptive fusion 실패, 하지만 P9의 좋은 고정비율과 달리 thermal 편향
- **원인**: Calibrated entropy + spatial에서도 aux mask 품질 부족(ISSUE-008) → thermal이 항상 낮은 entropy → Sky 영역에서 thermal 기반 예측 → sky 인식 불가

---

### P17 Experiments

#### P17-1: hardaug5 (night_epoch35, night-val best)

- **Config 학습**: `configs/bengio-multiaqua_rgbtl_P17_hardaug5.yaml`
  (학습은 levine 서버, 결과 디렉토리명은 `levine_multiaqua_rgbtl_P17_hardaug5`)
- **Config 평가**: `configs/eval_config/bengio-multiaqua_rgbtl_P17_hardaug5.yaml`
- **Checkpoint**: `outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/night_epoch35_90.34_top1_checkpoint.pth`
- **체크포인트 선택**: Night-Val 기준 (90.34 mIoU)
- **결과**: Val 92.60 / Test **53.86** / M **73.23**
- **Challenge result**: Submission #16107
- **Per-class Test IoU**: Static 61.46 / Dynamic 19.44 / Water 93.54 / **Sky 33.35**
- **Obstacle IoU**: Val 76.29 / Test 17.98
- **P17 아키텍처**: P16 + MultiScaleModalAuxDecoder
  - `ModalAuxDecoder`(fpn[0] 32ch만) → `MultiScaleModalAuxDecoder`(fpn[0,1,2] 352ch)
  - 3개 FPN 레벨: fpn[0](32ch,256²) + fpn[1](64ch,128²) + fpn[2](256ch,64²)
  - proj_dim=32 → concat(96ch) → 3×3 conv → 4class logits, ~53K/modality
- **Sky 부분 회복**: P16(3.17) → P17(33.35) = +30.18pp. Sky=0 프레임 157→62개
- **UAMM 분석 (test)**: img=0.864±0.030, lidar=0.787±0.032, thermal=0.864±0.029
  - P16 대비 thermal 지배 완화 (0.923→0.864)
  - CV 0.03-0.04 (P16: <0.02) — 2x 더 adaptive
- **Multi-Scale FPN 효과**: fpn[2](256ch) semantic context가 sky/static 구분 기여
- **그러나**: P9(M=81.47) 대비 여전히 -8.24. Static -20pp, Sky -43pp 갭 지속

#### P17-2: hardaug5 (epoch28, day-val best)

- **Checkpoint**: `outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/epoch28_93.77_top1_checkpoint.pth`
- **체크포인트 선택**: Day-Val 기준 (93.77 mIoU)
- **결과**: Val 92.99 / Test 52.69 / M 72.84
- **Challenge result**: Submission #16108
- **Per-class Test IoU**: Static **63.99** / Dynamic **27.62** / Water **94.26** / Sky 20.83
- **Night-val vs Day-val 체크포인트 비교**:
  - Night-val(ep35): Sky 33.35 (우세) / Static 61.46 / Dynamic 19.44
  - Day-val(ep28): Sky 20.83 / Static 63.99 (우세) / Dynamic 27.62 (우세)
  - mIoU 유사 (53.86 vs 52.69) 하지만 Sky vs Static+Dynamic 트레이드오프
- **교훈**: Night-val checkpoint이 Sky에 유리하고 M-score도 미세 우세 (73.23 vs 72.84)

---

### P14~P17 종합 분석: Dynamic Fusion의 실패

**핵심 패턴**: P9 이후 모든 adaptive fusion 시도가 P9의 고정 상수보다 나쁨

| 모델 | Fusion 방식 | M-score | vs P9 |
|------|-----------|---------|-------|
| P9 | 고정 상수 (img:27.5%, lidar:35.5%, thermal:37.0%) | 81.47 | — |
| P12 | Conditional MoE | 80.80 | -0.67 |
| P13 | Energy Score | 81.21 | -0.26 |
| P14 | Energy + aux decoder 독립 | 74.27 | -7.20 |
| P15 | Spatial energy (Fix3만) | 71.05 | -10.42 |
| P16 | Calibrated entropy + 4 Fixes | 68.42 | -13.05 |
| P17 | Multi-scale entropy + 4 Fixes | 73.23 | -8.24 |

**실패 원인 3가지**:
1. **Aux mask 품질 부족** (ISSUE-008): frozen backbone → GT 대비 부정확한 mask → entropy/energy 계산이 무의미
2. **Thermal 편향**: 야간에서 thermal이 전반적으로 confident → 과도한 가중치. 하지만 Sky에서 thermal은 무력
3. **Spatial amplification**: pixel-level fusion이 aux mask의 local error를 증폭 (P14→P15에서 -3.22pp 추가 하락이 증거)

**P9가 잘 작동하는 이유**: SAM2 memory attention이 이미 cross-modal implicit adaptation 수행. UAMM/AMF가 고정이어도 memory 내부에서 모달리티 간 정보가 선택적으로 활용됨.

---

### P19 Experiments

#### P19-1: hardaug5 (epoch36, day-val best)

- **Config 학습**: `configs/levine-multiaqua_rgbtl_P19_hardaug5.yaml`
- **Config 평가**: `configs/eval_config/levine-multiaqua_rgbtl_P19_hardaug5.yaml`
- **Checkpoint**: `outputs/MMSamP19/levine_multiaqua_rgbtl_P19_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/epoch36_94.23_top1_checkpoint.pth`
- **체크포인트 선택**: Day-Val 기준 (94.23 mIoU)
- **결과**: Val 93.44 / Test **45.82** / M **69.63**
- **Challenge result**: Submission #16313
- **Per-class Test IoU**: Static 60.75 / Dynamic 23.39 / Water 94.36 / **Sky 3.77**
- **Obstacle IoU**: Val 79.24 / Test 23.50
- **P19 아키텍처**: P9 base + SpatialCrossModalFusionHead (multi-scale FPN + DWConv + spatial softmax)
  - P9의 `(B,m)` scalar → `(B,m,H,W)` spatial 가중치
  - fpn[0,1,2] 3개 레벨 활용, proj_dim=32, DWConv context, zero-init compare head
  - Aux decoder 없음 — main loss만으로 학습
- **Sky 완전 붕괴**: 169/200 프레임 Sky=0, 191/200 프레임 Sky<10%
- **UAMM (test)**: img=0.741±0.011, lidar=0.992±0.003, thermal=0.750±0.034
  - LiDAR 독점 (0.992), P9(0.355)과 대조적
- **AMF (test)**: img=0.298, lidar=**0.403**, thermal=0.299
  - P9: img=0.275, lidar=0.355, **thermal=0.370** → P19는 lidar 편향으로 이동
- **Night_epoch49 (미제출)**: val_pred/test_pred 존재, eval_macvi 생성됨. Val 93.04
  - Night finetuning 후 thermal UAMM +0.100 상승 (0.750→0.850), test 불확실성도 증가
- **실패 원인**:
  1. **hardaug5 문제**: P14~P17과 동일한 augmentation → P9+hardaug6(M=75.95)도 하락, aug 자체 문제
  2. **Spatial fusion 과적합**: train night feature ≠ real night feature → 학습된 spatial pattern이 전이 실패
  3. **LiDAR 편향 수렴**: zero-init에서 학습하며 LiDAR 중심으로 수렴 → P9의 thermal 우세 균형 파괴
  4. **CRM/ZERO 부재**: P9에 유익했던 multimodal 강제 학습 신호 부재

---

### P9 hardaug6 Experiments

#### P9-h6-1: hardaug6 (epoch85, day-val best)

- **Config 학습**: `configs/bengio-multiaqua_rgbtl_P9_hardaug6.yaml`
- **Checkpoint**: `outputs/MMSamP9/bengio_multiaqua_rgbtl_P9_hardaug6/MULTIAQUA_CMNeXt-B2_ilt/epoch85_94.33_top1_checkpoint.pth`
- **체크포인트 선택**: Day-Val 기준 (94.33 mIoU)
- **결과**: Val 93.40 / Test 57.63 / M 75.51
- **Challenge result**: Submission #16339
- **Per-class Test IoU**: Static 67.88 / Dynamic 24.39 / Water 93.80 / **Sky 39.90**
- **Sky=0 프레임**: 42/200 (21%), Sky<10%: 73/200 (36.5%)
- **UAMM (test)**: img=0.778, lidar=1.000, thermal=0.993 — **완전한 고정 상수** (std≈0.000)
  - Val과 Test가 소수점 4자리까지 동일 → 입력 무관한 학습된 상수
- **AMF (test)**: img=0.281, lidar=0.361, thermal=0.358 — P9 h4(0.275, 0.355, 0.370)와 유사

#### P9-h6-2: hardaug6 (epoch20, periodic)

- **Checkpoint**: `outputs/MMSamP9/bengio_multiaqua_rgbtl_P9_hardaug6/MULTIAQUA_CMNeXt-B2_ilt/periodic_epoch20_checkpoint.pth`
- **결과**: Val 92.00 / Test **59.91** / M **75.95** (epoch85보다 M-score +0.44)
- **Challenge result**: Submission #16340
- **Per-class Test IoU**: Static 65.74 / Dynamic 19.54 / Water 93.40 / **Sky 56.87**
- **Sky=0 프레임**: **8/200** (4%) — epoch85(42/200)보다 훨씬 양호
- **AMF (test)**: img=0.241, lidar=0.393, thermal=0.366

#### P9 hardaug6 분석

- **Epoch20 > Epoch85 on test**: Sky가 결정적 차이 (56.87 vs 39.90 = -16.97pp)
- **학습이 길수록 Sky 하락**: 야간 aug가 sky texture 제거 → 오래 학습 시 sky 예측 포기로 과적합
- **hardaug6 vs hardaug4 실패 원인** (M 75.95 vs 81.47, -5.52):
  1. **너무 넓은 brightness [0.01, 0.60]**: 0.30~0.60 범위는 test에 없는 "밝은 야간" → capacity 낭비
  2. **gamma>1.0**: 이미지를 밝게 만드는 augmentation → 야간 학습에 반대 방향
  3. **CRM/ZERO 제거**: P9에는 aux decoder 없으므로 CRM/ZERO shortcut 문제 없음. 오히려 RGB 제거가 multimodal 학습 강화에 유익
  4. **Dark ratio 50%**: hardaug4(60%)보다 극저조도 노출 부족

---

### P9 Gamma TTA Experiment (실험 I — 실패)

#### P9-GammaTTA: hardaug4 + Gamma TTA [1.0, 1.5, 2.0, 2.5]

- **Config**: `configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml` (기존 P9 체크포인트 그대로)
- **Checkpoint**: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch47_94.18_checkpoint.pth`
- **실행**: `--test_gamma_tta "1.0,1.5,2.0,2.5"` (softmax probability averaging)
- **결과**: Val 93.30 / Test **58.89** / M **76.10**
- **Challenge result**: Submission #16412
- **Inference FPS**: 0.36~0.51 (baseline 대비 ~4x 느림, 4회 forward)

**Per-class Test IoU 비교 (Baseline vs Gamma TTA)**:

| Class | Baseline (γ=1.0) | Gamma TTA | Delta |
| --- | --- | --- | --- |
| Static | 81.30 | 72.92 | **-8.38** |
| Dynamic | 21.86 | 16.55 | **-5.31** |
| Water | 94.61 | 94.03 | -0.58 |
| **Sky** | **76.54** | **47.70** | **-28.84** |
| mIoU | 68.58 | 57.80 | **-10.78** |

**Sky 붕괴 통계**:
- Sky=0 프레임: 5 → **29** (5.8x 증가)
- Sky<10 프레임: 7 → **41** (5.9x 증가)
- Dynamic=0 프레임: 46 → **72** (1.6x 증가)

**실패 원인 분석**:

1. **높은 gamma가 OOD 입력 생성**: P9는 NightSim gamma [0.4, 0.8]로 학습. gamma=1.5~2.5로 밝아진 이미지는 학습 분포 밖 → confident but wrong 예측
2. **Equal-weight soft voting의 함정**: 4개 gamma의 softmax 확률을 1/4씩 동일 가중치로 평균. gamma=1.0의 정확한 예측이 gamma=2.0/2.5의 오예측에 의해 dilute
3. **Sky가 가장 취약**: 야간 하늘 = near-zero 픽셀. gamma=2.5 적용 시 `pixel^(1/2.5)`로 크게 밝아짐 → 수면/static과 시각적 혼동
4. **Memory Attention 연쇄 오염**: RGB gamma 변경 → memory embedding 변형 → LiDAR/Thermal cross-modal attention도 오염 → 모든 모달리티 예측 악화
5. **Val 무영향 (93.30 ≈ 93.32)**: Val은 주간 이미지, gamma 적용해도 이미 밝아서 변화 미미

**구현 방식** (`val_multiaqua.py`, `val_multiaqua_detailed.py`):
- `apply_test_gamma()`: 정규화된 텐서를 un-normalize → `x^(1/γ)` → re-normalize
- `_gamma_tta_forward()`: 각 gamma로 forward → softmax → 확률 평균
- Val/Test 모두 동일하게 적용

**교훈**:
1. **TTA는 학습 분포 내의 augmentation에서만 유효**: flip/scale은 학습 시 경험한 변형이므로 TTA 적합. 미경험 gamma > 1.0은 역효과
2. **Soft voting은 최약 예측에 취약**: 하나라도 나쁜 예측이 있으면 전체를 끌어내림
3. **MemorySAM의 sequential memory가 TTA를 더 어렵게 만듦**: RGB 변형이 downstream 모달리티에 연쇄 영향
4. **Single gamma (non-ensemble) 탐색 여지 있음**: γ=1.2~1.3 mild correction을 단독 적용하면 소폭 개선 가능성 잔존

---

### I2I Translation 실험 (실험 II, III — 실패)

#### 실험 II: Day-Trans — Test Night→Day (img2img-turbo)

- **방법**: img2img-turbo (https://github.com/GaParmar/img2img-turbo) 로 test 야간 RGB를 day-like로 변환 후, 기존 P9 hardaug4 모델로 인퍼런스
- **Checkpoint**: 기존 P9 hardaug4 epoch47 (변경 없음)
- **결과**: Val 93.30 / Test **64.50** / M **78.90**
- **Submission**: #16478

**Per-class Test IoU 비교 (Baseline vs Day-Trans)**:

| Class | Baseline | Day-Trans | Delta |
| --- | --- | --- | --- |
| Static | 81.30 | 75.97 | **-5.33** |
| Dynamic | 21.86 | 16.80 | **-5.06** |
| Water | 94.61 | 93.81 | -0.80 |
| Sky | 76.54 | 66.74 | **-9.80** |
| mIoU | 68.58 | 63.33 | **-5.25** |

**프레임별 통계**: 171/200 하락, 29/200 개선 (최대 +9.9pp), 최악 -32.6pp

**실패 원인**:
1. **정보 부재 영역의 hallucination**: 야간 near-zero 픽셀에 실제 정보 없음 → I2I 모델이 존재하지 않는 텍스처 날조 → 모델이 fabricated feature로 segmentation
2. **Cross-modal 불일치**: RGB만 day-like로 변환, thermal/lidar는 야간 그대로 → 학습 시 본 적 없는 모달리티 조합
3. **Boundary distortion**: Diffusion 기반 재구성이 object boundary 변형 → segmentation edge 정확도 하락

#### 실험 III: Night2 — Day→Night I2I 학습 데이터 확장

- **방법**: img2img-turbo로 train day RGB를 night-like로 변환 → 원본 day + night-translated로 학습 (NIGHT_TRANSLATION: true)
- **Config**: `configs/levine-multiaqua_rgbtl_P9_hardaug4_night2.yaml`
- **Checkpoint**: epoch49_93.49 (val 기준 top1)
- **결과**: Val 92.91 / Test **53.18** / M **73.04**
- **Submission**: #16482

**Per-class Test IoU 비교 (Baseline vs Night2)**:

| Class | Baseline | Night2 | Delta |
| --- | --- | --- | --- |
| Static | 81.30 | 67.17 | **-14.13** |
| Dynamic | 21.86 | 19.78 | -2.08 |
| Water | 94.61 | 93.57 | -1.04 |
| Sky | 76.54 | 26.92 | **-49.62** |
| mIoU | 68.58 | 51.86 | **-16.72** |

**Sky 붕괴 통계**: Sky IoU 30pp 이상 하락 프레임 **130/200 (65%)**, Sky 개선(>5pp) 프레임 **0/200장**

**프레임별 통계**: 181/200 하락, 19/200 개선

**실패 원인**:
1. **I2I artifact 학습**: 번역된 "야간" 이미지의 artifact(색 번짐, hallucinated 텍스처, boundary 변형)를 night feature로 과적합
2. **Cross-modal 불일치**: RGB만 night-like로 변환, thermal/lidar는 daytime measurement 그대로 → 학습 시 "어두운 RGB + 밝은 thermal" 조합 vs test의 "진짜 어두운 RGB + 진짜 야간 thermal"
3. **NightSim 이중 적용**: 이미 어두운 I2I 번역 이미지에 NIGHT_SIM_P=0.45로 추가 NightSim → 비현실적 극단 어둠
4. **Label noise**: I2I가 boundary 변형시키지만 annotation 동일 → 경계 영역 학습 혼란

**I2I 양방향 실패의 근본 원인 — 정보 비대칭**:

- **Day image**: 높은 정보량 (high SNR, rich texture). day→night→day roundtrip 시 원본과 거의 동일 (정보가 latent에 보존)
- **Real night image**: 낮은 정보량 (센서 단계에서 비가역적 소실). night→day 시 없는 정보를 hallucinate → 원본 day와 전혀 다른 결과
- **결론**: I2I 모델의 "synthetic night" ≠ "real night". Synthetic night은 pixel만 어두울 뿐 정보량은 day와 동일한 fake night. **Pixel-level domain bridging은 정보이론적 한계**

---

### Augmentation Ablation 종합 (P9 아키텍처)

| Aug | CRM/ZERO | Brightness | Dark Ratio | Sky IoU | Test mIoU | M-score |
| --- | --- | --- | --- | --- | --- | --- |
| **hardaug4** | **있음** | [0.03, 0.45] | 60% | **76.54** | **69.62** | **81.47** |
| hardaug4 Gamma TTA | 있음 | (test-time γ 1.0~2.5) | — | 47.70 | 58.89 | 76.10 |
| hardaug6 ep20 | 없음 | [0.01, 0.60] | 50% | 56.87 | 59.91 | 75.95 |
| hardaug6 ep85 | 없음 | [0.01, 0.60] | 50% | 39.90 | 57.63 | 75.51 |

**핵심 발견: CRM/ZERO는 P9에서 유익했을 가능성이 높다**
- P9에는 aux decoder 없음 → CRM/ZERO의 shortcut 학습 문제 해당 없음
- RGB가 0이 되면 thermal/lidar에서 학습 강제 → multimodal robustness 향상
- **P9+hardaug5 (CRM/ZERO만 제거, 나머지 비슷) 실험이 이 가설의 결정적 ablation**

**핵심 발견: "다양성 극대화" 전략은 실패**
- 넓은 범위가 test에 없는 조건에 capacity 분산
- "적절한 범위에 집중" (hardaug4) > "넓은 범위에 분산" (hardaug6)

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

| 파라미터 | basic-aug | hardaug2 | hardaug3 | hardaug4 | **hardaug5** |
| --- | --- | --- | --- | --- | --- |
| NIGHT_SIM_P | 0.35 | 0.50 | 0.40 | 0.45 | **0.60** |
| BRIGHTNESS | [0.03, 0.25] | [0.03, 0.50] | [0.020, 0.203] | [0.03, 0.45] | **[0.02, 0.20]** |
| SAMPLING | uniform | dark_biased | dark_biased | dark_biased | dark_biased |
| DARK_RATIO | - | 0.70 | 0.35 | 0.60 | **0.70** |
| DARK_RANGE | - | [0.03, 0.15] | [0.020, 0.035] | [0.03, 0.12] | **[0.02, 0.06]** |
| CRM_P | 0.30 | 0.30 | 0.25 | 0.35 | **제거** |
| ZERO_P | 0.12 | 0.08 | 0.06 | 0.09 | **제거** |
| CONTRAST | [0.3, 0.7] | [0.3, 0.7] | [0.3, 0.7] | [0.3, 0.7] | **[0.20, 0.65]** |
| GAMMA | [0.4, 0.8] | [0.4, 0.8] | [0.4, 0.8] | [0.4, 0.8] | **[0.30, 0.75]** |
| NOISE_STD | 0.02 | 0.02 | 0.02 | 0.02 | **0.025** |

**결론**: hardaug4가 P9 기준 최선 (M=81.47). hardaug5는 CRM/ZERO 제거 + 실측 밝기 정렬. P14(M=74.27), P15(M=71.05) 모두 hardaug5 사용했으나 하락은 아키텍처(energy fusion) 문제가 주원인.

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

## 진단 및 분석 도구

### interactive_gamma_viewer.py (EnhancementViewer)

- **목적**: P9 test worst 이미지들에 대해 gamma/brightness/contrast/denoise를 실시간 조절하며 인퍼런스 결과를 육안 비교
- **배경**: test 실패 케이스 대부분이 야간 극저조도 → 다양한 보정 조합으로 segmentation 개선 여부 탐색
- **대상 데이터**: P9 hardaug4 test 결과 (`frames_test.csv`) 중 mIoU 하위 N장
  - Worst 3: lj4_1_035090 (46.15), lj4_1_019840 (49.37), lj4_1_086900 (49.78)
  - mIoU<50: 3장, mIoU<60: 17장, mIoU<70: 128/200장
- **기능**:
  - 2×2 뷰: Original RGB | Enhanced RGB / Baseline Prediction | Enhanced Prediction
  - 4개 슬라이더: Gamma (0.5~4.0), Brightness (-0.3~+0.3), Contrast (0.5~3.0), Denoise (0~5)
  - Enhancement 적용 순서: Gamma → Brightness (additive) → Contrast (around mean) → Denoise (bilateral filter)
  - 슬라이더 조절 시 매번 재인퍼런스 (baseline은 캐시)
  - 슬라이더 값은 이미지 전환 시에도 유지 (동일 설정으로 여러 이미지 비교 가능)
  - Baseline 대비 변경 픽셀 수/비율 터미널 로깅
  - 클래스별 픽셀 분포 출력 (Static/Dynamic/Water/Sky)
- **사용법**:
  ```bash
  python interactive_gamma_viewer.py \
    --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \
    --model_path outputs/MMSamP9/.../epoch47_94.18_checkpoint.pth \
    --csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
    --dataset_root /ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night \
    --top_n 30
  ```
- **주요 발견**: brightness=+0.11 / contrast=×0.9 조합이 worst 이미지에서 육안상 개선 확인 → CV heuristic 실험으로 이어짐

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

### MISC/heuristic_enhancement_reinference.py

- **목적**: 성능 하위 test 이미지에 brightness/contrast/gamma 보정 적용 후 재인퍼런스하여 MACVi 제출 폴더 갱신
- **배경**: `interactive_gamma_viewer.py`로 worst case 이미지 육안 분석 결과, brightness +0.11 / contrast ×0.9 적용 시 야간 극저조도 이미지의 segmentation 개선 확인 → 해당 설정을 batch로 적용하는 자동화 스크립트
- **Enhancement 파이프라인** (적용 순서):
  1. Gamma correction: `img^(1/gamma)` — gamma>1 → 밝아짐 (기본값 1.0 = 미적용)
  2. Brightness (additive): `img + brightness` — 양수 → 밝아짐 (0-1 float scale)
  3. Contrast (around mean): `(img - mean) * contrast + mean` — <1 → 대비 감소 (어두운 영역 디테일 살림)
- **인퍼런스**: `val_multiaqua.py`와 동일 파이프라인 (hydra_overrides_extra, get_val_augmentation, softmax→:n_classes→argmax, _unpad_resize_to_orig)
- **주의**: RGB에만 보정 적용 (LiDAR/Thermal은 원본 그대로)
- **사용법**:
  ```bash
  # 기본: mIoU < 55인 이미지에 brightness=0.11, contrast=0.9 적용
  python MISC/heuristic_enhancement_reinference.py \
    --config configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \
    --checkpoint outputs/MMSamP9/.../epoch47_94.18_checkpoint.pth \
    --frames-csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
    --macvi-dir outputs/MMSamP9/.../epoch47_94.18_eval_macvi_CV2 \
    --miou-threshold 55 --brightness 0.11 --contrast 0.9

  # dry-run: 대상 이미지만 출력 (인퍼런스 안 함)
  python MISC/heuristic_enhancement_reinference.py \
    --frames-csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
    --miou-threshold 55 --dry-run
  ```

---

### P9 CV Heuristic Enhancement 실험 (실험 IV)

**개요**: P9 best checkpoint (hardaug4 ep47) 기반으로, test worst 이미지에 RGB brightness/contrast 보정 후 재인퍼런스하여 MACVi 제출.

**보정 파라미터**: brightness=+0.11 (밝게), contrast=×0.9 (대비 감소)
- Brightness +0.11: [0,1] float에서 additive → 전체적으로 밝아짐
- Contrast ×0.9: mean 기준 차이를 줄임 → 어두운 영역 디테일 개선, 밝은 영역 약간 어두워짐

#### CV-1: mIoU < 65 threshold (56장 수정)

- **Checkpoint**: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch47_94.18_checkpoint.pth`
- **제출 폴더**: `epoch47_94.18_eval_macvi_CV` (baseline macvi 345장 복사 후 56장 덮어쓰기)
- **결과**: Val 93.30 / Test **64.86** / Test Obstacle 17.16 / M **79.08**
- **Challenge result**: `outputs/MMSamP9/.../epoch47_94.18_eval_macvi_CV_P9_16485_results/`
- **Submission**: #16485
- **비고**: **실패** — baseline (test 69.62, M 81.47) 대비 test mIoU -4.76, M -2.39. 56장이 너무 많아 원래 잘 맞던 이미지까지 보정되어 성능 하락.

#### CV-2: mIoU < 55 threshold (5장 수정) — 제출 대기

- **Checkpoint**: 동일
- **제출 폴더**: `epoch47_94.18_eval_macvi_CV2` (baseline macvi 345장 복사 후 5장만 덮어쓰기)
- **수정 대상**:
  - lj4_1_035090 (mIoU 46.15)
  - lj4_1_019840 (mIoU 49.37)
  - lj4_1_086900 (mIoU 49.78)
  - lj4_1_027260 (mIoU 50.53)
  - lj4_1_009680 (mIoU 54.16)
- **비고**: CV-1 실패 후 더 적은 이미지만 타겟. 5장이면 전체 200장의 2.5%만 수정.
- **스크립트**: `MISC/heuristic_enhancement_reinference.py`

---

### P9 FDA Augmentation 실험 (실험 V — 취소)

#### P9-FDA: hardaug4 + FDA (Fourier Domain Adaptation)

- **Config**: `configs/levine-multiaqua_rgbtl_P9_hardaug4_fda.yaml`
- **모델**: P9 (LoRA_Sam_P9)
- **구현 파일**: `semseg/augmentations_mm.py` — `RandomFDA` 클래스
- **Ref**: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation" (CVPR 2020)
- **상태**: ~~구현 완료, 학습 대기~~ → **취소 (실험 불채택)**

**취소 사유 — FDA가 극단적 day↔night 밝기 차이에 부적합:**
- FDA는 소스의 low-freq FFT amplitude를 타겟의 것으로 교체하는 방식
- **주간↔야간은 저주파 amplitude 차이가 수십 배** → 교체 시 phase와 amplitude 간 에너지 충돌 발생
- day→night: 결과에 visible noise/artifact 발생 (beta=0.01~0.03 모두)
- night→day: 이미지가 완전히 깨짐 (인식 불가 수준)
- `clamp(0,1)` 문제: 범위 밖 값을 잘라내면서 추가 distortion 발생. min-max norm으로 대체하면 style transfer 효과 자체가 사라짐 (회색빛)
- FDA 원논문은 **Cityscapes↔GTA** 등 밝기가 유사한 도메인 간 adaptation 전제 → day↔night 극단적 gap에는 설계 자체가 맞지 않음
- **결론**: 주파수 도메인 접근은 이론적으로 타당하나, 현재 데이터셋의 극단적 밝기 gap에서는 유의미한 style transfer 없이 noise만 추가됨

**검토한 대안들:**
- PhysAug (AAAI 2025): Random conv + planar wave 기반 물리 augmentation → 날씨/대기 열화 시뮬레이션 목적이라 야간 조명 gap 해결과 무관. 보류.
- per-channel min-max norm: style transfer 효과(밝기/톤 변화) 자체를 파괴 → 부적합
- beta 극소화 (0.001~0.005): noise는 줄지만 style transfer 효과도 무의미해짐

```bash
# 학습 명령 (미실행)
# python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P9_hardaug4_fda.yaml
```

---

### P9 PhysAug 실험 (실험 VI — 학습 대기)

#### P9-PhysAug: hardaug4 + PhysAug (Physical-guided Augmentation)

- **Config**: `configs/levine-multiaqua_rgbtl_P9_hardaug4_physaug.yaml`
- **Eval config**: `configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4_physaug.yaml`
- **모델**: P9 (LoRA_Sam_P9), 새로 학습
- **Save dir**: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4_physaug`
- **핵심 변경**: 기존 hardaug4 NightSim(global scalar) + PhysAug(spatial-varying perturbation) 추가
- **동기**:
  - NightSim은 brightness/contrast/gamma/noise만 수행 → 공간적으로 균일한 변환
  - 실제 야간: 가로등, 수면 반사 등 비균일 조명 + 대기 산란 노이즈 패턴
  - FDA(실험 V)는 day↔night 극단적 밝기 차이로 artifact 발생 → 취소
  - PhysAug는 NightSim과 orthogonal: additive perturbation (원본 구조 유지)
- **PhysAug 두 모듈**:
  1. **Filter**: random convolution (identity + Gaussian noise) → 비균일 조명 시뮬레이션
  2. **Fourier**: planar sinusoidal wave + atmospheric light → 대기 산란/회절 패턴
- **PhysAug 파라미터** (segmentation용 보수적 설정):
  - P=0.40 (40% 확률 적용)
  - FILTER: SIGMA_RANGE=[0.0, 1.5], KERNEL_SIZE=3
  - FOURIER: GROUPS=[1,513], MEAN_STR=8.0, DECAY=0.3
- **augmentation 파이프라인 순서**:
  ResizeWidthPadToSquare → ColorJitter → **RandomPhysAug** → NightSim → CRM → ZeroOut → Flip → Blur → Crop → Normalize
- **설계 근거**:
  - PhysAug → NightSim 순서: filter의 min-max norm 후 NightSim이 어둡게 → 공간적 변화 보존
  - 반대 순서면 min-max norm이 NightSim의 어둡게 효과를 원복시킴
  - 원본 PhysAug(AAAI 2025, object detection) 대비 보수적 파라미터 (sigma_max 4→1.5, mean_str 5→8)
  - Segmentation boundary 보존을 위해 P=0.40 (60%는 원본 유지)
- **Ref**: PhysAug (AAAI 2025) — Physical-guided and Frequency-based Data Augmentation for Single-Domain Generalized Object Detection
- **구현 파일**: `semseg/augmentations_mm.py` — `RandomPhysAug` 클래스
- **뷰어**: `MISC/physaug_viewer.py`
- **상태**: 구현 완료, 학습 대기

```bash
# 학습 명령
python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P9_hardaug4_physaug.yaml

# 뷰어 (시각적 확인)
python MISC/physaug_viewer.py
```
