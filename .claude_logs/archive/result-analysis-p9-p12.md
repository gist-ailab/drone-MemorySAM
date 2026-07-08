---
legacy_id: 05
legacy_file: 05_result_analysis_P9_P12.md
moved: 2026-07-08
---

# P9 vs P12 실험결과 종합 분석 보고서

> 🗄 **[ARCHIVED — 2026-02 동결]** P9~P12 시절 1회성 심층 분석본. 현행 실험 요약은 [03_experiment_log.md](03_experiment_log.md) 참조.
> 분석일: 2026-02-25
> 분석 대상: P9 hardaug4 (Submission #15635), P12 hardaug4 (Submission #15949)

---

## 1. 핵심 요약 (Executive Summary)

| 지표 | P9 | P12 | Δ (P12-P9) | 비고 |
|------|------|------|-----------|------|
| **Val mIoU** (global) | 93.32 | 93.23 | -0.09 | 사실상 동일 |
| **Test mIoU** (global) | 69.62 | 68.37 | **-1.25** | P12 약간 열세 |
| **M-score** | **81.47** | 80.80 | **-0.67** | P9 유지 |
| Val Obstacle (Dynamic) | 78.85 | 78.47 | -0.38 | 동일 수준 |
| Test Obstacle (Dynamic) | 21.25 | 25.27 | **+4.02** | **P12 우세** |
| Inference FPS | 1.29 | 1.04 | -0.25 | P12 약간 느림 |

**결론**: P12는 Dynamic 클래스에서 P9 대비 개선되었으나, **Sky 클래스에서 대폭 하락** (-6.81pp)하여 전체 M-score가 오히려 낮아짐. P9이 여전히 최선 모델.

---

## 2. M-score 공식 수정사항

**문서 기록**: `M = 0.75 × val_mIoU + 0.25 × test_mIoU`
**실제 공식**: `M = (val_mIoU + test_mIoU) / 2`

검증:
- P9: (93.32 + 69.62) / 2 = **81.47** ✓
- P12: (93.23 + 68.37) / 2 = **80.80** ✓
- P10: (93.23 + 65.30) / 2 = **79.27** ✓
- P8 hardaug: (92.96 + 63.93) / 2 = **78.45** ✓

> ⚠️ `03_experiment_log.md`의 M-score 공식 수정 필요 (0.75/0.25 → 0.5/0.5)

---

## 3. Per-Class 상세 분석

### 3.1 클래스별 Val → Test 갭 (도메인 갭)

| 클래스 | P9 Val | P9 Test | P9 Gap | P12 Val | P12 Test | P12 Gap |
|--------|--------|---------|--------|---------|----------|---------|
| Static | 94.65 | 81.30 | -13.35 | 94.18 | 79.38 | -14.80 |
| **Dynamic** | 60.02 | 21.86 | **-38.16** | 58.89 | 25.59 | **-33.30** |
| Water | 98.94 | 94.61 | -4.32 | 98.90 | 94.20 | -4.70 |
| **Sky** | 98.03 | 76.54 | -21.49 | 97.47 | 69.73 | **-27.74** |

**분석**:
- **Dynamic**: P9의 최대 약점 (gap -38pp). P12가 gap을 5pp 줄임 → input-conditioning이 소형 객체 탐지에 일부 도움
- **Sky**: P12에서 gap이 6pp 추가 확대 (-21.5→-27.7). 야간 하늘 구분이 P12에서 더 어려워짐
- **Water/Static**: 양 모델 유사한 수준

### 3.2 Dynamic 클래스 심층 분석

**Test셋 Dynamic IoU 분포:**

| 구간 | P9 (200장) | P12 (200장) | 비고 |
|------|-----------|------------|------|
| **IoU = 0** | 38장 (19.0%) | **17장 (8.5%)** | P12 대폭 개선 |
| IoU < 20 | 120장 (60.0%) | **97장 (48.5%)** | P12 개선 |
| 20 ≤ IoU < 50 | 33장 (16.5%) | 42장 (21.0%) | P12 약간 더 많음 |
| IoU ≥ 50 | 25장 (12.5%) | 36장 (18.0%) | P12 개선 |
| IoU = 100 | 9장 (4.5%) | 8장 (4.0%) | 유사 |

**Dynamic 클래스에 대한 결론**:
- P12가 Dynamic IoU=0인 프레임을 38 → 17장으로 절반 이상 줄임
- 완전 미탐지(IoU=0) 문제 완화에 input-conditioning이 효과적
- 그러나 **mean Dynamic IoU 자체가 21.86 → 25.59로 여전히 매우 낮음** (25% 수준)
- 야간 Dynamic obstacle은 근본적으로 어려운 문제 (작고, 어두운 수면 위 물체)

### 3.3 Sky 클래스 문제 — P12의 치명적 약점

**Test셋 Sky IoU 분포:**

| 구간 | P9 | P12 | 비고 |
|------|-----|------|------|
| Sky IoU = 0 | 4장 | 4장 | 동일 |
| Sky IoU < 30 | 12장 | **18장** | P12 악화 |
| Sky IoU < 50 | 15장 | **23장** | P12 악화 |

**P12에서 Sky 하락 원인 추정**:
- P12의 Input-Conditioned gating이 야간에서 sky feature를 다른 클래스 feature와 혼동
- 야간 하늘은 매우 어두워서 water/static과 텍스처가 유사 → gating 혼란
- P9의 고정 가중치가 오히려 이 경우 더 안정적으로 작용

### 3.4 Per-Frame P9 vs P12 비교

| 비교 | 프레임 수 | 비율 |
|------|----------|------|
| P12 > P9 (0.5pp+) | 88 | 44.0% |
| P9 > P12 (0.5pp+) | 84 | 42.0% |
| Tie (±0.5pp) | 28 | 14.0% |

**P12 대폭 개선 (Δ > +5pp): 25장** — Dynamic 탐지 개선이 주원인
**P12 대폭 하락 (Δ < -5pp): 23장** — Sky 분류 실패가 주원인

최악 하락:
- `lj4_1_061980`: P9=67.0 → P12=36.1 (Δ=-30.9) — Sky 3.5%, Static 43.1%
- `lj4_1_072570`: P9=71.3 → P12=42.1 (Δ=-29.2) — Sky 10.2%
- `lj4_1_023080`: P9=65.9 → P12=37.3 (Δ=-28.5) — Sky 11.1%

→ P12의 tail-end 실패가 극심 (최대 -30.9pp). 안정성 부족.

### 3.5 성능 분포 비교

| 구간 | P9 | P12 |
|------|-----|------|
| Good (mIoU ≥ 75) | 29장 (14.5%) | 33장 (16.5%) |
| Mid (55-75) | 166장 (83.0%) | 149장 (74.5%) |
| **Bad (< 55)** | **5장 (2.5%)** | **18장 (9.0%)** |

**P12는 good 프레임이 약간 많지만, bad 프레임이 3.6배 증가** → 분산이 높고 안정성이 떨어짐

---

## 4. UAMM/AMF 분석 — 상수 출력 문제

### 4.1 P9 UAMM/AMF (ISSUE-003 확인)

| 지표 | img | lidar | thermal |
|------|-----|-------|---------|
| UAMM Val | 0.7453 ± 0.0001 | 0.9609 ± 0.00001 | 1.0000 ± 0.0000 |
| UAMM Test | 0.7450 ± 0.00006 | 0.9608 ± 0.00003 | 1.0000 ± 0.0000 |
| AMF Val | 0.2754 ± 0.00003 | 0.3551 ± 0.00001 | 0.3695 ± 0.00002 |
| AMF Test | 0.2753 ± 0.00003 | 0.3551 ± 0.00000 | 0.3696 ± 0.00002 |

**P9는 UAMM/AMF가 완전 상수** (std < 0.001). 모든 이미지에 동일한 가중치 적용.
- thermal이 항상 1위 (100%), lidar 2위 (96%), img 3위 (75%)
- 야간 이미지에서 RGB가 어두워도 항상 27.5% 가중치 → adaptive fusion이 아님

### 4.2 P12 UAMM/AMF (개선 여부)

| 지표 | img | lidar | thermal |
|------|-----|-------|---------|
| UAMM Val | 0.7702 ± 0.0041 | 0.9319 ± 0.0028 | 1.0000 ± 0.0000 |
| UAMM Test | 0.7373 ± **0.0105** | 0.9279 ± 0.0066 | 1.0000 ± 0.0000 |
| AMF Val | 0.2850 ± 0.0009 | 0.3449 ± 0.0005 | 0.3701 ± 0.0009 |
| AMF Test | 0.2766 ± 0.0022 | 0.3482 ± 0.0006 | 0.3752 ± 0.0024 |

**P12 관찰**:
- Val std가 P9 대비 ~30배 증가 (0.0001 → 0.004) — 약간의 이미지별 변동
- **Test std가 val 대비 더 큼** (0.0105 vs 0.0041 for img) — 야간에서 더 큰 변동
- 하지만 여전히 **UAMM img range [0.71, 0.78]** → 변동폭이 매우 작음
- thermal이 항상 1.0000으로 고정 → thermal 최우선 구조 불변

**결론**: P12가 P9 대비 UAMM/AMF 변동성을 약간 도입했으나, 실질적으로 의미 있는 adaptive fusion은 아직 달성되지 않음. range가 0.07 수준이면 사실상 near-constant.

---

## 5. MoE Expert Routing 분석

### 5.1 P9 Expert Collapse 현황 (ISSUE-002)

Block9_Q (핵심 레이어):
- **lidar E1 = 1.0%** ⚠ (collapsed)
- **thermal E1 = 0.5%** ⚠ (collapsed)
- img는 E0=50.1%, E1=8.3%, E2=41.6%로 상대적으로 분산

→ 3-expert MoE가 Block9에서 실질적으로 **2-expert로 동작**. 용량 1/3 낭비.

### 5.2 P12 Expert Collapse 현황 — 더 심각

P12 collapse 지점 (argmax < 5%):
- Block0_Q: img E2=3.3%, lidar **E0=0%, E2=0%**, thermal E0=0.9%
- Block9_Q: lidar E0=0.2%
- Block18_Q: img E0=3.2%, lidar **E0=1.6%, E1=0.4%**, thermal E0=4.9%

**P12가 P9보다 collapse가 더 심각**:
- Block0 lidar: 3개 expert 중 E1만 100% 사용 → 사실상 **단일 expert**
- Block18 lidar: E2가 98% 독점 → **단일 expert**
- input-conditioning이 collapse를 해소하지 못함 — `cond_dim` zero-init이 P9의 `experts_b` zero-init 문제를 물려받음

### 5.3 Val vs Test Routing Divergence (Block9_Q)

| 모달 | P9 Val → Test Δentropy | P12 Val → Test Δentropy |
|------|----------------------|------------------------|
| img | -0.026 | +0.023 |
| lidar | -0.030 | -0.026 |
| thermal | +0.013 | +0.011 |

- P9: Test에서 routing이 약간 더 결정적 (entropy 감소)
- P12: img에서 test entropy가 증가 (더 불확실) — 야간 RGB의 gating 혼란 반영

---

## 6. 실패 모드 분석 (Failure Modes)

### 6.1 공통 실패 모드

**FM-1: Dynamic obstacle 미탐지 (IoU = 0)**
- P9: 38장 (19%), P12: 17장 (8.5%)
- 원인: 야간 수면 위 소형 물체 → 극저조도, 낮은 contrast, 작은 pixel 수
- 이 프레임들에서 Static/Water/Sky는 정상 → Dynamic만 선택적 실패
- Night aug로 개선 여지 제한적 (구조적 한계)

**FM-2: Sky vs Dark region 혼동**
- 야간 하늘이 매우 어두움 → water/static 영역과 유사한 appearance
- P9: 12장에서 Sky<30, P12: 18장에서 Sky<30
- 특히 P12에서 악화 — input-conditioned gating이 어두운 sky를 water로 잘못 분류

**FM-3: Static obstacle 경계 부정확**
- Val Static=94.65, Test Static=81.30 (gap -13pp)
- 야간에서 건물/구조물 경계가 불분명 → 과소 segmentation

### 6.2 P12 고유 실패 모드

**FM-4: 극단적 프레임 성능 붕괴**
- P12에서 mIoU < 45 프레임: 8장 (P9: 0장)
- `lj4_1_061980` (36.1), `lj4_1_023080` (37.3) 등
- 공통 패턴: Sky IoU가 극저 (3.5%, 11.1%) — Sky를 완전히 놓침
- Input-conditioned routing이 특정 야간 조건에서 catastrophically 실패

---

## 7. 아키텍처 관점 원인 분석

### 7.1 왜 P12가 P9을 넘지 못하는가

1. **UAMM/AMF near-constant 문제 미해결**:
   - P12의 input-conditioning이 UAMM 변동성을 0.0001 → 0.01로 올렸지만, 여전히 실질적 상수
   - CrossModalFusionHead의 GAP + LayerNorm → constant output 구조가 근본 원인
   - `cond_dim` projection이 이 구조적 문제를 우회하지 못함

2. **Expert collapse 미해소 + 악화**:
   - `experts_b` zero-init 문제는 P12에서도 동일 (P12는 P9 구조 계승)
   - 오히려 P12의 Block0/Block18에서 collapse가 더 심각
   - Input-conditioning의 추가 파라미터가 기존 collapse를 고착화

3. **Sky 클래스 성능 하락의 구조적 원인**:
   - P12의 cond_proj가 모달리티 타입 정보를 gate에 주입
   - 그러나 야간에서 RGB의 sky region feature가 water feature와 유사
   - cond_proj가 이 혼동을 증폭시킬 수 있음 (잘못된 모달리티 편향 학습)

4. **안정성-성능 트레이드오프**:
   - P12는 Dynamic에서 +3.73pp 개선 (good)
   - 하지만 Sky에서 -6.81pp 하락 + tail-end failure 증가 (bad)
   - 추가 complexity가 robustness를 저해

### 7.2 왜 P9이 강한가 (역설적 분석)

P9의 **상수 가중치가 오히려 장점**:
- thermal=37%, lidar=35.5%, img=27.5%로 고정
- 야간에서 thermal이 가장 신뢰도 높음 → 높은 가중치 정당화
- RGB가 어두워도 27.5% → 약간의 texture 정보는 여전히 유용
- **일관된 fusion이 unstable adaptive fusion보다 나은 결과**를 낳음

이 패턴은 P10/P11에서도 확인:
- 복잡한 메커니즘(oracle KL, MI loss)이 test에서 악화
- simple + consistent > complex + adaptive (현 데이터셋 규모)

---

## 8. 한계점 및 개선 방향

### 8.1 현재 한계점

1. **근본적 도메인 갭** (93% val vs 69% test): Night aug만으로는 실제 야간 조건을 충분히 모사하지 못함
2. **Dynamic 클래스 구조적 한계** (mean 21-25%): 야간 수면 위 소형 물체는 모든 모달리티에서 약함
3. **UAMM/AMF 상수 출력**: CrossModalFusionHead의 GAP + LayerNorm 구조가 constant output을 유도
4. **Expert collapse**: experts_b zero-init이 Block 6-20에서 E1 사망 유발
5. **학습 데이터 부족**: Val 145장 (주간), Test 200장 (야간, 정답 없음) — 매우 작은 데이터셋

### 8.2 P13에 대한 기대와 리스크

**P13 설계 핵심**: ConfidenceAuxHead + Energy Score (학습 가능 파라미터 없는 fusion weight)

**기대**:
- Energy Score는 raw logit 기반 → GAP+LayerNorm의 constant 문제 우회
- experts_b kaiming*0.01 init → E1 collapse 해소 가능성
- 학습/추론 동일 메커니즘 → P10의 train≠test 문제 없음

**리스크**:
- P12의 교훈: Dynamic 개선이 Sky 하락으로 상쇄될 수 있음
- Energy Score가 야간에서 calibration 되어 있지 않을 수 있음
- Aux head의 야간 segmentation 품질이 낮으면 energy score도 부정확

### 8.3 개선 방향 제안

**단기 (P13 학습 후)**:
1. P13의 energy score가 이미지별로 실제 변동하는지 확인 (UAMM std > 0.05 목표)
2. Day-best vs Night-best checkpoint 비교 (ISSUE-001 해결 활용)
3. P9 + P12 ensemble 가능성 검토 (Dynamic과 Sky에서 상보적)

**중기**:
4. **Spatial-wise Energy Weighting** (ISSUE-004): 이미지 전체가 아닌 pixel-level에서 모달리티 가중치 → sky/water 경계에서 모달리티 선택적 활용
5. **Night-Val 기반 checkpoint selection**: Night-Val mIoU로 모델/epoch 선택 → test 상관관계 향상 예상

**장기**:
6. **Diffusion 기반 야간 합성 데이터** (ISSUE-005): Flux/SDXL로 주간 → 야간 변환하여 학습 데이터 확대
7. **Test Time Augmentation (TTA)**: `--tta` 플래그 효과 검증 — multi-scale + flip 평균
8. **Dynamic 클래스 전용 전략**: small object detection 기법 (FPN, ATSS 등) 또는 class-wise loss weighting

---

## 9. 전체 실험 순위 (업데이트)

| 순위 | 모델 | Config | Val mIoU | Test mIoU | M-score | 상태 |
|------|------|--------|----------|-----------|---------|------|
| 1 | **P9** | hardaug4 | 93.32 | 69.62 | **81.47** | **현재 최선** |
| 2 | **P12** | hardaug4 | 93.23 | 68.37 | **80.80** | 완료 |
| 3 | P10 | hardaug4 | 93.23 | 65.30 | 79.27 | 취소 |
| 4 | P8 | hardaug | 92.96 | 63.93 | 78.45 | 완료 |
| 5 | P8 | hardaug2 | 93.29 | 63.45 | 78.37 | 완료 |
| 6 | P8 | basic-aug | 93.13 | 62.50 | 77.82 | 완료 |
| 7 | P8 | hardaug3 | 93.36 | 61.57 | 77.46 | 완료 |
| 8 | P11 | hardaug4 | 93.17 | 61.01 | 77.09 | 취소 |
| 9 | P10 | hardaug3 | 93.18 | 58.93 | 76.05 | 취소 |
| 10 | P8 | no-aug | 93.10 | 35.93 | 64.51 | 기준선 |

---

## 10. 핵심 인사이트 요약

1. **모든 모델의 val mIoU가 93±0.5%** → val은 모델 비교에 무의미. test mIoU만이 유일한 비교 기준
2. **P9의 "상수 가중치"가 현 데이터셋 규모에서는 최적** — adaptive 메커니즘이 오히려 instability 유발
3. **Dynamic class (21-25%)가 M-score의 핵심 병목** — 이 클래스의 개선이 전체 성능의 키
4. **Sky class는 놓치기 쉬운 함정** — Dynamic 개선에 집중하다 Sky가 하락하면 전체 손해
5. **Expert collapse는 모든 P 버전의 공통 문제** — P13의 kaiming*0.01 init이 해결 열쇠
6. **M-score 공식은 0.5/0.5** (val과 test 동일 비중) → test 성능이 생각보다 중요
