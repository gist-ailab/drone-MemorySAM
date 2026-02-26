# P13 실험결과 종합 분석 보고서

> 분석일: 2026-02-26
> 분석 대상: P13 hardaug4 (Submission #15997)
> 비교 기준: P9 hardaug4 (Submission #15635, 현재 최선)

---

## 1. 핵심 요약

| 지표 | P9 | P13 | Δ (P13-P9) |
| --- | --- | --- | --- |
| Val mIoU | 93.32 | 92.45 | **-0.87** |
| Test mIoU | 69.62 | 69.98 | +0.36 |
| **M-score** | **81.47** | 81.21 | **-0.26** |
| Val Dynamic | 78.96 | 76.04 | -2.92 |
| Test Dynamic | 21.86 | 27.41 | **+5.55** |
| Test Static | 81.30 | 79.80 | -1.49 |
| Test Sky | 76.54 | 75.12 | -1.42 |
| Test Water | 94.61 | 94.27 | -0.34 |
| Inference FPS | 1.29 | 1.02 | -0.27 |

**결론**: P13은 Dynamic 탐지를 크게 개선(+5.55pp)했으나, val mIoU 하락(-0.87)으로 M-score가 P9보다 -0.26 낮음.

---

## 2. 체크포인트 선택 차이

| | P9 | P13 |
| --- | --- | --- |
| 선택 기준 | Best day-val (epoch47, 94.18) | Best **night-val** (epoch17, 87.71) |
| 학습 에폭 | ~47 epochs (from scratch) | ~17 epochs (resume) |
| Best day-val | epoch47: 93.32 (사용됨) | epoch14: **93.48** (사용 안 됨) |
| 비고 | Night-val 평가 없음 | Night-val 기준 선택 → val 희생 |

P13의 best day-val checkpoint (epoch14, 93.48)로 test 재평가 시 M-score 역전 가능성 있음.

---

## 3. 설계 목표 달성 여부

### 목표 1: Expert Collapse 해결 (kaiming*0.01 init)

**판정: 실패**

| 메트릭 | P12 Val | P13 Val | P12 Test | P13 Test |
| --- | --- | --- | --- | --- |
| Collapse rate | 16.0% | **17.4%** | 20.1% | 17.4% |
| LiDAR collapse | ~27% | ~27% | - | - |
| Q block collapse | ~25% | ~23-25% | - | - |
| V block collapse | ~10% | ~10-11% | - | - |

실패 원인:
- Resume 학습으로 이전 gate weights가 로드 → init 효과 무력화
- kaiming * 0.01 (~0.005 수준)은 zero-init과 실질적 차이 미미
- Collapse의 근본 원인은 init이 아니라 soft-MoE softmax의 winner-take-all 특성

Stage별 collapse rate (P13 val):
- S1 (blocks 0-2): 44-55% — 가장 심각
- S2 (blocks 3-5): ~20%
- S3 (blocks 6-20): 9-13% — 가장 건강
- S4 (blocks 21-23): ~30%

### 목표 2: Energy Score Fusion

**판정: 부분 성공 — 방향은 맞으나 효과 제한적**

#### 성공: UAMM/AMF 변동성 대폭 증가

| UAMM CV | P12 Val | P13 Val | P12 Test | P13 Test |
| --- | --- | --- | --- | --- |
| img | 0.005 | **0.112** (22x↑) | 0.014 | **0.073** (5x↑) |
| lidar | 0.003 | **0.015** (5x↑) | 0.007 | **0.000** |
| thermal | 0.000 | **0.042** (∞↑) | 0.000 | **0.009** |

P12 대비 img UAMM 변동성 22배 증가. Energy Score가 이미지별로 다른 fusion weight를 만들어내는 데 성공.

#### 한계: Test LiDAR 여전히 고정

- Test LiDAR UAMM = **1.0** (모든 200 프레임에서 상수), CV = 0.000
- Test LiDAR MoE routing: **48/48 블록 완전 고정** (P9/P12와 동일)
- 원인: 야간 LiDAR 데이터의 동질성 (조명 불변 센서 → 모든 야간 이미지에서 동일 feature)
- **이것은 P9, P12, P13 공통 현상 → 아키텍처가 아니라 LiDAR 데이터 특성**

---

## 4. Test Per-Class 상세 분석

### Frame-level 비교 (Test 200 frames)

| 분석 항목 | P9 | P13 |
| --- | --- | --- |
| P13 승리 프레임 | - | **118/200 (59%)** |
| P9 승리 프레임 | 82/200 (41%) | - |
| Dynamic=0 프레임 | 38 | **20** (-18) |
| Dynamic>50 프레임 | 25 | **30** (+5) |
| mIoU<55 프레임 | 5 | **3** (-2) |
| Sky=0 프레임 | 4 | 5 (+1) |
| 최대 P13 개선 | - | +11.72pp |
| 최대 P13 하락 | - | -18.90pp |
| Delta 평균 | - | +0.57pp |
| Delta std | - | 4.18pp |

### Dynamic 개선 상세

- P9에서 Dynamic=0이었던 38 프레임 중 19개에서 P13이 탐지 성공
- P13이 Dynamic을 잃은 프레임: 1개만
- P9-P13 Dynamic IoU 상관계수: 0.896

### Sky/Static 하락 상세

- Sky 하락 -1.42pp: P12(-6.81pp)보다는 경미하지만, 소수 프레임에서 catastrophic failure
  - 최악 예: `lj4_1_063370` Sky: 86.88 → 27.39 (-59.49pp)
- Static 하락 -1.49pp: 전반적으로 균일한 소폭 하락

---

## 5. P9 vs P13 학습 궤적 비교

| Epoch | P9 val mIoU | P9 loss | P13 val mIoU | P13 loss | P13 night-val |
| --- | --- | --- | --- | --- | --- |
| 0 | 83.16 | 1.279 | 85.82 | 1.545 | 73.72 |
| 5 | 92.28 | 0.636 | 92.69 | 0.910 | 87.28 |
| 14 | 93.35 | 0.543 | **93.48** | 0.750 | 87.24 |
| 16 | 93.00 | 0.508 | 93.27 | 0.743 | **87.71** |
| 17 | **93.57** | 0.515 | (학습 종료) | - | - |
| 46 | **94.18** | 0.462 | - | - | - |

P9: epoch 17(93.57) → epoch 46(94.18) = **+0.61pp만 추가 상승** (29 epochs 동안).
P13을 더 학습해도 val mIoU ~94.1 수준이 한계. M-score +0.3pp 정도 개선 기대.

P13의 night-val은 마지막 epoch(16)에서도 NEW BEST 기록 중 → 수렴 미완이지만 marginal.

---

## 6. Energy Score Fusion 메커니즘 분석

### compute_energy_confidence 동작

```python
# 각 모달리티의 aux logit에서 energy 계산
energy = -T * logsumexp(z / T, dim=1)  # (B, H, W)
conf = -energy.mean(dim=[1, 2])         # spatial average → (B,)
weights = softmax(conf / T, dim=1)      # (B, m) normalized
```

### P9 CrossModalFusionHead vs P13 Energy Score

| 속성 | P9 CrossModalFusionHead | P13 Energy Score |
| --- | --- | --- |
| 학습 파라미터 | compress + compare (Linear) | **없음** (parameter-free) |
| 출력 변동성 (val CV) | < 0.001 | **0.08-0.11** |
| 출력 변동성 (test CV) | < 0.001 | 0.02-0.07 |
| 장점 | 안정적, 좋은 기본 비율 | 이미지별 adaptive |
| 단점 | 상수 수렴 (ISSUE-003) | aux head 정확도에 의존 |

P9의 "상수 weight"가 우연히 좋은 비율(thermal 37% > lidar 35% > img 28%)이었기에 안정적.
P13의 adaptive weight는 방향은 맞지만, aux head의 energy 추정 정확도가 아직 부족.

### Test 시 모달리티별 weight

| 모달리티 | P9 UAMM | P13 UAMM | P9 AMF | P13 AMF |
| --- | --- | --- | --- | --- |
| img | 0.745 | 0.661 (var=0.048) | 27.5% | 27.1% |
| lidar | 0.961 | **1.000** (var=0.000) | 35.5% | 38.4% |
| thermal | 1.000 | 0.696 (var=0.006) | 37.0% | 34.5% |

P13에서 lidar UAMM이 1.0으로 고정 → lidar가 항상 "가장 confident"로 판정됨.
이는 aux head가 lidar의 불확실성을 정확히 추정하지 못하는 것을 의미.

---

## 7. Night Augmentation 포화 분석

### Augmentation 효과 정량화 (P8 동일 아키텍처, 모델 변경 없음)

| Aug | Test mIoU | vs no-aug | Marginal gain |
| --- | --- | --- | --- |
| no-aug | 35.93 | - | - |
| basic-aug | 62.50 | +26.57 | **+26.57** |
| hardaug | 63.93 | +28.00 | +1.43 |
| hardaug2 | 63.45 | +27.52 | -0.48 |
| hardaug3 | 61.57 | +25.64 | -1.88 |

no-aug → basic-aug: **+26.57pp** (전체 gain의 80%)
basic-aug → best hardaug: **+1.43pp** (전체 gain의 4%)

### M=85 달성 가능성 분석

목표: M = (val + test) / 2 ≥ 85 → val 93 유지 시 test ≥ 77 필요 → **+7.4pp 추가 필요**

| 접근 | 기대 효과 | 근거 |
| --- | --- | --- |
| Augmentation 추가 튜닝 | +1~2pp | P8에서 4개 변종 중 최대 +1.43pp |
| Night-val checkpoint | +0~1pp | P13에서 이미 적용, day-val과 trade-off |
| TTA (multi-scale+flip) | +0.5~2pp | 미검증, 기본 기대치 |
| Diffusion 기반 night 합성 | +3~8pp | 실 야간 조명 패턴 재현 가능 (미구현) |
| Ensemble (P9+P12/P13) | +1~2pp | Dynamic/Sky 상보성 활용 |

**Night Aug만으로 M=85는 불가능. 포화 상태.**

### 클래스별 Night Aug 한계

| Class | Val-Test Gap | Night Aug로 해결 가능? | 이유 |
| --- | --- | --- | --- |
| Water | -4.33pp | 거의 해결됨 | 수면 외관이 주야간 유사 |
| Static | -13.35pp | 부분적 | 밝기 해결되나 edge clarity 미해결 |
| Sky | -21.49pp | **한계** | 어두운 하늘/물 혼동은 밝기 변환으로 불가 |
| Dynamic | -38.16pp | **불가** | 소형 어두운 객체는 밝기와 무관 |

---

## 8. P13이 P9보다 낮은 원인 3가지

### 원인 1: Night-val 선택으로 val mIoU 희생

P13은 night-val best (epoch17, night-val 87.71)를 사용. 해당 체크포인트의 day-val은 93.27.
P13의 best day-val은 epoch14 (93.48) → 이 체크포인트로 test 재평가 시 결과가 달라질 수 있음.
M = (val + test) / 2 공식에서 val -0.87pp 하락이 M-score를 직접 -0.43pp 끌어내림.

### 원인 2: Energy Score 정확도 부족

Energy Score는 aux head의 raw logit quality에 의존.
- 17 epochs 학습으로는 aux head가 모달리티별 confidence를 정확히 추정하기 어려움
- Test에서 lidar UAMM = 1.0 고정 = lidar aux head가 항상 가장 높은 energy를 출력
- 실제로는 lidar 데이터 품질이 가장 낮은데(물 반사 없음, 원거리 미감지), aux head가 이를 반영 못함

### 원인 3: Static/Sky에서의 소폭 하락이 Dynamic 개선을 상쇄

Dynamic +5.55pp 개선은 유의미하지만, Static -1.49pp + Sky -1.42pp + Water -0.34pp로 상쇄.
118/200 프레임에서 P13이 이기지만, 소수 프레임의 큰 폭 하락(-18.9pp max)이 평균을 끌어내림.

---

## 9. 종합 판정 및 향후 방향

### P13 설계 평가

| 설계 변경 | 판정 | 비고 |
| --- | --- | --- |
| kaiming*0.01 init | **실패** | Resume 학습 + 스케일 미미 → 무효 |
| Energy Score fusion | **부분 성공** | UAMM CV 5-22x↑, Dynamic +5.55pp, 하지만 lidar 고정 |
| Aux loss (λ=0.3) | **효과 제한** | Energy 추정 기반 제공하나 정확도 부족 |
| Night-val checkpoint | **효과적** | test +0.36pp, 하지만 M-score 공식에서 불리 |

### 향후 실험 우선순위 (epoch39 crash 이전)

1. **P13 best day-val checkpoint (epoch14_93.48)로 test 재평가** — M-score 역전 가능성 확인
2. ~~P13 epoch 40-50까지 추가 학습~~ — **취소: epoch39에서 test crash 확인 (ISSUE-007)**
3. **Diffusion 기반 night 데이터 합성** (ISSUE-005) — M=85 도달을 위한 필수 접근
4. **TTA 적용** — P9/P13 모두에 미검증, 저비용 개선 가능
5. **P9 + P13 Ensemble** — Dynamic(P13 우세) + Sky(P9 우세) 상보성 활용

---

## 10. P13 Epoch39 Test Crash 분석 (Submission #16044)

> 분석일: 2026-02-26
> 비교 대상: Epoch17 (#15997) vs Epoch39 (#16044)

### 10.1 핵심 수치

| 지표 | Epoch17 | Epoch39 | Δ |
| --- | --- | --- | --- |
| Val mIoU | 92.45 | 92.86 | +0.41 |
| Night-val | 87.71 | **89.53** | +1.82 |
| **Test mIoU** | 69.98 | **50.48** | **-19.50** |
| **M-score** | 81.21 | **71.67** | **-9.54** |

Val과 night-val 모두 개선, test 폭락 → **전형적인 overfitting**

### 10.2 Per-Class Test 비교

| Class | Epoch17 | Epoch39 | Δ | Crash 기여도 |
| --- | --- | --- | --- | --- |
| **Sky** | 75.12 | 23.36 | **-51.76** | **67%** |
| Dynamic | 27.41 | 15.44 | -11.97 | 15% |
| Static | 79.80 | 66.27 | -13.53 | 17% |
| Water | 94.27 | 94.42 | +0.15 | 0% |

Sky 붕괴가 crash의 67%. Epoch17에서 Sky=0 프레임 5개 → epoch39에서 **80/200 프레임**.
192/200 프레임에서 전체적으로 성능 하락.

### 10.3 근본 원인: CRM/ZERO Overfitting (ISSUE-007)

#### 메커니즘

1. **CRM** (p=0.35): RGB에 1-4개 랜덤 직사각형을 exact 0으로 마스킹 (면적 20-50%)
2. **ZERO** (p=0.09): RGB 전체를 exact 0으로 대체
3. 합산 **학습 샘플 ~44%**에 exact-zero 픽셀 존재

#### Exact zero가 문제인 이유

- 실제 야간 센서: noise가 있는 near-zero (0.001~0.01), 절대 exact 0이 아님
- ImageNet normalize 후: `(0-mean)/std = (-2.118, -2.036, -1.804)` — 자연 이미지에서 불가능한 극단값
- 모델이 학습하는 shortcut: "exact zero 감지 → RGB 무시" — train/night-val에서는 유효, test에서는 무효

#### Night-val 오염

`get_nightval_augmentation()`에도 CRM/ZERO가 동일 확률로 적용됨.

| | Night-val | Real Test |
| --- | --- | --- |
| CRM/ZERO 적용 | 있음 (p=0.35+0.09) | **없음** |
| RGB 값 패턴 | exact zero 포함 | noisy near-zero |
| Shortcut 유효 | **유효** | **무효** |

Night-val 87.71→89.53 개선 = shortcut을 더 잘 학습한 결과. 하지만 이 shortcut은 test에서 무효.

#### Sky가 가장 취약한 이유

야간 하늘 = near-zero RGB → CRM/ZERO의 exact zero와 가장 유사 → shortcut이 Sky 영역에서 가장 강하게 활성화 → Sky→Water 오분류 폭증.
Water는 thermal/LiDAR로 충분히 구분 가능 → RGB shortcut 영향 적음.

### 10.4 시사점

| 발견 | 대응 |
| --- | --- |
| Night-val↑ + Test↓ = CRM/ZERO shortcut 과적합 | Night-val에서 CRM/ZERO 제거 |
| Epoch17→39로 학습할수록 shortcut 강화 | Early stopping 필수 (epoch17이 sweet spot) |
| Sky 붕괴가 crash의 67% | Sky/Water 구분은 RGB의 미세한 차이에 의존 |
| 44% exact-zero 패턴 = train-test 분포 불일치 | CRM/ZERO 확률 대폭 축소 또는 noisy near-zero로 대체 |

### 10.5 수정된 향후 실험 우선순위

1. **Night-val에서 CRM/ZERO 제거** — 즉시 적용 가능, 신뢰할 수 있는 test proxy 확보
2. **CRM/ZERO 확률 축소 (CRM 0.35→0.10, ZERO 0.09→0.03)** 또는 noisy near-zero로 대체
3. **P13 epoch17 checkpoint 확정** — 추가 학습은 역효과
4. **Diffusion 기반 night 합성** (ISSUE-005) — M=85 달성의 유일한 경로
5. **TTA / Ensemble** — 저비용 개선
