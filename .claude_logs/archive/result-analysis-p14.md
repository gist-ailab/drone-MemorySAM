---
legacy_id: 07
legacy_file: 07_result_analysis_P14.md
moved: 2026-07-08
---

# P14 실험 결과 분석

> 🗄 **[ARCHIVED — 2026-02 동결]** P14 1회성 분석본. 현행 실험 요약은 [experiments/log.md](../experiments/log.md) 참조.
> 생성일: 2026-02-27

## 1. P14 실험 개요

- **모델**: LoRA_Sam_P14 (Per-Modality Separate Aux Decoders)
- **Config**: `configs/bengio-multiaqua_rgbtl_P14_hardaug5.yaml`
- **Checkpoint**: `night_epoch47_90.75_top1_checkpoint.pth` (night-val best)
- **Submission**: #16062
- **M-score: 74.27** (P9: 81.47, **-7.20pp**)

## 2. Per-class 성능

| 클래스 | Val mIoU | Test mIoU | Gap | 비고 |
| --- | --- | --- | --- | --- |
| Static | 96.65 | 62.57 | -34.08 | 중간 |
| Dynamic | 78.80 | 22.87 | -55.93 | 심각 |
| Water | 98.93 | 92.92 | -6.01 | 양호 |
| Sky | 98.44 | 36.47 | -61.97 | **최악** |
| **mIoU** | **93.18** | **55.36** | **-37.82** | |

Sky 36.47%: 73/200 프레임 <10%, 56/200 프레임 <1%.

## 3. UAMM/AMF 분석

| 모달리티 | Val UAMM | Test UAMM | Val AMF | Test AMF |
| --- | --- | --- | --- | --- |
| img | 0.752 (std 0.135) | 0.555 (std 0.044) | 0.284 | 0.226 |
| lidar | 0.994 (std 0.019) | **1.000 (std 0.000)** | 0.380 | 0.408 |
| thermal | 0.880 (std 0.062) | 0.898 (std 0.020) | 0.335 | 0.366 |

핵심: Test LiDAR UAMM = 1.000 고정 (200장 전부, stdev=0.000). RGB 억제 (0.555).

## 4. MoE Routing 전체 블록 분석 (Test, 200장)

### 4.1. Expert Collapse 심각도 (Q projection)

16개 block-modality 조합에서 expert <5%:

| Block | Modality | Dead Expert | Weight |
| --- | --- | --- | --- |
| B00 | img | E1 | 1.4% |
| B01 | lidar | E1 | 0.3% |
| B01 | thermal | E2 | 2.9% |
| B04 | lidar | E1 | 3.0% |
| B05 | lidar | E1 | 1.3% |
| B05 | thermal | E1 | 1.4% |
| B12 | lidar | E0 | 0.8% |
| B12 | lidar | E2 | 0.2% |
| B12 | thermal | E2 | 0.9% |
| B16 | img | E2 | 2.9% |
| B16 | lidar | E2 | 2.4% |
| B20 | lidar | E1 | 0.8% |
| B20 | lidar | E2 | 2.2% |
| B20 | thermal | E1 | 3.8% |

**B12, B20 최악**: lidar에서 2개 expert 사실상 사망 → 1개 expert 독점.

### 4.2. 블록별 패턴

- 초기 블록 (B00-B05): **E0 독점** (img 90%, lidar 85%)
- 중간 블록 (B09-B12): **E1 독점** (lidar 64%, thermal 71%)
- 후기 블록 (B16-B20): 혼재 (E0 또는 E1 독점)

### 4.3. Q vs V

| 지표 | Q | V |
| --- | --- | --- |
| Severe collapse (<5%) | 39% (28/72) | 17% (12/72) |
| Balanced (>70% ratio) | 4% (3/72) | 7% (5/72) |

### 4.4. LiDAR routing 완전 정적

모든 블록(0-23), 모든 Q/V에서 LiDAR routing stdev = 0.0000. 200장 야간 이미지에서 routing이 소수점 4자리까지 동일.

### 4.5. 블록 수준 specialization

블록마다 다른 expert 조합을 사용하지만, **이미지 간에는 변하지 않는 고정 패턴**. 입력 적응형이 아니라 학습 중 고정된 lookup table로 퇴화.

## 5. Aux Mask 품질 분석

시각화 확인 (val 145장, test 200장):

- **Aux (img)**: Water/Static 대략적 분할. Sky/Dynamic 경계 부정확.
- **Aux (lidar)**: 거의 전체를 Water로 예측. Sky/Static 구분 불가.
- **Aux (thermal)**: 대략적 Water/Sky 분할. Dynamic 객체 못 잡음.
- **모달리티 간 비교**: GT 대비 성능이 모두 낮아서 "어느 것이 낫다" 판단 불가.

P13(공유 aux head) 대비 P14(독립 aux head)에서 소폭 개선됐으나, 여전히 Energy Score로 모달리티 품질을 비교하기에는 불충분한 수준.

## 6. Model Uncertainty

| 지표 | Val | Test | 배율 |
| --- | --- | --- | --- |
| mean_entropy | 0.178 | 0.570 | 3.2x |
| high_uncertainty_ratio | 13.5% | 63.4% | 4.7x |

Test에서 전체 픽셀의 63%가 high uncertainty.

## 7. 실패 원인 종합

### 1차 원인: LiDAR UAMM = 1.0 고정 + RGB 억제

Energy Score가 LiDAR를 항상 "최고 confident"로 판정 → UAMM=1.0.
Sky 영역에서 LiDAR는 상공 포인트가 없어 무의미 → Sky IoU 붕괴.
RGB는 UAMM 0.555로 억제 → Sky 인식에 핵심인 RGB 정보 절반 감소.

### 2차 원인: Aux mask 품질 부족 (ISSUE-008)

Frozen backbone feature 위의 경량 aux decoder로는 모달리티별 품질을 정확히 측정할 수 없음.
Decoder를 독립화(P14)해도 입력 feature 자체의 정보량 부족이 병목.

### 3차 원인: Image-level scalar fusion 한계

이미지 전체에 동일 가중치 → Sky/Water/Dynamic 영역별 최적 모달리티 반영 불가.
→ P15 spatial-wise 접근 동기.

## 8. P9 vs P13 vs P14 비교

| 버전 | Val mIoU | Test mIoU | M-score | Test Sky | Test Dynamic | 핵심 특징 |
| --- | --- | --- | --- | --- | --- | --- |
| **P9** | 93.32 | **69.62** | **81.47** | 76.54 | 21.86 | Near-constant fusion (안정적) |
| P13 (ep17) | 92.45 | 69.98 | 81.21 | 75.12 | **27.41** | Energy Score (Dynamic 개선) |
| **P14** | 93.18 | 55.36 | 74.27 | **36.47** | 22.87 | 독립 aux + CRM/ZERO 제거 |

P9의 near-constant fusion이 "좋은 기본 비율"로 안정적으로 작동.
Energy Score fusion은 Dynamic을 개선하지만 Sky를 희생.
P14는 CRM/ZERO 제거 + aux 독립화 모두 Sky collapse를 막지 못함.

## 9. Energy Score Fusion의 구조적 한계

### 9.1. Energy Score = confidence, not correctness

Energy Score는 logit magnitude 기반 confidence. "자신있게 틀리는" 경우 오히려 해로움.
P14에서 LiDAR aux head가 모든 영역에서 높은 energy → "confident but wrong" for Sky.

### 9.2. Gradient 경로 문제

현재 `compute_energy_confidence()`에 `.detach()` 없음.
Main loss gradient가 energy → aux heads → LoRA로 역전파.
LoRA가 두 가지 목표를 동시에 최적화 (main seg + energy score 조정) → 충돌 가능.
권장: `compute_energy_confidence([z.detach() for z in aux_logits_list])`

### 9.3. Prototype-based aux 대안 검토

현재 PrototypeSegmentation은 fused feature(`m_feat`)에 대해서만 동작.
모달리티별로 분리하면 gradient 오염 없이 (`.data` EMA) 품질 측정 가능.
하지만 prototype matching은 선형 분류 수준 → aux mask 품질이 conv-based보다 낮을 가능성.
근본 병목은 decoder 종류가 아니라 frozen backbone feature 정보량.
