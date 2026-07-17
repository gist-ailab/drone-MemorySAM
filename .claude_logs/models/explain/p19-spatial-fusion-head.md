---
legacy_file: outputs_model_explain/P19_SpatialFusionHead.md
moved: 2026-07-08
---

# P19: Learned Spatial Cross-Modal Fusion (SpatialCrossModalFusionHead)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P19` |
| 파일 | `sam_lora_image_encoder_seg.py` |
| Fusion Head | `sam_lola_utils.py` — `SpatialCrossModalFusionHead` |
| 베이스 | P9 (P14~P17의 aux decoder 접근 포기) |
| 상태 | 완료 (실패, M=69.63) |
| 최선 M-score | 69.63 (hardaug5, ep36) |

## 변경 동기

P14~P17은 aux decoder 기반 adaptive fusion → 모두 실패 (aux mask 품질 한계).
**P19**: aux decoder **없이**, backbone feature에서 직접 학습 가능한 spatial 가중치를 생성.

P9 CrossModalFusionHead는 GAP으로 공간정보 소실 → scalar (B,m).
P10은 GAP+GMP+Std 시도 → 실패 (M -2.2).
P17은 aux entropy로 spatial → aux mask 품질 의존.

**P19 접근**: multi-scale FPN feature를 직접 학습 가능한 conv network로 처리하여 spatial `(B, m, H, W)` 가중치 생성. Aux decoder 불필요.

## 아키텍처

### SpatialCrossModalFusionHead

```
Phase A: Multi-Scale FPN Projection (shared across modalities)
  fpn[0] (32ch, 256²) → Conv1×1(32→32) → BN → ReLU  ──────────→ (B, 32, 256, 256)
  fpn[1] (64ch, 128²) → Conv1×1(64→32) → BN → ReLU → ×2 upsample → (B, 32, 256, 256)
  fpn[2] (256ch, 64²) → Conv1×1(256→32) → BN → ReLU → ×4 upsample → (B, 32, 256, 256)
                                                          concat → (B, 96, 256, 256)

Phase B: Per-Modality Spatial Context (shared)
  DWConv 3×3(96, groups=96) → BN → ReLU → Conv1×1(96→32) → BN → ReLU
  → (B, 32, 256, 256)  -- LiDAR density, Thermal padding, RGB illumination 패턴

Phase C: Cross-Modal Spatial Comparison
  concat m modalities → (B, 96, 256, 256)
  → Conv1×1(96→64) → BN → ReLU
  → DWConv 3×3(64, groups=64) → BN → ReLU  -- spatial coherence
  → Conv1×1(64→3) [zero-init]
  → softmax(dim=1) → (B, 3, 256, 256)
```

## P9 대비 변경 요약

| 구분 | P9 | P19 |
|------|----|----|
| Fusion Head | CrossModalFusionHead (GAP) | **SpatialCrossModalFusionHead (DWConv)** |
| FPN Input | fpn[0] only (32ch) | **fpn[0]+[1]+[2] (352ch)** |
| Weight Shape | (B, m) scalar | **(B, m, H, W) spatial** |
| UAMM | scalar broadcast | **per-level F.interpolate** |
| AMF | `.view(-1,1,1,1)` | **`_resize_weight()` spatial** |
| Aux Decoder | 없음 | **없음** (P14~P17과 다름) |
| Fusion Params | ~15K | ~23K |
| Augmentation | hardaug4 | **hardaug5** |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug5 (ep36) | 93.44 | **45.82** | **69.63** | **-11.84** | #16313 |

### Per-Class Test IoU

| Class | P9 | P19 | Delta |
|-------|-----|-----|-------|
| Static | 81.30 | 60.75 | -20.55 |
| Dynamic | 21.86 | 23.39 | +1.53 |
| Water | 94.61 | 94.36 | -0.25 |
| Sky | 76.54 | **3.77** | **-72.77** |

### Sky 완전 붕괴

- Sky=0 프레임: **169/200** (84.5%)
- Sky<10% 프레임: **191/200** (95.5%)

### UAMM/AMF 분석

| 지표 | P9 | P19 |
|------|-----|-----|
| AMF img | 0.275 | 0.298 |
| AMF lidar | 0.355 | **0.403** |
| AMF thermal | 0.370 | 0.299 |
| UAMM lidar (test) | ~0.96 | **0.992** (거의 독점) |

- **LiDAR 편향 수렴**: P9의 thermal 우세 균형(0.370) → P19의 lidar 우세(0.403)

## 해결하려는 문제를 해결했는가?

### 학습 가능한 spatial fusion으로 위치별 모달리티 선택? **완전 실패**

실패 원인 분석:

1. **hardaug5 문제**: CRM/ZERO 제거 + 좁은 brightness → P9+hardaug6(M=75.95)도 하락. Aug 자체 문제
2. **Spatial fusion 과적합**: train(주간) night feature ≠ real(야간) night feature → 학습된 spatial pattern이 전이 실패
3. **LiDAR 편향 수렴**: zero-init에서 학습하며 LiDAR 중심으로 수렴 → P9의 thermal 우세 균형 파괴
4. **CRM/ZERO 부재**: P9에서 유익했던 multimodal 강제 학습 신호 부재

### P14~P17(aux 기반) vs P19(학습 기반) — 동일 패턴

| 접근 | 대표 모델 | M-score | 실패 모드 |
|------|----------|---------|----------|
| Aux energy → scalar | P13 | 81.21 | LiDAR UAMM=1.0 |
| Aux energy → spatial | P15 | 71.05 | Spatial amplification |
| Aux entropy → 4 Fixes | P16 | 68.42 | Thermal 지배 |
| Aux multi-scale → spatial | P17 | 73.23 | 부분 회복, 부족 |
| **Learned spatial (no aux)** | **P19** | **69.63** | **LiDAR 편향 수렴** |

## 핵심 교훈

1. **학습 가능 fusion은 계속 실패**: P12~P19 8개 실험 전부 P9(고정 상수)보다 나쁨
2. **Aux 없이 direct learning도 실패**: aux mask 품질 문제를 우회하더라도, spatial fusion 자체가 주간→야간 일반화 실패
3. **LiDAR 편향은 모든 adaptive fusion의 공통 패턴**: LiDAR의 consistent spatial structure가 학습 시 가장 쉬운 최적화 경로
4. **P9의 "학습된 상수"가 최적인 이유**: SAM2 memory attention이 implicit cross-modal adaptation 수행 → 외부 explicit fusion이 오히려 이를 방해

## 이후 방향 전환

P12~P19의 일관된 실패 → **UAMM/AMF scoring 개선은 포기**하고 다른 축으로 접근:
- P20: MoE gate 구조 개선 (MLP + rank 상향)
- P21: DeBA-FP (feature pyramid refinement)
- Augmentation 개선: hardaug8 + PhysAug
