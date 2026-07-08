---
legacy_file: outputs_model_explain/P10_CrossModalFusionHeadV2.md
moved: 2026-07-08
---

# P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P10` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 1859) |
| 베이스 | P9 |
| 상태 | **취소** (test 성능 하락) |
| 최선 M-score | 79.27 (hardaug4) |

## P9에서의 변경 동기

P9의 CrossModalFusionHead가 near-constant weight를 출력 → gating이 충분히 adaptive하지 않음.

**가설**:
1. GAP만으로는 품질 정보 부족 → multi-pool(GAP + GMP + Std)로 풍부한 통계 추출
2. Scoring 함수에 직접적인 supervision이 없음 → GT 기반 oracle weight로 KL loss 추가

## 아키텍처 변경

### CrossModalFusionHeadV2 (V1 교체)

```python
class CrossModalFusionHeadV2:
    # Multi-pool: GAP + GMP + Channel Std
    gap = AdaptiveAvgPool2d(1)       # 평균 신호
    gmp = AdaptiveMaxPool2d(1)       # 최대 활성화
    # std = channel-wise std          # 텍스처/노이즈 indicator

    # Per-modality compress (V1과 달리 각 모달리티별 독립)
    compress_per_modal = ModuleList[Linear(in_ch * 3, in_ch // 4)]

    # Cross-modal compare
    compare = Linear(in_ch // 4 * num_modalities, num_modalities)
```

**V1 대비 차이**:
- GAP만 → GAP + GMP + Std (3배 정보)
- Shared compress → Per-modality compress (모달리티별 독립 특징 추출)

### ModalAuxHead (신규)

```python
class ModalAuxHead:
    # 각 모달리티별 경량 segmentation head
    # conv1x1 → BN → ReLU → conv1x1 → num_classes
    # 출력: raw logits (B, C, H, W)
```

- 각 모달리티의 backbone feature로 독립 segmentation 수행
- GT와 비교하여 per-modal IoU 계산 → oracle weight 생성

### Oracle KL Loss

```python
# 학습 시:
per_modal_iou = [compute_iou(aux_pred[i], gt) for i in range(m)]
oracle_weights = softmax(per_modal_iou / τ)           # GT 기반 "정답" 가중치
kl_loss = KL_divergence(amf_weights || oracle_weights) # scoring을 정답에 맞추는 loss

total_loss = seg_loss + λ_gate * kl_loss  # LAMBDA_GATE = 0.5
```

## P9 대비 변경 요약

| 구분 | P9 | P10 |
|------|----|----|
| Fusion Head | CrossModalFusionHead (GAP only) | **CrossModalFusionHeadV2 (GAP+GMP+Std)** |
| Compress | Shared across modalities | **Per-modality independent** |
| Aux Head | 없음 | **ModalAuxHead (per-modal seg)** |
| Loss | seg_loss only | **seg_loss + λ_gate × KL(amf \|\| oracle)** |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug4 | 93.23 | 65.30 | 79.27 | **-2.20** | #15731 |
| hardaug3 | 93.18 | 58.93 | 76.05 | **-5.42** | #15757 |

## 해결하려는 문제를 해결했는가?

### Near-constant weight → Adaptive weight? **실패**

- Multi-pool(GAP+GMP+Std)을 추가했으나, frozen encoder가 이미 quality 차이를 정규화
- Per-modality compress가 추가 파라미터로 과적합 유발

### Oracle KL로 scoring 학습? **역효과**

- Oracle supervision이 **주간(Val) 데이터에 과적합**
- 학습 시: oracle가 정확한 가중치 제공 → scoring이 잘 학습
- 테스트 시: oracle 없음, scoring이 **학습 분포(주간)의 패턴만 기억** → 야간에서 잘못된 가중치
- **Train-test 불일치**: 학습 시 GT 의존 → 추론 시 GT 부재

## 핵심 교훈

1. **Multi-pool도 해결책이 아님**: GAP+GMP+Std를 합쳐도 frozen encoder 출력의 quality 차이는 여전히 미미
2. **Oracle supervision은 domain gap을 악화**: 주간 데이터의 oracle weight 패턴이 야간으로 일반화되지 않음
3. **복잡도 증가 = 과적합 위험**: Per-modality compress + Aux head + KL loss → MULTIAQUA의 작은 데이터셋(2,952장)에서 과적합
4. **Val mIoU는 유지되지만 Test mIoU가 하락**: 주간 성능 유지 + 야간 일반화 실패의 전형적 패턴

## 취소 이유

- P9 대비 M-score -2.20 하락 (hardaug4 기준)
- Test mIoU 65.30으로 P8의 hardaug2(63.45)보다 약간 나은 수준
- 아키텍처 복잡도 대비 효과가 전혀 없음
