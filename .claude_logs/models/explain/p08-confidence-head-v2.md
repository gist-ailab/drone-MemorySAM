---
legacy_file: outputs_model_explain/P8_ConfidenceHeadV2.md
moved: 2026-07-08
---

# P8: ConfidenceHeadV2 + Sigmoid UAMM

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P8` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 1134) |
| 베이스 | MemorySAM 최초 구현 |
| 상태 | 완료 (5가지 aug 실험) |
| 최선 M-score | **78.45** (hardaug 기본) |

## 해결하려는 문제

MemorySAM 아키텍처의 최초 구현. SAM2의 시간축 메모리 어텐션을 모달리티 축으로 전용하여 RGB + LiDAR + Thermal 멀티모달 fusion을 수행하는 기본 프레임워크 구축.

## 아키텍처

### Scoring 함수: ConfidenceHeadV2

```
backbone_feats → ConfidenceHeadV2(fusion_dim) → logits → sigmoid → scores (0~1)
                                                         ↓
UAMM: scores 그대로 적용 (각 모달리티 독립)
AMF:  normalized_scores = scores / sum(scores)
```

```python
class ConfidenceHeadV2(nn.Module):
    # Conv2d(in_ch, 64, 3×3, stride=2) → BN → ReLU
    # Conv2d(64, 64, 3×3, stride=2) → BN → ReLU
    # GAP → Linear(64, 32) → ReLU → Linear(32, 1)
    # 출력: logit (스칼라) → sigmoid → 0~1 score
```

- 각 모달리티에 대해 **독립적**으로 0~1 점수 산출
- 모달리티 간 상대 비교 없음 (relative comparison 부재)

### MoE LoRA

- `SoftMoE_LoRA_Layer`: softmax gating, 모든 expert 참여
- Gate: `Linear(dim, num_experts=3)`, rank=4
- 48개 layer (24 blocks × Q/V)

### Forward 흐름

```
Phase 1: 모달리티별 인코딩
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)  → Hiera-B+ + SoftMoE_LoRA
    memory_attention(backbone_feat, memory)
    memory.append(backbone_feat)

Phase 2: Sigmoid Scoring
  for each modality:
    score = sigmoid(ConfidenceHeadV2(backbone_feat))  → (B,) ∈ [0, 1]

Phase 3: UAMM
  modulated_feats = backbone_feats * scores  → 각 모달리티 독립 스케일링

Phase 4: AMF
  normalized = scores / sum(scores)  → 합=1 정규화
  final = sum(normalized[i] * output[i])
```

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | Submission |
|--------|----------|-----------|---------|------------|
| no-aug (beforeAug) | 93.10 | 35.93 | 64.51 | #15509 |
| basic-aug | 93.13 | 62.50 | 77.82 | #15541 |
| **hardaug (기본)** | 92.96 | **63.93** | **78.45** | #15561 |
| hardaug2 | 93.29 | 63.45 | 78.37 | #15589 |
| hardaug3 | 93.36 | 61.57 | 77.46 | #15616 |

### Night Augmentation 효과

- no-aug → basic-aug: **+26.57pp** (전체 test gain의 80%)
- basic-aug → best hardaug: **+1.43pp** (전체 gain의 4%)
- **Augmentation은 이 시점에서 이미 포화 상태**

## 발견된 문제점

### 1. Sigmoid Saturation

학습이 진행되면 모든 모달리티의 logit이 양수로 수렴 → `sigmoid(logit > 3) ≈ 1.0`

```
학습 초기: score = [0.6, 0.7, 0.8]  → 의미 있는 차이
학습 후기: score = [0.99, 0.98, 0.99] → 전부 ~1.0, 차이 소멸
```

### 2. AMF Uniform 수렴

모든 score ≈ 1.0 → `normalized = [1/3, 1/3, 1/3]` → 단순 평균과 동일

### 3. UAMM 무의미

모든 feature에 ~1.0 곱함 → modulation 효과가 사실상 없음

## 핵심 교훈

1. **독립 scoring은 상대 비교 불가**: 각 모달리티를 독립적으로 평가하면, 모든 모달리티가 "충분히 좋다"고 판단하여 차별화 실패
2. **Sigmoid는 saturation에 취약**: 학습이 길어질수록 logit이 커져 정보 소실
3. **NIGHT_AUG가 핵심**: aug 없이는 test 35.93%, basic-aug만으로 62.50% → aug가 야간 generalization의 80% 담당
4. **hardaug3(실측 정렬)이 hardaug2(넓은 범위)보다 나빴음**: test 분포에 맞추는 것이 항상 좋진 않음

## 다음 모델 (P9)로의 동기

- Sigmoid 독립 scoring → **모든 모달리티를 동시에 비교하는 cross-modal relative comparison** 필요
- AMF가 항상 1/3씩 → **softmax 기반으로 합=1 보장 + 상대적 가중치** 필요
- UAMM이 ~1.0 → **최선 모달리티=1.0, 나머지는 상대적 억제하는 max-norm** 필요
