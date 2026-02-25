# 모델 아키텍처 상세 (Model Architecture Details)

> 최종 업데이트: 2026-02-25

## 공통 기반: MemorySAM

### 핵심 아이디어

SAM2의 시간축 메모리 어텐션을 **모달리티 축**으로 전용:
1. 각 모달리티(RGB, LiDAR, Thermal)를 별도 "프레임"으로 인코딩
2. SAM2의 memory attention으로 모달리티 간 상호 참조
3. 모달리티별 가중치(UAMM/AMF)로 adaptive fusion

### SAM2 Backbone: Hiera-B+

- `embed_dim=112`, stages=(2,3,16,3) = 24 blocks, `dim_mul=2.0`
- Block별 차원:
  - Blocks 0-2: dim=112 (3개)
  - Blocks 3-5: dim=224 (3개)
  - Blocks 6-20: dim=448 (15개)
  - Block 21: dim=448→896 (전환)
  - Blocks 22-23: dim=896 (2개)
- Pretrained: `semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt`

### Soft-MoE LoRA Layer (공통)

파일: `semseg/models/sam2/sam2/sam_lola_utils.py` (line 521)

```python
class SoftMoE_LoRA_Layer:
    gate: Linear(dim, num_experts)         # routing network
    experts_a: ModuleList[Linear(dim, rank)]   # down-projection (LoRA A)
    experts_b: ModuleList[Linear(rank, dim)]   # up-projection (LoRA B)
```

- **Soft-MoE**: softmax gating → 모든 expert가 참여 (top-k 아님)
- **초기화**: gate.weight N(0, 0.01), gate.bias=0, experts_a=kaiming, experts_b=zeros
- **총 48개 layer**: 24 blocks × 2 (Q, V)
- `rank=4`, `num_experts=3` (모달리티 수와 동일)

### Forward 흐름 (공통)

```
Phase 1: 모달리티별 인코딩
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)        # Hiera-B+ + SoftMoE_LoRA
    memory_attention(backbone_feat, memory)     # cross-modal attention
    memory.append(backbone_feat)

Phase 2: 모달리티 가중치 계산
  cross_weights = Head(all_backbone_feats)      # 방법은 버전별 상이

Phase 3: UAMM (Unified Attention Modulation Memory)
  modulated_feats = backbone_feats * uamm_scores  # feature 조절

Phase 4: Tracking + AMF (Adaptive Modality Fusion)
  outputs = [track(modulated_feat) for feat in modulated_feats]
  final = sum(amf_weights[i] * outputs[i])     # 가중 평균
```

---

## P8: ConfidenceHeadV2 + Sigmoid UAMM

파일: `sam_lora_image_encoder_seg.py` line 1134, 클래스: `LoRA_Sam_P8`

### 아키텍처

```
backbone_feats → ConfidenceHeadV2(fusion_dim) → logits → sigmoid → scores
                                                         ↓
UAMM: scores (0~1, 각 모달리티 독립)
AMF:  normalized_scores = scores / sum(scores)
```

### ConfidenceHeadV2

- GAP(backbone_feat) → Linear → sigmoid
- 각 모달리티에 대해 **독립적**으로 0~1 점수 산출
- 모달리티 간 상대 비교 없음

### 한계점

1. **Sigmoid saturation**: logit > 3 → score ≈ 1.0, logit < -3 → score ≈ 0.0
   - 학습 진행 시 모든 모달리티의 logit이 양수로 → 전부 ~1.0
2. **AMF uniform**: 모든 score ≈ 1.0 → normalized = 1/3씩 uniform 분배
3. **UAMM 무의미**: 모든 feature에 ~1.0 곱함 → modulation 효과 없음

### 실험 결과 요약

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| no-aug (beforeAug) | 93.10 | 35.93 | 64.51 |
| basic-aug | 93.13 | 62.50 | 77.82 |
| hardaug (기본) | 92.96 | 63.93 | 78.45 |
| hardaug2 | 93.29 | 63.45 | 78.37 |
| hardaug3 | 93.36 | 61.57 | 77.46 |

---

## P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선)

파일: `sam_lora_image_encoder_seg.py` line 1355, 클래스: `LoRA_Sam_P9`

### P8에서의 변경 동기

P8의 sigmoid 독립 평가 → 모달리티 간 상대 비교 부재 → uniform AMF
→ **해결**: 모든 모달리티를 동시에 비교하는 cross-modal head

### 아키텍처

```
all_backbone_feats → CrossModalFusionHead → softmax → cross_weights (B, m)
                                                       ↓
UAMM: max_w = max(cross_weights)
       uamm_scores = cross_weights / max_w  → 최선 모달리티=1.0, 나머지 상대적
AMF:  amf_weights = cross_weights (softmax 출력 그대로)
```

### CrossModalFusionHead

```python
class CrossModalFusionHead:
    # GAP → compress → 모든 모달리티 concat → compare → softmax
    gap = AdaptiveAvgPool2d(1)
    compress = Linear(in_channels, in_channels // 4)  # 차원 축소
    compare = Linear(in_channels // 4 * num_modalities, num_modalities)  # 상대 비교
```

- 핵심: **모든 모달리티의 feature를 concat** 후 비교 → 상대적 품질 평가
- softmax 출력 → 합=1 보장, 상대적 가중치

### Max-Norm UAMM

```python
max_w = cross_weights.max(dim=1, keepdim=True)[0]
uamm_scores = cross_weights / (max_w + 1e-8)
# 최선 모달리티 = 1.0 (feature 보존), 나머지 < 1.0 (억제)
```

- P8의 sigmoid와 달리, 최선 모달리티의 feature는 **완전 보존**
- 나쁜 모달리티만 상대적으로 억제

### 한계점 (관찰됨)

1. **Cross-modal weight near-constant**: 특정 이미지에서 thermal≈1.0, lidar≈0.96, img≈0.74 패턴 반복
2. 단순 GAP만 사용 → 텍스처/노이즈 정보 반영 부족
3. 그러나 test generalization은 P8 대비 크게 향상 → 이 방식이 효과적

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.32 | 69.62 | **81.47** |

---

## P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨)

파일: `sam_lora_image_encoder_seg.py` line 1859, 클래스: `LoRA_Sam_P10`

### P9에서의 변경 동기

P9의 cross-modal weight가 near-constant → gating이 충분히 adaptive하지 않음
→ **시도**: quality-aware multi-pool + oracle supervision으로 gating 학습 강화

### 아키텍처 변경

```
all_backbone_feats → CrossModalFusionHeadV2 → softmax → cross_weights
                  ↘ ModalAuxHead(각 모달리티) → per-modal segmentation
                     ↓
                  oracle_weights = softmax(per_modal_iou)  # 학습 시 GT와 비교
                  KL(amf_weights || oracle_weights)        # gating 지도학습
```

### CrossModalFusionHeadV2

```python
class CrossModalFusionHeadV2:
    # Multi-pool: GAP + GMP + Channel Std
    gap = AdaptiveAvgPool2d(1)
    gmp = AdaptiveMaxPool2d(1)
    # Std = channel-wise std (텍스처/노이즈 indicator)
    compress_per_modal = ModuleList[Linear(in_ch * 3, in_ch // 4)]  # per-modality
    compare = Linear(in_ch // 4 * num_modalities, num_modalities)
```

- GAP (평균) + GMP (최대값) + Std (변동성) → 품질 정보 풍부
- Per-modality compress → 각 모달리티 독립 특징 추출

### ModalAuxHead

```python
class ModalAuxHead:
    # 각 모달리티별 경량 segmentation head
    conv1x1 → BN → ReLU → conv1x1 → num_classes
```

- 각 모달리티의 backbone feature로 독립 segmentation 수행
- GT와 비교하여 per-modal IoU 계산 → oracle weight 생성
- `LAMBDA_GATE: 0.5`

### 취소 이유

1. **Test 성능 하락**: M-score 79.27 (P9: 81.47, **-2.2**)
2. Test mIoU 65.30 (P9: 69.62, **-4.3**)
3. Oracle supervision이 주간(Val) 데이터에 과적합
4. Multi-pool의 Std feature가 야간에서 부정확한 quality estimation
5. Aux head 추가로 파라미터 증가 → overfitting 가속

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.23 | 65.30 | 79.27 |
| hardaug3 | 93.18 | 58.93 | 76.05 |

---

## P11: P10 + MI Routing Loss (취소됨)

파일: `sam_lora_image_encoder_seg.py` line 2130, 클래스: `LoRA_Sam_P11`

### P10에서의 변경 동기

MoE gate weights가 "uniform"으로 수렴하는 문제 (당시 spatial mean 기준)
→ **시도**: Mutual Information (MI) loss로 expert 분화 강제

### 아키텍처 변경

```
P10 구조 그대로 +
MI loss = H(gate|input) - H(gate_marginal)
LAMBDA_MI: 1.0

UAMM: softmax with temperature (τ=2.0) 로 변경 (max-norm 대신)
```

- Gate distribution을 gradient 유지한 채 수집 (`_grad_gate_collector`)
- Per-modal gate distribution → MI loss 계산
- UAMM: `softmax(logits / τ) * m` (temperature-scaled)

### 취소 이유

1. **Test 성능 더 악화**: M-score 77.09 (P10: 79.27, P9: 81.47)
2. Test mIoU 61.01 → P10보다도 나쁨
3. 지도교수 피드백: "loss를 넣어볼게 아니라 왜 gating이 안되는지 분석이 먼저"
4. **후속 진단에서 핵심 발견**: MoE gate는 이미 정상 작동!
   - "Uniform"은 spatial mean의 CLT artifact
   - Per-token entropy_ratio=0.55, max_weight=0.72
   - MI loss가 불필요하고, 오히려 이미 잘 작동하는 routing을 방해

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.17 | 61.01 | 77.09 |

---

## P12: Input-Conditioned Soft MoE LoRA (설계만 완료)

파일: `sam_lola_utils.py` 내 `SoftMoE_LoRA_Layer` 에 `cond_dim` 파라미터 추가

### 설계 아이디어

- 기존: gate(x) → softmax → weights (입력 x만으로 routing)
- P12: gate(x) + cond_proj(condition) → softmax → weights
- `condition` = 모달리티 타입 임베딩 또는 외부 신호
- `cond_proj`: Linear(cond_dim, num_experts), zero-init (초기에는 P9과 동일)

### 미실행 이유

- MoE gate가 이미 정상 작동한다는 진단 결과 → 구조 변경 우선순위 낮아짐
- P10/P11의 실패로 복잡도 추가에 대한 경계심

---

## 버전 비교 총괄

| 구분 | P8 | P9 | P10 | P11 |
| --- | --- | --- | --- | --- |
| Head | ConfidenceHeadV2 | CrossModalFusionHead | CrossModalFusionHeadV2 | CrossModalFusionHeadV2 |
| UAMM | sigmoid (0~1) | max-norm | max-norm | softmax+temperature |
| AMF | norm(sigmoid) | raw softmax | raw softmax | raw softmax |
| Aux Head | 없음 | 없음 | ModalAuxHead×3 | ModalAuxHead×3 |
| 추가 Loss | 없음 | 없음 | oracle KL (λ=0.5) | oracle KL + MI (λ=1.0) |
| 학습 반환 | (out, feat) | (out, feat) | (out, feat, aux, amf_w) | (out, feat, aux, amf_w, gates) |
| 최선 M-score | 78.45 | **81.47** | 79.27 | 77.09 |
| 교훈 | sigmoid saturation | 상대비교가 핵심 | oracle 과적합 | 진단이 먼저 |
