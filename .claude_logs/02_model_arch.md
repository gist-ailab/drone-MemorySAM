# 모델 아키텍처 상세 (Model Architecture Details)

> 최종 업데이트: 2026-02-26

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

## P12: Input-Conditioned Soft MoE LoRA

파일: `sam_lora_image_encoder_seg.py` line 1585, 클래스: `LoRA_Sam_P12`

### P9에서의 변경 동기

MoE gate 진단 결과 정상이었으나, 모달리티별로 다른 routing 패턴이 필요하다는 가설
→ RGB 채널 통계(mean+std)를 gate에 condition으로 주입

### 아키텍처 변경

```
gate(x) + cond_proj(condition) → softmax → weights
condition = RGB channel mean+std (cond_dim=6), lidar/thermal은 cond=None
cond_proj: Linear(cond_dim, num_experts), zero-init
```

### 실험 결과

- M-score 80.80 (P9: 81.47, **-0.67**)
- Dynamic +4.02pp 개선, Sky -6.81pp 하락
- Expert collapse P9보다 심화 (15% → 20%)
- Test LiDAR routing 48/48 블록 완전 고정

---

## P13: Energy Score Fusion + Expert Collapse Fix

파일: `sam_lora_image_encoder_seg.py` line 2483, 클래스: `LoRA_Sam_P13`

### P9에서의 변경 동기

1. CrossModalFusionHead의 near-constant 출력 문제 (ISSUE-003) → 학습 가능 파라미터 없는 fusion weight
2. SoftMoE_LoRA_Layer의 expert collapse (ISSUE-002) → 비영 초기화로 대칭 깨기

### 아키텍처

```
Phase 2: Aux Prediction + Energy Confidence (P9 Phase 2 대체)
  all_backbone_feats → ConfidenceAuxHead(공유) → aux_logits_list
  aux_logits_list → compute_energy_confidence(T=1.0) → cross_weights (B, m)

나머지 Phase (1, 3, 4)는 P9과 동일
```

### ConfidenceAuxHead

```python
class ConfidenceAuxHead(nn.Module):
    # 공유 1개 (모든 모달리티가 동일 head 사용)
    head = Sequential(
        Conv2d(in_ch, in_ch//4, 1),  # mid_channels = max(in_ch//4, 32)
        BatchNorm2d, ReLU,
        Conv2d(mid_ch, num_classes, 1),
    )
    # 출력: raw logits (B, C, H, W)
```

### compute_energy_confidence

```python
def compute_energy_confidence(aux_logits_list, temperature=1.0):
    for z in aux_logits_list:
        energy = -T * logsumexp(z / T, dim=1)  # (B, H, W)
        conf = -energy.mean(dim=[1, 2])          # (B,) spatial average
    weights = softmax(stack(confs) / T, dim=1)   # (B, m)
    return weights
```

핵심 특징:
- **학습 가능 파라미터 없음** — computed signal이므로 상수 수렴 불가
- **학습/추론 동일 메커니즘** — P10의 oracle-at-train / guess-at-test 불일치 없음
- aux head는 학습됨 (seg loss + λ_aux * aux_CE)

### Expert Collapse Fix

```python
# P13 __init__에서 experts_b 재초기화
for expert_b in moe_q.experts_b:
    nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
    expert_b.weight.data *= 0.01
```

### 실험 결과 및 설계 목표 달성 여부

| 설계 목표 | 판정 | 결과 |
| --- | --- | --- |
| Expert collapse 해결 | **실패** | collapse rate 17.4% (P12: 16.0%와 동일) |
| Energy Score fusion | **부분 성공** | UAMM CV 5-22x 증가, Dynamic +5.55pp |

- M-score 81.21 (P9: 81.47, **-0.26**)
- Val mIoU 92.45 (-0.87), Test mIoU 69.98 (+0.36)
- Night-val checkpoint 선택으로 test 개선 but val 희생

### 한계점 (관찰됨)

1. **Expert collapse 미해결**: kaiming*0.01 init은 resume 학습으로 무력화, 스케일도 미미
2. **Test LiDAR UAMM = 1.0 고정**: aux head가 LiDAR를 항상 "가장 confident"로 판정 (실제 LiDAR 품질은 가장 낮음)
3. **Val mIoU 하락**: Energy Score의 adaptive weight가 P9의 안정적 상수 비율보다 val에서 불리
4. **17 epochs 학습**: P9(47 epochs) 대비 짧지만, P9도 epoch 17(93.57) → 46(94.18)은 +0.61pp만 개선

---

## P14: Per-Modality Separate Aux Decoders

파일: `sam_lora_image_encoder_seg.py` line 2780, 클래스: `LoRA_Sam_P14`

### P13에서의 변경 동기

P13의 ConfidenceAuxHead는 **공유 1개** → 모든 모달리티가 동일 decoder를 공유.
RGB 텍스처, LiDAR 점군, Thermal gradient는 특성이 완전히 다름 → 공유 head로는 각 모달리티에 특화된 예측 불가.
시각화에서 aux mask 품질이 모두 GT와 큰 괴리 확인.

### 아키텍처 변경

```
P13: ConfidenceAuxHead×1 (공유) → 모든 모달리티 동일 head
P14: ModalAuxDecoder×3 (독립) → 모달리티별 전용 head
     · 첫 conv를 3×3으로 변경 → 텍스처/경계 패턴 특화
     · 각 모달리티가 고유 파라미터 → inter-modality gradient interference 제거
```

나머지(Energy Score fusion, UAMM max-norm, AMF, MoE init)는 P13과 동일.

### 상태

- **구현 완료**, 학습 대기 (hardaug5 config 준비됨)
- hardaug5: CRM/ZERO 완전 제거 + test셋 실측 밝기 분포 정렬

---

## P15: Calibrated Spatial Entropy Fusion (설계 단계)

### 변경 동기 — P12~P14 실패 분석에서의 교훈

**1. UAMM/AMF 개념은 유효하다**

| 모델 | Fusion | Val mIoU | 비고 |
| --- | --- | --- | --- |
| Baseline (LoRA_Sam) | 단순 평균 (1/3) | 92.86 | AMF 없음 |
| **P9** | UAMM + AMF (학습된 가중치) | 93.32 | **Baseline 대비 개선** |

Baseline(단순 평균) < P9(UAMM/AMF) → modality fusion 개념 자체의 가치 확인.

**2. Energy Score 방향은 맞지만 정확도가 부족**

P13의 Energy Score fusion은 **낮/밤 적응을 실제로 수행**:

| 모달리티 | P9 Val AMF | P9 Test AMF | P13 Val AMF | P13 Test AMF |
| --- | --- | --- | --- | --- |
| img | 0.275 | 0.275 (**동일**) | 0.404 | **0.289 (↓28%)** |
| lidar | 0.355 | 0.355 (**동일**) | 0.429 | **0.517 (↑20%)** |
| thermal | 0.370 | 0.370 (**동일**) | 0.167 | 0.194 |

P9는 345장 전체에서 소수점 4자리까지 동일한 **학습된 상수** (std ≈ 0.0000).
P13은 밤에 RGB↓ LiDAR↑ 적응 → **방향은 맞지만** LiDAR Sky 맹신으로 실패.

**3. 실패의 직접 원인 3가지**

1. **Energy Score = confidence, not correctness** → "confident but wrong" (ISSUE-008)
2. **Gradient 오염**: `.detach()` 없음 → main loss가 aux head 왜곡
3. **Image-level scalar**: 위치별 모달리티 차이 무시

P15는 이 3가지를 동시에 수정.

### P15 핵심 변경 4가지

#### Fix 1: Gradient 격리 — `.detach()`

```python
# P13/P14 (현재 — gradient 오염)
cross_weights = compute_energy_confidence(aux_logits_list, ...)

# P15 (수정 — gradient 차단)
cross_weights = compute_spatial_entropy_confidence(
    [z.detach() for z in aux_logits_list], ...
)
```

aux head는 **자기 자신의 CE loss만으로** 학습 → 정직한 confidence 출력.
Main loss gradient가 energy→aux→LoRA로 역전파되는 경로 차단.

#### Fix 2: Energy Score → Calibrated Entropy 교체

Energy Score 문제: `E(x) = -T * logsumexp(z/T)` → logit magnitude 기반.
LiDAR가 4클래스 중 하나에 높은 logit → 높은 energy → "confident" → **하지만 틀림** (Sky에서).

Entropy 기반 대안: **예측 분포의 불확실성**을 직접 측정.

```python
# P15: Calibrated Spatial Entropy Confidence
def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0, num_classes=4):
    """
    Energy Score 대신 calibrated entropy로 per-pixel confidence 계산.

    핵심 차이:
    - Energy: logit magnitude → "자신있게 틀리면" 높은 점수 (dangerous)
    - Entropy: 확률 분포 균등도 → 4클래스에 골고루 분산 = 낮은 confidence (safe)

    LiDAR가 Sky에서 Water로 확신있게 오예측 → Energy 높음 (나쁨)
    LiDAR가 Sky에서 불확실 → Entropy 높음 → confidence 낮음 (좋음)
    """
    conf_maps = []
    for z in aux_logits_list:  # z: (B, C, H, W), C=num_classes
        # Temperature scaling for calibration
        probs = F.softmax(z / temperature, dim=1)               # (B, C, H, W)
        log_probs = F.log_softmax(z / temperature, dim=1)       # (B, C, H, W)
        entropy = -(probs * log_probs).sum(dim=1)               # (B, H, W)
        # Normalize: 0 (완전 확신) ~ 1 (완전 균등)
        max_entropy = math.log(num_classes)
        confidence = 1.0 - entropy / max_entropy                # (B, H, W)
        conf_maps.append(confidence)

    stacked = torch.stack(conf_maps, dim=1)                     # (B, m, H, W)
    weights = F.softmax(stacked / temperature, dim=1)           # (B, m, H, W)
    return weights
```

Entropy의 장점:
- **"자신있게 틀리는" 케이스 감지**: LiDAR가 Sky에서 단일 클래스(Water)에 높은 확률을 주면 aux head가 정확해야만 높은 confidence → aux head가 부정확하면 자연스럽게 분산된 예측 → 높은 entropy → 낮은 confidence
- **Calibration 가능**: temperature T를 val에서 최적화하여 confidence를 보정

#### Fix 3: Spatial-wise (공간별 가중치)

기존 `(B, m)` 스칼라 → `(B, m, H, W)` spatial map:

```python
# UAMM: vision_feats 각 level에 spatial weight 적용
spatial_score = uamm_scores[:, frame_idx]                 # (B, H, W)
for level, feat in enumerate(vision_feats):
    h, w = feat_sizes[level]
    score_resized = F.interpolate(
        spatial_score.unsqueeze(1), size=(h, w), mode='bilinear'
    )  # (B, 1, h, w)
    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (h*w, B, 1)
    modulated_feat = feat * score_flat

# AMF: output fusion에 spatial weight 적용
w_i = F.interpolate(
    amf_weights[:, i:i+1], size=output[0].shape[2:], mode='bilinear'
)  # (B, 1, H_out, W_out)
m_output += output[i] * w_i
```

#### Fix 4: Aux Warmup Schedule

Aux head가 충분히 학습된 후에 UAMM/AMF 활성화:

```python
# Config
TRAIN:
  AUX_WARMUP_EPOCHS: 10    # 초기 N epoch는 aux CE만 학습
  LAMBDA_AUX: 0.3

# Forward에서
if current_epoch < aux_warmup_epochs:
    # Uniform weights (P9의 near-constant와 유사)
    cross_weights = torch.ones(B, m, H, W) / m
else:
    # Calibrated entropy weights
    cross_weights = compute_spatial_entropy_confidence(
        [z.detach() for z in aux_logits_list], ...
    )
```

첫 N epoch 동안:
- Aux head: CE loss로 학습 → 기본적인 segmentation 능력 확보
- UAMM/AMF: uniform(1/m) → P9처럼 안정적 학습
- Main decoder: 정상 학습

N epoch 이후:
- Aux head의 entropy가 UAMM/AMF에 반영 시작
- 점진적 전환 (abrupt하지 않도록 linear ramp 고려)

### 전체 Forward 흐름 (P15)

```
Phase 1: 모달리티별 인코딩 (P14 동일)
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)  # Hiera-B+ + SoftMoE_LoRA

Phase 2: Spatial Entropy Confidence
  aux_logits[i] = aux_heads[i](backbone_feat[i])        # 독립 aux decoder × 3
  conf_maps = entropy_confidence([z.detach() for z])     # (B, m, H, W)

Phase 3: Spatial UAMM + Tracking
  for each modality:
    spatial_uamm = conf_maps[:, i, :, :]                 # (B, H, W)
    modulated_vision_feats = vision_feats * spatial_uamm  # level별 interpolate
    output[i] = track_step(modulated_vision_feats, memory)

Phase 4: Spatial AMF
  amf_weights = conf_maps                                # (B, m, H, W)
  final = sum(output[i] * interpolate(amf_weights[:, i]))
```

### P15 vs 이전 버전 차이 요약

| 구분 | P13 | P14 | **P15** |
| --- | --- | --- | --- |
| Confidence 방식 | Energy Score (logit) | Energy Score (logit) | **Calibrated Entropy** |
| Gradient 격리 | 없음 (오염) | 없음 (오염) | **`.detach()` 적용** |
| Weight 형태 | `(B, m)` 스칼라 | `(B, m)` 스칼라 | **`(B, m, H, W)` spatial** |
| Aux Decoder | 공유 1개 | 독립 3개 | 독립 3개 (P14 유지) |
| Warmup | 없음 | 없음 | **AUX_WARMUP_EPOCHS** |
| UAMM | max-norm 스칼라 | max-norm 스칼라 | **spatial max-norm** |
| AMF | energy softmax 스칼라 | energy softmax 스칼라 | **spatial entropy softmax** |

### 구현 시 주의사항

1. **해상도 정합**: aux head 출력 `(H_feat, W_feat)`와 vision_feats/output의 해상도가 다름 → `F.interpolate` 필수
2. **vision_feats 형상**: SAM2 Hiera는 `(num_tokens, B, C)` 형태의 flattened feature 사용 → reshape/flatten 처리 필요
3. **feat_sizes**: `_prepare_backbone_features()`에서 반환하는 각 level의 (h, w) 사용
4. **backward compatibility**: train 시 `(output, m_feat, aux_logits_list)` 반환 형식 유지
5. **Temperature 최적화**: `temperature` 파라미터를 config에 노출 (기본 1.0, val에서 grid search 가능)
6. **Warmup→Active 전환**: abrupt 전환은 학습 불안정 유발 가능 → linear ramp (N~N+5 epoch) 고려

---

## 버전 비교 총괄

| 구분 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | ConfidenceHeadV2 | CrossModalFusionHead | CrossModalFusionHeadV2 | CrossModalFusionHeadV2 | CrossModalFusionHead | ConfidenceAuxHead + Energy Score | ModalAuxDecoder×3 + Energy Score | ModalAuxDecoder×3 + **Calibrated Entropy** |
| UAMM | sigmoid (0~1) | max-norm | max-norm | softmax+temperature | max-norm | max-norm (energy) | max-norm (energy) | **spatial max-norm** `(B,m,H,W)` |
| AMF | norm(sigmoid) | raw softmax | raw softmax | raw softmax | raw softmax | energy softmax | energy softmax | **spatial entropy softmax** |
| Aux Head | 없음 | 없음 | ModalAuxHead×3 | ModalAuxHead×3 | 없음 | ConfidenceAuxHead×1 (공유) | **ModalAuxDecoder×3** | ModalAuxDecoder×3 |
| 추가 Loss | 없음 | 없음 | oracle KL (λ=0.5) | oracle KL + MI (λ=1.0) | 없음 | aux CE (λ=0.3) | aux CE (λ=0.3) | aux CE (λ=0.3) |
| Gradient 격리 | N/A | N/A | N/A | N/A | N/A | 없음 | 없음 | **`.detach()` 적용** |
| Warmup | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | **AUX_WARMUP (10ep)** |
| MoE init | zero | zero | zero | zero | zero | kaiming*0.01 | kaiming*0.01 | kaiming*0.01 |
| 학습 반환 | (out, feat) | (out, feat) | (out, feat, aux, amf_w) | (out, feat, aux, amf_w, gates) | (out, feat) | (out, feat, aux_list) | (out, feat, aux_list) | (out, feat, aux_list) |
| 최선 M-score | 78.45 | **81.47** | 79.27 | 77.09 | 80.80 | 81.21 | 74.27 | 설계 단계 |
| 교훈 | sigmoid saturation | 상대비교가 핵심 | oracle 과적합 | 진단이 먼저 | cond 효과 미미 | 방향 유효, 정확도 부족 | aux 독립화만 불충분 | **정확도+spatial+격리** |
