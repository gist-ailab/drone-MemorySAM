---
legacy_file: outputs_model_explain/P13_EnergyScore.md
moved: 2026-07-08
---

# P13: Energy Score Fusion + Expert Collapse Fix

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P13` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 2483) |
| 베이스 | P9 (P12의 condition 접근 포기) |
| 상태 | 완료 (P9 거의 근접) |
| 최선 M-score | 81.21 (hardaug4, ep17) |

## P9에서의 변경 동기

1. **CrossModalFusionHead의 near-constant 출력**: 학습 가능 파라미터 기반 → 항상 상수 수렴
2. **Expert collapse**: P9/P12에서 일부 expert 미사용 (collapse rate ~15-20%)

**핵심 전환**: 학습 파라미터 기반 scoring → **computed signal** 기반 scoring (Energy Score)

## 아키텍처 변경

### Energy Score Fusion (CrossModalFusionHead 교체)

```python
# P9: 학습된 MLP → 상수 수렴
cross_weights = CrossModalFusionHead(backbone_feats)  # 항상 [0.275, 0.355, 0.370]

# P13: Auxiliary prediction의 confidence를 직접 측정
aux_logits = ConfidenceAuxHead(backbone_feat)           # 경량 seg head
energy = -T * logsumexp(aux_logits / T, dim=1)          # (B, H, W)
conf = -energy.mean(dim=[1, 2])                         # (B,) spatial average
cross_weights = softmax(stack(confs) / T, dim=1)        # (B, m)
```

**핵심 특성**:
- **학습 가능 파라미터 없는 fusion weight**: computed signal이므로 상수 수렴 원천 불가
- **학습/추론 동일 메커니즘**: P10의 oracle-at-train / guess-at-test 불일치 없음
- Aux head는 별도 CE loss로 학습 (λ_aux)

### ConfidenceAuxHead

```python
class ConfidenceAuxHead(nn.Module):
    # 공유 1개 (모든 모달리티가 동일 head 사용)
    head = Sequential(
        Conv2d(in_ch, max(in_ch//4, 32), 1),
        BatchNorm2d, ReLU,
        Conv2d(mid_ch, num_classes, 1),
    )
```

### Expert Collapse Fix

```python
# P9: experts_b zero-init → 대칭 → collapse
# P13: experts_b를 kaiming*0.01로 재초기화 → 대칭 깨기
for expert_b in moe_q.experts_b:
    nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
    expert_b.weight.data *= 0.01  # 0.01 스케일 유지
```

## P9 대비 변경 요약

| 구분 | P9 | P13 |
|------|----|----|
| Scoring | CrossModalFusionHead (학습 MLP) | **Energy Score (computed signal)** |
| Aux Head | 없음 | **ConfidenceAuxHead (공유 1개)** |
| Expert Init | zeros (experts_b) | **kaiming*0.01 (대칭 깨기)** |
| 상수 수렴 가능성 | 있음 (학습 파라미터) | **원천 불가 (computed)** |

## 실험 결과

| Config | Epoch | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|-------|----------|-----------|---------|-------|------------|
| hardaug4 | 17 (night-val best) | 92.45 | 69.98 | **81.21** | **-0.26** | #15997 |
| hardaug4 | 39 (night-val best) | 92.86 | **50.48** | 71.67 | -9.80 | #16044 |

### Per-Class Test IoU (ep17, P9 대비)

| Class | P9 | P13 | Delta |
|-------|-----|-----|-------|
| Static | 81.30 | 79.80 | -1.50 |
| Dynamic | 21.86 | **27.41** | **+5.55** |
| Water | 94.61 | 94.27 | -0.34 |
| Sky | 76.54 | 75.12 | -1.42 |

## 해결하려는 문제를 해결했는가?

### 상수 수렴 방지? **성공 (부분적)**

- UAMM CV(변동계수)가 P12 대비 5-22x 증가 → 실제로 이미지별 다른 가중치
- 밤에 RGB↓ LiDAR↑ 적응을 **실제로 수행** (P9는 val/test 동일 상수)

| 모달리티 | P9 Val AMF | P9 Test AMF | P13 Val AMF | P13 Test AMF |
|----------|------------|-------------|-------------|--------------|
| img | 0.275 | 0.275 (**동일**) | 0.404 | **0.289 (↓28%)** |
| lidar | 0.355 | 0.355 (**동일**) | 0.429 | **0.517 (↑20%)** |
| thermal | 0.370 | 0.370 (**동일**) | 0.167 | 0.194 |

### Expert collapse 해결? **실패**

- Collapse rate 17.4% (P12: 16.0%와 동일 수준)
- kaiming*0.01 init은 resume 학습으로 무력화, 스케일도 미미

### Epoch39 Test Crash

- Night-val 87.71→89.53 개선되었지만, **Test 69.98→50.48 폭락 (-19.50pp)**
- **원인: CRM/ZERO Overfitting** — 학습 44%에 exact-zero RGB 패턴 → test에 없는 shortcut 학습
- Sky -51.76pp 붕괴 (crash의 67%), 80/200 프레임 Sky=0

## 핵심 교훈

1. **Energy Score "방향은 맞다"**: val→test 적응 실제 수행, Dynamic +5.55pp 최대 개선
2. **하지만 "confident but wrong" 문제**: aux head가 LiDAR를 항상 "가장 confident"로 판정 → Sky에서 LiDAR 맹신
3. **Val mIoU 하락이 M-score에 불리**: Energy의 adaptive weight가 P9의 안정적 상수보다 val에서 불리 (92.45 vs 93.32)
4. **CRM/ZERO Overfitting 발견**: epoch39 crash로 처음 발견. 이후 hardaug5(CRM/ZERO 제거) 도입의 계기
5. **P13이 P9에 가장 근접** (-0.26) — 이후 P14~P19 모두 P13보다 나쁨

## 다음 모델 (P14)로의 동기

- Aux head가 **공유 1개**라서 모달리티별 특화 불가 → **독립 3개**로 분리
- Sky에서 LiDAR 맹신 → 모달리티별 전용 decoder가 각 모달의 특성을 더 정확히 반영할 것이라는 기대
