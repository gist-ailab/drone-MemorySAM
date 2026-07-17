---
legacy_file: outputs_model_explain/P9_CrossModalFusionHead.md
moved: 2026-07-08
---

# P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P9` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 1355) |
| 베이스 | P8 |
| 상태 | **현재 최선 모델** |
| 최선 M-score | **81.98** (hardaug8_physaug ep131) |

## P8에서의 변경 동기

P8의 Sigmoid 독립 평가 → 모달리티 간 상대 비교 부재 → uniform AMF 수렴

**해결 방향**: 모든 모달리티를 **동시에 비교**하는 cross-modal head + softmax 기반 상대 가중치

## 아키텍처 변경

### CrossModalFusionHead (P8의 ConfidenceHeadV2 교체)

```python
class CrossModalFusionHead(nn.Module):
    # GAP: AdaptiveAvgPool2d(1)           — (B, C, H, W) → (B, C)
    # compress: Linear(C, 64) → ReLU      — 차원 축소 (shared across modalities)
    # compare: Linear(64*3, 64) → ReLU → Linear(64, 3)  — 상대 비교
    # 출력: softmax(logits / τ) → (B, 3), sum=1

    # Zero-init: compare[-1].weight=0, bias=0
    # → 학습 시작 시 softmax([0,0,0]) = [1/3, 1/3, 1/3]
```

**핵심 차이**: P8은 각 모달리티를 독립 평가(sigmoid) → P9은 모든 모달리티를 **concat 후 비교**(softmax)

### Max-Norm UAMM (P8의 raw sigmoid 교체)

```python
# P8: uamm_scores = sigmoid(logits)  → 전부 ~1.0
# P9:
max_w = cross_weights.max(dim=1, keepdim=True)[0]
uamm_scores = cross_weights / (max_w + 1e-8)
# → 최선 모달리티 = 1.0 (보존), 나머지 < 1.0 (상대적 억제)
```

### AMF

```python
# P8: scores / sum(scores) → sigmoid saturation으로 항상 1/3
# P9: amf_weights = cross_weights  → softmax 출력 그대로 사용
```

## P8 대비 변경 요약

| 구분 | P8 | P9 |
|------|----|----|
| Scoring | ConfidenceHeadV2 (독립 sigmoid) | **CrossModalFusionHead (상대 softmax)** |
| UAMM | raw sigmoid (~1.0) | **max-norm (best=1.0, 나머지 상대적)** |
| AMF | scores/sum (항상 1/3) | **softmax 출력 직접 사용** |
| 모달리티 비교 | 없음 (독립) | **concat 후 MLP로 상대 비교** |

## 실험 결과

| Config | Epoch | Val mIoU | Test mIoU | M-score | Submission |
|--------|-------|----------|-----------|---------|------------|
| **hardaug8_physaug** | **131** | **93.54** | **70.41** | **81.98** | **#16683** |
| hardaug4 | 47 | 93.32 | 69.62 | 81.47 | #15635 |
| hardaug8_physaug | 188 (과적합) | 93.49 | 68.20 | 80.84 | #16702 |
| hardaug8 | 94 | 93.36 | 68.13 | 80.75 | #16640 |
| hardaug8 | 83 | 93.21 | 67.94 | 80.57 | #16624 |
| hardaug4 Day-Trans | 47 | 93.30 | 64.50 | 78.90 | #16478 |
| hardaug4 Gamma TTA | 47 | 93.30 | 58.89 | 76.10 | #16412 |
| hardaug6 | 20 | 92.00 | 59.91 | 75.95 | #16340 |
| hardaug6 | 85 | 93.40 | 57.63 | 75.51 | #16339 |
| hardaug4 Night2 | 49 | 92.91 | 53.18 | 73.04 | #16482 |

### Per-Class Test IoU (best: hardaug8_physaug ep131)

| Class | IoU |
|-------|-----|
| Static | 76.64 |
| Dynamic | **33.50** |
| Water | 94.80 |
| Sky | 73.75 |

### hardaug8_physaug 학습 궤적 (Non-Monotonic)

- ep83(M=80.57) → ep94(80.75) → **ep131(M=81.98) ★ peak** → ep188(M=80.84, 과적합)
- **ep131이 sweet spot** — 추가 학습은 역효과
- Dynamic IoU: 21.86(hardaug4) → **33.50**(hardaug8 ep131) = **+11.64pp**

## 발견된 문제점 (관찰됨)

### 1. Scoring 함수의 상수 수렴

```
AMF weights (all images): img=0.239, lidar=0.371, thermal=0.390
Standard deviation: ≈ 0.0000 (소수점 4자리까지 동일)
```

345장(val+test) 전체에서 **완전한 학습된 상수**. 입력에 따른 adaptive fusion이 아님.

### 2. 상수 수렴의 구조적 원인

1. **GAP이 품질 정보 소멸**: AdaptiveAvgPool2d(1)로 spatial 전체 압축 → 품질 단서 사라짐
2. **SAM2 frozen encoder 정규화**: 어두운 RGB든 밝은 RGB든 encoder 출력 통계가 유사
3. **Zero-init + uniform이 안정적 고정점**: logits=[0,0,0]에서 탈출하는 gradient가 매우 약함

### 3. Val vs Test 갭

- Val mIoU ~93-94% (주간) vs Test mIoU ~70% (야간)
- 모든 후속 모델에서도 이 갭은 동일하게 존재

## 핵심 발견

1. **P9의 "학습된 상수"가 P8~P21 전체에서 최선**: 모든 adaptive fusion 시도가 이 상수보다 나쁨
2. **SAM2 memory attention이 implicit cross-modal adaptation 수행**: UAMM/AMF가 상수여도 memory 내부에서 모달리티 정보가 선택적으로 활용됨
3. **MoE gate는 정상 분화**: per-token entropy_ratio=0.55, max_weight=0.72 → "uniform"은 spatial mean의 CLT artifact
4. **장기 학습 + 다양한 aug = 추가 개선**: hardaug4(ep47, M=81.47) → hardaug8_physaug(ep131, M=81.98)

## 다음 모델들로의 동기

P9 이후 모든 모델은 "scoring 함수의 상수 수렴"을 해결하려 시도했으나, P21까지 전부 실패:

| 모델 | 시도 | M-score | vs P9 |
|------|------|---------|-------|
| P10 | Multi-pool + Oracle KL | 79.27 | -2.20 |
| P11 | MI routing loss | 77.09 | -4.38 |
| P12 | Input-conditioned MoE | 80.80 | -0.67 |
| P13 | Energy Score | 81.21 | -0.26 |
| P14 | 독립 Aux Decoder | 74.27 | -7.20 |
| P15 | Spatial Energy | 71.05 | -10.42 |
| P16 | Calibrated Entropy | 68.42 | -13.05 |
| P17 | Multi-scale Aux | 73.23 | -8.24 |
| P19 | Learned Spatial | 69.63 | -11.84 |
| P20 | MLP Gate + Rank 8 | 학습 대기 | - |
| P21 | DeBA-FP | ~81 (중간) | ~-1 |
