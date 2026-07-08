---
legacy_file: outputs_model_explain/P11_MI_Routing_Loss.md
moved: 2026-07-08
---

# P11: P10 + MI Routing Loss (취소됨)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P11` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 2130) |
| 베이스 | P10 |
| 상태 | **취소** (MoE gate 진단 결과 불필요 판명) |
| 최선 M-score | 77.09 (hardaug4) |

## P10에서의 변경 동기

MoE gate weights가 spatial mean 기준으로 "uniform"하게 보임 → expert 분화가 안 되고 있다는 가설.

**시도**: Mutual Information (MI) loss로 expert 분화를 gradient로 강제.

## 아키텍처 변경

### MI Routing Loss (신규)

```python
# Gate distribution 수집 (gradient 유지)
gate_distributions = collect_gate_weights()  # (batch, layers, experts)

# MI loss = H(gate|input) - H(gate_marginal)
# expert 사용 분포가 uniform하면 penalty → 특정 expert에 집중하도록 유도
MI_loss = compute_mi_loss(gate_distributions)

total_loss = seg_loss + λ_gate * kl_loss + λ_mi * MI_loss
# LAMBDA_MI = 1.0
```

### UAMM 변경

```python
# P9/P10: max-norm
uamm_scores = cross_weights / max(cross_weights)

# P11: temperature-scaled softmax
uamm_scores = softmax(logits / τ) * m   # τ=2.0, m=3
```

## P10 대비 변경 요약

| 구분 | P10 | P11 |
|------|-----|-----|
| Gate Loss | 없음 | **MI loss (LAMBDA_MI=1.0)** |
| UAMM | max-norm | **temperature softmax (τ=2.0)** |
| 나머지 | 동일 | 동일 (V2 head + KL + aux) |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug4 | 93.17 | 61.01 | 77.09 | **-4.38** | #15851 |

## 해결하려는 문제를 해결했는가?

### MoE gate "uniform" 해결? **문제 자체가 잘못된 진단**

P11 이후 `diagnose_moe_gate.py`로 정밀 분석한 결과:

| 측정 방식 | 결과 | 해석 |
|-----------|------|------|
| Spatial mean (기존) | uniform (~1/3씩) | CLT artifact — 수천 토큰 평균 → 중심극한정리 |
| **Per-token 분석** | entropy_ratio=0.55, max_weight=0.72 | **정상 분화!** |

- MoE gate는 이미 token 수준에서 잘 분화되어 있었음
- Spatial mean으로 보면 "uniform"이지만, 개별 토큰에서는 명확한 expert 선호 존재
- **MI loss가 불필요했고, 오히려 이미 잘 작동하는 routing을 방해**

### 성능 영향

- P9 대비 -4.38, P10 대비 -2.18 추가 하락
- MI loss가 gate를 인위적으로 분화 → 자연스러운 routing 패턴 파괴

## 핵심 교훈

1. **진단 없이 loss 추가하지 말 것**: "uniform처럼 보인다" → loss 추가가 아니라 **왜 그렇게 보이는지 분석이 먼저** (지도교수 피드백)
2. **측정 artifact 주의**: spatial mean의 CLT 효과를 uniform으로 오해
3. **loss 추가 ≠ 해결책**: 이미 정상인 메커니즘에 loss를 넣으면 오히려 악화
4. **MoE gate는 P9에서 이미 정상**: per-token entropy_ratio=0.55, 충분히 분화

## 취소 이유

- P10보다도 더 나쁜 성능 (M 77.09)
- 근본 원인 분석(MoE 진단) 결과 MI loss가 불필요함이 판명
- 이후 MoE gate 관련 실험은 gate 구조 개선(P20) 방향으로 전환
