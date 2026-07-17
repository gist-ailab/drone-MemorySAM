---
legacy_file: outputs_model_explain/P20_SharedMLPGate.md
moved: 2026-07-08
---

# P20: Shared MLP Gate + Higher Rank MoE (실험 J-A)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P20` |
| 파일 | `sam_lora_image_encoder_seg.py`, `sam_lola_utils.py` |
| 신규 모듈 | `SharedGateMLP`, `SoftMoE_LoRA_Layer_V2` |
| 베이스 | P9 (UAMM/AMF scoring 개선 포기, MoE 구조 개선) |
| 상태 | 구현 완료 → **gate 공유 제거 수정 완료** → 학습 대기 |
| Augmentation | hardaug8_physaug |

## 변경 동기 — 방향 전환

P12~P19: UAMM/AMF scoring 개선 → **8개 실험 전부 실패**.
→ Scoring이 아닌 **MoE LoRA 자체**를 강화하는 방향으로 전환.

P9 MoE의 한계:
1. **Gate**: `Linear(C→3)` 단일 선형 레이어 → 비선형 결정경계 학습 불가
2. **Rank**: 4로 매우 낮음 → expert 간 specialization 여지 부족
3. **Gate 독립성**: 48개 독립 gate → 파라미터 비효율 (당시 공유가 더 낫다고 판단)

## 아키텍처 변경 3가지

### 1. SharedGateMLP (2-layer MLP Gate)

```python
class SharedGateMLP(nn.Module):
    # Linear(C → C//4) → ReLU → Linear(C//4 → num_experts)
    # 비선형 결정경계 학습 가능
    # init: kaiming(first) + normal(0.01, last weight) + zeros(bias)
```

### 2. Gate 공유 전략

동일 `in_features` 차원의 블록들이 1개 MLP gate 공유:

| Stage | Blocks | dim | Q/V layers | 공유 MLP |
|-------|--------|-----|-----------|---------|
| 0 | 0-1 | 112 | 4 | 1개 |
| 1 | 2-4 | 224 | 6 | 1개 |
| 2 | 5-20 | 448 | 32 | 1개 |
| 3 | 21-23 | 896 | 6 | 1개 |
| **합계** | | | **48** | **4개** |

- P9: 48개 독립 Linear → P20: **4개 공유 MLP**
- 파라미터는 비슷 (~268K) 하지만 과적합 방지

### 3. Rank 상향: 4 → 8

- Expert capacity 2배 증가
- Expert 간 실질적 차이 발생 가능 → gate 분화에 대한 gradient 신호 강화

### SoftMoE_LoRA_Layer_V2

```python
class SoftMoE_LoRA_Layer_V2(nn.Module):
    # 초기 구현: 외부 shared gate 참조 (_shared_gate)
    # → 수정 후: 내부 독립 MLP gate 보유 (per-layer)

    # 수정 전:
    self._shared_gate = None  # set_shared_gate()로 외부 gate 연결
    gate_logits = self._shared_gate(x)

    # 수정 후 (현재):
    self.gate = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_experts),
    )
    gate_logits = self.gate(x)  # 내부 독립 gate
```

## Gate 공유 → 독립으로 수정된 이유

### 학습 결과 (ep47) 분석

| 지표 | P9 (독립 Linear) | P20 (공유 MLP) |
|------|-----------------|---------------|
| Val mIoU | 93.32 | **85.82** |
| MoE entropy_ratio (Q) | 0.52 | **0.93-0.95** |
| MoE entropy_ratio (V) | 0.64 | **0.93-0.95** |

- Gate를 공유하니 **entropy_ratio가 0.93-0.95로 uniform에 수렴**
- 특히 Stage 2-3의 **38개 레이어가 1개 gate 공유** → gate가 모든 레이어에 타협 → uniform compromise
- P9의 per-layer gate(entropy 0.52-0.64)보다 분화가 크게 후퇴

### 수정 후 구조 (현재)

| 항목 | 수정 전 (shared) | 수정 후 (per-layer) |
|------|-----------------|-------------------|
| Gate 수 | 4개 (dim별 1개) | **96개** (48 layers × Q/V) |
| Stage 2-3 Gate:Layer 비율 | 1:38 | **1:1** |
| Gate 파라미터 | ~268K | **~2.8M** |
| Gate 구조 | SharedGateMLP | **내부 2-layer MLP** (동일 구조) |

## P9 대비 변경 요약 (최종)

| 구분 | P9 | P20 (최종) |
|------|----|----|
| Gate | `Linear(C→3)` × 48 | **`MLP(C→C//4→3)` × 96** (per-layer 독립) |
| Gate 표현력 | 선형 | **비선형 (2-layer MLP)** |
| Rank | 4 | **8** |
| Expert 파라미터 | ~700K | **~1.4M** |
| Gate 파라미터 | ~268K | **~2.8M** |
| Fusion Head | CrossModalFusionHead | 동일 |
| UAMM/AMF | 동일 | 동일 |

## 실험 결과

ep47 기준 (gate 공유 버전): Val 85.82 — **gate 공유 문제로 인한 성능 하락**

수정 후(per-layer 독립 MLP gate) 재학습 필요.

## 기대 효과

- P9의 per-layer gate 독립성 + P20의 MLP gate 표현력 결합
- 비선형 결정경계 → 모달리티/공간/컨텐츠 기반 의미 있는 routing
- Rank 8 → expert 간 실질적 차이 → gate 분화에 대한 gradient 신호 강화

## 핵심 교훈

1. **Gate 공유는 해로움**: 38개 레이어가 1개 gate 공유 → uniform compromise. 레이어별 독립 gate가 필수
2. **MLP gate의 잠재력은 있으나 공유 전략이 병목**: MLP 자체의 비선형 표현력은 좋지만, 공유로 인해 발휘 불가
3. **P9의 48개 독립 Linear가 의외로 효과적**: 단순하지만 레이어별 독립성이 핵심
4. **Rank 상향 효과는 gate 수정 후 재평가 필요**
