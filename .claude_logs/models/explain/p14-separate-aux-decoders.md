---
legacy_file: outputs_model_explain/P14_SeparateAuxDecoders.md
moved: 2026-07-08
---

# P14: Per-Modality Separate Aux Decoders

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P14` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 2780) |
| 베이스 | P13 |
| 상태 | 완료 (심각한 하락) |
| 최선 M-score | 74.27 (hardaug5) |

## P13에서의 변경 동기

P13의 ConfidenceAuxHead는 **공유 1개** → 모든 모달리티가 동일 decoder 사용.
RGB 텍스처, LiDAR 점군, Thermal gradient는 특성이 완전히 다름 → 공유 head로는 모달리티별 정확한 confidence 추정 불가.

추가로 **CRM/ZERO 제거** (hardaug5): P13 ep39 crash의 원인이었던 CRM/ZERO overfitting 방지.

## 아키텍처 변경

### ModalAuxDecoder × 3 (공유 → 독립)

```python
# P13: ConfidenceAuxHead × 1 (공유)
aux_logits = shared_head(backbone_feat)  # 모든 모달리티 동일 head

# P14: ModalAuxDecoder × 3 (독립)
aux_logits[i] = aux_heads[i](backbone_feat[i])  # 각 모달리티 전용 head
```

```python
class ModalAuxDecoder(nn.Module):
    # P13의 1×1 conv → 3×3 conv로 변경 (텍스처/경계 패턴 특화)
    head = Sequential(
        Conv2d(in_ch, mid_ch, 3, padding=1),  # 3×3
        BatchNorm2d, ReLU,
        Conv2d(mid_ch, num_classes, 1),
    )
```

### Augmentation 변경: hardaug4 → hardaug5

| 파라미터 | hardaug4 | hardaug5 |
|----------|----------|----------|
| CRM_P | 0.35 | **0.0 (제거)** |
| ZERO_P | 0.09 | **0.0 (제거)** |
| NIGHT_SIM_P | 0.45 | **0.60** |
| BRIGHTNESS | [0.03, 0.45] | **[0.02, 0.20]** |

## P13 대비 변경 요약

| 구분 | P13 | P14 |
|------|------|-----|
| Aux Head | ConfidenceAuxHead × 1 (공유) | **ModalAuxDecoder × 3 (독립)** |
| Aux Conv | 1×1 | **3×3** (텍스처 포착) |
| CRM/ZERO | 있음 (hardaug4) | **제거 (hardaug5)** |
| 나머지 | Energy Score, max-norm UAMM | 동일 |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug5 | 93.18 | 55.36 | 74.27 | **-7.20** | #16062 |

### Per-Class Test IoU (P9/P13 대비)

| Class | P9 | P13 | P14 | P14 vs P9 |
|-------|-----|-----|-----|-----------|
| Static | 81.30 | 79.80 | 62.57 | **-18.73** |
| Dynamic | 21.86 | 27.41 | 22.87 | +1.01 |
| Water | 94.61 | 94.27 | 92.92 | -1.69 |
| Sky | 76.54 | 75.12 | **36.47** | **-40.07** |

## 해결하려는 문제를 해결했는가?

### 독립 Aux Decoder로 aux mask 품질 향상? **미미한 개선**

- 독립 head가 모달리티별 특성을 약간 더 포착하지만, 근본적으로 **frozen backbone feature 위의 경량 decoder**라는 한계는 동일
- GT 대비 여전히 매우 부정확 — 모달리티 간 비교 불가 수준

### CRM/ZERO 제거로 crash 방지? **overfitting은 방지했으나 Sky collapse 지속**

- P13 ep39처럼 극단적 crash는 발생하지 않음
- 하지만 **Sky 36.47%** (P9: 76.54%) → CRM/ZERO는 Sky 문제의 일부 원인이었으나 유일 원인은 아님

### 핵심 실패 원인

1. **LiDAR UAMM = 1.000 고정** (test 200장 전부, stdev=0.000)
   - aux head가 LiDAR를 항상 "가장 confident"로 판정
   - Sky 영역에서 LiDAR는 무의미한데도 최대 가중치
2. **RGB 억제**: test UAMM img=0.555 (val=0.752) → Sky 인식의 핵심인 RGB가 절반으로 감소
3. **Image-level scalar fusion의 근본 한계**: Sky/Water 영역별 최적 모달리티가 다르지만 이미지 전체에 동일 가중치

## 핵심 교훈

1. **Aux decoder 독립화만으로는 부족**: frozen feature의 한계가 decoder 구조보다 본질적
2. **Energy Score "confident but wrong" 지속**: LiDAR가 Sky에서 단일 클래스(Water)에 높은 logit → 높은 energy → "confident" → 실제로는 틀림
3. **Image-level scalar는 공간 이질성 대응 불가**: 이 발견이 P15(spatial) 설계의 동기
4. **Augmentation 변경과 아키텍처 변경을 동시에 하면 원인 분리 불가**: hardaug5 + 독립 decoder 두 가지를 동시에 바꿈

## 다음 모델 (P15)로의 동기

- Image-level scalar `(B, m)` → **Spatial-wise `(B, m, H, W)`** 가중치로 전환
- Sky 영역에서 RGB 유지, Water 영역에서 LiDAR 활용 등 **위치별** 모달리티 선택
