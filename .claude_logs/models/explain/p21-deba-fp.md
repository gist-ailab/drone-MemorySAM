---
legacy_file: outputs_model_explain/P21_DeBA_FP.md
moved: 2026-07-08
---

# P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P21` |
| 파일 | `sam_lora_image_encoder_seg.py` |
| 신규 모듈 | `DeBAFP` (`sam_lola_utils.py`) |
| 베이스 | P9 (MoE LoRA 동일, DeBA-FP만 추가) |
| 상태 | 학습 중 (ep51 중간 분석 완료) |
| Ref | CVPR 2026 — "Rethinking Deformable Convolution as an Adapter with Cross-layer Weight Sharing" |
| Augmentation | hardaug8_physaug |

## 변경 동기

P9의 FPN feature(fpn[0])는 SAM2 encoder에서 나온 그대로 → GAP → CrossModalFusionHead.
Spatial refinement 없이 global average만으로 모달리티 중요도 산출.

**Day→Night domain gap에서 경계/형태 같은 구조적 정보는 domain-invariant인데, 이를 명시적으로 포착하는 메커니즘 부재.**

DeBA (CVPR 2026)는 deformable convolution으로 domain-invariant structural information을 포착.
특히 **LaRS(수면 환경) 벤치마크에서 SOTA** → MULTIAQUA와 직접 관련.

## 아키텍처 변경

### DeBA-FP: FPN Feature Refinement

```
P9:  fpn[0] ──────────────────→ CrossModalFusionHead → UAMM/AMF
P21: fpn[0] → DeBA-FP(shared) → CrossModalFusionHead → UAMM/AMF
```

```python
class DeBAFP(nn.Module):
    """
    feat' = feat + α_m × W_u(GELU(LN(DCM(W_d(feat)))))

    Shared across modalities: W_d, DCM, LN, W_u
    Per-modality: α (init=0 → identity at start)
    """
    W_d: Conv2d(256→64, 1×1)              # bottleneck down projection
    offset_mask_conv: Conv2d(64→27, 3×3)  # DCNv2 offset+mask prediction
    dcm_weight: Parameter(64, 64, 3, 3)   # deformable conv weight
    norm: LayerNorm(64)                     # shared normalization
    W_u: Conv2d(64→256, 1×1)              # up projection
    alpha: ParameterList([zeros(1)] × 3)   # per-modality scale (init=0)
```

### 핵심 설계 결정

1. **Cross-modal weight sharing**: W_d, DCM, LN, W_u 모두 3개 모달리티가 공유
   - 2,952 학습 샘플로 최대한 regularization
   - α만 per-modality → 각 모달리티가 다른 강도로 adaptation
2. **α=0 init**: 학습 시작 시 DeBA-FP = identity → P9과 동일 출발점
3. **Offset zero-init**: DCM offset이 0부터 시작 → regular conv → 점진적으로 deformable
4. **fpn[0] only**: P9가 fpn[0]만 사용하므로 다른 FPN 레벨은 불필요

### 원본 DeBA와의 차이

| 항목 | 원본 DeBA | P21 |
|------|----------|-----|
| Backbone | DINOv2 ViT | SAM2 Hiera B+ |
| DeBA-BB | ViT 블록 사이 삽입 | **미적용** |
| DeBA-FP | FPN 4-level | **fpn[0] only** |
| Cross-layer sharing | 레이어 간 | **모달리티 간** |
| DCN version | DCNv4 | **DCNv2** (torchvision) |

### 파라미터 추가량

| 구성 | 파라미터 |
|------|---------|
| W_d: Conv2d(256→64, 1×1) | 16,448 |
| offset_mask_conv: Conv2d(64→27, 3×3) | 15,579 |
| dcm_weight: (64, 64, 3, 3) | 36,864 |
| LayerNorm(64) | 128 |
| W_u: Conv2d(64→256, 1×1) | 16,640 |
| α × 3 | 3 |
| **합계** | **~85K** |

P9 LoRA ~700K 대비 **12% 증가**. 전체 trainable ~785K.

## P9 대비 변경 요약

| 구분 | P9 | P21 |
|------|----|----|
| FPN 처리 | raw fpn[0] 직접 사용 | **DeBA-FP로 refine 후 사용** |
| Deformable Conv | 없음 | **DCNv2 bottleneck adapter** |
| 추가 파라미터 | 0 | **~85K** |
| MoE LoRA | 동일 | 동일 |
| Fusion Head | CrossModalFusionHead | 동일 |
| UAMM/AMF | 동일 | 동일 |

## 실험 결과 (ep51, 학습 중간)

| 지표 | P9 (ep131) | P21 (ep51) | Delta |
|------|-----------|-----------|-------|
| Val mIoU | 93.54 | 93.94 | +0.40 |
| **분석 기반 추정 Test mIoU** | 70.41 | ~67-68 (추정) | ~-3 |

### Per-Class Val IoU (P9 대비)

| Class | P9 ep131 | P21 ep51 | Delta |
|-------|----------|----------|-------|
| Static | 87.78 | 87.88 | +0.10 |
| Dynamic | **53.41** | **45.06** | **-8.35** |
| Water | 96.93 | 97.64 | +0.71 |
| Sky | 95.27 | 94.79 | -0.48 |

### detailed_log.json 분석

#### MoE Routing

| 지표 | P9 | P21 | 판정 |
|------|-----|-----|------|
| entropy_ratio (Q) | 0.48 | 0.47 | 정상 (P9 수준) |
| entropy_ratio (V) | 0.60 | 0.61 | 정상 |
| max_weight (Q) | 0.73 | 0.73 | 동일 |
| top2_gap (Q) | 0.47 | 0.48 | 동일 |

→ MoE routing은 P9과 거의 동일하게 작동. DeBA-FP가 MoE에 간섭하지 않음.

#### UAMM/AMF

| 모달리티 | P9 AMF | P21 AMF | Delta |
|----------|--------|---------|-------|
| img | 0.239 | **0.296** | +0.057 (RGB 가중치 상승) |
| lidar | 0.371 | 0.345 | -0.026 |
| thermal | 0.390 | 0.359 | -0.031 |

- **더 uniform한 분포**: P9의 beneficial asymmetry(thermal 우세) → P21은 거의 1/3씩
- std ≈ 0.0000 (여전히 완전 상수)

#### Prediction Confidence

- P21의 prediction confidence가 P9보다 약간 낮음
- P20(ep47)보다는 높음 → 중간 수준

## 해결하려는 문제를 해결했는가?

### FPN feature의 structural refinement? **판단 보류 (학습 중)**

- ep51은 P9 ep131 대비 학습량 부족 (ep131까지 학습 시 개선 가능)
- Dynamic IoU -8.35pp가 우려 — DeBA-FP가 Dynamic class에 부정적 영향?

### Scoring 함수 상수 수렴 해결? **해결 대상 아님 (의도적)**

- P21은 scoring 문제를 건드리지 않음
- DeBA-FP가 feature를 바꿔도, CrossModalFusionHead의 GAP→softmax 경로가 동일하므로 상수 수렴은 지속
- **이것이 P21 한계의 근본**: DeBA-FP가 아무리 좋은 feature를 만들어도, scoring 병목에서 차이가 소실

### UAMM/AMF 상수 패턴 변화

- P9: img=0.239, lidar=0.371, thermal=0.390 (thermal 우세)
- P21: img=0.296, lidar=0.345, thermal=0.359 (거의 균등)
- DeBA-FP가 feature 분포를 변경 → scoring이 새로운 (더 uniform한) 고정점으로 수렴
- **P9이 우연히 찾은 좋은 비율(thermal 약간 우세)을 잃음**

## 핵심 교훈

1. **DeBA-FP 자체는 무해**: MoE routing에 간섭 없이 feature를 refine. α=0 init 덕분에 안전한 출발
2. **하지만 scoring 병목 통과 후 차이 소실**: DeBA-FP → refined feature → GAP(품질 차이 사라짐) → softmax(상수 수렴) → scoring의 구조적 한계가 DeBA-FP의 효과를 상쇄
3. **AMF가 더 uniform해지면 나쁨**: P9의 beneficial asymmetry가 DeBA-FP로 인해 파괴
4. **Dynamic class에 부정적**: 이유 불명, 추가 학습 필요

## 다음 방향

- ep131+까지 장기 학습하여 P9 hardaug8_physaug와 공정 비교 필요
- P22: Multi-Scale DeBA-FP (fpn[0,1,2] 전부, Phase 1 적용) — feature refinement의 범위 확대
- **근본 문제**: DeBA-FP든 MoE 강화든, scoring 함수(CrossModalFusionHead)의 구조적 한계(GAP + frozen encoder normalization + zero-init 고정점)를 해결하지 않는 한 UAMM/AMF는 상수로 수렴
