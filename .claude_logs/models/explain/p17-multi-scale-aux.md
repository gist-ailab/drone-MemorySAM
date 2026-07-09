---
legacy_file: outputs_model_explain/P17_MultiScaleAux.md
moved: 2026-07-08
---

# P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P17` |
| 파일 | `sam_lora_image_encoder_seg.py` |
| 베이스 | P16 |
| 상태 | 완료 (P16 대비 개선, P9 대비 -8.24) |
| 최선 M-score | 73.23 (hardaug5, night ep35) |

## P16에서의 변경 동기

P16의 aux decoder는 `backbone_fpn[0]` (32ch, 256×256) **하나만** 사용.
SAM2 Hiera B+는 3개 FPN 레벨을 이미 계산하지만 나머지 2개는 미활용:
- `fpn[1]`: 64ch, 128×128
- `fpn[2]`: 256ch, 64×64

**핵심**: 32ch → 352ch(32+64+256) = **11배 정보량 증가, 추가 backbone 연산 0**

## 아키텍처 변경

### MultiScaleModalAuxDecoder (단일 FPN → 3-level FPN)

```python
class MultiScaleModalAuxDecoder(nn.Module):
    # 각 FPN 레벨을 proj_dim(32)으로 project
    proj_layers = [Conv2d(ch, 32, 1) + BN + ReLU for ch in (32, 64, 256)]

    # 모든 레벨을 fpn[0] 해상도로 upsample → concat → decode
    # Concat(32×3=96) → Conv2d(96, 48, 3×3) → BN → ReLU → Conv2d(48, 4, 1×1)

    def forward(self, fpn_feats):  # [fpn0, fpn1, fpn2]
        projected = [proj(feat) for proj, feat in zip(self.proj_layers, fpn_feats)]
        # 해상도 맞춤 → concat → decode
```

**파라미터**: ~53K/modality × 3 = ~159K total (기존 ModalAuxDecoder ~290/modality 대비 대폭 증가)

## P16 대비 변경 요약

| 구분 | P16 | P17 |
|------|------|-----|
| Aux Decoder | ModalAuxDecoder (fpn[0] only, 32ch) | **MultiScaleModalAuxDecoder (fpn[0,1,2], 352ch)** |
| Aux 파라미터 | ~290/modality | **~53K/modality** |
| Confidence | Calibrated Entropy (동일) | 동일 |
| `.detach()` | 적용 | 동일 |
| Warmup | 10ep+5ep ramp | 동일 |
| Spatial | (B, m, H, W) | 동일 |

## 실험 결과

| Config | Epoch | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|-------|----------|-----------|---------|-------|------------|
| hardaug5 | night ep35 | 92.60 | 53.86 | **73.23** | -8.24 | #16107 |
| hardaug5 | ep28 (day-val) | 92.99 | 52.69 | 72.84 | -8.63 | #16108 |

### Per-Class Test IoU (P9/P16 대비)

| Class | P9 | P16 | P17 | P17 vs P16 |
|-------|-----|-----|-----|------------|
| Static | 81.30 | 58.19 | 61.46 | +3.27 |
| Dynamic | 21.86 | 20.76 | 19.44 | -1.32 |
| Water | 94.61 | 92.24 | 93.54 | +1.30 |
| Sky | 76.54 | 3.17 | **33.35** | **+30.18** |

### Sky 부분 회복

- Sky=0 프레임: P16 157/200 → P17 **62/200** (60% 감소)
- Sky +30.18pp 개선은 fpn[2](256ch) semantic context가 sky/static 구분에 기여

## 해결하려는 문제를 해결했는가?

### Multi-scale FPN으로 aux mask 품질 향상? **부분적 성공**

- Sky: 3.17% → 33.35% (+30.18pp) — 의미 있는 개선
- Thermal 지배 완화: 0.923 → 0.864
- UAMM CV 2배 증가 (0.02 → 0.04) — 약간 더 adaptive

### P9 수준 회복? **실패**

- M-score 73.23 (P9: 81.47, **-8.24**)
- Static -20pp, Sky -43pp 갭 지속
- aux mask 품질은 개선됐지만 **여전히 GT 대비 부정확**

### Checkpoint 선택 분석

| 선택 기준 | Epoch | Sky | Static | Dynamic | mIoU |
|-----------|-------|-----|--------|---------|------|
| Night-val | 35 | **33.35** | 61.46 | 19.44 | **53.86** |
| Day-val | 28 | 20.83 | **63.99** | **27.62** | 52.69 |

Night-val checkpoint이 Sky에 유리 → M-score도 미세 우세 (73.23 vs 72.84)

## 핵심 교훈

1. **Multi-scale FPN은 효과적**: fpn[2](256ch)의 semantic context가 aux mask 품질을 실질적으로 개선
2. **하지만 근본 해결은 아님**: frozen backbone feature의 한계는 정보량 증가만으로 극복 불가
3. **P13→P14→P15→P16→P17: adaptive fusion의 한계 확정**
   - P13: Energy scalar → M=81.21 (-0.26)
   - P14: 독립 decoder → M=74.27 (-7.20)
   - P15: Spatial energy → M=71.05 (-10.42)
   - P16: 4 Fixes → M=68.42 (-13.05)
   - P17: Multi-scale → M=73.23 (-8.24)
   - **모든 adaptive fusion이 P9(상수)보다 나쁨**
4. **SAM2 memory attention이 이미 implicit adaptation 수행**: 외부 explicit adaptive fusion이 불필요하거나 해로움

## Dynamic Fusion 실패 종합 결론

| 모델 | Fusion 방식 | M-score | vs P9 |
|------|-----------|---------|-------|
| **P9** | **고정 상수** | **81.47** | **—** |
| P12 | Conditional MoE | 80.80 | -0.67 |
| P13 | Energy Score | 81.21 | -0.26 |
| P14 | Energy + 독립 decoder | 74.27 | -7.20 |
| P15 | Spatial energy (Fix3만) | 71.05 | -10.42 |
| P16 | 4 Fixes 통합 | 68.42 | -13.05 |
| P17 | Multi-scale + 4 Fixes | 73.23 | -8.24 |

**결론**: P9의 CrossModalFusionHead가 학습한 "상수 비율"이 현재 아키텍처에서 최적. Adaptive fusion은 aux mask 품질 한계로 오히려 해로움.
