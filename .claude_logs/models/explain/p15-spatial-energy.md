---
legacy_file: outputs_model_explain/P15_SpatialEnergy.md
moved: 2026-07-08
---

# P15: Spatial Energy Fusion (역대 최악)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P15` |
| 파일 | `sam_lora_image_encoder_seg.py` |
| 베이스 | P14 |
| 상태 | 완료 (**역대 최악 M-score 71.05**, 이후 P16이 경신) |
| 최선 M-score | 71.05 (hardaug5) |

## P14에서의 변경 동기

P14에서 image-level scalar fusion의 근본 한계 확인 → **Spatial-wise `(B, m, H, W)` 가중치**로 전환.
Sky 영역에서 RGB, Water 영역에서 LiDAR 등 위치별 최적 모달리티를 선택하겠다는 의도.

## 아키텍처 변경

### Spatial Energy Confidence (scalar → spatial)

```python
# P14: Energy Score → (B, m) scalar
conf = -energy.mean(dim=[1, 2])  # spatial 전체 평균

# P15: Energy Score → (B, m, H, W) spatial
# spatial 평균 없이 pixel-level confidence 유지
conf_map = -energy  # (B, H, W) per-pixel confidence
cross_weights = softmax(stack(conf_maps), dim=1)  # (B, m, H, W)
```

### Spatial UAMM

```python
# P14: scalar broadcast
modulated_feats = feats * uamm_score  # (B,) → broadcast

# P15: spatial interpolate
spatial_score = F.interpolate(uamm_scores[:, i], size=(h, w))
modulated_feats = feats * spatial_score  # level별 해상도 맞춤
```

### 적용한 Fix (4개 중 1개만)

설계 문서에서 4가지 Fix를 제안했으나, **Fix 3(spatial-wise)만 단독 적용**:

| Fix | 내용 | P15 적용 |
|-----|------|---------|
| Fix 1 | `.detach()` gradient 격리 | **미적용** |
| Fix 2 | Energy → Calibrated Entropy | **미적용** |
| **Fix 3** | **Scalar → Spatial** | **적용** |
| Fix 4 | Aux Warmup Schedule | **미적용** |

## P14 대비 변경 요약

| 구분 | P14 | P15 |
|------|------|-----|
| Weight 형태 | `(B, m)` scalar | **`(B, m, H, W)` spatial** |
| UAMM | scalar broadcast | **spatial interpolate** |
| AMF | `view(-1,1,1,1)` | **`F.interpolate` spatial** |
| Confidence | Energy Score scalar | **Energy Score spatial** |
| `.detach()` | 없음 | 없음 (동일) |
| Warmup | 없음 | 없음 (동일) |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug5 | 93.17 | **48.94** | **71.05** | **-10.42** | #16087 |

### Per-Class Test IoU

| Class | P9 | P14 | P15 | P15 vs P14 |
|-------|-----|-----|-----|------------|
| Static | 81.30 | 62.57 | 60.94 | -1.63 |
| Dynamic | 21.86 | 22.87 | 26.58 | +3.71 |
| Water | 94.61 | 92.92 | 92.27 | -0.65 |
| Sky | 76.54 | 36.47 | **16.66** | **-19.81** |

### Sky 붕괴 통계

- Sky=0 프레임: **111/200** (55.5%)
- Sky<10% 프레임: **152/200** (76%)

## 해결하려는 문제를 해결했는가?

### Spatial-wise 가중치로 위치별 모달리티 선택? **오히려 악화**

UAMM 분석에서 val→test 적응은 실제로 발생:

| 모달리티 | Val UAMM | Test UAMM | 변화 |
|----------|----------|-----------|------|
| img | 0.716 | 0.566 | -21% (야간 RGB 억제) |
| lidar | 0.834 | 0.956 | +15% (LiDAR 강화) |
| thermal | 0.554 | 0.630 | +14% |

- 적응 "방향"은 맞지만, **부정확한 energy를 pixel-level로 전파**한 결과:

### "Spatial Amplification Effect"

```
P14 (scalar): 부정확한 energy → 이미지 전체 평균 → 에러가 smooth됨
P15 (spatial): 부정확한 energy → pixel-level 전파 → aux mask의 모든 local error 증폭
```

- P14(scalar, Sky 36.47%) → P15(spatial, Sky **16.66%**): spatial 적용 후 **-19.81pp 추가 악화**
- Aux mask의 pixel-level error가 UAMM/AMF를 통해 직접 전파 → noise amplifier

## 핵심 교훈

1. **Fix 3만 단독 적용하면 역효과**: aux mask 정확도 개선(Fix 1,2,4)이 선행되어야 spatial이 효과
2. **Energy Score "confident but wrong"이 spatial에서 더 치명적**: scalar면 평균으로 희석되지만, spatial이면 모든 잘못된 pixel이 그대로 전파
3. **4가지 Fix는 동시 적용 필요**: Fix 3만 단독 → 최악. 이 교훈이 P16(4 Fixes 통합) 설계로 연결
4. **Adaptive UAMM 자체는 작동**: LiDAR가 1.0에 미고정(0.956), img -21% 적응 → 하지만 aux quality가 낮아서 적응 방향이 틀림
