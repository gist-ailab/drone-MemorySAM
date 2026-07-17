---
legacy_file: outputs_model_explain/P16_CalibratedEntropy.md
moved: 2026-07-08
---

# P16: Calibrated Spatial Entropy Fusion (역대 최악 경신)

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P16` |
| 파일 | `sam_lora_image_encoder_seg.py` |
| 베이스 | P14/P15 설계 통합 |
| 상태 | 완료 (**역대 최악 M=68.42**) |
| 최선 M-score | 68.42 (hardaug5) |

## 변경 동기

P15(Fix 3만 단독 적용)가 역대 최악 → P12~P14 실패 분석에서 도출한 **4가지 Fix를 전부 통합**.
P15의 교훈: 4가지 Fix는 동시 적용해야 효과.

## 아키텍처 변경: 4가지 Fix 통합

### Fix 1: `.detach()` Gradient 격리

```python
# P13/P14/P15: gradient 오염 — main loss가 aux head를 왜곡
cross_weights = compute_energy(aux_logits_list)

# P16: gradient 차단 — aux head는 자체 CE loss만으로 학습
cross_weights = compute_entropy([z.detach() for z in aux_logits_list])
```

### Fix 2: Energy Score → Calibrated Entropy

```python
# Energy: -T × logsumexp(z/T) → logit magnitude 기반
# → "자신있게 틀리면" 높은 점수 (dangerous)

# Entropy: -(p × log p).sum() → 확률 분포 균등도
# → 불확실하면 낮은 confidence (safe)

def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0):
    for z in aux_logits_list:
        probs = F.softmax(z / temperature, dim=1)
        entropy = -(probs * F.log_softmax(z / temperature, dim=1)).sum(dim=1)
        confidence = 1.0 - entropy / log(num_classes)  # 0(확신)~1(균등)
    weights = F.softmax(stack(conf_maps) / temperature, dim=1)  # (B, m, H, W)
```

### Fix 3: Spatial-wise (B, m, H, W) — P15에서 유지

### Fix 4: Aux Warmup Schedule (신규)

```python
# 3단계: uniform → linear ramp → full entropy
if epoch < 10:
    cross_weights = uniform(1/m)                  # P9처럼 안정적
elif epoch < 15:
    ramp = (epoch - 10) / 5.0
    cross_weights = (1-ramp)*uniform + ramp*entropy  # 점진적 전환
else:
    cross_weights = entropy                       # full adaptive
```

## P15 대비 변경 요약

| 구분 | P15 | P16 |
|------|------|-----|
| Confidence | Energy Score | **Calibrated Entropy** |
| Gradient | 격리 없음 | **`.detach()` 적용** |
| Warmup | 없음 | **10ep uniform + 5ep ramp** |
| Weight 형태 | (B, m, H, W) | (B, m, H, W) (동일) |
| Aux Decoder | ModalAuxDecoder × 3 | 동일 |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug5 (night ep31) | 93.14 | **43.70** | **68.42** | **-13.05** | #16106 |

### Per-Class Test IoU

| Class | P9 | P15 | P16 | P16 vs P15 |
|-------|-----|-----|-----|------------|
| Static | 81.30 | 60.94 | 58.19 | -2.75 |
| Dynamic | 21.86 | 26.58 | 20.76 | -5.82 |
| Water | 94.61 | 92.27 | 92.24 | -0.03 |
| Sky | 76.54 | 16.66 | **3.17** | **-13.49** |

### Sky 완전 붕괴

- Sky=0 프레임: **157/200** (78.5%)
- Sky<10% 프레임: **191/200** (95.5%)

## 해결하려는 문제를 해결했는가?

### 4가지 Fix 통합으로 adaptive fusion 작동? **완전 실패**

UAMM 분석:
- img=0.758, lidar=0.819, **thermal=0.923** → thermal 지배
- CV < 0.02 → P9처럼 거의 고정 비율이지만, **나쁜 고정 비율**

**실패 원인 분석**:
1. `.detach()` + entropy로 바꿔도 **aux mask 품질 자체가 부족** (ISSUE-008)
2. Thermal이 전반적으로 낮은 entropy → "confident" → 과도한 가중치
3. Sky 영역에서 thermal은 무의미한데 thermal 기반 예측
4. **4가지 Fix가 모두 aux mask 품질에 의존** — aux 품질 개선 없이 집계 방법만 바꿔서는 불가

### P16이 P15보다 나빴던 이유

- P15: UAMM 적응 수행 (img -21%, lidar +15%) → 방향은 맞음
- P16: entropy 기반 → thermal 편향으로 수렴 → P15보다 더 나쁜 고정점
- Warmup이 10ep 동안 uniform 유지 → 이미 aux head가 thermal 편향으로 학습 → ramp 후에도 편향 유지

## 핵심 교훈

1. **Aux mask 품질이 근본 한계** (ISSUE-008 확정): frozen backbone 위의 경량 decoder → entropy/energy 어느 것이든 신뢰도 부족
2. **4 Fixes 통합도 해결 불가**: 문제는 Fix 대상(aux mask)이 아니라 aux mask 자체의 품질
3. **Thermal 편향 패턴**: entropy 기반에서 thermal이 항상 "confident" → P16 고유의 실패 모드
4. **Scoring 접근 전체의 한계**: P13(energy scalar) → P14(독립 decoder) → P15(spatial) → P16(4 fixes) 순차적으로 모든 가능한 개선을 시도했으나, **aux mask 품질이라는 근본 병목**을 해결하지 못함

## 다음 모델 (P17)로의 동기

- Aux decoder의 정보량 문제: fpn[0] (32ch, 256×256) **하나만** 사용
- SAM2는 fpn[0,1,2] 3개 레벨을 계산하지만 나머지 2개 미활용
- 32ch → 352ch(32+64+256) = **11배 정보량 증가, 추가 backbone 연산 0**
