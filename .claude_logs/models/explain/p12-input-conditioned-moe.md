---
legacy_file: outputs_model_explain/P12_InputConditioned_MoE.md
moved: 2026-07-08
---

# P12: Input-Conditioned Soft MoE LoRA

## 기본 정보

| 항목 | 내용 |
|------|------|
| 클래스 | `LoRA_Sam_P12` |
| 파일 | `sam_lora_image_encoder_seg.py` (line 1585) |
| 베이스 | P9 (P10/P11 취소 후 P9으로 복귀) |
| 상태 | 완료 (P9 미달) |
| 최선 M-score | 80.80 (hardaug4) |

## P9에서의 변경 동기

MoE gate 진단(P11 이후)에서 gate가 정상 분화됨을 확인. 하지만 **모달리티별로 동일한 routing 패턴** 관찰 → 모달리티의 입력 품질에 따라 routing을 달리해야 한다는 가설.

**핵심 아이디어**: RGB 채널의 raw 통계(mean + std)를 gate에 condition으로 주입하여, 야간(어두운 RGB)에서 다른 routing을 학습.

## 아키텍처 변경

### Input Conditioning

```python
# RGB 통계 추출 (정규화 전 raw input에서)
condition = [ch0_mean, ch1_mean, ch2_mean, ch0_std, ch1_std, ch2_std]  # (B, 6)

# Gate에 condition 추가
gate(x) + cond_proj(condition) → softmax → weights

# cond_proj: Linear(6, num_experts=3), zero-init
# LiDAR/Thermal은 condition=None (RGB만 조건부)
```

### CrossModalFusionHead에도 condition 추가

```python
class CrossModalFusionHead:
    # 기존 compare(concat) → logits
    # + cond_compare(condition) → bias
    logits = compare(concat) + cond_compare(condition)  # condition이 있을 때만
```

## P9 대비 변경 요약

| 구분 | P9 | P12 |
|------|----|----|
| MoE Gate | Linear(C→3) | **Linear(C→3) + Linear(6→3) condition** |
| CrossModalFusionHead | compare only | **compare + cond_compare** |
| Input 정보 | 없음 | **RGB mean+std (cond_dim=6)** |
| Conditioning 대상 | - | **RGB만** (LiDAR/Thermal은 None) |

## 실험 결과

| Config | Val mIoU | Test mIoU | M-score | vs P9 | Submission |
|--------|----------|-----------|---------|-------|------------|
| hardaug4 | 93.23 | 68.37 | 80.80 | **-0.67** | #15949 |

### Per-Class Test IoU (P9 대비)

| Class | P9 | P12 | Delta |
|-------|-----|-----|-------|
| Static | 81.30 | 78.47 | -2.83 |
| Dynamic | 21.25 | **25.27** | **+4.02** |
| Water | 94.61 | 94.36 | -0.25 |
| Sky | 76.54 | **69.73** | **-6.81** |

## 해결하려는 문제를 해결했는가?

### 모달리티별 adaptive routing? **부분적으로만**

- Dynamic class +4.02pp 개선 → condition이 일부 효과
- 하지만 **Sky -6.81pp 하락**이 Dynamic 개선을 상쇄

### UAMM/AMF 상수 수렴 해결? **실패**

- UAMM/AMF 변동성 소폭 증가 (std 0.0001→0.01)하지만 여전히 near-constant
- `cond_dim=6`(RGB mean+std만)은 너무 약한 신호

### Expert collapse

- P9보다 심화 (collapse rate 15% → 20%)
- Block0 lidar: E0=E2=0% (2개 expert 완전 미사용)
- Condition 추가가 특정 expert로의 편향을 강화

### Tail-end failure

- mIoU < 55% 프레임: P9 5장 → P12 18장 (3.6배 증가)
- Test LiDAR routing: 48/48 블록 완전 고정
- 최악 케이스에서 P9보다 더 취약

## 핵심 교훈

1. **RGB 통계만으로는 부족**: mean/std 6차원은 모달리티 품질을 판별하기에 너무 약한 신호
2. **Dynamic ↔ Sky 트레이드오프**: Dynamic 개선 시도가 Sky를 희생 — 이후 P13~P17에서도 반복되는 패턴
3. **cond_dim=6 zero-init**: 아주 작은 bias로 시작하지만, 학습 과정에서 잘못된 방향으로 증폭
4. **Expert collapse 악화**: condition이 특정 expert로의 routing 편향을 강화

## 다음 모델 (P13)로의 동기

- CrossModalFusionHead(학습된 파라미터 기반) 대신 **학습 파라미터 없는 computed signal**로 fusion weight 생성
- Energy Score: aux prediction의 confidence를 직접 측정 → 상수 수렴 원천 차단
