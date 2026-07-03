---
title: HyperDUM — Hyperdimensional Uncertainty Quantification for Multimodal Fusion (CVPR 2025)
tags: [related-work, uncertainty, hyperdimensional-computing, multimodal-segmentation, deliver, rbma, competitor]
created: 2026-07-02
source: Track 2 deep-research 2026-07-02 (07_parallel_research_prompts_2026-07-02.md); arXiv:2503.20011 [VERIFIED-PDF]; adversarially verified (skeptic1 + skeptic2 both reproduced Table 4 from primary HTML)
status: verified-draft
---

# HyperDUM (CVPR 2025) — 상세 검증 노트

- **arXiv:** 2503.20011 · **Venue:** CVPR 2025 · **Code:** 논문 본문에 명시 없음
- 이 노트는 [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]의 ledger 행을 원문(PDF/HTML) 수준으로 확장한 per-paper note.

## Problem setting

Autonomous-vehicle 멀티모달 perception(3D detection + semantic segmentation)에서 sensor degradation/corruption에 강건한 fusion을 위해 feature-level epistemic uncertainty를 싸게(deterministic, single-pass) 추정하는 문제. 기존 UQ(MC-Dropout, ensembles, PostNet, LDU)는 연산 비용이 크거나 재학습이 필요.

## Novelty (논문의 명시적 주장)

Deterministic uncertainty method (DUM): **hyperdimensional computing (HDC)** 으로 feature를 channel/patch-wise projection & bundling하여 hypervector prototype과의 거리로 uncertainty를 계산. SOTA UQ 대비 2.36× fewer FLOPs, 최대 38.3× fewer params. [VERIFIED-PDF]

## Method (equations)

- Uncertainty = feature hypervector와 label-bundled prototype 간 similarity distance: `𝕌_m = ⋃_{l=1..L} {δ(ℋ_m^z, ℋ_m^l)}`; projection `ϕ_m(z_m) = Φ^{(d×C)} ⊗ z_m^{pooled}`. [VERIFIED-PDF]
- **Fusion 주입점:** uncertainty로 feature를 재가중한 `ẑ_m`을 만드는 **learnable uncertainty-weighting layer를 end-to-end fine-tune** ("fine-tune the uncertainty weighting layer... along with other post-fusion layers" — skeptic1이 원문에서 재확인). 주입 위치는 **feature level, pre-fusion**. 닫힌 형태의 가중 식은 본문에 없음(블랙박스 학습 레이어).
- **Mechanism class: learned feature reweighting (feature-multiply). Attention logit은 건드리지 않음.** [VERIFIED-PDF]
- **Training-free 아님:** prototype은 labeled training data로 구축 + weighting layer/post-fusion layer fine-tune 필요.

## Quantitative results (verbatim)

DELIVER semantic seg (Table 4, mIoU, CMNeXt backbone, RGB-D-E-L) — **[VERIFIED-PDF] [val]** (논문이 "DeLiVER validation set performance"라고 명시; 두 skeptic 모두 원문에서 독립 재확인):

| Scenario | CMNeXt | InfMCD | InfNoise | PostNet | LDU | HyperDUM |
|---|---|---|---|---|---|---|
| Cloudy | 68.70 | 69.23 | 69.21 | 69.28 | 68.94 | **69.76** |
| Night | 62.46 | 63.44 | 63.14 | 62.97 | 63.06 | **64.21** |
| Motion Blur | 62.91 | 63.61 | 63.55 | 63.55 | 63.16 | **64.28** |
| Lidar-Jitter | 65.92 | 66.12 | 66.25 | 66.33 | 66.40 | **66.93** |
| **Mean** | 66.30 | 66.90 | 66.86 | 66.77 | 66.60 | **67.59** |

- CMNeXt mean **66.30** ≈ CAFuser Table III의 CMNeXt(MiT-B2) **66.3 [val]** — DELIVER two-cluster 논쟁의 프로토콜 앵커 (→ [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]]). 같은 모델이 test-cluster(CLDE)에서는 53.0. [VERIFIED-PDF]
- aiMotive 3D detection (Table 1): HyperDUM mean 66.70/66.00 AP vs LDU 64.69/64.73 (+2.01 all-point AP); corner cases (Table 2) 65.16/64.62 mean. [VERIFIED-PDF]

## Limitations

- Labeled data 기반 prototype + fine-tuning 필요 → training-free 아님.
- Uncertainty가 epistemic-feature-level: per-modality **predictive** uncertainty가 아니고, fusion weight는 학습된 블랙박스 레이어(해석성 낮음).
- DELIVER 평가가 val-cluster 프로토콜 — test-cluster 방법(CAFuser/DGFusion/우리)과 직접 비교 불가; 표에 프로토콜 명기 필수.

## Improvement directions

- 학습 없는 reliability 신호(예: raw predictive entropy)로 prototype/fine-tuning 제거.
- Feature reweighting 대신 attention 결정점(logit)에 주입해 해석 가능한 λ·B 형태로.
- val/test 두 프로토콜 모두 보고해 비교 가능성 확보.

## Comparison to RBMA-P29-P30 (mechanism-class)

| 축 | HyperDUM | RBMA (ours) |
|---|---|---|
| Signal source | HDC prototype distance (label-supervised) | raw per-modality decoder softmax entropy |
| Training-free? | 아니오 (prototype + FT) | **예** |
| Injection | learned feature reweighting (pre-fusion) | **additive PRE-softmax bias on SAM2 memory cross-attn logits** |
| Mechanism class | feature-multiply | logit-additive-bias |

## Application to ours (RBMA/P29/P30 적용방향)

- **필수 비교 행:** HyperDUM은 DELIVER *val* 프로토콜에서 reliability-signal SOTA(67.59) — RBMA 실험표에 val-cluster 행으로 반드시 인용하되 프로토콜 차이를 명기.
- **평가 슬라이스 차용:** Table 4의 per-corruption 행(Motion Blur, LiDAR-Jitter, Night, Cloudy)이 우리 condition-adaptive 스토리의 평가 슬라이스 템플릿 — RBMA도 동일 슬라이스로 보고.
- **Ablation 상대:** "learned feature reweighting vs additive logit bias" ablation의 대표 상대 (42번 노트의 ablation 표 3행).

## Related-work paragraph candidate (English)

HyperDUM (CVPR 2025) estimates feature-level epistemic uncertainty via hyperdimensional prototype distances and injects it through a learnable uncertainty-weighting layer that reweights sensor features before fusion, improving CMNeXt on the DELIVER validation clusters from 66.30 to 67.59 mIoU, including sensor-failure slices such as motion blur and LiDAR jitter. However, its prototypes are built from labeled training data and the weighting layer must be fine-tuned end-to-end; the reliability signal never reaches the attention computation itself. In contrast, our reliability is training-free (raw per-modality predictive entropy) and enters as an additive pre-softmax bias on memory-attention logits.

## Links

- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]]
