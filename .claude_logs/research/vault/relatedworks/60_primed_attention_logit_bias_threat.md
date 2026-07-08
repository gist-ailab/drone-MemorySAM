---
title: PRIMED — Modality-Prior Additive Attention-Logit Bias for Referring Audio-Visual Segmentation (RBMA HIGH threat)
tags: [related-work, threat-watch, rbma, attention, logit-bias, audio-visual-segmentation, high-threat]
created: 2026-07-02
source: "arXiv:2605.07154 (2026-05); found by skeptic2 adversarial pass of Track 8, [[sources/08_threat_watch_2026H2]]"
status: verified-draft
---

# PRIMED (arXiv:2605.07154) — HIGH threat to RBMA's logit-bias novelty claim

**Why this note exists:** the Track 8 sweep initially concluded that "additive pre-softmax attention-logit bias driven by reliability/uncertainty/condition in multimodal dense-prediction fusion" was an unoccupied cell. Adversarial verification (skeptic2) **refuted** that claim with this paper. PRIMED does exactly the RBMA injection mechanism, in a segmentation (dense-prediction) task. It was missed by the original 2026H1 query battery — a material gap. It MUST be cited in the RBMA paper.

## Citation

- arXiv:2605.07154, May 2026. Task: Referring Audio-Visual Segmentation (RAVS). Venue/code: not yet extracted (full read pending — blocking follow-up #1 in [[sources/08_threat_watch_2026H2]] §5).

## Problem setting

Referring audio-visual segmentation: segment the object a (language) query refers to in video, using both audio and visual modalities. Different queries rely on different modalities ("the thing making the sound" vs "the red car"), so cross-modal attention should be steered by which modality the query actually depends on.

## Novelty (theirs)

A **modality prior** P — an estimate of which modality the query relies on — is converted into an additive bias on cross-attention logits, steering attention toward the relied-on modality's tokens before softmax normalization. The prior is **learned, distilled from Qwen3-omni soft labels** (an LMM teacher provides per-query modality-reliance supervision).

## Method (with equations)

Verbatim mechanism as quoted from the paper during verification [VERIFIED-PDF via skeptic2 quote]:

> "The bias b_M is broadcast and added to the attention logits before softmax: MHCA(Q,K,V) = Softmax(QK^T/√d + b_M)V"

with the bias defined from the modality prior P:

$$
b_M = \gamma_p \cdot \log\!\frac{P}{1-P}
$$

- P ∈ (0,1): modality prior (probability the query relies on modality M), learned by distillation from Qwen3-omni soft labels.
- log-odds transform makes the bias symmetric around P = 0.5 and unbounded at the extremes; γ_p scales it (analogous to our λ).
- Injection point: **pre-softmax cross-modal attention logits** — the same control point as RBMA.

## Quantitative results

- Not yet extracted — full PDF read is a **blocking follow-up**. No rows can be quoted here. [ABSTRACT-ONLY / equations VERIFIED-PDF via skeptic quote] [unknown split]
- RAVS benchmarks do not overlap DELIVER/MUSES/MCubeS/MULTIAQUA, so no direct leaderboard collision.

## Limitations (relative to our setting; some inferred pending full read)

1. **Signal is learned/distilled**, requiring an LMM teacher (Qwen3-omni) and a training stage — not training-free, not derivable at test time from the model's own predictions.
2. Signal semantics = *modality reliance of a query* (which modality answers this query), not *modality reliability under sensor degradation* (is this sensor trustworthy right now). Reliance ≠ reliability: a query may rely on RGB even when RGB is degraded.
3. Injection site is a bespoke MHCA in a RAVS architecture, not a VFM's native memory attention; no SAM2/SAM3, no memory bank.
4. Domain: audio-visual video with language queries; two modalities; no adverse-condition axis, no multi-sensor (RGB-D-E-L) setting.

## Improvement directions (what PRIMED leaves open — our territory)

- Replace the distilled prior with a **training-free, calibration-grounded signal** (our B_i = 1 − H/log C predictive entropy) — removes the LMM-teacher dependency and adapts per test sample.
- Move the bias into a **foundation model's existing attention** (SAM2 memory attention over modality memory tokens) instead of adding a bespoke MHCA — parameter-free reuse of pretrained fusion machinery.
- Condition the bias on **sensor degradation** (night/rain/fog/snow) rather than query semantics; evaluate per-condition.
- N-modality (4+) generalization: P/(1−P) log-odds is binary-flavored; per-modality entropy scales to any N.

## Comparison to RBMA-P29-P30 (mechanism-class)

| Axis | PRIMED | RBMA |
|---|---|---|
| Mechanism class | **logit-additive-bias** (same class — this is the threat) | logit-additive-bias |
| Signal source | learned modality prior, distilled from Qwen3-omni soft labels | training-free per-modality decoder predictive entropy, B_i = 1 − H(softmax(Dec_i(f_i)))/log C |
| Signal semantics | query→modality reliance | modality reliability under degradation |
| Injection site | bespoke MHCA in RAVS model | SAM2 memory cross-attention over modality memory tokens |
| Task | referring audio-visual segmentation (2 modalities) | multi-sensor semantic segmentation (RGB-D-E-L), adverse conditions |
| Training | needs distillation stage | training-free signal (λ optionally tuned) |

P29/P30: no overlap (no MoE/LoRA, no query-classification decoder in the P30 sense).

## Application to ours (RBMA/P29/P30 적용방향)

1. **Novelty 문장 재작성 (필수):** "no prior work biases attention logits by modality reliability" 문장 전면 금지. 새 문장: PRIMED가 dense prediction에서 additive pre-softmax modality bias를 최초 점유 — RBMA의 방어선은 (i) training-free entropy signal, (ii) SAM2 memory-attention 주입 지점, (iii) multi-sensor adverse-condition seg 과제의 **conjunction**. [[relatedworks/42_attention_logit_bias_novelty_defense]] 업데이트 반영됨.
2. **b(r) 함수형 차용 검토:** PRIMED의 log-odds 형태 b = γ·log(P/(1−P))는 우리 B_i ∈ [0,1]에도 적용 가능 (b = λ·log(B_i/(1−B_i))). 기존 오픈 퀘스천("b(r)를 log(r+ε)로 할까?")에 대한 실험 후보 — ablation 표에 log-odds 변형 추가.
3. **Ablation 추가:** "learned/distilled prior (PRIMED-style) vs training-free entropy (ours)" 비교 — 같은 주입 지점에서 signal source만 바꾸는 ablation이 리뷰어 방어에 결정적.
4. **인용 위치:** related work의 attention-bias 문단 첫 항목으로 인용; ViSymRe·SAE와 함께 3-way distinguish.

## Related-work paragraph candidate (English)

Recent work has begun to inject modality-level priors directly into attention logits. PRIMED [arXiv:2605.07154] adds a modality-prior bias b_M = γ_p·log(P/(1−P)) to cross-attention logits before softmax for referring audio-visual segmentation, where the prior P — which modality a query relies on — is distilled from an omni-modal LLM teacher. While PRIMED shares our injection point (pre-softmax additive bias), its signal is a *learned reliance prior* requiring teacher distillation, and it targets query–modality alignment in audio-visual video. In contrast, RBMA derives a *training-free reliability* signal from each sensor modality's own predictive entropy and injects it into SAM2's native memory attention, suppressing degraded sensors at retrieval time in multi-sensor semantic segmentation under adverse conditions.

## Links

- [[sources/08_threat_watch_2026H2]] · [[relatedworks/42_attention_logit_bias_novelty_defense]] · [[relatedworks/58_sae_entropy_logit_bias_threat]]
