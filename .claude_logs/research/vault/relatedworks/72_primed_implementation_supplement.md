---
title: PRIMED — Additive Attention-Logit Bias Implementation Supplement
tags: [relatedwork, threat, additive-logit-bias, attention, referring-audio-visual-segmentation, implementation, gap-fill]
created: 2026-07-02
source: "[arXiv:2605.07154, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# PRIMED — Additive Attention-Logit Bias Implementation Supplement

## Citation / verification

- **Paper:** *PRIMED: Adaptive Modality Suppression for Referring Audio-Visual Segmentation via Biased Competition*.
- **arXiv:** `2605.07154`.
- **Code:** no official code link found in arXiv HTML/PDF.
- **Verification tag:** `[VERIFIED-PDF]`.

## Why this matters

PRIMED is a strong threat to broad claims such as “we are first to add modality reliability as attention-logit bias.” It is less threatening to the narrower RBMA claim because it is not RGB-X semantic segmentation, not SAM2 memory cross-attention for modality frames, and not entropy-derived/training-free.

## Exact mechanics

PRIMED computes a text-derived modality prior:

```text
z_M = W2 σ(W1 t_g + b1) + b2
P_M = Softmax(z_M) = [p_A, p_V, p_AV]
```

It computes visual/text and visual/audio similarities:

```text
Sim_vis = (<T_g, F_V> + 1) / 2
Sim_aud = (<A_g, F_V> + 1) / 2
```

It forms the prior map:

```text
P_hat = p_V Sim_vis + p_A Sim_aud + p_AV (Sim_vis * Sim_aud)
```

Then converts it to logit-space bias:

```text
b_M = rho * log(P_hat / (1 - P_hat))
```

And injects it before softmax:

```text
MHCA_P(Q,K,V) = Softmax(QK/sqrt(d) + b_M) V
```

## Implementation implication

Bias injection point is **cross-modal attention logits**, not final segmentation logits and not LVLM token logits. The bias is trained through a modality prior decoder supervised by Qwen3-omni soft labels and audit/human verification.

## Novelty boundary for ours

Safe phrasing:

> Unlike PRIMED's learned text/audio/vision modality prior for referring audio-visual segmentation, RBMA uses per-modality predictive entropy as a training-free reliability signal and injects it into SAM2 memory cross-attention for RGB-X semantic segmentation.

## Limitations / caveats

- No official code found.
- Task is referring audio-visual segmentation, not RGB-X semantic segmentation.
- Uses learned modality prior supervision, not raw predictive entropy.
