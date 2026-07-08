---
title: RSGMamba — Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation
tags: [relatedwork, multimodal-segmentation, reliability, mamba, rgb-d, rgb-t, gap-fill]
created: 2026-07-02
source: "[arXiv:2604.12319, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# RSGMamba — Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation

## Citation / verification

- **Paper:** *RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation*.
- **arXiv:** `2604.12319v2`.
- **Status:** arXiv only in the checked sources.
- **Code:** no official GitHub URL found in the checked arXiv/PDF source.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task and evaluation

RSGMamba targets RGB-D and RGB-T semantic segmentation. The verified datasets are NYUDepth V2, SUN RGB-D, MFNet, and PST900. Reported headline mIoU values from the checked PDF are:

| Dataset | Reported mIoU |
|---|---:|
| NYUDepth V2 | 58.8 |
| SUN RGB-D | 54.0 |
| MFNet | 61.1 |
| PST900 | 88.9 |

## Methodology

The core block is the **Reliability-aware Self-Gated Mamba Block (RSGMB)**. Instead of directly concatenating or mixing RGB and auxiliary features, it performs cross-state fusion in the SSM readout stage.

Two learned gates control fusion:

| Gate | Role |
|---|---|
| uncertainty-aware gate | estimates modality-specific reliability |
| consistency-aware gate | measures spatial cross-modal agreement |

The paper also uses low-rank cross projection and learnable scaling to limit fusion capacity, reducing negative transfer from noisy or misaligned auxiliary modalities. A **Local Cross-Gated Modulation (LCGM)** module adds local detail.

## Implementation details verified

- Optimizer: AdamW.
- Weight decay: 0.01.
- Initial LR: `6e-5`.
- Schedule: polynomial decay, 10-epoch warm-up.
- Training: 500 epochs, batch size 8.
- Hardware: 4× NVIDIA H100.
- RSGMamba-B: 48.55M parameters.
- Loss: standard cross-entropy.

## Novelty relative to RBMA / P29 / P30

RSGMamba is a close **reliability-aware fusion** neighbor, but its reliability is learned and embedded inside an SSM/Mamba fusion block. RBMA differs because it uses per-modality predictive entropy as a **training-free additive pre-softmax bias** in SAM2 memory cross-attention logits.

P29 differs because it uses an unsupervised image-derived condition prototype to modulate a Soft-MoE LoRA router. RSGMamba has learned reliability gates, not unsupervised condition routing.

P30 does not directly collide because RSGMamba does not implement a class-token decoder on SAM2 fused memory features.

## Limitations

- Not training-free.
- Reliability gate calibration can drift under unseen domain shifts.
- Does not use SAM2 memory attention.
- Does not directly test RBMA-style entropy logit bias.

## Ours application direction

Use RSGMamba as a strong learned-reliability baseline. In writing, position RBMA as a lighter, training-free, SAM2-memory-attention-level reliability prior. A useful ablation is to add an RSGMamba-style consistency score as a second term next to RBMA entropy:

```text
attention_logits = QK^T/sqrt(d) + lambda_entropy * B_entropy + lambda_consistency * B_consistency
```

## ⚠️ CoRB collision (2026-07-06)

**Correction of scope.** The "Novelty relative to RBMA / P29 / P30" section above scopes the collision to P30 and concludes "P30 does not directly collide." That reading **understates the risk**: an internal audit (2026-07-06) finds RSGMamba is the **#1 prior-art threat to CoRB (P32-B)**, not merely a P30 neighbor. CoRB's entire premise — reliability from cross-modal agreement — is exactly what RSGMamba's **consistency-aware gate** also targets. This is the collision that matters for the paper. (Existing content above kept unchanged; this section supersedes its "does not directly collide" wording for the CoRB axis.)

### 4 confirmed facts (all `[VERIFIED-PDF]`)

1. **Uncertainty self-gate is LEARNED:** `g_u = σ(MLP(f))`.
2. **Consistency gate is FEATURE-space abs-diff through a LEARNED MLP, NOT a posterior:** `g_c = σ(𝒢_c([f_rgb, f_x, |f_rgb − f_x|]))`.
3. **PAIRWISE only** (RGB + one X modality), **never a joint N≥3 consensus**.
4. **Injection is MULTIPLICATIVE into the Mamba SSM C-matrices:** `C_eff_rgb = g_u^rgb · (1 − g_c) · C_rgb`.

### 4 axes on which CoRB differs (we beat it on all four)

| Axis | RSGMamba | CoRB (P32-B) |
|---|---|---|
| Reliability estimator | **Learned** gates (MLP) | **Training-free**, closed-form |
| Agreement space | **Feature** abs-diff | **Posterior-space Bhattacharyya** ← cleanest discriminator |
| Modality scope | **Pairwise** RGB+X | **Leave-one-out consensus, N≥3** |
| Injection | **Multiplicative** into SSM C-matrices | **Additive pre-softmax** into SAM2 memory attention |

Plus CoRB's **unique-info veto** (training-free protect-the-dissenter) has no counterpart in RSGMamba.

**Verdict: NEAR-MISS, and RSGMamba is MUST-CITE in the CoRB related-work.** Full defense: [[relatedworks/49_corb_novelty_defense]]; consolidated ranking: [[P32_CoRB/P32_CoRB_novelty_risk_register]].
