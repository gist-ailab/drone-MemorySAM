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
