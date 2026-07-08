---
title: TASeg — Text-Aware RGB-T Semantic Segmentation with SAM and CLIP
tags: [relatedwork, rgb-t, sam, clip, lora, semantic-segmentation, gap-fill]
created: 2026-07-02
source: "[arXiv:2506.21975, venue: IROS 2025, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# TASeg — Text-Aware RGB-T Semantic Segmentation with SAM and CLIP

## Citation / verification

- **Paper:** *TASeg: Text-aware RGB-T Semantic Segmentation based on Fine-tuning Vision Foundation Models*.
- **arXiv:** `2506.21975v1`.
- **Venue:** IROS 2025.
- **Code:** no official code URL found in checked PDF.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

RGB-T semantic segmentation using SAM/CLIP-style foundation models.

## Methodology

TASeg fine-tunes the SAM image encoder with LoRA and uses CLIP-generated text embeddings for semantic alignment.

Key components:

| Component | Role |
|---|---|
| SAM image encoder + LoRA | visual foundation-model adaptation |
| DFFM — Dynamic Feature Fusion Module | combines RGB and thermal features |
| CLIP text embeddings | class semantic alignment / classification correction |
| Loss | cross-entropy + Dice |

The checked PDF describes a frozen RGB branch and trainable thermal patch embedding.

## Reported facts

- Trainable params: 28.77M.
- MFNet overall: 77.6.
- PST900: 86.09.

## Novelty relative to RBMA / P29 / P30

TASeg is important for the VFM/SAM + RGB-T line. It uses LoRA fine-tuning and text embeddings, while RBMA uses SAM2 memory reliability bias. P30's class-token decoder can use CLIP text class embeddings as initialization or regularization.

## Limitations

- SAM image encoder, not SAM2 memory attention.
- RGB-T only.
- Text embeddings depend on class taxonomy.

## Ours application direction

Use TASeg as a foundation-model adaptation baseline. P30 can extend the class-token idea by decoding from fused memory features rather than only SAM image features.
