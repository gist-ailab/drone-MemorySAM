---
title: TUNI — Modality-Aware Mutual Learning and Rectification for RGB-T Segmentation
tags: [relatedwork, rgb-t, semantic-segmentation, pretraining, foundation-model-adaptation, gap-fill]
created: 2026-07-02
source: "[arXiv:2509.10005, code: https://github.com/xiaodonguo/TUNI-v2, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# TUNI — Modality-Aware Mutual Learning and Rectification for RGB-T Segmentation

## Citation / verification

- **Paper:** *TUNI: Unifying Pre-training and Fine-tuning with Modality-Aware Mutual Learning and Rectification for RGB-T Semantic Segmentation*.
- **arXiv:** `2509.10005`.
- **Venue/status:** extended ICRA work; to appear in IEEE TCSVT, DOI `10.1109/TCSVT.2026.3701706`.
- **Code:** https://github.com/xiaodonguo/TUNI-v2
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

RGB-T semantic segmentation.

## Methodology

TUNI unifies pre-training and fine-tuning for RGB-T segmentation. It uses:

- RGB-T local module for salient consistent/distinct local features.
- Modality-inverted contrastive mutual learning between RGB-dominated and thermal-dominated encoders.
- Modality rectification learning to focus on correct but divergent prediction regions across modality-specific decoders.

The paper uses generated aligned RGB-T image pairs for ImageNet-1K pretraining.

## Implementation details verified

- Hardware: 8× NVIDIA H20.
- Paired RGB-T data generated using RGB-T translation.
- Variants: TUNI-T / TUNI-S / TUNI-B for real-time, balanced, and high-performance settings.

## Novelty relative to RBMA / P29 / P30

TUNI is a pretraining/representation route. RBMA is an inference-time memory-attention bias route. P29/P30 can be framed as adaptation modules that do not require RGB-T synthetic pretraining.

## Limitations

- RGB-T specific.
- Requires paired/generated RGB-T pretraining pipeline.
- Not SAM2 memory based.

## Ours application direction

Use TUNI as a strong thermal-aware pretraining baseline. In our method section, emphasize that RBMA/P29 can be added without requiring synthetic RGB-T pretraining.
