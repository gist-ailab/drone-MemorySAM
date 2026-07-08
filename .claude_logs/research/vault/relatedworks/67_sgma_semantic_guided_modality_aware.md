---
title: SGMA — Semantic-Guided Modality-Aware Segmentation with Incomplete Multimodal Data
tags: [relatedwork, remote-sensing, missing-modality, prototypes, semantic-segmentation, gap-fill]
created: 2026-07-02
source: "[arXiv:2603.02505, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# SGMA — Semantic-Guided Modality-Aware Segmentation with Incomplete Multimodal Data

## Citation / verification

- **Paper:** *SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data*.
- **arXiv:** `2603.02505v1`.
- **Status:** arXiv only in checked sources.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Remote-sensing semantic segmentation under incomplete multimodal data.

## Methodology

SGMA addresses modality imbalance, intra-class variation, and cross-modal heterogeneity.

Key components:

| Component | Role |
|---|---|
| SGF — Semantic-Guided Fusion | extracts multi-scale class-wise semantic prototypes |
| MAS — Modality-Aware Sampling | dynamically reweights fragile modality/challenging samples |

SGF estimates modality robustness through prototype-feature alignment, then uses that robustness score for adaptive fusion.

## Implementation details verified

- Hardware: 4× NVIDIA A100.
- Optimizer: AdamW.
- LR: `6e-5`.
- Weight decay: `1e-2`.
- Epsilon: `1e-8`.
- Training: 200 epochs, 10-epoch warm-up.
- Loss coefficients: `lambda_SGF = 2`, `lambda_MAS = 1`.

## Novelty relative to RBMA / P29 / P30

SGMA's robustness score is supervised semantic-prototype based. RBMA uses predictive entropy as a training-free prior. P29 uses unsupervised condition prototypes. P30 class-token memory decoding can borrow SGMA's prototype alignment intuition.

## Limitations

- Remote-sensing specific.
- Class prototype alignment depends on semantic labels.
- No SAM2 memory attention.

## Ours application direction

Use SGMA as evidence that class prototypes are a strong robustness signal, then argue that P30's class-token decoder on fused memory features is a more direct VFM/memory-compatible way to enforce class structure.
