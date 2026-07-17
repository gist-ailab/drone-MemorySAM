---
title: CBC-SLP — Structured Latent Projection for Robust Multispectral Segmentation
tags: [relatedwork, remote-sensing, missing-modality, contrastive-learning, abstract-only, gap-fill]
created: 2026-07-02
source: "[arXiv:2604.15856, status: [ABSTRACT-ONLY + PDF existence verified]]"
status: gap-fill-verified
---

# CBC-SLP — Structured Latent Projection for Robust Multispectral Segmentation

## Verification

- **Paper:** *Robust Multispectral Semantic Segmentation under Missing or Full Modalities via Structured Latent Projection* / CBC-SLP.
- **arXiv:** `2604.15856v1`.
- **Verification tag:** `[ABSTRACT-ONLY + PDF existence verified]`.

## Method idea

CBC-SLP addresses a trade-off: shared representations help missing modalities but may lose full-modality complementary information. It uses structured latent projection / class-balanced contrastive structure to maintain robustness across missing and full modality settings.

## Novelty relative to RBMA/P29/P30

This is most relevant to P30. Class-level latent structure supports the rationale for a class-token decoder on fused memory features.

## Limitations

- Remote-sensing focus.
- Insufficient formula extraction in this run.
- Not SAM2 based.

## Ours application direction

Use as support for class-structured latent alignment in P30. Do not overclaim benchmark numbers until tables are visually verified.
