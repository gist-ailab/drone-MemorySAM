---
title: Latent Space Guided Scenario Sampling for Missing-Modality Segmentation
tags: [relatedwork, missing-modality, remote-sensing, scenario-sampling, abstract-only, gap-fill]
created: 2026-07-02
source: "[arXiv:2605.20372, status: [ABSTRACT-ONLY + PDF existence verified]]"
status: gap-fill-verified
---

# Latent Space Guided Scenario Sampling for Missing-Modality Segmentation

## Verification

- **Paper:** *Latent Space Guided Scenario Sampling for Multimodal Segmentation Under Missing Modalities*.
- **arXiv:** `2605.20372v1`.
- **Status:** arXiv only in checked sources.
- **Verification tag:** `[ABSTRACT-ONLY + PDF existence verified]`.

## Method idea

The paper argues that not all modality-availability scenarios are equally informative. Instead of uniformly sampling missing-modality scenarios, it uses latent-space informativeness to guide scenario sampling.

## Novelty relative to RBMA/P29/P30

This is relevant to P29 because a scenario latent can be treated as a condition prototype. It is complementary to RBMA: scenario sampling improves training coverage; RBMA handles inference-time reliability.

## Limitations

- Remote-sensing focus.
- Formula and implementation details were not sufficiently extracted in this run.

## Ours application direction

Use as a training-schedule baseline for P29 condition prototype learning: uniform missing-modality sampling vs latent-scenario sampling.
