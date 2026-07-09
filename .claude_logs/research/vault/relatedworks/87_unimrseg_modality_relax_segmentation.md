---
title: UniMRSeg — Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation
tags: [relatedwork, missing-modality, segmentation, neurips-2025, abstract-only, gap-fill]
created: 2026-07-02
source: "[arXiv:2509.16170, code: https://github.com/Xiaoqi-Zhao-DLUT/UniMRSeg, status: [ABSTRACT-ONLY + metadata verified]]"
status: gap-fill-verified
---

# UniMRSeg — Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation

## Verification

- **Paper:** *UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation*.
- **arXiv:** `2509.16170`.
- **Venue/status:** NeurIPS 2025 accepted according to metadata returned in the gap-fill run.
- **Code:** https://github.com/Xiaoqi-Zhao-DLUT/UniMRSeg
- **Verification tag:** `[ABSTRACT-ONLY + metadata verified]`.

## Method idea

UniMRSeg targets segmentation with incomplete or corrupted modalities without deploying a separate model for every modality subset. The key module is **Hierarchical Self-Supervised Compensation (HSSC)**, which compensates gaps between complete and incomplete modalities at input, feature, and prediction levels.

## Novelty relative to RBMA/P29/P30

UniMRSeg shares deployment motivation with P29/P30: avoid modality-subset-specific models. It differs because RBMA is a plug-in attention reliability bias and P29 is condition-routed LoRA rather than hierarchical compensation.

## Limitations

- Detailed equations were not extracted in this run.
- Need PDF/table verification before citing quantitative results.

## Ours application direction

Use as a recent NeurIPS anchor for unified missing/corrupted modality segmentation. Compare against RBMA/P29 as lighter plug-in alternatives.
