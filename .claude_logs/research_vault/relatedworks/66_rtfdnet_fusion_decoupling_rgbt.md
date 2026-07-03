---
title: RTFDNet — Fusion-Decoupling for Robust RGB-T Segmentation
tags: [relatedwork, rgb-t, semantic-segmentation, missing-modality, decoupling, gap-fill]
created: 2026-07-02
source: "[arXiv:2603.09149, code: https://github.com/curapima/RTFDNet, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# RTFDNet — Fusion-Decoupling for Robust RGB-T Segmentation

## Citation / verification

- **Paper:** *RTFDNet: Fusion-Decoupling for Robust RGB-T Segmentation*.
- **arXiv:** `2603.09149v1`.
- **Code stated in PDF:** https://github.com/curapima/RTFDNet
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

RGB-T semantic segmentation on MFNet, FMB, and PST900, with robustness to missing or degraded modalities.

## Methodology

RTFDNet supports full fusion and single-modality fallback with a three-branch architecture:

1. RGB branch.
2. Thermal branch.
3. Fusion branch.

Key modules:

| Module | Role |
|---|---|
| SFF — Synergistic Feature Fusion | selective cross-modal interaction |
| CMDR — Cross-modal Decouple Regularization | separates modality-specific and shared features |
| RDR — Region Decouple Regularization | aligns branch outputs at region level |

Total loss:

```text
L_ALL = lambda L_CMDR + lambda' L_RDR + lambda'' L_CrossEntropy
```

## Novelty relative to RBMA / P29 / P30

RTFDNet is close to P29 because it explicitly supports modality fallback. However it does so with branch decoupling/regularization rather than condition-prototype LoRA routing. RBMA is more lightweight because it does not require three persistent branches.

## Limitations

- RGB-T only.
- Multiple branches increase management and compute complexity.
- Does not use SAM2 memory attention.

## Ours application direction

Use RTFDNet as a missing-modality/fallback baseline. Compare branch decoupling with RBMA's dynamic attention reweighting.
