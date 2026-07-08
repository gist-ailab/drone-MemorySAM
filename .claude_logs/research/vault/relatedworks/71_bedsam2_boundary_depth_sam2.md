---
title: BED-SAM2 — Boundary-Enhanced-Depth SAM2 via Monocular Geometric Priors
tags: [relatedwork, sam2, depth, geometric-prior, dense-prediction, gap-fill]
created: 2026-07-02
source: "[arXiv:2605.24893, code: https://github.com/TylerRust-1/BED-SAM2, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# BED-SAM2 — Boundary-Enhanced-Depth SAM2 via Monocular Geometric Priors

## Citation / verification

- **Paper:** *BED-SAM2: Boundary-Enhanced-Depth SAM2 via Monocular Geometric Priors*.
- **arXiv:** `2605.24893v1`.
- **Venue:** CVPR 2026 Workshop on Computer Vision in the Wild poster.
- **Code:** https://github.com/TylerRust-1/BED-SAM2
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Salient object detection / camouflaged object detection / binary dense segmentation, not multi-class semantic segmentation.

## Methodology

BED-SAM2 adds a fourth-channel geometric prior to SAM2 Hiera encoder. It computes Sobel edge maps from RGB, raw depth, inverse depth, and centered/enhanced depth, then combines them into a cumulative structure map.

Depth enhancement:

```text
D'(x,y) = |D(x,y)-0.5| * 2
```

It replaces the SAM2 mask decoder with a U-Net style decoder.

Loss:

```text
L = L_IoU^w + L_BCE^w
L_total = sum_{i=1}^3 L(G, S_i)
```

## Novelty relative to RBMA / P29 / P30

BED-SAM2 is a useful SAM2 dense-prediction neighbor for geometric priors. It is not a reliability-bias method and not SAM2 memory cross-modal semantic segmentation.

## Limitations

- Binary segmentation, not semantic segmentation.
- Monocular depth priors can introduce spurious discontinuities.
- Uses U-Net decoder rather than stock SAM2 mask decoder.

## Ours application direction

P30 can cite BED-SAM2 as evidence that replacing/augmenting the SAM2 decoder is common in dense prediction, then justify class-token decoding from fused memory features as a multi-class extension.
