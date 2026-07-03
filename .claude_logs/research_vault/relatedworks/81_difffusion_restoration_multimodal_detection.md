---
title: DiffFusion — Diffusion-Based Restoration for Multimodal 3D Detection in Adverse Weather
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, diffusion, restoration, adverse-weather]
created: 2026-07-02
source: "[arXiv:2512.13107, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# DiffFusion — Diffusion-Based Restoration for Multimodal 3D Detection in Adverse Weather

## Verification

- **Paper:** *Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather* / DiffFusion.
- **arXiv:** `2512.13107v2`.
- **Code:** implementation promised but not verified as released.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Adverse-weather camera-LiDAR 3D detection on KITTI, KITTI-C, and DENSE zero-shot settings.

## Methodology

DiffFusion uses two main modules:

1. **Diffusion-based Restoration Module**
   - Diffusion-IR restores degraded images.
   - PCR restores/completes point clouds using 2D boxes and depth.

2. **BAFAM — Bidirectional Adaptive Fusion and Alignment Module**
   - Cross-attention adaptive fusion between camera and LiDAR BEV features.
   - Bidirectional offset learning for BEV alignment.

DDPM forward process:

```text
q(x_t | x_c) = N(x_t; sqrt(alpha_bar_t) x_c, (1-alpha_bar_t) I)
```

Camera-LiDAR cross-attention is standard:

```text
A(Q,K,V) = softmax(QK^T/sqrt(d)) V
```

## Novelty relative to RBMA/P29/P30

DiffFusion attacks adverse weather through restoration and alignment, not reliability-logit bias. It is complementary but computationally heavier.

## Limitations

- Diffusion restoration may be slow.
- No explicit condition router or entropy reliability signal verified.
- Not semantic segmentation.

## Ours application direction

Position DiffFusion under restoration/alignment methods. RBMA can be combined after restoration to decide how much to trust restored camera vs LiDAR features.
