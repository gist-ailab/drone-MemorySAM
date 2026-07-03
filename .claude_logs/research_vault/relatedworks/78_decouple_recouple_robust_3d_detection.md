---
title: Multi-Modal Decouple and Recouple Network for Robust 3D Object Detection
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, corruption, expert-router]
created: 2026-07-02
source: "[arXiv:2603.07486, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# Multi-Modal Decouple and Recouple Network for Robust 3D Object Detection

## Verification

- **Paper:** *Multi-Modal Decouple and Recouple Network for Robust 3D Object Detection*.
- **arXiv:** `2603.07486v1`.
- **Code:** no official code release verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Robust camera-LiDAR BEV 3D detection under camera/LiDAR/both corruption. Training is on clean nuScenes and testing uses Robo3D/RoboBEV-style corruptions.

## Methodology

The method first decouples modality-invariant and modality-specific features with deformable attention, then recouples them through self-attention and cross-attention.

A router predicts expert weights for three enhanced BEV features:

```text
W = Softmax(Router([F_c, F_l])) = [W_ec, W_el, W_ef]
F_out = W_ec F_ec + W_el F_el + W_ef F_ef
```

Entropy regularization prevents collapse:

```text
L_reg = - sum_i W_i log W_i
```

## Novelty relative to RBMA/P29/P30

This is an implicit corruption-adaptive expert router. It is feature/expert-level and learned, not attention-logit bias. It is a useful P29/P30 detector baseline.

## Limitations

- Synthetic corruption benchmark focus.
- Severe fog where all modalities fail remains difficult.
- Learned router and expert fusion are heavier than RBMA.

## Ours application direction

Compare RBMA detector bias with expert weighted-sum routing. The entropy regularization idea can inform P29 router anti-collapse loss.
