---
title: SAMFusion — Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, adverse-weather]
created: 2026-07-02
source: "[arXiv:2508.16408, project: https://light.princeton.edu/samfusion/, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# SAMFusion — Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather

## Verification

- **Paper:** *SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather*.
- **arXiv:** `2508.16408v1`.
- **Project:** https://light.princeton.edu/samfusion/
- **Code:** no GitHub code link verified in the checked full text.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task / modalities / datasets

SAMFusion targets adverse-weather 3D object detection using RGB camera, LiDAR, NIR gated camera, and radar. It evaluates daytime/nighttime, rain/snow/fog/twilight/night settings and reports KITTI-style 3D-AP/BEV-AP.

## Methodology

The method builds a four-sensor adverse-weather fusion stack. Encoder-level early fusion uses local cross-modal attention. A verified equation uses camera/gated features as query and sampled LiDAR context as key/value:

```text
phi_CG* = sum_{phi_L,CG in J_s} softmax(phi_CG phi_L,CG^T / sqrt(d)) phi_L,CG
```

It also performs intra-modal attention in parallel and later combines image/range features in BEV through learned refinement. The transformer decoder weighs modalities according to distance and visibility.

## Novelty relative to RBMA/P29/P30

SAMFusion is a strong adverse-weather detection baseline, but it does not inject reliability as additive pre-softmax attention-logit bias. Its adaptation is mainly sensor-stack design plus learned visibility/distance-aware weighting.

## Limitations

- Requires four sensors, increasing deployment cost.
- Adverse-weather uncertainty propagation is mentioned as future work.
- Not semantic segmentation and not SAM2 memory attention.

## Ours application direction

For detection extension, compare RBMA against SAMFusion-style distance/visibility weighting. Our distinction: reliability-derived bias controls token/query competition directly inside attention logits.
