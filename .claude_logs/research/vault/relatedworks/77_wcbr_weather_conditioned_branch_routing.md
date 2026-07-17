---
title: WCBR — Weather-Conditioned Branch Routing for LiDAR-Radar 3D Detection
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, weather-routing, lidar-radar]
created: 2026-07-02
source: "[arXiv:2604.05405, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# WCBR — Weather-Conditioned Branch Routing for LiDAR-Radar 3D Detection

## Verification

- **Paper:** *Weather-Conditioned Branch Routing for Robust LiDAR-Radar 3D Object Detection*.
- **arXiv:** `2604.05405v1`.
- **Code:** source code promised but no released URL verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task / modalities / datasets

Adverse-weather 3D object detection on K-Radar using LiDAR + 4D radar. A front-view camera image is used to infer a weather/condition token.

## Methodology

WCBR has three branches: LiDAR-only, radar-only, and condition-gated fusion. A visual condition token is extracted from a front-view image, and a semantic weather token is built with CLIP text prompts over seven weather classes.

Semantic weather vector:

```text
alpha = softmax(p W^T) in R^7
c_p = Proj(alpha W) in R^512
```

A two-layer MLP predicts branch routing weights:

```text
w = [w_L, w_R, w_F]
```

These weights aggregate the three branch BEV features before the detection head. The paper also uses auxiliary weather classification and diversity regularization to prevent branch collapse.

## Novelty relative to RBMA/P29/P30

WCBR is a strong P29 near-neighbor because it explicitly uses weather-conditioned routing. The difference is that WCBR is supervised/prompt-vocabulary based and branch-level, while P29 should be scoped as unsupervised condition latent / entropy-derived routing and possibly attention-logit-level modulation.

## Limitations

- Depends on weather taxonomy and prompt vocabulary.
- Uses camera for condition extraction rather than purely sensor reliability.
- Unknown/unlabeled conditions are not the core claim.

## Ours application direction

Use WCBR as the strongest detector-domain condition-router baseline. Ablate branch-level routing vs token/query-level RBMA bias.
