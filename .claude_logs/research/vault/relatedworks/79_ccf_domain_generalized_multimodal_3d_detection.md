---
title: CCF — Complementary Collaborative Fusion for Domain-Generalized Multimodal 3D Detection
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, domain-generalization, query-balance]
created: 2026-07-02
source: "[arXiv:2603.23276, code: https://github.com/IMPL-Lab/CCF.git, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# CCF — Complementary Collaborative Fusion for Domain-Generalized Multimodal 3D Detection

## Verification

- **Paper:** *Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection*.
- **arXiv:** `2603.23276v1`.
- **Code:** https://github.com/IMPL-Lab/CCF.git
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Domain-generalized camera-LiDAR 3D detection, including rain/night/Boston domain shifts.

## Methodology

CCF addresses imbalance between LiDAR-originated 3D queries and image-originated 2D queries.

Key components:

| Component | Role |
|---|---|
| Query Decoupled Loss | reduces 2D/3D supervision imbalance |
| LiDAR-Guided Depth Prior | fuses image and LiDAR-derived depth distributions |
| Complementary Cross-Modal Masking | encourages modalities/queries to compete and complement |

Depth prior fusion:

```text
d_fused_i = softmax(lambda_i log d_2d_i + (1-lambda_i) log d_3d_i)
```

## Novelty relative to RBMA/P29/P30

CCF is not a reliability-logit-bias method, but it is important because detector extension must handle query-supervision imbalance. RBMA attention bias may need to be paired with query decoupled loss.

## Limitations

- Assumes dual-branch proposal-level detector.
- LiDAR-derived depth prior can fail under severe LiDAR degradation.
- Cross-modal masking is training-time augmentation.

## Ours application direction

For detector RBMA, evaluate gains separately for 2D-originated and 3D-originated queries. Combine RBMA with query-decoupled supervision as an ablation.
