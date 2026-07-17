---
title: DGFusion — Dual-Guided Fusion for Robust Multimodal 3D Object Detection
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, hard-instances, lidar-camera]
created: 2026-07-02
source: "[arXiv:2511.10035, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# DGFusion — Dual-Guided Fusion for Robust Multimodal 3D Object Detection

## Verification

- **Paper:** *DGFusion: Dual-guided Fusion for Robust Multi-Modal 3D Object Detection*.
- **arXiv:** `2511.10035v1`.
- **Code:** no official code verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Camera-LiDAR 3D object detection on nuScenes, focused on hard instances such as distant, small, and occluded objects.

## Methodology

DGFusion argues that point-guide-image and image-guide-point fusion each fail on different hard-instance regimes. It uses:

| Module | Role |
|---|---|
| Instance Match Modules | produce camera/LiDAR proposals and instance-level features |
| DIPM | difficulty-aware instance pair matching |
| Dual-guided modules | let easy/hard pairs guide feature fusion |

Instance proposal scores below threshold gamma are discarded. Key sample positions around boxes are used to form instance-level features.

## Novelty relative to RBMA/P29/P30

DGFusion is not condition-adaptive or reliability-logit-bias. It is important because hard-instance modality imbalance can exist even without weather corruption.

## Limitations

- Instance-level feature quality limits performance.
- Instance feature generation is the largest extra time cost.
- Not designed for missing modality or adverse weather specifically.

## Ours application direction

Evaluate RBMA detector extension on hard subsets: long distance, small objects, occlusion, low LiDAR point count. Use DGFusion as a hard-instance fusion comparator.
