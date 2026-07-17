---
title: RadarXFormer — Cross-Dimension Fusion of 4D Radar Spectra and Images
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, radar-camera, deformable-attention]
created: 2026-07-02
source: "[arXiv:2603.14822, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# RadarXFormer — Cross-Dimension Fusion of 4D Radar Spectra and Images

## Verification

- **Paper:** *RadarXFormer: Robust Object Detection via Cross-Dimension Fusion of 4D Radar Spectra and Images for Autonomous Driving*.
- **arXiv:** `2603.14822v1`.
- **Code:** no official code release verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Adverse-weather 3D object detection with 4D mmWave radar spectrum and camera images, evaluated on K-Radar.

## Methodology

Unlike radar-camera methods using sparse radar point clouds, RadarXFormer uses pre-CFAR 4D radar spectra. It projects object-query reference points into both image feature maps and radar feature cubes and applies multi-scale deformable cross-attention.

Verified attention form:

```text
MSDA(f_q, p_hat_q, {x^l}) = sum_m W_m (sum_l sum_k A_mlqk W'_m x^l(phi_l(p_hat_q)+Delta p_mlqk))
```

## Novelty relative to RBMA/P29/P30

RadarXFormer's robustness comes from radar spectrum representation and cross-dimension deformable attention. It does not use explicit reliability bias or condition routing.

## Limitations

- Requires raw 4D radar spectrum access.
- Sensor-specific and data-heavy.
- No explicit uncertainty calibration.

## Ours application direction

For radar-camera detection extension, inject RBMA-style reliability bias into deformable attention weights or reference-point query attention.
