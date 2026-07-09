---
title: MambaFusion — Adaptive State-Space Fusion for Multimodal 3D Detection
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, mamba, uncertainty, reliability]
created: 2026-07-02
source: "[arXiv:2602.08126, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# MambaFusion — Adaptive State-Space Fusion for Multimodal 3D Detection

## Verification

- **Paper:** *MambaFusion: Adaptive State-Space Fusion for Multimodal 3D Object Detection*.
- **arXiv:** `2602.08126v2`.
- **Code:** no official code verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Camera-LiDAR BEV 3D object detection on nuScenes, Argoverse 2, and nuScenes-C corruption evaluation.

## Methodology

MambaFusion combines Mamba state-space blocks, windowed transformers, temporal camera BEV aggregation, token alignment, uncertainty modeling, and diffusion-style detection refinement.

The most relevant part is reliability-aware fusion. It computes spatial descriptors such as point density, camera depth variance, occlusion score, multi-view consistency, and ego distance, then predicts a gate:

```text
g(x,y) = sigmoid(Phi_gate([g_descriptor(x,y), Q_C_att(x,y), Q_L_att(x,y)]))
```

It also estimates modality log variance and fuses by inverse variance:

```text
Q_fused = (g Q_C_att/sigma_C^2 + (1-g) Q_L_att/sigma_L^2) / (g/sigma_C^2 + (1-g)/sigma_L^2 + eps)
```

## Novelty relative to RBMA/P29/P30

MambaFusion is the closest detector-side reliability formula among the gap-fill papers. It still performs feature-level gate/inverse-variance fusion after attention, not additive pre-softmax attention-logit bias.

## Limitations

- Many coupled components make causal attribution difficult.
- arXiv-only and code unavailable in checked source.
- Reliability descriptors are learned/trained, not training-free predictive entropy.

## Ours application direction

Use MambaFusion as the feature-level reliability baseline. Compare descriptor-based gate, predictive entropy, and RBMA logit bias in the same BEV/query attention framework.
