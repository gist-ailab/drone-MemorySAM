---
title: CLLAP — LiDAR-Augmented Contrastive Pretraining for Radar-Camera Fusion
tags: [relatedwork, object-detection, 3d-detection, multimodal-fusion, gap-fill, radar-camera, pretraining, contrastive-learning]
created: 2026-07-02
source: "[arXiv:2604.24044, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# CLLAP — LiDAR-Augmented Contrastive Pretraining for Radar-Camera Fusion

## Verification

- **Paper:** *CLLAP: Contrastive Learning-based LiDAR-Augmented Pretraining for Enhanced Radar-Camera Fusion*.
- **arXiv:** `2604.24044v1`.
- **Code:** no official code verified.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

Radar-camera 3D object detection with LiDAR used as privileged data during pretraining.

## Methodology

CLLAP converts LiDAR to pseudo-radar and uses dual-stage, dual-modality contrastive learning to improve radar-camera detector features. It includes local contrastive learning, global contrastive learning, sliding-window feature matching, and bidirectional channel-spatial attention.

InfoNCE-style local loss:

```text
L_local = -log exp(sim(z_i,z_i+)/tau) / sum_j exp(sim(z_i,z_j)/tau)
```

## Novelty relative to RBMA/P29/P30

CLLAP improves representation alignment through pretraining. RBMA improves online reliability-aware fusion decisions. They are complementary.

## Limitations

- Requires LiDAR-rich pretraining data.
- Pseudo-radar fidelity may not capture real radar noise/Doppler fully.
- No condition/reliability attention bias.

## Ours application direction

Use CLLAP as pretraining baseline and combine with RBMA at fusion time.
