---
title: MULTIAQUA — Maritime Multimodal Dataset and Robust Training Strategies
tags: [relatedwork, dataset, maritime, semantic-segmentation, rgb-thermal-lidar, robustness, gap-fill]
created: 2026-07-02
source: "[arXiv:2512.17450, dataset: https://lmi.fe.uni-lj.si/en/MULTIAQUA, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# MULTIAQUA — Maritime Multimodal Dataset and Robust Training Strategies

## Citation / verification

- **Paper:** *MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation*.
- **arXiv:** `2512.17450v1`.
- **Dataset page:** https://lmi.fe.uni-lj.si/en/MULTIAQUA
- **GitHub/page repo:** https://github.com/JonNatanael/MULTIAQUA-dataset-page
- **Verification tag:** `[VERIFIED-PDF]`.

## Task and modalities

Maritime semantic segmentation for unmanned surface vehicles. The dataset includes synchronized/calibrated multimodal data such as RGB, thermal, IR, and LiDAR.

## Methodology

The paper proposes robust training for nighttime maritime perception. It uses full-modality and RGB-zeroed passes to force auxiliary modalities to carry useful semantics.

Full input loss:

```text
L_f = CE(Z_GT, Z_ITL) + CE(Z_GT, Z_I) + CE(Z_GT, Z_A)
```

RGB-zeroed loss:

```text
L_s = CE(Z_GT, Z(empty, X_T, X_L)) + CE(Z_GT, Z_A)
```

Total:

```text
L = L_f + L_s
```

This reduces RGB dominance and encourages thermal/LiDAR to work under night conditions.

## Novelty relative to RBMA / P29 / P30

MULTIAQUA is training-time robustness through RGB masking and auxiliary heads. RBMA is inference-time reliability bias. P29 can use day/night/maritime visibility as condition-prototype structure. P30 can measure whether fused memory features remain class-separable when RGB is zeroed.

## Limitations

- Maritime-specific domain.
- Requires supervised segmentation annotations.
- RGB zeroing improves missing RGB but does not directly calibrate noisy RGB reliability.

## Ours application direction

Use MULTIAQUA as a benchmark/domain to demonstrate that RBMA can shift attention away from unreliable/night RGB toward thermal/LiDAR memory. The paper's RGB-zero training is a strong complementary baseline.
