---
title: CrossWeaver — Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation
tags: [relatedwork, arbitrary-modality, semantic-segmentation, reliability, gap-fill]
created: 2026-07-02
source: "[arXiv:2604.02948, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# CrossWeaver — Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation

## Citation / verification

- **Paper:** *CrossWeaver: Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation*.
- **arXiv:** `2604.02948v2`.
- **Status:** arXiv only in checked sources.
- **Code:** no official code URL found in checked source.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task and datasets

Arbitrary-modality semantic segmentation on MCubeS, DeLiVER, and MUSES.

## Methodology

CrossWeaver introduces:

| Module | Function |
|---|---|
| MIB — Modality Interaction Block | token-adaptive cross-modal interaction in the encoder |
| SAF — Seam-Aligned Fusion | spatially coherent aggregation of enhanced features |

The method tries to distinguish reliable complementary cues from noisy/redundant information through learned reliability-aware interaction.

## Implementation details verified

- Hardware: 8× RTX 4090.
- Optimizer: AdamW.
- LR: `6e-5`.
- Weight decay: `1e-2`.
- Training: 200 epochs, 10-epoch warm-up.
- Crop: DeLiVER 1024×1024, MCubeS 512×512.
- Backbone: SegFormer-B0 / MiT-B0.

## Reported results

| Dataset | mIoU |
|---|---:|
| MCubeS | 48.76 |
| DeLiVER | 64.69 |
| MUSES | 53.19 |

Reported gains over strongest available baselines are +0.52 / +0.66 / +0.88.

## Novelty relative to RBMA / P29 / P30

CrossWeaver is a strong arbitrary-modality baseline. It uses learned token-adaptive interaction, not training-free entropy logit bias and not SAM2 memory attention. Therefore it is a competitor for performance, but not a direct novelty blocker for RBMA.

## Limitations

- Learned fusion module, not training-free.
- No SAM2 memory mechanism.
- Small reported margins mean protocol and backbone matching are crucial.

## Ours application direction

Use CrossWeaver as a recent arbitrary-modality baseline. In writing, distinguish:

```text
CrossWeaver: learned encoder interaction
RBMA: reliability bias inside SAM2 memory cross-attention logits
```
