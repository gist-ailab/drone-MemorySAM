---
title: Detector Gap-Fill — Adaptive Fusion for Multimodal 3D Detection after 2025-06
tags: [relatedwork, object-detection, 3d-detection, adaptive-fusion, reliability, adverse-weather, gap-fill]
created: 2026-07-02
source: "[sources: arXiv 2508.16408,2604.05405,2603.07486,2603.23276,2602.08126,2512.13107,2603.14822,2604.24044,2511.10035, status: mixed verified] "
status: gap-fill-verified
---

# Detector Gap-Fill — Adaptive Fusion for Multimodal 3D Detection after 2025-06

## Scope

This note consolidates Track C gap-fill papers for extending RBMA/P29/P30 from semantic segmentation to object/3D detection. It avoids duplicating older notes on BEVFusion, TransFusion, DeepInteraction, FUTR3D, ReliFusion, ModalPatch, and AW-MoE.

## Papers verified by the gap-fill run

| Paper | arXiv | Modalities | Mechanism | Reliability affects logits? | Status |
|---|---|---|---|---|---|
| SAMFusion | 2508.16408 | RGB, LiDAR, NIR gated, radar | distance/visibility-aware fusion stack | No | [VERIFIED-PDF] |
| WCBR | 2604.05405 | LiDAR, 4D radar, camera condition | weather token -> branch weights | No | [VERIFIED-PDF] |
| Multi-Modal Decouple and Recouple Network | 2603.07486 | camera, LiDAR | decouple/recouple + expert router | No | [VERIFIED-PDF] |
| CCF | 2603.23276 | camera, LiDAR | query decoupled loss + depth prior + masking | No | [VERIFIED-PDF] |
| MambaFusion | 2602.08126 | camera, LiDAR | spatial reliability gate + inverse variance fusion | No | [VERIFIED-PDF] |
| DiffFusion | 2512.13107 | camera, LiDAR | diffusion restoration + cross-attention alignment | No | [VERIFIED-PDF] |
| RadarXFormer | 2603.14822 | radar spectrum, camera | cross-dimension deformable attention | No | [VERIFIED-PDF] |
| CLLAP | 2604.24044 | LiDAR pretrain, radar-camera finetune | contrastive pretraining | No | [VERIFIED-PDF] |
| DGFusion | 2511.10035 | camera, LiDAR | hard-instance dual-guided fusion | No | [VERIFIED-PDF] |
| PEFT-DML | 2512.00060 | LiDAR/radar/camera/IMU/GNSS claimed | PEFT + metric learning | unclear | [ABSTRACT-ONLY] |

## Most important near-neighbors

### WCBR — Weather-conditioned branch routing

WCBR uses a weather token derived from a camera image and CLIP text prompts. A two-layer MLP predicts branch weights:

```text
w = [w_L, w_R, w_F]
```

These weights aggregate LiDAR-only, radar-only, and fusion branch BEV features. It is a strong P29 near-neighbor, but it uses supervised weather/prompt structure and branch-level weighted sum rather than label-free condition latent or attention-logit bias.

### MambaFusion — spatial reliability gate

MambaFusion has the closest reliability-fusion formula. It uses descriptors such as point density, camera depth variance, occlusion score, multi-view consistency, and distance to compute a spatial gate and inverse variance fusion:

```text
Q_fused = (g Q_C/sigma_C^2 + (1-g) Q_L/sigma_L^2) / (g/sigma_C^2 + (1-g)/sigma_L^2 + eps)
```

This is reliability-aware feature fusion, not additive pre-softmax attention-logit bias.

### CCF — query imbalance under domain shift

CCF corrects 2D/3D query supervision imbalance and fuses image/LiDAR depth distributions in log space:

```text
d_fused_i = softmax(lambda_i log d_2d_i + (1-lambda_i) log d_3d_i)
```

This is useful for RBMA detector extension because attention bias alone may not solve query-supervision imbalance.

## Cross-paper conclusion

After the targeted search, the detector literature has many forms of adaptive weighting:

- branch/expert weighted sums;
- feature-level gates;
- inverse-variance feature fusion;
- restoration/alignment;
- contrastive pretraining;
- query supervision balancing.

However, the gap-fill run did **not** verify a 2025-06~2026-07 detector paper that injects reliability/uncertainty/condition directly as an additive pre-softmax bias into detector attention logits. This should be stated cautiously as “in our targeted search,” not as an absolute universal negative.

## Ours application direction

For a detection extension, implement RBMA in BEV/query/deformable attention:

```text
attention_logits = QK^T/sqrt(d) + lambda * reliability_bias
```

Compare against:

1. WCBR-style branch weights.
2. MambaFusion-style feature-level inverse variance fusion.
3. Decouple/Recouple expert weighted sum.
4. CCF query decoupled loss.
5. DGFusion hard-instance stratified evaluation.

The novelty should be scoped as **token/query-level competition control through reliability-derived logit bias**, not generic adaptive fusion.
