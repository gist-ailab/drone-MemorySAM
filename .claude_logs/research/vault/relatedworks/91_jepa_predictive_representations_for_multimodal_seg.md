---
title: JEPA and Predictive Representation Models for Multimodal Segmentation
tags: [relatedwork, multimodal-segmentation, representation-learning, jepa, self-supervised-learning, foundation-model]
created: 2026-06-28
source: [arXiv:2301.08243, arXiv:2404.08471, arXiv:2506.09985]
status: draft-source-gated
---

# JEPA and Predictive Representation Models for Multimodal Segmentation

## Scope

This note tracks **Joint-Embedding Predictive Architecture (JEPA)** models and derivatives that may matter for [[../00_MOC_26_MultimodalSeg|26_MultimodalSeg]]. The focus is not that JEPA is already a semantic-segmentation method, but that it offers a useful pretraining principle for robust **latent representation prediction** across missing/corrupted visual evidence.

## Core idea

JEPA predicts **representations** rather than raw pixels. In an image/video setting, the model sees a context region and predicts the latent embedding of hidden target regions.

Compared with pixel reconstruction, this encourages semantic abstraction:

- ignore exact low-level texture when not needed,
- preserve object/scene information,
- learn representations useful for downstream tasks.

## Source map

SOURCE_TABLE_PLACEHOLDER

## Why this matters for multimodal segmentation

### 1. Feature-level prediction fits multimodal missing/corrupted sensors

Multimodal segmentation often has RGB, thermal, depth/LiDAR, or event streams. At test time, one modality may be noisy or unavailable. JEPA suggests a training objective:

> Given reliable context from some regions/modalities, predict the latent representation of hidden regions/modalities.

This can be adapted as:

- RGB context predicts thermal/depth/event latent targets,
- thermal/depth context predicts RGB-like semantic targets,
- multi-sensor context predicts masked spatial regions,
- teacher feature targets from SAM/SegFormer/DINO-style encoders are predicted rather than pixels.

### 2. JEPA is complementary to segmentation heads

JEPA itself does not output segmentation masks. It is best framed as **pretraining or auxiliary representation learning**. A segmentation decoder/head is still required.

Possible integration:

| Location | JEPA-style role |
|---|---|
| Backbone pretraining | Learn robust object/scene features before segmentation fine-tuning |
| Multimodal fusion encoder | Predict missing modality/region embeddings |
| Auxiliary loss | Encourage modality-invariant semantic latent consistency |
| Foundation-model adaptation | Add JEPA pretext on top of frozen or LoRA-tuned encoders |

### 3. Relation to current multimodal segmentation themes

| Current axis | JEPA relevance |
|---|---|
| Unimodal bias | Predicting masked targets from non-dominant modalities may reduce RGB over-reliance |
| Missing/corrupted modality robustness | JEPA naturally trains latent prediction under masking |
| Adapter/LoRA adaptation | JEPA loss can train adapters without full pixel-level reconstruction |
| Foundation models | I-JEPA/V-JEPA show non-generative predictive pretraining can produce useful visual features |

## Paper notes

### I-JEPA — arXiv:2301.08243

- Predicts target block embeddings from a context block in the same image.
- Avoids hand-crafted data augmentations, pixel reconstruction, text supervision, and negative pairs.
- Key design: large target blocks and appropriate masking guide the model toward semantic representations.
- For multimodal segmentation: use as an analogy for predicting **semantic latent blocks**, not pixels.

### V-JEPA — arXiv:2404.08471

- Extends feature prediction to video.
- Learns from public video data using a feature-prediction objective alone.
- Important for motion/dynamics and temporal consistency, relevant to event streams and sequential sensor fusion.

### V-JEPA 2 — arXiv:2506.09985

- Combines large-scale video/image pretraining with a small amount of robot trajectory data.
- Claims a bridge from passive observation to understanding, prediction, and planning.
- For multimodal segmentation: suggests pretrained video predictive representations could help dynamic scenes, but direct semantic segmentation gains require verification.

## Proposed use in 26_MultimodalSeg

1. **Literature framing**: JEPA belongs under self-supervised/foundation representation learning, not direct segmentation SOTA.
2. **Method idea**: add JEPA-style latent target prediction across modalities/regions.
3. **Ablation idea**:
   - baseline multimodal segmentation;
   - + masked spatial latent prediction;
   - + cross-modal latent prediction;
   - + reliability-gated JEPA targets.
4. **Claim boundary**: do not claim JEPA solves multimodal segmentation directly unless verified by a segmentation paper.

## Open verification tasks

- [ ] Search for direct “multimodal JEPA” papers beyond I-JEPA/V-JEPA/V-JEPA2.
- [ ] Verify whether any JEPA derivative reports semantic segmentation transfer numbers.
- [ ] Check if V-JEPA/Hiera features have public segmentation fine-tuning benchmarks.
- [ ] Compare with MAE, DINO, data2vec, ImageBind-style multimodal embedding methods.

## Ready-to-use related-work paragraph

Joint-Embedding Predictive Architectures (JEPAs) learn representations by predicting latent embeddings of masked targets from observed context rather than reconstructing pixels. I-JEPA demonstrates this principle for images, while V-JEPA and V-JEPA 2 extend feature prediction to video and physical-world prediction/planning settings. Although these models are not semantic segmentation methods by themselves, their latent prediction objective is relevant to multimodal segmentation: it suggests a way to train encoders to infer missing or corrupted modality/region representations without overemphasizing low-level reconstruction. In multimodal segmentation, a JEPA-style auxiliary loss can be positioned as representation-level robustness training complementary to segmentation heads and modality-fusion modules.



## Async source sweep addendum — derivative models and dense/multimodal relevance

The later source sweep added several JEPA-family or JEPA-adjacent papers that sharpen the multimodal segmentation interpretation.

| Year | Model / paper | Source | What it adds | Relevance to segmentation |
|---:|---|---|---|---|
| 2023 | MC-JEPA — *A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features* | arXiv:2307.12698 | Jointly learns motion/optical-flow-like features and content features | Stronger link to video segmentation, motion boundaries, moving-object perception |
| 2026 | V-JEPA 2.1 — *Unlocking Dense Features in Video Self-Supervised Learning* | arXiv:2603.14482 | Dense predictive loss, deep self-supervision, image/video unified training, multimodal tokenizers | Most directly relevant JEPA derivative for dense prediction; still needs mIoU/table verification |
| 2026 | JEPA-VLA — *Video Predictive Embedding is Needed for VLA Models* | arXiv:2602.11832 | Integrates predictive video embeddings, especially V-JEPA 2, into VLA policies | Useful if segmentation is part of a robot/VLA perception stack; not direct segmentation SOTA |

### Updated interpretation

For [[../00_MOC_26_MultimodalSeg|26_MultimodalSeg]], the strongest JEPA-related claims should be phrased carefully:

1. **I-JEPA** supports semantic image representation learning, but is not inherently multimodal or dense.
2. **V-JEPA** supports temporal/motion-aware representation learning, but most reported results are not segmentation mIoU.
3. **MC-JEPA** is important because it explicitly connects content and motion, and its abstract mentions semantic segmentation of images/videos as a downstream evaluation axis.
4. **V-JEPA 2.1** is the most relevant derivative for dense features; if used in related work, verify actual benchmark tables before claiming segmentation superiority.
5. For RGB-depth/thermal/event/LiDAR segmentation, JEPA is still best described as a **latent prediction principle**: predict missing or future modality/region embeddings, then attach a task-specific segmentation decoder.

### Claim boundary

Use wording such as:

> JEPA-family models motivate representation-level prediction objectives for robust multimodal perception, especially under spatial masking, temporal prediction, and missing-modality conditions. However, most JEPA results are reported as representation, video-understanding, or planning benchmarks rather than direct multimodal semantic segmentation SOTA; dense segmentation gains must be verified model-by-model.


## References

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. arXiv:2301.08243.
- Bardes, A. et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471.
- Assran, M. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985.
