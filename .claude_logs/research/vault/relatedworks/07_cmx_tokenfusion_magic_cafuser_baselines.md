---
title: CMX, TokenFusion, MAGIC++, CAFuser — Core Multimodal Fusion Baselines
aliases: [CMX, TokenFusion, MAGIC, MAGIC++, CAFuser]
tags: [related-work, baseline, multimodal-fusion, rgb-x, token-fusion]
created: 2026-06-24
source: arXiv:2203.04838, 2204.08721, 2412.16876, 2410.10791; downloaded PDFs in [[sources/pdfs/priority_a]]
status: verified-draft
---

# CMX, TokenFusion, MAGIC++, CAFuser — Core Multimodal Fusion Baselines

## Purpose

This note groups several high-importance baselines for RGB-X semantic segmentation and multimodal fusion. These are not all direct competitors to RBMA, but they define the reviewer’s expected background.

## CMX — Cross-Modal Fusion for RGB-X Semantic Segmentation

Source: Zhang et al., **“CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers,”** arXiv:2203.04838; IEEE TITS 2023. PDF archived at [[sources/pdfs/priority_a/2203.04838.pdf]].

Verified from PDF:

- CMX targets RGB-X segmentation across RGB-depth, RGB-thermal, RGB-polarization, RGB-event, and RGB-LiDAR combinations.
- It proposes a unified cross-modal fusion framework.
- It uses a Cross-Modal Feature Rectification Module (CM-FRM) and Feature Fusion Module (FFM) to calibrate/fuse bi-modal features.

Relevance:

- Strong baseline for RGB-X semantic segmentation.
- Useful for framing fixed RGB-X fusion vs arbitrary/memory-based multimodal fusion.

## TokenFusion — Multimodal Token Fusion for Vision Transformers

Source: Wang et al., **“Multimodal Token Fusion for Vision Transformers,”** CVPR 2022, arXiv:2204.08721. PDF archived at [[sources/pdfs/priority_a/2204.08721.pdf]].

Verified from PDF:

- It studies multimodal inputs in vision transformers.
- It observes that naive multimodal feeding can dilute intra-modal attention weights.
- It proposes token-level fusion for transformers.

Relevance:

- Important transformer-era baseline for multimodal token fusion.
- Useful contrast to SAM2 memory attention: TokenFusion fuses tokens inside a ViT pipeline, whereas RBMA biases memory attention over modality tokens.

## MAGIC++ — Hierarchical Modality Selection

Source: Zheng et al., **“MAGIC++: Efficient and Resilient Modality-Agnostic Semantic Segmentation via Hierarchical Modality Selection,”** arXiv:2412.16876. PDF archived at [[sources/pdfs/priority_a/2412.16876.pdf]].

Verified from PDF:

- MAGIC++ addresses modality-agnostic semantic segmentation.
- It argues that RGB-centered asymmetric architectures are limiting, especially at night when non-RGB modalities may be more reliable.
- It introduces hierarchical modality selection and multimodal interaction.
- It compares MAGIC and MAGIC++ on DeLiVER and MUSES.

Relevance:

- Strong baseline for arbitrary / modality-agnostic segmentation.
- Important for positioning our method against modality selection approaches.

## CAFuser — Condition-Aware Multimodal Fusion

Source: Brödermann et al., **“CAFuser: Condition-Aware Multimodal Fusion for Robust Semantic Perception of Driving Scenes,”** arXiv:2410.10791; IEEE RA-L 2025. PDF archived at [[sources/pdfs/priority_a/2410.10791.pdf]].

Verified from PDF:

- CAFuser uses RGB input to classify environmental conditions and generate a **Condition Token**.
- The condition token guides multimodal fusion.
- It introduces modality-specific feature adapters to align diverse sensor inputs into a shared latent space with a single pretrained backbone.
- It reports robust semantic perception of driving scenes and sets a state of the art on DeLiVER according to the abstract text.

Relevance:

- Very relevant to our adapter + condition/reliability fusion story.
- DGFusion extends this line by adding depth-guided local tokens.

## Comparison matrix

| Method | Fusion level | Reliability / adaptivity | Modalities | Direct gap for RBMA |
|---|---|---|---|---|
| CMX | feature rectification + feature fusion | cross-modal calibration, not explicit uncertainty | RGB-X | no SAM2 memory; no logit reliability bias |
| TokenFusion | token-level ViT fusion | token fusion, not predictive reliability | multimodal ViT inputs | no semantic uncertainty signal |
| MAGIC++ | hierarchical modality selection | modality selection for arbitrary modality availability | multi-modal / anymodal | selection rather than memory-attention logit bias |
| CAFuser | condition-aware fusion + adapters | global condition token | driving sensors | global condition, not local predictive reliability |
| DGFusion | condition + local depth tokens | depth-guided spatial reliability | camera/LiDAR/depth | depth-specific; not SAM2 memory |
| RBMA | SAM2 memory-attention logits | predictive reliability as additive pre-softmax bias | RGB/Thermal/Event/LiDAR | proposed contribution |

## Related-work paragraph candidate

RGB-X semantic segmentation has been advanced by transformer-based fusion methods such as CMX, which rectifies and fuses RGB and auxiliary modality features, and TokenFusion, which performs multimodal token fusion in vision transformers. More recent modality-agnostic methods such as MAGIC++ select modalities hierarchically, while CAFuser conditions fusion on a scene-level environmental token and uses modality-specific adapters. These methods show that multimodal fusion must adapt to sensor type and environmental condition, but they typically operate at the feature, token, or modality-selection level. Our work instead targets SAM2-style memory attention and injects reliability as a pre-softmax logit bias.
