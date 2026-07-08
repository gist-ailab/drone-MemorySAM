---
title: StitchFusion — MultiAdapter Weaving for Multimodal Semantic Segmentation
tags: [related-work, adapter, multimodal-segmentation, parameter-efficient]
created: 2026-06-24
source: arXiv:2408.01343; downloaded PDF text in [[sources/pdfs/priority_a/text/2408.01343.txt]]
status: verified-draft
---

# StitchFusion — Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation

## Citation

Li et al., **“StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation,”** arXiv:2408.01343, 2024/2025 version. PDF archived at [[sources/pdfs/priority_a/2408.01343.pdf]].

## Problem setting

StitchFusion studies multimodal semantic segmentation with arbitrary visual modalities. Its motivation is that separate pretrained encoders contain useful modality-specific representations, but late fusion may not sufficiently align multiscale information.

## Verified method summary

The PDF text describes a **StitchFusion framework** with a lightweight low-rank adaptation module used as a modality-of-adapter / MultiAdapter component. The paper reports comparison on NYUDv2, DeLiVER, MCubeS, and SUN. The imported memo notes that the module is inserted around transformer blocks to weave features between encoders.

## Novelty

The contribution is early/mid-encoder **feature weaving**: instead of only fusing after encoders, StitchFusion propagates information between modality-specific pretrained encoders using lightweight adapter modules.

## Main claims

- Adapter-style cross-modal weaving can improve multimodal semantic segmentation.
- Pretrained encoders should retain their own modality modeling capacity.
- Parameter-efficient adapters can provide cross-modal communication without expensive full fusion modules.

## Why it matters for our project

StitchFusion is the key adapter-side related work for our LoRA/SAM2 direction. It supports the argument that adapter modules are a natural way to adapt pretrained visual backbones to multimodal inputs.

## Limitations / gaps for our project

- It focuses on encoder-level feature exchange, not SAM2 memory attention.
- It does not directly model input-dependent reliability as an attention-logit bias.
- It is complementary to RBMA: adapters can adapt features, while RBMA controls fusion attention.

## Related-work paragraph candidate

StitchFusion shows that lightweight adapter modules can weave information between pretrained modality encoders during feature extraction, improving multimodal semantic segmentation without relying solely on late fusion. This adapter-based strategy is closely related to LoRA and other parameter-efficient adaptation methods for foundation models. Our work follows the same parameter-efficient philosophy but places the key multimodal reasoning step in SAM2-style memory attention, where reliability can bias cross-modality token selection.
