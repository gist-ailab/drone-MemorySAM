---
title: "M$^4$-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection"
tags: [source-note, weekly-sweep, sam_lora_adapter]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# M$^4$-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `sam_lora_adapter`
- Venue/year: arXiv cs.CV / 2026
- Date: 2026-05-12
- Primary URL: https://arxiv.org/abs/2605.11760v1
- arXiv/DOI: 2605.11760v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

SAM/SAM2 adaptation 및 adapter/LoRA 기반 dense prediction baseline 후보.

## Abstract / description snapshot

> The Segment Anything Model 2 (SAM2) has emerged as a foundation model for universal segmentation. Owing to its generalizable visual representations, SAM2 has been successfully applied to various downstream tasks. However, extending SAM2 to the RGB-D video salient object detection (RGB-D VSOD) task encounters three challenges including limited spatial modeling of linear LoRA, insufficient employment of SAM's multi-scale features, and dependence of initialization on explicit prompts. To address the issues, we present Multi-Modal Mixture-of-Experts with Memory-Augmented SAM (M$^4$-SAM), which equips SAM2 with modality-related PEFT, hierarchical feature fusion, and prompt-free memory initialization. Firstly, we inject Modality-Aware MoE-LORA, which employs convolutional experts to encode local spatial priors and introduces a modality dispatcher for efficient multi-modal fine-tuning, into SAM2's encoder. Secondly, we deploy Gated Multi-Level Feature Fusion, which hierarchically aggregates multi-scale encoder features with an adaptive gating mechanism, to balance spatial details and semantic context. Finally, to conduct zero-shot VSOD without manual prompts, we utilize a Pseudo-Guided Initialization, where a coarse mask is regarded as a pseudo prior and used to bootstrap the memory bank. Extensive experiments demonstrate that M$^4$-SAM achieves the state-of-the-art performance across all evaluation metrics on three public RGB-D VSOD datasets.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
