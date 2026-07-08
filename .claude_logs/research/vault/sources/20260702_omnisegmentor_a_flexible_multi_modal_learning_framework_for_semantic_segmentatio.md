---
title: "OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation"
tags: [source-note, weekly-sweep, multimodal_semantic_segmentation]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_semantic_segmentation`
- Venue/year: arXiv cs.CV / 2025
- Date: 2025-09-18
- Primary URL: https://arxiv.org/abs/2509.15096v1
- arXiv/DOI: 2509.15096v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

프로젝트 핵심 범위와 관련된 신규/누락 source candidate.

## Abstract / description snapshot

> Recent research on representation learning has proved the merits of multi-modal clues for robust semantic segmentation. Nevertheless, a flexible pretrain-and-finetune pipeline for multiple visual modalities remains unexplored. In this paper, we propose a novel multi-modal learning framework, termed OmniSegmentor. It has two key innovations: 1) Based on ImageNet, we assemble a large-scale dataset for multi-modal pretraining, called ImageNeXt, which contains five popular visual modalities. 2) We provide an efficient pretraining manner to endow the model with the capacity to encode different modality information in the ImageNeXt. For the first time, we introduce a universal multi-modal pretraining framework that consistently amplifies the model's perceptual capabilities across various scenarios, regardless of the arbitrary combination of the involved modalities. Remarkably, our OmniSegmentor achieves new state-of-the-art records on a wide range of multi-modal semantic segmentation datasets, including NYU Depthv2, EventScape, MFNet, DeLiVER, SUNRGBD, and KITTI-360.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
