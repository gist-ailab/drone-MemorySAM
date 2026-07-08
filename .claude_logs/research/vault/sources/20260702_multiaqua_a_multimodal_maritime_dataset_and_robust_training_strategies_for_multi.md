---
title: "MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation"
tags: [source-note, weekly-sweep, multimodal_semantic_segmentation]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_semantic_segmentation`
- Venue/year: arXiv cs.CV,cs.LG / 2025
- Date: 2025-12-19
- Primary URL: https://arxiv.org/abs/2512.17450v1
- arXiv/DOI: 2512.17450v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

멀티모달 semantic segmentation 데이터셋/robust training 후보.

## Abstract / description snapshot

> Unmanned surface vehicles can encounter a number of varied visual circumstances during operation, some of which can be very difficult to interpret. While most cases can be solved only using color camera images, some weather and lighting conditions require additional information. To expand the available maritime data, we present a novel multimodal maritime dataset MULTIAQUA (Multimodal Aquatic Dataset). Our dataset contains synchronized, calibrated and annotated data captured by sensors of different modalities, such as RGB, thermal, IR, LIDAR, etc. The dataset is aimed at developing supervised methods that can extract useful information from these modalities in order to provide a high quality of scene interpretation regardless of potentially poor visibility conditions. To illustrate the benefits of the proposed dataset, we evaluate several multimodal methods on our difficult nighttime test set. We present training approaches that enable multimodal methods to be trained in a more robust way, thus enabling them to retain reliable performance even in near-complete darkness. Our approach allows for training a robust deep neural network only using daytime images, thus significantly simplifying data acquisition, annotation, and the training process.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
