---
title: "ClustViT: Clustering-based Token Merging for Semantic Segmentation"
tags: [source-note, weekly-sweep, token_efficiency]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# ClustViT: Clustering-based Token Merging for Semantic Segmentation

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `token_efficiency`
- Venue/year: arXiv cs.CV / 2025
- Date: 2025-10-02
- Primary URL: https://arxiv.org/abs/2510.01948v2
- arXiv/DOI: 2510.01948v2
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

dense prediction에서 token pruning/merging 효율화 축의 신규/누락 후보.

## Abstract / description snapshot

> Vision Transformers can achieve high accuracy and strong generalization across various contexts, but their practical applicability on real-world robotic systems is limited due to their quadratic attention complexity. Recent works have focused on dynamically merging tokens according to the image complexity. Token merging works well for classification but is less suited to dense prediction. We propose ClustViT, where we expand upon the Vision Transformer (ViT) backbone and address semantic segmentation. Within our architecture, a trainable Cluster module merges similar tokens along the network guided by pseudo-clusters from segmentation masks. Subsequently, a Regenerator module restores fine details for downstream heads. Our approach achieves up to 2.18x fewer GFLOPs and 1.64x faster inference on three different datasets, with comparable segmentation accuracy. Our code and models will be made publicly available.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
