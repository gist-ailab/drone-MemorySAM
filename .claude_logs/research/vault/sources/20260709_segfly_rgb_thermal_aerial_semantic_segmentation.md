---
title: SegFly: A Dataset and 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale
tags: [source-candidate, weekly-sweep, multimodal_semantic_segmentation,_dataset,_rgb_thermal,_aerial]
created: 2026-07-09
source: https://arxiv.org/abs/2603.17920 ; https://github.com/markus-42/SegFly ; https://huggingface.co/datasets/markus-42/SegFly
status: verified-candidate
---

# SegFly: A Dataset and 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale

## Verification

- **Venue/year:** ECCV 2026 / arXiv cs.CV 2026
- **Primary source(s):** https://arxiv.org/abs/2603.17920 ; https://github.com/markus-42/SegFly ; https://huggingface.co/datasets/markus-42/SegFly
- **Verification status:** Verified arXiv metadata (2603.17920v2) and official GitHub README/API metadata on 2026-07-09; dataset link present in README.
- **Priority:** high

## Why this matters for [[00_MOC_26_MultimodalSeg]]

20k+ aerial RGB images and 15k+ aligned RGB-T pairs; official repo states ECCV 2026 acceptance and July 2026 HuggingFace dataset release. Strong benchmark/dataset candidate for RGB-T segmentation and scalable 2D-3D-2D label propagation.

## Project category

- multimodal_semantic_segmentation; dataset; rgb_thermal; aerial

## Connections

- [[sources/04_weekly_source_sweep_log]]
- [[relatedworks/90_clustered_relatedwork_synthesis]]
- [[PROJECT_TRACKING_26_MultimodalSeg]]

## Next extraction tasks

- Verify full paper PDF/project page details before citing exact numbers.
- Extract benchmark tables, datasets, metrics, and model components if this source becomes part of the final related-work matrix.
- Compare mechanism against RBMA/CGMoD positioning: reliability-aware attention bias, SAM2 memory adaptation, multimodal fusion, or efficient dense prediction as applicable.
