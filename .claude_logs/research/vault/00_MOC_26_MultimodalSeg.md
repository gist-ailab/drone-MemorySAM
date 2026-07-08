---
title: 26_MultimodalSeg — Map of Content
tags: [moc, project, multimodal-segmentation, semantic-segmentation, detection, obsidian]
created: 2026-06-24
source: user project brief + imported Claude memo + first-pass API source search
status: active
---

# 26_MultimodalSeg — Map of Content

## Research goal

Write a paper on **semantic segmentation using multimodal sensor data** — RGB, thermal, event, LiDAR/depth — with an applied emphasis on solving object-detection/perception tasks using multiple sensors.

The current conceptual center is robust multimodal fusion under sensor unreliability: daytime/RGB-dominant training can collapse at night or adverse conditions, so the system needs input-adaptive reliability handling rather than static modality averaging.

## Core project folders

| Folder | Role |
|---|---|
| [[sources]] | Raw and first-pass source collection: papers, datasets, API search results, copied notes |
| [[relatedworks]] | Per-paper research synthesis: novelty, claims, method, figures, limitations, comparisons |
| [[material]] | Study materials and PDF-ready explanations, always English + Korean versions |

## Current notes

- [[PROJECT_TRACKING_26_MultimodalSeg]] — live project status board
- [[sources/00_imported_claude_related_work_2026-06-24]] — imported Claude/NotebookLM memo from the user
- [[sources/01_source_index_multimodal_segmentation]] — first-pass Semantic Scholar paper/source index
- [[sources/02_openalex_top_venue_literature_database]] — expanded top-conference / vision-journal literature database
- [[sources/03_seed_paper_verification_candidates]] — candidate metadata matches for DGFusion, MemorySAM, StitchFusion, AnySeg, DeLiVER/MUSES/MCubeS, and key baselines
- [[sources/02_source_map_multimodal_semantic_segmentation]] — category source map
- [[sources/02_source_map_multimodal_object_detection]] — category source map
- [[sources/02_source_map_adapter_lora_foundation_seg_det]] — category source map
- [[sources/02_source_map_segmentation_detection_heads]] — category source map
- [[relatedworks/00_relatedworks_index]] — planned related-work synthesis index

## Key research axes

### A. Multimodal semantic segmentation datasets and benchmarks

- [[DELIVER]] / DeLiVER
- [[MUSES]]
- [[MCubeS]]
- RGB-T urban semantic segmentation datasets
- RGB-D semantic segmentation datasets
- RGB-event and event-camera segmentation datasets
- LiDAR-camera semantic segmentation datasets

### B. Robust multimodal fusion methods

- [[DGFusion]]
- [[StitchFusion]]
- [[AnySeg]]
- [[MemorySAM]]
- [[MAGIC]] / [[MAGIC++]]
- [[CMX]]
- [[TokenFusion]]
- [[CAFuser]]
- [[UTFNet]]
- [[HyperDUM]]
- [[ReliFusion]] / [[READ]] / confidence-based baselines

### C. Proposed novelty direction

Working hypothesis:

> **Reliability-Biased Memory Attention (RBMA)**: estimate per-modality or per-region reliability from decoder predictive uncertainty, then inject reliability as an additive pre-softmax bias into SAM2-style memory-attention logits.

Important distinction for related work:

- Many prior works do feature multiplication, output scaling, learned gates, or loss weighting.
- The novelty claim should focus on **attention-logit-level reliability bias** inside memory-attention fusion.
- Predictive uncertainty is the signal; the attention-logit bias is the stronger mechanism-level contribution.

### D. ViT / SAM / adapter axis

- ViT segmentation backbones: [[SegFormer]], [[Mask2Former]], [[OneFormer]], [[SETR]], [[DPT]]
- SAM/SAM2/SAM3 adaptation for dense multimodal segmentation
- [[LoRA]], adapters, prompt tuning, parameter-efficient adaptation
- Token pruning/merging for dense prediction: [[Expedit]], [[DToP]], [[ToMe]], [[PiToMe]], [[Token Transforming]]

### E. Detection axis

Because the applied task includes multimodal object detection, also track:

- Camera-LiDAR 3D detection
- BEV fusion
- RGB-T detection
- Robust detection under adverse weather/night
- Whether segmentation-side reliability mechanisms transfer to detection heads

## Next work queue

- [x] Verify seed papers and arXiv IDs for Priority A: DGFusion, MemorySAM, StitchFusion, AnySeg, Reducing Unimodal Bias, CMX, TokenFusion, MAGIC++, CAFuser.
- [ ] Extract exact benchmark tables for DELIVER/MUSES/MCubeS.
- [x] Create verified-draft per-paper notes in [[relatedworks]] for top-priority papers.
- [ ] Build a comparison matrix: method, modalities, dataset, fusion level, uncertainty/reliability handling, train/test requirements, metrics.
- [ ] Draft related-work paragraphs grouped by datasets, fusion mechanisms, uncertainty/reliability, SAM/ViT adaptation, and detection transfer.
- [ ] Produce study material PDFs in English and Korean once the first source set is stabilized.
