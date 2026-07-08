---
title: 26_MultimodalSeg — Map of Content
aliases: [26_MultimodalSeg]
tags: [moc, project, multimodal-segmentation, semantic-segmentation, detection, obsidian]
created: 2026-06-24
updated: 2026-07-08
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
| [[sources/00_MOC_sources\|sources]] | Raw and first-pass source collection: source maps, discovery DBs, sweep logs, stubs, archive (see folder MOC) |
| [[relatedworks/00_relatedworks_index\|relatedworks]] | Per-paper research synthesis: novelty, claims, method, figures, limitations, comparisons |
| [[material]] | Study materials and PDF-ready explanations, always English + Korean versions (currently `01_multimodal_seg_clustered_relatedwork_{en,ko}.md/.pdf`) |
| [[P32_CoRB/00_P32_CoRB_index\|P32_CoRB]] | Our-method reports & figures for **P32-B CoRB** (reliability-signal redesign): figure report, text report, PDF, figure assets |

## Current notes (updated 2026-07-08)

### Status / our-method
- [[PROJECT_TRACKING_26_MultimodalSeg]] — live project status board
- [[P32_CoRB/00_P32_CoRB_index\|P32_CoRB index]] — P32-B CoRB folder MOC (reports, PDF, figures)
- [[P32_CoRB/P32_CoRB_리포트\|P32_CoRB 리포트 (그림판)]] — our-method report: why P28 failed and what P32-B changes (self-entropy → corroboration), with figures
- [[P32_CoRB/P32_CoRB_novelty_risk_register]] — consolidated novelty risk register across RBMA-mechanism / RBMA-signal / CoRB (most-dangerous-first threat table)
- [[relatedworks/49_corb_novelty_defense]] — CoRB (P32-B) novelty defense: 4-pillar claim, RSGMamba/MAGIC++ near-misses, posterior-Bhattacharyya discriminator

### Related work (synthesis)
- [[relatedworks/00_relatedworks_index]] — related-work synthesis index (77 notes + index: baselines 01–09, detection 10–15, adapters 20–23, heads 30–34, novelty defense 40–49, 2026-07-02 deep-research 50–60, gap-fill 61–88, synthesis 90–93)
- [[relatedworks/90_clustered_relatedwork_synthesis]] — 6-cluster synthesis + related-work paragraph candidates (exported to material/)
- [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] — **canonical benchmark number tables** (split-tagged, §U1–U9)
- [[relatedworks/93_benchmark_protocol_split_resolution]] — DELIVER/MUSES protocol forensics + dual-split reporting rules
- [[material/01_multimodal_seg_clustered_relatedwork_ko|클러스터 related-work 자료 (ko)]] / [[material/01_multimodal_seg_clustered_relatedwork_en|(en)]] — PDF-ready study material

### Sources / discovery
- [[sources/00_MOC_sources]] — **sources folder MOC** (소스맵 / DB / 스윕로그 / 스텁 / archive 분류)
- [[sources/00_imported_claude_related_work_2026-06-24]] — imported Claude/NotebookLM memo from the user
- [[sources/01_source_index_multimodal_segmentation]] — first-pass Semantic Scholar paper/source index
- [[sources/02_openalex_top_venue_literature_database]] — expanded top-conference / vision-journal literature database
- [[sources/03_seed_paper_verification_candidates]] — candidate metadata matches for DGFusion, MemorySAM, StitchFusion, AnySeg, DeLiVER/MUSES/MCubeS, and key baselines
- [[sources/02_source_map_multimodal_semantic_segmentation]] · [[sources/02_source_map_multimodal_object_detection]] · [[sources/02_source_map_adapter_lora_foundation_seg_det]] · [[sources/02_source_map_segmentation_detection_heads]] — category source maps
- [[sources/07_parallel_research_prompts_2026-07-02]] — 8-track parallel deep-research prompts + completion record
- [[sources/08_threat_watch_2026H2]] — 2026H2 scoop/threat watch triage

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
- [x] Extract exact benchmark tables for DELIVER/MUSES/MCubeS. → [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] (+ protocol resolution [[relatedworks/93_benchmark_protocol_split_resolution]])
- [x] Create verified-draft per-paper notes in [[relatedworks/00_relatedworks_index|relatedworks]] for top-priority papers.
- [x] Build a comparison matrix: method, modalities, dataset, fusion level, uncertainty/reliability handling, train/test requirements, metrics. → [[relatedworks/08_priority_a_comparison_matrix]]
- [x] Draft related-work paragraphs grouped by datasets, fusion mechanisms, uncertainty/reliability, SAM/ViT adaptation, and detection transfer. → [[relatedworks/90_clustered_relatedwork_synthesis]] (paragraph candidates per cluster)
- [x] Produce study material PDFs in English and Korean once the first source set is stabilized. → `material/01_multimodal_seg_clustered_relatedwork_{en,ko}.pdf`
