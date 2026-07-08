---
title: "GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth"
tags: [source-note, weekly-sweep, multimodal_semantic_segmentation]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_semantic_segmentation`
- Venue/year: arXiv cs.CV,cs.RO / 2026
- Date: 2026-04-13
- Primary URL: https://arxiv.org/abs/2604.11585v1
- arXiv/DOI: 2604.11585v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

프로젝트 핵심 범위와 관련된 신규/누락 source candidate.

## Abstract / description snapshot

> Multimodal perception systems for robotics and embodied AI often assume reliable RGB-D sensing, but in practice, depth is frequently missing, noisy, or corrupted. We thus present GeomPrompt, a lightweight cross-modal adaptation module that synthesizes a task-driven geometric prompt from RGB alone for the fourth channel of a frozen RGB-D semantic segmentation model, without depth supervision. We further introduce GeomPrompt-Recovery, an adaptation module that compensates for degraded depth by predicting the fourth channel correction relevant for the frozen segmenter. Both modules are trained solely with downstream segmentation supervision, enabling recovery of the geometric prior useful for segmentation, rather than estimating depth signals. On SUN RGB-D, GeomPrompt improves over RGB-only inference by +6.1 mIoU on DFormer and +3.0 mIoU on GeminiFusion, while remaining competitive with strong monocular depth estimators. For degraded depth, GeomPrompt-Recovery consistently improves robustness, yielding gains up to +3.6 mIoU under severe depth corruptions. GeomPrompt is also substantially more efficient than monocular depth baselines, reaching 7.8 ms latency versus 38.3 ms and 71.9 ms. These results suggest that task-driven geometric prompting is an efficient mechanism for cross-modal compensation under missing and degraded depth inputs in RGB-D perception.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
