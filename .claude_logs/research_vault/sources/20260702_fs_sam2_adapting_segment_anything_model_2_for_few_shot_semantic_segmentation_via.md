---
title: "FS-SAM2: Adapting Segment Anything Model 2 for Few-Shot Semantic Segmentation via Low-Rank Adaptation"
tags: [source-note, weekly-sweep, sam_lora_adapter]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

> ⚠️ **abstract-only 미검증** — arXiv metadata만 수집된 스텁. 원문 정독 전 정량 인용 금지.

# FS-SAM2: Adapting Segment Anything Model 2 for Few-Shot Semantic Segmentation via Low-Rank Adaptation

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `sam_lora_adapter`
- Venue/year: arXiv cs.CV / 2025
- Date: 2025-09-15
- Primary URL: https://arxiv.org/abs/2509.12105v1
- arXiv/DOI: 2509.12105v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

SAM/SAM2 adaptation 및 adapter/LoRA 기반 dense prediction baseline 후보.

## Abstract / description snapshot

> Few-shot semantic segmentation has recently attracted great attention. The goal is to develop a model capable of segmenting unseen classes using only a few annotated samples. Most existing approaches adapt a pre-trained model by training from scratch an additional module. Achieving optimal performance with these approaches requires extensive training on large-scale datasets. The Segment Anything Model 2 (SAM2) is a foundational model for zero-shot image and video segmentation with a modular design. In this paper, we propose a Few-Shot segmentation method based on SAM2 (FS-SAM2), where SAM2's video capabilities are directly repurposed for the few-shot task. Moreover, we apply a Low-Rank Adaptation (LoRA) to the original modules in order to handle the diverse images typically found in standard datasets, unlike the temporally connected frames used in SAM2's pre-training. With this approach, only a small number of parameters is meta-trained, which effectively adapts SAM2 while benefiting from its impressive segmentation performance. Our method supports any K-shot configuration. We evaluate FS-SAM2 on the PASCAL-5$^i$, COCO-20$^i$ and FSS-1000 datasets, achieving remarkable results and demonstrating excellent computational efficiency during inference. Code is available at https://github.com/fornib/FS-SAM2

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
