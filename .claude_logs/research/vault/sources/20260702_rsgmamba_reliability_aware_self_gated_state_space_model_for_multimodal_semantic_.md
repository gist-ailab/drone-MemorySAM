---
title: "RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation"
tags: [source-note, weekly-sweep, multimodal_semantic_segmentation]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_semantic_segmentation`
- Venue/year: arXiv cs.CV / 2026
- Date: 2026-04-14
- Primary URL: https://arxiv.org/abs/2604.12319v2
- arXiv/DOI: 2604.12319v2
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

RBMA의 reliability-aware fusion/모달 기여도 균형화 논지와 직접 연결됨.

## Abstract / description snapshot

> Multimodal semantic segmentation has emerged as a powerful paradigm for enhancing scene understanding by leveraging complementary information from multiple sensing modalities (e.g., RGB, depth, and thermal). However, existing cross-modal fusion methods often implicitly assume that all modalities are equally reliable, which can lead to feature degradation when auxiliary modalities are noisy, misaligned, or incomplete. In this paper, we revisit cross-modal fusion from the perspective of modality reliability and propose a novel framework termed the Reliability-aware Self-Gated State Space Model (RSGMamba). At the core of our method is the Reliability-aware Self-Gated Mamba Block (RSGMB), which explicitly models modality reliability and dynamically regulates cross-modal interactions through a self-gating mechanism. Unlike conventional fusion strategies that indiscriminately exchange information across modalities, RSGMB enables reliability-aware feature selection and enhancing informative feature aggregation. In addition, a lightweight Local Cross-Gated Modulation (LCGM) is incorporated to refine fine-grained spatial details, complementing the global modeling capability of RSGMB. Extensive experiments demonstrate that RSGMamba achieves state-of-the-art performance on both RGB-D and RGB-T semantic segmentation benchmarks, resulting 58.8% / 54.0% mIoU on NYUDepth V2 and SUN-RGBD (+0.4% / +0.7% over prior best), and 61.1% / 88.9% mIoU on MFNet and PST900 (up to +1.6%), with only 48.6M parameters, thereby validating the effectiveness and superiority of the proposed approach.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
