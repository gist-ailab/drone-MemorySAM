---
title: "Robust Multimodal Semantic Segmentation with Balanced Modality Contributions"
tags: [source-note, weekly-sweep, multimodal_semantic_segmentation]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

> → 승격: [[relatedworks/62_equiseg_balanced_modality_contributions]] — 이 스텁은 relatedworks gap-fill 노트로 승격됨 (2026-07-08 archive 이동). 인용은 승격 노트를 사용할 것.

# Robust Multimodal Semantic Segmentation with Balanced Modality Contributions

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_semantic_segmentation`
- Venue/year: arXiv cs.CV / 2025
- Date: 2025-09-29
- Primary URL: https://arxiv.org/abs/2509.24505v1
- arXiv/DOI: 2509.24505v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

RBMA의 reliability-aware fusion/모달 기여도 균형화 논지와 직접 연결됨.

## Abstract / description snapshot

> Multimodal semantic segmentation enhances model robustness by exploiting cross-modal complementarities. However, existing methods often suffer from imbalanced modal dependencies, where overall performance degrades significantly once a dominant modality deteriorates in real-world scenarios. Thus, modality balance has become acritical challenge for practical multimodal segmentation. To address this issue, we propose EQUISeg, a multimodal segmentation framework that balances modality contributions through equal encoding of modalities. Built upon a four-stage Cross-modal Transformer Block(CMTB), EQUISeg enables efficient multimodal fusion and hierarchical selection. Furthermore, we design a Self-guided Module(SGM) that mitigates modality imbalance by introducing a mutual guidance mechanism, enabling each modality to adaptively adjust its contribution and enhance robustness under degraded conditions. Extensive experiments on multiple datasets demonstrate that EQUISeg achieves significant performance gains and effectively alleviates the adverse effects of modality imbalance in segmentation tasks.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
