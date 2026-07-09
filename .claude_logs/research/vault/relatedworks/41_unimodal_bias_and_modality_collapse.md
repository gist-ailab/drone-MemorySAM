---
title: Unimodal Bias and Modality Collapse in Multimodal Segmentation
tags: [related-work, unimodal-bias, modality-collapse, multimodal-segmentation, robustness, rbma]
created: 2026-06-24
source: [[relatedworks/03_unimodal_bias_entropy_relatedwork]]; arXiv:2505.06635; arXiv:2412.16876; arXiv:2411.17141; arXiv:2505.12861; DOI:10.1007/978-3-031-72754-2_9
status: verified-draft
---

# Unimodal Bias and Modality Collapse in Multimodal Segmentation

## Scope

This note supports the RBMA motivation that multimodal models may **collapse onto an easy/dominant modality** (often RGB), even when auxiliary modalities become more reliable under adverse conditions. It distinguishes train-time loss regularization, modality selection, distillation, and reliability-aware attention.

## Verified source ledger

| Method / paper | Verification source | Venue / year | Mechanism | Relevance to modality collapse |
|---|---|---:|---|---|
| Reducing Unimodal Bias with Multi-Scale Functional Entropy Regularization | arXiv:2505.06635; [[relatedworks/03_unimodal_bias_entropy_relatedwork]] | arXiv 2025 | Prediction- and feature-scale functional entropy / Fisher-information regularization. | Directly frames over-reliance on easily learnable modalities and provides loss-level anti-collapse baseline. |
| MAGIC++ — Efficient and Resilient Modality-Agnostic Semantic Segmentation via Hierarchical Modality Selection | arXiv:2412.16876; OpenAlex abstract | arXiv 2024 | Multi-modal interaction plus hierarchical modality selection; argues RGB-centered asymmetric architectures fail especially at night when event or non-RGB modalities may be stronger. | Strong evidence that modality dominance/RGB centrality is a recognized limitation. |
| AnySeg — Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation | arXiv:2411.17141; [[relatedworks/05_anyseg_relatedwork]] | arXiv 2024/2025 | Multimodal teacher, unimodal and cross-modal distillation, modality-agnostic prediction distillation. | Handles missing arbitrary modalities; reduces dependence on any single modality through distillation. |
| Any2Seg / Learning Modality-Agnostic Representation for Semantic Segmentation from Any Modalities | DOI: [10.1007/978-3-031-72754-2_9](https://doi.org/10.1007/978-3-031-72754-2_9); OpenAlex | LNCS / ECCV workshop-style proceedings 2024 | Modality-agnostic representation learning for segmentation from any modalities. | Relevant anymodal baseline; exact module details require PDF verification. |
| RMMSS — Robust Multi-Modal Semantic Segmentation with Hybrid Prototype Distillation and Feature Selection | arXiv:2505.12861; OpenAlex abstract | arXiv 2025 | Two-stage framework with Hybrid Prototype Distillation Module and Feature Selection Module; robust to incomplete/degraded/missing sensor data. | Addresses robustness without sacrificing full-modality performance; uses feature/logit selection rather than attention-logit bias. |
| CAFuser | arXiv:2410.10791; [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]] | arXiv 2024 / RA-L 2025 per existing note | Condition token + modality-specific adapters. | Counters uniform fusion across conditions, but does not explicitly regularize unimodal bias. |
| DGFusion | arXiv:2509.09828; [[relatedworks/02_dgfusion_relatedwork]] | RA-L 2026 per existing note | Local depth tokens + condition token. | Addresses spatially varying sensor reliability rather than train-time modality collapse. |

## Comparison table: anti-collapse mechanisms

| Category | Representative methods | Unit of control | When it acts | What it prevents | Gap for RBMA |
|---|---|---|---|---|---|
| **Loss weighting / regularization** | Reducing Unimodal Bias | Loss terms, Fisher information / functional entropy | Training | Dominant modality overfitting | Does not decide attention over memory tokens at inference. |
| **Modality selection** | MAGIC / MAGIC++ | Modality or feature granularity selection | Training + inference | RGB-centered asymmetry and missing-modality brittleness | Selection may be discrete/coarse compared with local reliability bias. |
| **Distillation** | AnySeg, RMMSS | Teacher-student features/logits/prototypes | Training | Missing-modality failure and weak unimodal branches | Robustness is learned indirectly; no explicit SAM2 memory attention control. |
| **Learned gating / condition modulation** | CAFuser, DGFusion | Feature adapters, condition/depth tokens | Inference | Uniform fusion under adverse conditions | Gates features; does not add reliability prior to attention logits. |
| **Feature/logit selection** | RMMSS | Feature and logits layers selected by score | Training/inference depending on module | Missing/degraded modality degradation | Selection is outside SAM2 memory attention. |
| **Pre-softmax attention-logit bias** | RBMA proposed | Attention logits over modality memory tokens | Inference / model forward pass | Attention collapse onto unreliable modality tokens | Novel axis requiring ablation against all above categories. |

## Established facts

- [[relatedworks/03_unimodal_bias_entropy_relatedwork]] verifies that recent MMSS work explicitly names **unimodal bias** and proposes functional entropy regularization to reduce it.
- MAGIC++ explicitly criticizes RGB-centered asymmetric architectures and motivates dynamic modality adaptation, especially for nighttime and adverse conditions.
- AnySeg and RMMSS demonstrate that missing/degraded modality robustness is often handled by **distillation**, not by changing the attention scoring rule.
- The anti-collapse literature is therefore mostly training-objective, selection, or distillation oriented; this creates room for a memory-attention reliability mechanism.

## Open questions

- Does functional entropy regularization remain effective when a modality is present but locally corrupted, rather than globally missing or globally weak?
- Can modality selection methods provide sufficiently fine spatial control for small objects, shadows, glare, thermal saturation, event noise, or LiDAR sparsity?
- Do distillation methods improve calibration of per-modality uncertainty, or only average segmentation accuracy?
- For RBMA, the critical ablation is whether an attention-logit bias improves over post-hoc output scaling and feature gating under controlled modality corruptions.

## Ready-to-use related-work paragraph candidates

### Paragraph A — unimodal bias motivation

Recent multimodal segmentation work shows that robustness is limited not only by fusion architecture but also by unimodal bias. Zheng et al. address this directly with multi-scale functional entropy regularization, which reduces over-reliance on easily learned modalities by balancing modality contributions through prediction- and feature-level regularizers. MAGIC++ makes a related architectural observation: RGB-centered designs can fail in adverse settings such as nighttime, where auxiliary modalities may be more informative. These works motivate RBMA’s central premise that modality usage should be reliability-dependent rather than fixed by training-set dominance.

### Paragraph B — why RBMA is not just anymodal segmentation

Anymodal methods such as AnySeg, Any2Seg, and RMMSS improve robustness to missing or degraded modalities through distillation, modality-agnostic representation learning, and feature/logit selection. They are important baselines, but their main intervention is training-time robustness or modality-subset generalization. RBMA targets a different failure mode: when modalities are present but locally unreliable, SAM2 memory attention may still attend to the wrong modality tokens. RBMA therefore biases the pre-softmax attention logits using predictive reliability, providing a mechanism-level complement to anymodal distillation.

## Links

- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/03_unimodal_bias_entropy_relatedwork]]
- [[relatedworks/05_anyseg_relatedwork]]
- [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
