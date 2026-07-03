---
title: Reducing Unimodal Bias — Functional Entropy Regularization
tags: [related-work, key-paper, uncertainty, multimodal-segmentation, loss]
created: 2026-06-24
source: arXiv:2505.06635; downloaded PDF text in [[sources/pdfs/priority_a/text/2505.06635.txt]]
status: verified-draft
---

# Reducing Unimodal Bias with Multi-Scale Functional Entropy Regularization

## Citation

Zheng et al., **“Reducing Unimodal Bias in Multi-Modal Semantic Segmentation with Multi-Scale Functional Entropy Regularization,”** arXiv:2505.06635, 2025. PDF archived at [[sources/pdfs/priority_a/2505.06635.pdf]].

## Problem setting

The paper studies **unimodal dominance / unimodal bias** in multimodal semantic segmentation. A multimodal model may over-rely on the easiest or most predictive modality during training, such as RGB in normal lighting. This causes failures when that dominant modality becomes unreliable at test time.

## Verified method summary

The PDF abstract states that the paper proposes a **simple plug-and-play regularization term** to address over-reliance on easily learnable modalities. The paper visualizes Fisher information on MUSES and DeLiVER and argues that regularization reduces the gap between modality Fisher information values, encouraging the model to use both input modalities.

From the imported memo and PDF text, the method uses functional entropy / Fisher-information regularization at prediction and feature scales. The objective augments supervised cross entropy with prediction-level and feature-level regularizers.

## Novelty

Rather than designing a new fusion block, the method attacks unimodal bias at the **optimization level**. It regularizes modality contribution so that the model does not collapse onto a single dominant modality.

## Main claims

- Multimodal segmentation failures can arise from train-time imbalance, not only poor architecture.
- Functional entropy/Fisher-information regularization can reduce modality dominance.
- A plug-and-play loss can improve robustness without adding new inference modules.

## Why it matters for our project

This is one of the strongest papers for motivating the RGB-collapse / night-domain-gap problem. It provides a complementary strategy:

- Their method: training-time regularization against modality dominance.
- Our method: inference-time or model-internal reliability-biased attention in SAM2 memory fusion.

## Limitations / gaps for our project

- It does not define a SAM2/MemorySAM-specific fusion mechanism.
- It regularizes modality usage but does not directly decide which modality tokens should receive attention under input-specific corruption.
- It is complementary to, not a replacement for, reliability-aware attention.

## Related-work paragraph candidate

Recent work on unimodal bias shows that multimodal segmentation models can overfit to a dominant modality and underuse complementary sensors. Zheng et al. address this by adding multi-scale functional entropy regularization, which balances modality contributions through prediction- and feature-level Fisher-information terms. This optimization-level approach is complementary to architecture-level fusion methods: it reduces train-time dominance, whereas our method changes the memory-attention mechanism itself so that modality reliability can bias token selection at inference.
