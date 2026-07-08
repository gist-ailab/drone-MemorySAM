---
title: AnySeg — Robust Anymodal Segmentation via Distillation
tags: [related-work, anymodal, distillation, multimodal-segmentation]
created: 2026-06-24
source: arXiv:2411.17141; downloaded PDF text in [[sources/pdfs/priority_a/text/2411.17141.txt]]
status: verified-draft
---

# AnySeg — Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation

## Citation

Zheng et al., **“Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation,”** arXiv:2411.17141, 2024/2025. PDF archived at [[sources/pdfs/priority_a/2411.17141.pdf]].

## Problem setting

AnySeg addresses semantic segmentation when arbitrary subsets of modalities may be available. This is the **anymodal** setting: a model should work with any combination of RGB, depth, event, LiDAR, etc.

## Verified method summary

The PDF abstract states that AnySeg:

- trains a strong multimodal teacher using parallel modality learning;
- distills unimodal and cross-modal knowledge to an anymodal student;
- uses multiscale feature-level distillation;
- adds prediction-level modality-agnostic distillation;
- explicitly handles missing modalities by learning unimodal and cross-modal correspondence.

The PDF also contains DeLiVER anymodal evaluation tables for RGB, depth, event, LiDAR combinations.

## Novelty

AnySeg’s novelty is a distillation framework for arbitrary-modality generalization. It treats missing or partial modality availability as a core training objective.

## Main claims

- Multimodal teacher knowledge can supervise a student that generalizes to arbitrary modality subsets.
- Unimodal and cross-modal distillation reduce dependence on any single modality.
- Prediction-level modality-agnostic distillation improves semantic consistency.

## Why it matters for our project

AnySeg is relevant because our method may also need to operate under partial modality reliability or missing modalities. It is also an important baseline for reviewers who ask whether the problem is simply “anymodal segmentation.”

## Limitations / gaps for our project

- It is a distillation-based training framework, not a SAM2 memory-attention modification.
- It handles missing modalities but does not necessarily handle spatially varying reliability within all available modalities.
- It does not inject reliability into attention logits.

## Related-work paragraph candidate

AnySeg studies robust anymodal segmentation by distilling a multimodal teacher into a student that can handle arbitrary modality subsets. Its unimodal, cross-modal, and modality-agnostic distillation losses directly address missing-modality robustness. Our setting differs in that all modalities may be present but have spatially varying reliability; therefore, rather than only distilling arbitrary modality combinations, we bias SAM2 memory attention according to predictive reliability.
