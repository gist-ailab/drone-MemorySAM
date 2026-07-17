---
title: DeLiVER, MUSES, and MCubeS — Multimodal Segmentation Benchmarks
aliases: [DELIVER, DeLiVER, MUSES, MCubeS]
tags: [related-work, dataset, benchmark, multimodal-segmentation]
created: 2026-06-24
source: arXiv:2303.01480; MUSES ECCV 2024 DOI; MCubeS-related papers; downloaded source PDFs in [[sources/pdfs/priority_a]]
status: verified-draft-needs-table-extraction
---

# DeLiVER, MUSES, and MCubeS — Multimodal Segmentation Benchmarks

## Purpose of this note

This note tracks the datasets and benchmark settings that repeatedly appear in recent multimodal semantic segmentation papers. It is not yet a full dataset-card; benchmark tables still need exact extraction from the original papers.

## DeLiVER / “Delivering Arbitrary-Modal Semantic Segmentation”

Source: Zhang et al., **“Delivering Arbitrary-Modal Semantic Segmentation,”** arXiv:2303.01480. PDF archived at [[sources/pdfs/priority_a/2303.01480.pdf]].

Verified from PDF:

- The work frames arbitrary-modal semantic segmentation.
- It evaluates combinations of RGB, depth, event, and LiDAR-like modalities.
- It is a central benchmark for modality-agnostic and arbitrary-modality segmentation.

Why it matters:

- DeLiVER is the most directly relevant benchmark for RGB/depth/event/LiDAR multimodal segmentation.
- It is repeatedly used by MemorySAM, AnySeg, MAGIC++, DGFusion, StitchFusion, and related methods.

## MUSES

Source candidate: **“MUSES: The Multi-sensor Semantic Perception Dataset for Driving Under Uncertainty,”** ECCV/LNCS 2024 candidate in [[sources/03_seed_paper_verification_candidates]].

Verified from DGFusion and CAFuser PDFs:

- MUSES is used as a challenging driving benchmark for robust semantic perception.
- DGFusion reports state-of-the-art panoptic and semantic segmentation on MUSES and DeLiVER.
- CAFuser also positions robust multimodal semantic perception around MUSES/DeLiVER-style adverse-condition settings.

Needs follow-up:

- Download and parse the official MUSES paper.
- Extract modalities, classes, train/val/test split, adverse conditions, metrics, and official baselines.

## MCubeS / MCubes

Source candidates:

- MCubeS appears in MemorySAM, StitchFusion, AnySeg, and related OpenAlex candidate matches.
- Related method paper: MMSFormer / material and semantic segmentation, depending on exact benchmark usage.

Needs follow-up:

- Verify the official MCubeS dataset paper and exact spelling/version.
- Extract sensor modalities and class definitions.
- Record whether MCubeS is material segmentation, semantic segmentation, or both in each benchmark table.

## Dataset comparison matrix — current draft

| Dataset | Modalities seen in related papers | Main task | Why relevant | Status |
|---|---|---|---|---|
| DeLiVER | RGB, depth, event, LiDAR-like modalities | arbitrary/multimodal semantic segmentation | Closest to our RGB/depth/event/LiDAR setting | PDF downloaded; table extraction pending |
| MUSES | multi-sensor driving perception | semantic/panoptic segmentation under uncertainty | Real-world adverse driving benchmark | candidate verified via DGFusion/CAFuser; official paper extraction pending |
| MCubeS / MCubes | multimodal visual modalities; often appears with DeLiVER | semantic/material segmentation | Used in MemorySAM/StitchFusion/AnySeg-style comparisons | official source verification pending |

## Related-work paragraph candidate

Recent multimodal semantic segmentation papers increasingly evaluate on benchmarks that stress arbitrary modality availability and adverse-condition robustness. DeLiVER provides RGB/depth/event/LiDAR-style combinations for arbitrary-modal segmentation, while MUSES targets multi-sensor driving perception under uncertainty. MCubeS/MCubes appears in several RGB-X segmentation comparisons and broadens evaluation beyond driving-only scenes. These benchmarks are essential for evaluating whether a fusion method can generalize beyond clean RGB-dominant settings.
