---
title: MemorySAM — SAM2 Memory for Multimodal Semantic Segmentation
aliases: [MemorySAM]
tags: [related-work, key-paper, memorysam, sam2, multimodal-segmentation]
created: 2026-06-24
source: arXiv:2503.06700; downloaded PDF text in [[sources/pdfs/priority_a/text/2503.06700.txt]]
status: verified-draft
---

# MemorySAM — Memorize Modalities and Semantics with SAM2

## Citation

Liao et al., **“MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation,”** arXiv:2503.06700, 2025. PDF archived at [[sources/pdfs/priority_a/2503.06700.pdf]].

## Problem setting

MemorySAM addresses **multi-modal semantic segmentation (MMSS)** where pixel-level semantic labels are predicted from multiple sensor modalities. The paper asks how SAM2 can be adapted from image/video segmentation to multimodal semantic segmentation and how SAM2 can better encode semantic information.

## Verified method summary

- Treats multimodal data as a **sequence of frames representing the same scene**.
- Uses SAM2’s video-style memory mechanism to correlate modalities.
- Applies LoRA fine-tuning to adapt SAM2 image encoding to multimodal data.
- Adds semantic adaptation through memory/prototype mechanisms rather than simply averaging modalities.
- PDF text explicitly describes “treat multi-modal data as a sequence of frames representing the same scene” and “memorize the modality-agnostic information.”

## Novelty

The key novelty is the mapping:

> modality fusion ≈ cross-frame memory reasoning in SAM2.

This makes MemorySAM the closest architectural ancestor for our project. It provides the baseline that our proposed [[RBMA]] modifies.

## Main claims

- SAM2 can be adapted to multimodal semantic segmentation by treating modalities as frames.
- Memory attention can fuse modality-agnostic information.
- LoRA-style adaptation is sufficient to avoid full SAM2 fine-tuning.

## Limitations / gaps for our project

- The current fusion mechanism does not explicitly model **per-modality reliability**.
- It does not directly inject uncertainty or confidence into the attention logits.
- If fusion reduces to roughly equal treatment of modality frames, it can still fail when one modality is corrupted or domain-shifted.

## Comparison to our project

Our proposed direction, [[Reliability-Biased Memory Attention]], should be framed as a direct extension of MemorySAM:

| Axis | MemorySAM | Our direction |
|---|---|---|
| Modality representation | Modalities as SAM2 frames | Same |
| Fusion mechanism | SAM2 memory attention | Reliability-biased memory attention |
| Reliability signal | Not explicit | Predictive uncertainty / confidence |
| Injection point | Memory mechanism, no explicit logit bias | Additive pre-softmax attention-logit bias |
| Adaptation | LoRA-tuned SAM2 encoder | LoRA/adapters + reliability-aware attention |

## Related-work paragraph candidate

MemorySAM adapts SAM2 to multi-modal semantic segmentation by interpreting different sensor modalities as frame-like observations of the same scene and using SAM2’s memory mechanism to fuse their information. This reframes multimodal fusion as a memory-attention problem and provides a strong foundation-model baseline for RGB-X segmentation. However, MemorySAM does not explicitly estimate the reliability of each modality under adverse conditions or inject such reliability into the attention logits. Our work builds on this modality-as-frame formulation but makes the memory attention reliability-aware by biasing attention toward more trustworthy modality tokens.
