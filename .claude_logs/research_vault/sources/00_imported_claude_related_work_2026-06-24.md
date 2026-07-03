---
title: Claude/NotebookLM Imported Related Work Memo — Multimodal Fusion and Attention Dynamics
tags:
  - source
  - imported-note
  - related-work
  - multimodal-segmentation
  - memorysam
  - rbma
created: 2026-06-24
source: user-provided Claude/NotebookLM memo pasted in Telegram
status: imported-summary
---

# Claude/NotebookLM Imported Related Work Memo — Multimodal Fusion and Attention Dynamics

> Imported from the user's pasted Claude/NotebookLM working note. This file preserves the actionable research map and will be used as seed context for [[00_MOC_26_MultimodalSeg]], [[sources/01_source_index_multimodal_segmentation]], and per-paper notes in [[relatedworks]].

## Project task framing

The project target is a paper on **semantic segmentation using multimodal sensor data** such as RGB, thermal, event, and LiDAR. The applied/business task is also related to solving **object detection** using multi-sensor data.

Important existing related-work anchors mentioned by the user:

- [[DELIVER]] dataset and DeLiVER semantic segmentation tasks
- [[MUSES]] dataset and multimodal semantic segmentation tasks
- [[DGFusion]] and related depth-guided fusion work
- [[MemorySAM]] / SAM2 memory-attention style modality-as-frame fusion
- [[ViT]]-based segmentation and detection
- [[LoRA]] and adapter-based model adaptation

## Seed bibliography from pasted memo

| #   |                                                                          Paper / source | Year      | Topic                                                                  | Project relevance        |
| --- | --------------------------------------------------------------------------------------: | --------- | ---------------------------------------------------------------------- | ------------------------ |
| 1   |                             Reducing Unimodal Bias in Multi-Modal Semantic Segmentation | 2025      | Functional entropy regularization; anti-RGB/unimodal dominance         | Very high                |
| 2   |                     DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception | 2026      | Depth-guided multimodal semantic segmentation; condition/depth tokens  | High                     |
| 3   | StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation | 2024      | MultiAdapter-style early encoder fusion                                | Medium-high              |
| 4   |                                                  AnySeg: Anymodal semantic segmentation | 2024      | Multimodal teacher → anymodal student; distillation                    | High                     |
| 5   |                                                                               MemorySAM | 2025      | SAM2 modality-as-frame memory attention; baseline for current idea     | Very high                |
| 6   |                                                      Fast SAM2 / token pruning for SAM2 | 2025      | Memory-engine-adjacent token pruning                                   | Medium-high              |
| 7   |                                                                   Expedit / Expedit-SAM | 2022      | Training-free token clustering/reconstruction for dense prediction/SAM | Medium-high              |
| 8   |                                                                                  PiToMe | 2024      | Protect informative tokens before merging                              | Medium                   |
| 9   |                                                 ToMe / ToMeSD / PPT / DynamicViT / DToP | 2021–2025 | Token pruning/merging and efficiency                                   | Medium                   |
| 10  |                                                      Vision Transformers Need Registers | 2024      | ViT attention-sink/register tokens; dense feature quality              | Medium                   |
| 11  |                                                          UTFNet / HyperDUM / TMC / ETMC | 2021–2025 | Uncertainty-aware multimodal fusion baselines                          | High for novelty defense |
| 12  |                     ReliFusion / READ / AG-Fusion / MAGIC++ / Any2Seg / RMMSS / EQUISeg | 2024–2025 | Reliability/confidence fusion baselines                                | High for novelty defense |

## Key synthesis from pasted memo

### 1. Robust multimodal segmentation is primarily an adaptive reliability problem

A recurring problem is **unimodal dominance**, especially RGB dominance: during normal daytime training, the model learns to over-rely on RGB. When RGB becomes unreliable under night, glare, blur, rain, fog, or adverse illumination, performance collapses even if other modalities remain informative.

Relevant mechanisms:

- Entropy or information-theoretic regularization to discourage unimodal dominance.
- Spatially local reliability estimation rather than global modality weights.
- Depth/LiDAR auxiliary supervision to stabilize local fusion.
- Prediction-uncertainty signals to estimate modality reliability at inference.

### 2. DGFusion and entropy regularization are near-term related-work anchors

[[DGFusion]] is especially relevant because it treats multimodal segmentation as a multimodal + multitask problem and uses depth supervision to guide fusion.

[[Reducing Unimodal Bias in Multi-Modal Semantic Segmentation]] is especially relevant because it attacks the train-time optimization cause of modality dominance using functional-entropy regularization, with prediction-level and feature-level terms.

### 3. Adapter and LoRA work are relevant to parameter-efficient multimodal adaptation

[[StitchFusion]] is important because it uses lightweight MultiAdapter modules to weave information between pretrained encoders early in the encoder stack, rather than relying only on late fusion.

This connects directly to LoRA/adapters for adapting SAM2, ViT, SegFormer, Mask2Former, OneFormer, or other pretrained segmentation backbones to multimodal inputs.

### 4. RBMA novelty hypothesis from pasted memo

The strongest proposed novelty direction is:

> **Reliability-Biased Memory Attention (RBMA): inject per-modality reliability as an additive pre-softmax bias into SAM2-style memory attention logits.**

The pasted memo's novelty defense:

- Existing methods often use reliability/confidence at the feature level, attention-output scaling level, or loss level.
- The proposed mechanism operates inside the attention logits, before softmax.
- Decoder predictive uncertainty can be used as a lightweight, training-free reliability signal.
- The signal itself is not wholly novel because uncertainty-aware multimodal fusion exists; the stronger novelty is the **attention-logit bias mechanism** in a SAM2/MemorySAM-style memory fusion system.

### 5. SAM3 portability note

The pasted memo claims SAM3 tracker memory uses a transformer encoder where memory is passed as prompt tokens and cross-attention receives a `memory_mask` / `attn_mask` path. If true, RBMA may be easier to port to SAM3 than to SAM2 by adding reliability bias into the memory-token columns of this mask.

This should be verified against the actual SAM3 code before being cited.

## Open questions for follow-up

- Verify exact quantitative tables for DeLiVER, MCubeS, MUSES in DGFusion, AnySeg, MemorySAM, MAGIC/MAGIC++, CMX, TokenFusion, CAFuser, and related baselines.
- Confirm arXiv IDs and publication venues for every seed source.
- Build a broad source library covering:
  - RGB-D / RGB-T / event / LiDAR multimodal segmentation
  - multimodal object detection and 3D detection
  - ViT-based segmentation/detection backbones
  - adapter/LoRA/prompt tuning for segmentation and SAM/SAM2
  - uncertainty/reliability-aware fusion
  - token pruning/merging/attention-efficiency for dense prediction
- Create per-paper [[relatedworks]] notes with novelty, claims, method, figures, limitations, and comparison points.

## Links

- [[00_MOC_26_MultimodalSeg]]
- [[sources/01_source_index_multimodal_segmentation]]
- [[relatedworks/00_relatedworks_index]]
