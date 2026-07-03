---
title: Multimodal Segmentation Clustered Related-Work Review
tags: [material, pdf-ready, related-work, multimodal-segmentation, rbma]
created: 2026-06-25
source: [[relatedworks/90_clustered_relatedwork_synthesis]]
status: pdf-ready
---

# Multimodal Segmentation Clustered Related-Work Review

## Abstract

This review converts the current [[26_MultimodalSeg]] related-work vault into a paper-oriented study document. The literature clusters into six groups: direct multimodal semantic segmentation, multimodal object detection, foundation-model adaptation with adapters/LoRA, segmentation and detection heads, uncertainty/reliability/novelty, and benchmarks/datasets. The main conclusion is that existing methods provide strong baselines for feature fusion, SAM/SAM2 adaptation, anymodal distillation, condition-aware perception, and BEV/query detection, but they do not fully cover reliability-biased SAM2 memory attention. This leaves a defensible contribution space for RBMA: reliability estimated from modality evidence is injected as an additive pre-softmax attention-logit bias during multimodal memory fusion.

## 1. Research question

**How should a new multimodal semantic segmentation paper position an RBMA-style method against existing multimodal segmentation, multimodal detection, adapter/LoRA, foundation-model, uncertainty, and benchmark literature?**

The most defensible answer is to make the related work about the *location of fusion control*. Prior work often improves features, decoders, training objectives, modality selection, or late fusion. RBMA should be described as changing attention competition itself.

## 2. Clustered literature map

| Cluster | Representative works | Main lesson | Gap for RBMA |
|---|---|---|---|
| Direct multimodal segmentation | MemorySAM, DGFusion, CMX, TokenFusion, MAGIC++, CAFuser, StitchFusion, AnySeg | Strong baselines for memory, feature, token, adapter, condition, and anymodal fusion | Reliability is implicit, proxy-based, or outside memory-attention logits |
| Multimodal detection | BEVFusion, TransFusion, DeepInteraction, FUTR3D | Robust multimodal perception benefits from shared spaces, learned queries, and preserved modality identity | Detection mechanisms must be adapted to dense semantic prediction |
| Adapters / LoRA / foundation models | LoRA, AdaptFormer, VPT, ViT-Adapter, SAM-Adapter, SAMed, MedSAM, MoE-LoRA SAM | PEFT adapts large encoders cheaply to domains and sensors | Adaptation does not decide sensor trust under corruption |
| Heads | SegFormer, Mask2Former, OneFormer, DETR, Deformable DETR, DINO, MaskDINO, YOLO, Mask R-CNN | Heads determine task outputs and metrics | Heads are evaluation interfaces, not reliability mechanisms |
| Reliability / novelty | UTFNet, HyperDUM, TMC, DGFusion, CAFuser, unimodal-bias regularization | Uncertainty and modality collapse are recognized problems | Most methods weight features/outputs or losses, not pre-softmax memory attention |
| Benchmarks | DeLiVER, MUSES, MCubeS | Relevant multimodal datasets for semantic/panoptic segmentation | Numeric claims require source-table verification |

## 3. Direct multimodal semantic segmentation

The closest baselines include MemorySAM, DGFusion, CMX, TokenFusion, MAGIC++, CAFuser, StitchFusion, AnySeg, and Reducing Unimodal Bias. MemorySAM is the closest architectural baseline because it maps modalities into a SAM2 memory formulation. DGFusion and CAFuser use condition-aware or depth-guided fusion for robust driving-scene perception. StitchFusion shows that pretrained encoders can be connected through lightweight adapters. AnySeg and related anymodal methods use distillation to handle missing or arbitrary modalities.

The synthesis is straightforward: these methods improve multimodal semantic segmentation, but they do not directly implement reliability as a pre-softmax prior in SAM2 memory attention. Therefore, RBMA should be positioned as a reliability-control layer that can sit on top of or alongside these representation and adaptation strategies.

## 4. Multimodal object detection as supporting context

Object detection papers are not the main semantic segmentation baselines, but they provide strong design principles. BEVFusion shows the value of a shared BEV space for camera-LiDAR fusion. TransFusion shows that learned query attention can be more robust than hard geometric association. DeepInteraction shows that preserving modality-specific streams can avoid loss of useful evidence. FUTR3D shows that query-based feature sampling can support flexible sensor configurations.

For the paper, use detection work to justify *why learned multimodal association matters*. Do not overclaim detection papers as direct semantic segmentation competitors.

## 5. Foundation-model adaptation

LoRA, AdaptFormer, VPT, and ViT-Adapter explain how to adapt transformer backbones. SAM-Adapter, MedSAM, SAMed, MemorySAM, MoE-LoRA SAM, SAM-FuseNet, and ClassWise-SAM-Adapter show that SAM-family models need domain-specific customization. This literature motivates using LoRA/adapters in the proposed model.

However, adaptation and reliability are different. Adapters answer: “How can the model represent this modality or domain?” RBMA answers: “How much should this modality be trusted at this moment and location?” This distinction should appear clearly in the introduction and related work.

## 6. Heads and evaluation interfaces

A clean semantic segmentation experiment can use SegFormer, UPerNet, or DeepLabv3+-style heads to report mIoU. Mask2Former and OneFormer support semantic, instance, and panoptic segmentation. DETR, Deformable DETR, DINO, MaskDINO, YOLO, and Mask R-CNN support object detection or instance segmentation extensions.

For a focused paper, keep the first submission centered on semantic segmentation and reliability-aware fusion. Detection/panoptic heads can be framed as future extensions or secondary experiments.

## 7. Reliability and novelty defense

The strongest novelty defense is mathematical. Feature scaling changes feature magnitudes. Late fusion changes output aggregation. Evidential fusion changes branch-level confidence. Modality selection chooses sensors or feature groups. Loss regularization changes training dynamics. Condition/depth tokens add context. RBMA is different because it adds reliability to attention logits before softmax, directly changing memory-token competition.

A credible paper should include ablations for:

1. MemorySAM-style fusion without reliability bias.
2. Feature-level reliability scaling.
3. Output-level uncertainty weighting.
4. Learned gate without explicit uncertainty.
5. Global modality reliability versus local patch/token reliability.
6. Corruption-specific tests: dark RGB, thermal saturation, event noise, sparse LiDAR/depth.
7. Calibration tests such as ECE and uncertainty-error correlation.

## 8. Benchmarks and quantitative reporting

The current benchmark notes support DeLiVER, MUSES, and MCubeS as the main datasets. Existing extracted tables include MemorySAM, DGFusion, Reducing Unimodal Bias, StitchFusion, AnySeg, MAGIC++, and CAFuser. Use these benchmark rows for motivation and related-work comparison, but preserve source table identifiers for every numeric claim.

## 9. Ready-to-use related-work paragraph

Recent multimodal semantic segmentation methods improve robustness through feature rectification, token fusion, modality selection, adapter exchange, condition-aware modulation, and distillation. CMX and TokenFusion represent feature and token fusion baselines; MAGIC++ and AnySeg address arbitrary or missing modality settings; CAFuser and DGFusion introduce condition and depth-guided fusion for driving-scene perception; StitchFusion uses adapters to weave pretrained encoders; and MemorySAM maps modalities into SAM2 memory attention. These works establish strong multimodal baselines, but they generally model reliability implicitly or outside the attention competition itself. In contrast, RBMA injects an explicit reliability prior into memory-attention logits before softmax, allowing unreliable modality-memory tokens to be down-weighted during fusion while retaining the benefits of foundation-model adaptation.

## 10. References

See [[relatedworks/90_clustered_relatedwork_synthesis]] and the linked per-paper notes for full citation metadata and verification status. Final manuscript citations should be copied from verified per-paper notes, not from unverified discovery database entries.
