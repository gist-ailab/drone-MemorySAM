---
title: Multimodal SAM/adapter matrix for segmentation and detection adaptation
tags: [related-work, comparison-matrix, sam, sam2, adapter, lora, multimodal-segmentation, dense-prediction]
created: 2026-06-24
source: [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/21_vit_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], [[sources/02_source_map_adapter_lora_foundation_seg_det]]
status: verified-draft
---

# Multimodal SAM/adapter matrix for segmentation and detection adaptation

## Purpose

This matrix compares adapter/LoRA/foundation-model adaptation candidates for [[26_MultimodalSeg]]. It is written as a practical selection guide for a SAM2/SAM-style multimodal semantic segmentation project, especially one involving RGB-X sensors and reliability-aware fusion.

## Executive comparison

| Candidate | Foundation model | Modality target | Mechanism | Parameter/update strategy | Dense prediction relevance | Main limitation | Project fit |
|---|---|---|---|---|---|---|---|
| LoRA | Transformer / ViT / SAM | Any domain | Low-rank residual updates in selected projections | Freeze base; train rank-$r$ matrices | Strong when inserted in attention/MLP blocks and paired with decoder | Rank/placement sensitive | Default PEFT building block |
| AdaptFormer | ViT | Image/video domains | Bottleneck residual adapters | Freeze backbone; train adapters | Moderate-to-strong; good adaptation capacity | Adds modules and latency | Good ViT baseline |
| VPT | ViT | Recognition/domain adaptation | Learned prompt tokens | Freeze backbone; train prompts | Weaker for pixel-level tasks unless combined with dense head | Limited spatial correction | Cheap ablation |
| ViT-Adapter | Plain ViT | Detection/segmentation | Adapter introduces spatial/multi-scale inductive bias | Train dense adapter/head around ViT | Very strong; designed for dense prediction | Not multimodal fusion | Best dense-prediction ViT comparator |
| SAM-Adapter | SAM | Camouflage, shadow, medical, underperformed scenes | Domain/task adapters or prompts | Avoid full SAM fine-tuning | Strong evidence that SAM needs adaptation | Not inherently multimodal | Motivation source |
| MedSAM | SAM | Medical | Medical SAM adaptation | Domain-adapted SAM workflow | Strong medical segmentation source | Prompt/domain-specific | Domain adaptation evidence |
| SAMed | SAM | Medical semantic segmentation | LoRA image encoder + prompt/mask tuning | Small fraction of SAM parameters updated | Direct semantic segmentation use | Medical-specific | LoRA-SAM recipe |
| MemorySAM | SAM2 | Multimodal semantic segmentation | Modalities as frames; SAM2 memory mechanism; semantic prototype memory | LoRA + memory/prototype training | Closest SAM2 MMSS baseline | No explicit reliability bias | Must compare |
| StitchFusion | Pretrained visual encoders | Any visual modalities | MultiAdapter cross-modal feature weaving | Lightweight adapter exchange among encoders | Strong multimodal segmentation baseline | Not SAM2 memory-based | Strong adapter baseline |
| MoE-LoRA SAM | SAM | Multimodal semantic segmentation | Modality-specific LoRA experts + routing | Frozen SAM; train LoRA experts/router | Direct multimodal SAM LoRA precedent | Router/expert complexity | Very strong comparator |
| SAM-FuseNet | SAM-guided | RGB–thermal aerial perception | SAM-guided multimodal fusion | Full details require paper extraction | Top-journal RGB-T aerial perception | Exact adapter strategy to verify | High-priority source |
| ClassWise-SAM-Adapter | SAM | SAR semantic segmentation | Class-wise PEFT SAM adaptation | PEFT for SAR domain | Sensor-domain segmentation | SAR-specific | Useful sensor-adaptation example |

## Ranked adapter candidates for [[26_MultimodalSeg]]

1. **MemorySAM + reliability-biased memory attention.** Best architectural starting point because it already maps modalities to SAM2 memory frames. Add reliability as a pre-softmax attention-logit bias.
2. **MoE-LoRA for SAM.** Best LoRA expert comparator because it explicitly customizes SAM for multimodal semantic segmentation with modality-specific experts.
3. **StitchFusion MultiAdapter.** Best non-SAM adapter baseline for arbitrary visual modalities; useful for arguing that early/mid feature exchange is a strong alternative to memory-attention fusion.
4. **ViT-Adapter.** Best dense-prediction ViT baseline; helps justify spatial/multi-scale adapters when using plain ViT/SAM image encoders.
5. **SAMed / SAM-Adapter / MedSAM.** Best evidence that SAM requires PEFT/domain adaptation in non-natural-image or underperformed domains.
6. **VPT.** Useful low-parameter ablation but unlikely to be sufficient alone for robust multimodal dense prediction.

## Mechanism matrix by design choice

| Design question | Best-supported answer | Evidence | How to use in our method |
|---|---|---|---|
| How to adapt a huge frozen segmentation model cheaply? | LoRA or adapters | LoRA, AdaptFormer, SAMed, SAM-Adapter | Train small modules while preserving SAM/SAM2 priors |
| How to recover dense spatial detail from ViT tokens? | Dense adapters / multi-scale decoder | ViT-Adapter | Add spatial/multi-scale adaptation before segmentation head |
| How to handle different visual modalities? | Modality-specific adapters/LoRA experts | MoE-LoRA SAM, StitchFusion | Separate modality parameters and fuse adaptively |
| How to use SAM2 for multimodal inputs? | Treat modalities as frame-like observations | MemorySAM | Use memory attention across modalities |
| How to handle corrupted sensors? | Reliability-aware fusion is still underdeveloped | Gap across MemorySAM/MoE-LoRA/StitchFusion | Inject reliability into attention logits rather than only features |

## Detection/segmentation PEFT notes

- For **semantic segmentation**, the adapter must preserve pixel alignment and class semantics. LoRA-only SAM adaptation should be paired with semantic heads, prototypes, or memory modules.
- For **instance segmentation/detection**, adapters must support object localization and multi-scale cues. ViT-Adapter is more directly relevant than classification-only PEFT papers.
- For **RGB-T / RGB-D / Event / SAR**, modality-specific LoRA or adapters are preferable to a single shared update because sensor statistics differ sharply.
- For **robust fusion**, parameter-efficient adaptation should be separated from reliability estimation: adapters answer “how to represent this modality,” while reliability-aware attention answers “how much should this modality be trusted now.”

## Limitations across the field

| Limitation | Appears in | Why it matters |
|---|---|---|
| No explicit uncertainty/reliability in fusion | MemorySAM, MoE-LoRA SAM, StitchFusion | Fusion may still attend to corrupted modalities |
| Semantic segmentation mismatch | Vanilla SAM, some SAM adapters | Promptable masks are not category labels |
| Weak dense spatial bias | VPT, plain ViT LoRA | Pixel boundaries and small objects suffer |
| Expert/router instability | MoE-LoRA | Dominant modalities can collapse routing |
| Domain specificity | SAMed, MedSAM, ClassWise-SAM-Adapter | Medical/SAR gains may not transfer directly to outdoor RGB-X |
| SAM3 unverified | Project seed memo only | Do not build citations on unverified implementation claims |

## Paragraph candidates

**Matrix synthesis paragraph.** The adapter literature suggests a division of labor for multimodal foundation-model segmentation. LoRA and bottleneck adapters provide parameter-efficient domain specialization, ViT-Adapter-style modules recover dense spatial and multi-scale features, and SAM-specific adapters show that promptable segmentation models require downstream customization in medical, SAR, and underperformed natural-image domains. Multimodal works such as MemorySAM, StitchFusion, and MoE-LoRA for SAM extend this idea to sensor fusion, but they mainly adapt or exchange features rather than explicitly modeling sensor reliability.

**Novelty-defense paragraph.** Existing SAM/SAM2 adaptation methods provide strong baselines but leave a gap in reliability-aware multimodal fusion. MemorySAM treats modalities as frame-like inputs to SAM2 memory attention, while MoE-LoRA trains modality-specific SAM experts and StitchFusion weaves features across pretrained encoders. These methods can improve representation and fusion capacity, yet none directly injects a calibrated modality reliability signal into the attention logits. Reliability-biased memory attention is therefore best positioned as a complementary mechanism: it can use LoRA/adapters for efficient adaptation while making the fusion step robust to corrupted or uninformative sensors.

## References

See detailed source notes: [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/21_vit_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], [[relatedworks/01_memorysam_relatedwork]], and [[relatedworks/04_stitchfusion_relatedwork]].
