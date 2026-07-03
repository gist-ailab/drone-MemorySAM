---
title: SAM, SAM2, and SAM3 adapter/LoRA related work for segmentation
tags: [related-work, sam, sam2, sam3, lora, adapter, medsam, samed, memorysam, multimodal-segmentation, key-paper]
created: 2026-06-24
source: arXiv:2304.09148; DOI:10.1038/s41467-024-44824-z; arXiv:2304.13785; arXiv:2503.06700; arXiv:2412.04220; DOI:10.1109/TGRS.2025.3648127
status: verified-draft
---

# SAM, SAM2, and SAM3 adapter/LoRA related work for segmentation

## Scope and verification note

This note covers adapter and LoRA adaptation of Segment Anything family models for dense segmentation, with emphasis on SAM/SAM2 and multimodal sensor adaptation. The requested **SAM3** coverage is treated cautiously: the project seed memo mentions a SAM3 portability hypothesis, but I did not find a stable primary SAM3 paper/source in the project database or arXiv queries during this cron run. Therefore, SAM3 is not cited as established related work here; it is listed as an open verification item.

## Citation metadata

| Method | Primary source | Venue / status | Adaptation mechanism | Relevance to [[26_MultimodalSeg]] |
|---|---|---|---|---|
| SAM-Adapter | Chen et al., “SAM Fails to Segment Anything? — SAM-Adapter,” arXiv:2304.09148v3; DOI:10.1109/ICCVW60793.2023.00361 | ICCV Workshop 2023 / arXiv | Adds domain-specific adapters/prompts instead of full SAM fine-tuning | First-wave evidence that SAM needs lightweight domain adaptation for underperformed scenes |
| MedSAM | Ma et al., “Segment anything in medical images,” *Nature Communications*, 2024; DOI:10.1038/s41467-024-44824-z | Nature Communications 2024 | Medical-domain SAM adaptation, commonly box-prompted for medical segmentation | Strong primary source showing SAM adaptation in a high-value non-natural-image domain |
| SAMed | Zhang and Liu, “Customized Segment Anything Model for Medical Image Segmentation,” arXiv:2304.13785v2 | arXiv 2023 | LoRA fine-tuning of SAM image encoder plus prompt encoder and mask decoder tuning | Direct LoRA-on-SAM precedent |
| MemorySAM | Liao et al., “MemorySAM,” arXiv:2503.06700v2 | arXiv 2025 | Treats modalities as SAM2 frame sequence; uses SAM2 memory mechanisms and LoRA adaptation | Closest architectural baseline for SAM2 multimodal semantic segmentation |
| MoE-LoRA SAM | Zhu et al., arXiv:2412.04220v1 | arXiv 2024 | Mixture of LoRA experts tailored to visual modalities; frozen SAM weights | Direct SAM multimodal semantic segmentation LoRA baseline |
| SAM-FuseNet | Zhu et al., “SAM-FuseNet: Segment Anything Guided Multimodal Fusion for RGB–Thermal Aerial Robotic Perception,” DOI:10.1109/TGRS.2025.3648127 | IEEE TGRS 2026 metadata / project DB lists 2025/2026 | SAM-guided RGB–thermal multimodal fusion | Important top-journal multimodal SAM-guided segmentation/detection-adjacent source |
| ClassWise-SAM-Adapter | Pu et al., “ClassWise-SAM-Adapter,” DOI:10.1109/JSTARS.2025.3532690; arXiv:2401.02326 | IEEE JSTARS 2025 | Class-wise parameter-efficient SAM adaptation for SAR semantic segmentation | Sensor-domain adaptation example for SAR |

## Method mechanisms

### SAM-Adapter

SAM-Adapter argues that SAM performs poorly in underrepresented scenarios such as camouflage, shadow, and medical images. Rather than fine-tuning the full SAM network, it injects task/domain information through adapters or visual prompts. This establishes the basic adapter thesis for SAM: the foundation model is powerful but not sufficient when the target domain has different appearance statistics or task semantics.

### MedSAM and SAMed

MedSAM provides a strong medical-image adaptation reference in a peer-reviewed journal. It demonstrates that SAM-style foundation segmentation can be adapted to medical imaging, where image statistics and annotation protocols differ substantially from natural-image data. SAMed is especially relevant for PEFT because it applies LoRA to the SAM image encoder and tunes the prompt encoder and mask decoder for medical semantic segmentation. SAMed's mechanism is a clean example of using low-rank updates to make SAM domain-aware without full encoder fine-tuning.

### MemorySAM

MemorySAM adapts SAM2 to multimodal semantic segmentation by treating modalities as a sequence of frames representing the same scene. It uses SAM2's video memory mechanism to capture modality-agnostic information and applies LoRA-style adaptation. This is the closest known baseline for a SAM2 multimodal method and should be cited whenever describing modality-as-frame fusion. See [[relatedworks/01_memorysam_relatedwork]].

### MoE-LoRA for multimodal SAM

MoE-LoRA for SAM keeps SAM weights frozen and trains modality-tailored LoRA experts. Its routing strategy adaptively weights features across modalities to reduce cross-modal inconsistencies. This is directly relevant to the project because it combines three design choices also under consideration: frozen SAM, LoRA experts, and multimodal semantic segmentation. It is less directly memory-based than MemorySAM, so it is a complementary baseline rather than a replacement.

### SAM-FuseNet

SAM-FuseNet is listed in the project database and Crossref as an IEEE TGRS paper titled “Segment Anything Guided Multimodal Fusion for RGB–Thermal Aerial Robotic Perception.” It should be treated as a high-priority primary source for RGB–thermal aerial perception. The title indicates SAM-guided multimodal fusion rather than generic adapter tuning; therefore it is most useful as evidence that SAM priors are being integrated into multimodal remote-sensing/robotics perception pipelines.

## Parameter / update strategy

| Method | Frozen base? | Trainable/adapted parts | Fusion/adaptation level | Notes |
|---|---|---|---|---|
| SAM-Adapter | Mostly yes | Adapter/prompt modules | Domain/task adaptation | Good underperformed-scene baseline |
| MedSAM | Adapted SAM family model | Medical-specific adaptation and prompt workflow | Domain-specific segmentation | Peer-reviewed medical evidence |
| SAMed | Mostly frozen SAM image encoder with LoRA updates; prompt/mask parts tuned | LoRA + prompt encoder + mask decoder | Medical semantic segmentation | Direct LoRA-SAM recipe |
| MemorySAM | SAM2 adapted with LoRA | SAM2 memory mechanisms + semantic prototype memory module during training | Multimodal fusion as frame/memory reasoning | Closest to RBMA direction |
| MoE-LoRA SAM | Frozen SAM base | Multiple LoRA experts + router | Modality-specific adaptation and integration | Best LoRA-expert comparator |
| SAM-FuseNet | Needs full paper extraction for exact freeze/update details | SAM-guided RGB–thermal fusion components | Aerial robotic perception | Top-journal source; verify architecture before detailed claims |

## Dense prediction relevance

SAM-family methods are inherently segmentation-oriented, but vanilla SAM is promptable mask generation rather than supervised semantic segmentation. The adaptation literature is therefore important because it bridges foundation-model masks to task-specific semantic outputs. SAMed and MemorySAM explicitly target semantic segmentation settings. ViT/SAM encoder LoRA adapts representations, mask-decoder tuning adapts output semantics, and memory/adapter modules adapt multimodal interactions.

## Limitations and caveats

- **SAM is not automatically semantic.** SAM produces masks; semantic segmentation requires category supervision, prototypes, or a semantic decoder.
- **Prompt dependence.** Some SAM methods depend on boxes/points, which may not be available in autonomous multimodal perception.
- **Domain overfitting.** Medical/SAR/thermal adapters may not generalize across sensors without modular routing.
- **Memory vs. reliability.** MemorySAM fuses modalities through SAM2 memory but does not explicitly bias attention by sensor reliability.
- **SAM3 uncertainty.** No stable primary SAM3 source was verified in this run; do not cite SAM3 implementation details until code/paper is inspected.

## Comparison table

| Axis | SAM-Adapter | SAMed / MedSAM | MemorySAM | MoE-LoRA SAM | SAM-FuseNet | Our RBMA direction |
|---|---|---|---|---|---|---|
| Main target | Underperformed scenes | Medical images | Multimodal semantic segmentation | Multimodal semantic segmentation | RGB–thermal aerial perception | Robust multimodal SAM2/SAM-style fusion |
| PEFT type | Adapter/prompt | LoRA/domain adaptation | LoRA + memory/prototypes | Mixture of LoRA experts | SAM-guided fusion; exact PEFT details to verify | LoRA/adapters + reliability-biased attention |
| Foundation model | SAM | SAM | SAM2 | SAM | SAM-guided | SAM2 first, possible SAM/SAM3 port later |
| Fusion mechanism | Not central | Not central | Modalities as frames in memory | Expert routing/fusion | RGB–thermal fusion | Additive reliability bias in memory attention logits |
| Best citation use | Need for SAM adaptation | Domain-specific SAM PEFT | Closest baseline | Closest LoRA expert baseline | Top-journal multimodal SAM-guided precedent | Proposed novelty |

## Paragraph candidates

**SAM adaptation paragraph.** Although SAM provides strong generic segmentation priors, several studies show that direct zero-shot use is insufficient in specialized domains. SAM-Adapter introduces lightweight task/domain adapters for underperformed scenes, while MedSAM and SAMed adapt SAM to medical-image segmentation, with SAMed using LoRA updates in the image encoder. These works support the view that foundation segmentation models require parameter-efficient specialization when target images differ substantially from the natural-image distribution.

**Multimodal SAM paragraph.** Recent work extends SAM-family models from single-modality adaptation to multimodal semantic segmentation. MemorySAM treats different modalities as frame-like observations of the same scene and uses SAM2 memory mechanisms to fuse modality-agnostic information. In parallel, MoE-LoRA customizes SAM with modality-specific low-rank experts and adaptive routing, and SAM-FuseNet investigates SAM-guided RGB–thermal fusion for aerial robotic perception. These methods establish SAM adaptation as a viable route for multimodal segmentation, but they leave open how to explicitly estimate and inject modality reliability into the fusion attention mechanism.

## References

- Chen, T., Zhu, L., Ding, C., Cao, R., Wang, Y., Li, Z., Sun, L., Mao, P., and Zang, Y. (2023). *SAM Fails to Segment Anything? — SAM-Adapter: Adapting SAM in Underperformed Scenes*. arXiv:2304.09148v3; ICCV Workshops. DOI:10.1109/ICCVW60793.2023.00361.
- Ma, J., He, Y., Li, F., Han, L., You, C., and Wang, B. (2024). *Segment anything in medical images*. Nature Communications. DOI:10.1038/s41467-024-44824-z.
- Zhang, K., and Liu, D. (2023). *Customized Segment Anything Model for Medical Image Segmentation*. arXiv:2304.13785v2.
- Liao, C., Zheng, X., Lyu, Y., Xue, H., Cao, Y., Wang, J., Yang, K., and Hu, X. (2025). *MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation*. arXiv:2503.06700v2.
- Zhu, C., Xiao, B., Shi, L., Xu, S., and Zheng, X. (2024). *Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts*. arXiv:2412.04220v1.
- Zhu, C., Wang, J., Zhang, L., et al. (2026). *SAM-FuseNet: Segment Anything Guided Multimodal Fusion for RGB–Thermal Aerial Robotic Perception*. IEEE Transactions on Geoscience and Remote Sensing. DOI:10.1109/TGRS.2025.3648127.
- Pu, X., Jia, H., Zheng, L., et al. (2025). *ClassWise-SAM-Adapter: Parameter-Efficient Fine-Tuning Adapts Segment Anything to SAR Domain for Semantic Segmentation*. IEEE JSTARS. DOI:10.1109/JSTARS.2025.3532690.
