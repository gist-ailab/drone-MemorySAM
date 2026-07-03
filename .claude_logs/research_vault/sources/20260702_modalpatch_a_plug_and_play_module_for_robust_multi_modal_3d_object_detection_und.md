---
title: "ModalPatch: A Plug-and-Play Module for Robust Multi-Modal 3D Object Detection under Modality Drop"
tags: [source-note, weekly-sweep, multimodal_object_detection]
created: 2026-07-02
source: arXiv
status: candidate-verified-metadata
---

# ModalPatch: A Plug-and-Play Module for Robust Multi-Modal 3D Object Detection under Modality Drop

- Project link: [[00_MOC_26_MultimodalSeg]]; weekly log: [[sources/04_weekly_source_sweep_log]]
- Category: `multimodal_object_detection`
- Venue/year: arXiv cs.CV / 2026
- Date: 2026-03-03
- Primary URL: https://arxiv.org/abs/2603.02481v1
- arXiv/DOI: 2603.02481v1
- Discovery channel: arXiv
- Verification status: Primary arXiv metadata verified via arXiv API; full PDF/method details still need reading.

## Why this matters

camera-LiDAR/radar multimodal detection fusion 및 adverse/missing-modality 강건성 비교축.

## Abstract / description snapshot

> Multi-modal 3D object detection is pivotal for autonomous driving, integrating complementary sensors like LiDAR and cameras. However, its real-world reliability is challenged by transient data interruptions and missing, where modalities can momentarily drop due to hardware glitches, adverse weather, or occlusions. This poses a critical risk, especially during a simultaneous modality drop, where the vehicle is momentarily blind. To address this problem, we introduce ModalPatch, the first plug-and-play module designed to enable robust detection under arbitrary modality-drop scenarios. Without requiring architectural changes or retraining, ModalPatch can be seamlessly integrated into diverse detection frameworks. Technically, ModalPatch leverages the temporal nature of sensor data for perceptual continuity, using a history-based module to predict and compensate for transiently unavailable features. To improve the fidelity of the predicted features, we further introduce an uncertainty-guided cross-modality fusion strategy that dynamically estimates the reliability of compensated features, suppressing biased signals while reinforcing informative ones. Extensive experiments show that ModalPatch consistently enhances both robustness and accuracy of state-of-the-art 3D object detectors under diverse modality-drop conditions.

## Follow-up extraction checklist

- [ ] Read full paper/project page.
- [ ] Extract method diagram, fusion/adaptation mechanism, datasets, and metrics.
- [ ] Compare against [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/20_lora_adapter_relatedwork]], [[relatedworks/22_sam_adapter_relatedwork]], and [[relatedworks/30_segformer_relatedwork]] as applicable.
- [ ] Decide whether to promote into a full [[relatedworks/00_relatedworks_index]] note.
