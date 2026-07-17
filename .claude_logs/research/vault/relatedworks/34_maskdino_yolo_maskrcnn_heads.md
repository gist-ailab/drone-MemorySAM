---
title: MaskDINO, YOLO, Mask R-CNN, and practical detection/segmentation heads
tags: [related-work, key-paper, detection-head, segmentation-head, instance-segmentation, yolo, mask-rcnn]
created: 2026-06-24
source: [MaskDINO arXiv 2206.02777](https://arxiv.org/abs/2206.02777), [Mask R-CNN arXiv 1703.06870](https://arxiv.org/abs/1703.06870), [YOLO arXiv 1506.02640](https://arxiv.org/abs/1506.02640)
status: verified-draft
---

# MaskDINO, YOLO, Mask R-CNN, and practical detection/segmentation heads

## Citation metadata

| Method | Main citation | Venue / year | Core tasks | Link |
|---|---|---:|---|---|
| MaskDINO | Li et al., “Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation” | CVPR 2023 | detection, instance segmentation, panoptic segmentation | https://arxiv.org/abs/2206.02777 |
| Mask R-CNN | He et al., “Mask R-CNN” | ICCV 2017 | detection, instance segmentation | https://arxiv.org/abs/1703.06870 |
| YOLO | Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection” | CVPR 2016 | real-time object detection | https://arxiv.org/abs/1506.02640 |
| YOLO family | YOLOv3/v4/v5/v7/v8/v9/v10/v11-style descendants | 2018–2025 | real-time detection, sometimes segmentation/pose | varies by implementation |
| BEV/3D heads | CenterPoint, TransFusion, BEVFusion, FUTR3D, BEVFormer families | 2020–2023 | 3D detection / BEV perception | see [[relatedworks/14_multimodal_detection_survey_note]] |

Related project links: [[relatedworks/31_mask2former_relatedwork]], [[relatedworks/33_detr_deformable_detr_dino_relatedwork]], [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/11_transfusion_relatedwork]].

## Mechanism

**MaskDINO** unifies DETR-style detection and mask prediction. It combines DINO-style object detection improvements with mask prediction so that the same query framework can output boxes and masks. This makes it a bridge between [[relatedworks/33_detr_deformable_detr_dino_relatedwork]] and [[relatedworks/31_mask2former_relatedwork]].

**Mask R-CNN** is the canonical two-stage instance segmentation baseline. A Region Proposal Network proposes boxes; RoIAlign extracts aligned region features; parallel heads predict class, refined box, and binary mask. Its modularity makes it a strong reference for separating detection and mask losses, even when newer transformer heads outperform it.

**YOLO family** represents one-stage real-time detection. YOLO predicts dense boxes/classes directly over feature maps. Later versions add stronger backbones/necks, anchor-free variants, decoupled heads, distributional box losses, segmentation branches, and deployment-oriented optimizations. For this note, the key point is not one specific version but the family’s role as the speed/deployment baseline for box detection.

**3D/BEV detection heads** include anchor/center-based heads such as CenterPoint, query-based heads such as TransFusion and FUTR3D, and BEV transformer heads such as BEVFormer/BEVFusion. They predict 3D boxes in ego/world coordinates and are evaluated by nuScenes/Waymo metrics rather than 2D mIoU.

## Supported tasks and metrics

| Head | Semantic segmentation | Instance segmentation | Panoptic segmentation | 2D detection | 3D detection | Metrics |
|---|:---:|:---:|:---:|:---:|:---:|---|
| MaskDINO | yes via mask aggregation | yes | yes | yes | via extensions | mIoU, mask AP, PQ, box AP |
| Mask R-CNN | no | yes | via Panoptic FPN extensions | yes | via 3D variants | box AP, mask AP |
| YOLO family | no mostly | some versions yes | no | yes primary | via YOLO3D variants | FPS/latency, box AP, mAP50-95 |
| CenterPoint / BEV heads | no unless extended | no | no | no | yes primary | nuScenes mAP/NDS, Waymo mAP/mAPH |

## Strengths

- MaskDINO is the strongest conceptual fit when the project wants one transformer head for boxes and masks. It can evaluate detection AP and segmentation mask quality from a shared query representation.
- Mask R-CNN is stable, interpretable, modular, and widely recognized; useful as a conservative baseline or citation anchor for instance segmentation.
- YOLO is real-time and deployment-oriented; useful if the multimodal system must run on embedded/robotics hardware.
- 3D BEV heads are necessary for camera-LiDAR/radar detection tasks where outputs are 3D boxes rather than image masks.

## Limitations

- MaskDINO is more complex than semantic-only heads and requires both detection and mask supervision for full benefit.
- Mask R-CNN is region-proposal based and less aligned with transformer/foundation-model query architectures.
- YOLO is not naturally a semantic segmentation head; YOLO-seg variants output instance masks but not full stuff/semantic parsing in the same way as Mask2Former/OneFormer.
- 3D/BEV heads depend on calibration, coordinate frames, and dataset-specific assumptions, so they are not plug-and-play for 2D multimodal segmentation.

## Relevance to semantic segmentation vs object detection

MaskDINO is the most relevant method in this note for a **joint segmentation-detection** project. Mask R-CNN and YOLO are primarily object-detection / instance-segmentation baselines. For pure semantic segmentation mIoU, use [[relatedworks/30_segformer_relatedwork]] or [[relatedworks/31_mask2former_relatedwork]]. For detection AP and real-time constraints, use YOLO/Mask R-CNN/DINO. For BEV autonomous-driving detection, use BEVFormer/TransFusion/BEVFusion-style heads.

## Attachment to multimodal / SAM-style encoders

| Head | Attachment strategy | Multimodal/SAM implication |
|---|---|---|
| MaskDINO | Add query detector-mask decoder on fused/SAM features | Best unified box+mask candidate; reliability can bias query attention. |
| Mask R-CNN | Use fused feature pyramid as backbone output | Strong modular baseline but less SAM-native. |
| YOLO | Use fused feature pyramid/neck before YOLO head | Good for real-time RGB-X detection; not ideal for semantic classes/stuff. |
| BEV/3D heads | Transform multimodal features into BEV, then detect | Needed for LiDAR-camera/radar 3D object detection. |

## Related-work paragraph candidates

**Concise.** MaskDINO extends the DETR/DINO query-detection paradigm to unified object detection and segmentation, making it a natural bridge between box AP and mask metrics. Mask R-CNN remains the canonical two-stage detection-plus-instance-mask architecture, while YOLO-family detectors define the practical real-time detection baseline. In autonomous-driving settings, BEV/3D heads such as CenterPoint, BEVFormer, TransFusion, FUTR3D, and BEVFusion adapt these ideas to calibrated multi-sensor 3D boxes.

**Project-specific.** For [[26_MultimodalSeg]], MaskDINO should be considered if the final system needs both object detection and mask outputs from a SAM-style multimodal encoder. YOLO is appropriate for a speed/robotics detection branch, and Mask R-CNN is appropriate as a conservative instance-segmentation baseline. None of these replace a semantic mIoU head unless the dataset and inference pipeline explicitly convert masks into semantic labels.

## References

- Li, F. et al. (2023). *Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation*. CVPR. arXiv:2206.02777.
- He, K., Gkioxari, G., Dollar, P., and Girshick, R. (2017). *Mask R-CNN*. ICCV. arXiv:1703.06870.
- Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR. arXiv:1506.02640.
- Redmon, J. and Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767.
- Li, Z. et al. (2022). *BEVFormer*. ECCV. arXiv:2203.17270.
- Bai, X. et al. (2022). *TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers*. CVPR. See [[relatedworks/11_transfusion_relatedwork]].
- Liang, T. et al. (2022). *BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework*. See [[relatedworks/10_bevfusion_relatedwork]].
