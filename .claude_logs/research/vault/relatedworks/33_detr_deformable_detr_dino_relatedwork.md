---
title: DETR, Deformable DETR, DINO, and transformer detection heads
tags: [related-work, key-paper, detection-head, object-detection, transformer, bev-detection]
created: 2026-06-24
source: [DETR arXiv 2005.12872](https://arxiv.org/abs/2005.12872), [Deformable DETR arXiv 2010.04159](https://arxiv.org/abs/2010.04159), [DINO arXiv 2203.03605](https://arxiv.org/abs/2203.03605), [BEVFormer arXiv 2203.17270](https://arxiv.org/abs/2203.17270)
status: verified-draft
---

# DETR, Deformable DETR, DINO, and transformer detection heads

## Citation metadata

| Method | Main citation | Venue / year | Core task | Link |
|---|---|---:|---|---|
| DETR | Carion et al., “End-to-End Object Detection with Transformers” | ECCV 2020 | 2D object detection | https://arxiv.org/abs/2005.12872 |
| Deformable DETR | Zhu et al., “Deformable DETR: Deformable Transformers for End-to-End Object Detection” | ICLR 2021 | 2D object detection | https://arxiv.org/abs/2010.04159 |
| DINO | Zhang et al., “DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection” | ICLR 2023 | 2D object detection | https://arxiv.org/abs/2203.03605 |
| BEVFormer | Li et al., “BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers” | ECCV 2022 | 3D detection / BEV perception | https://arxiv.org/abs/2203.17270 |
| DETR3D | Wang et al., “DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries” | CoRL 2021 | 3D object detection | https://arxiv.org/abs/2110.06922 |
| PETR | Liu et al., “PETR: Position Embedding Transformation for Multi-View 3D Object Detection” | ECCV 2022 | 3D object detection | https://arxiv.org/abs/2203.05625 |

Related project links: [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/11_transfusion_relatedwork]], [[relatedworks/12_deepinteraction_relatedwork]], [[relatedworks/13_futr3d_relatedwork]], [[relatedworks/14_multimodal_detection_survey_note]].

## Mechanism

**DETR** casts object detection as set prediction. A CNN/Transformer encoder produces image features, a Transformer decoder updates a fixed set of object queries, and bipartite Hungarian matching assigns predictions to ground-truth boxes. The head predicts class labels and boxes without anchors or hand-designed NMS.

**Deformable DETR** addresses DETR's slow convergence and difficulty with small objects by replacing dense global attention with multi-scale deformable attention: each query attends to a sparse set of learned sampling points across feature levels. This is the most relevant DETR variant for attaching to high-resolution multimodal encoders because it can sample from multi-scale fused features efficiently.

**DINO** improves DETR training through contrastive denoising, improved query initialization, and anchor-box formulation. It is a strong canonical modern DETR-family detector and is often a better default than original DETR if the goal is competitive 2D AP.

**3D BEV/transformer detection heads** adapt the query idea to autonomous driving. DETR3D uses 3D object queries projected into camera views; BEVFormer constructs BEV queries and uses spatial/temporal attention to aggregate multi-camera features; PETR encodes 3D positional information into image features; FUTR3D/TransFusion/BEVFusion extend query fusion to LiDAR/radar/camera settings.

## Supported tasks and metrics

| Head family | 2D detection | 3D detection | Semantic segmentation | Instance / panoptic segmentation | Metrics |
|---|:---:|:---:|:---:|:---:|---|
| DETR | yes | via extensions | no | limited via extensions | box AP, AP50/AP75 |
| Deformable DETR | yes | via extensions | no | limited via extensions | box AP, APsmall/medium/large |
| DINO | yes | via extensions | no | limited via extensions | box AP |
| DETR3D / BEVFormer / PETR | no 2D-only focus | yes | optional map/occupancy extensions | no | nuScenes mAP, NDS, Waymo mAP/mAPH |

## Strengths

- Object queries are a natural interface for multimodal fusion: each query can attend to RGB, thermal, LiDAR, radar, or BEV features.
- End-to-end set prediction reduces anchor/NMS heuristics and can be extended to open-vocabulary or prompt-conditioned detection.
- Deformable attention is multimodal-friendly because sparse sampling across scales/views/modalities is a direct design pattern for RGB-X or BEV fusion.
- BEVFormer/DETR3D/PETR show how query heads can consume multi-camera features and geometric projections, which is relevant if [[26_MultimodalSeg]] expands to detection/BEV tasks.

## Limitations

- Original DETR has slow convergence and weak small-object performance relative to later variants.
- Box detectors do not directly produce semantic masks or mIoU; segmentation requires Mask2Former/MaskDINO/Mask R-CNN-style branches.
- 3D BEV heads require calibration, geometry, temporal alignment, and dataset-specific metrics; they are not drop-in for 2D RGB-X segmentation.
- Query attention can still learn modality shortcuts; robustness requires reliability-aware sampling or explicit modality dropout/conflict handling.

## Relevance to semantic segmentation vs object detection

DETR-family heads are the canonical transformer **object detection** counterpart to SegFormer/Mask2Former segmentation heads. They are appropriate when the output is boxes/classes and the metric is AP/NDS, not when the primary output is pixel-level semantic mIoU. For a multimodal SAM-style encoder, DETR/DINO can be attached as a detection branch in parallel with a semantic head.

## Attachment to multimodal / SAM-style encoders

| Project setting | Best head option | Rationale |
|---|---|---|
| 2D object detection from fused RGB-X features | Deformable DETR or DINO | Stronger convergence and multi-scale sampling than original DETR. |
| Joint segmentation + detection | DINO/Deformable DETR plus Mask2Former or MaskDINO | Separates box AP and mask/mIoU evaluation. |
| Camera-LiDAR / BEV detection | BEVFormer, DETR3D, PETR, FUTR3D, TransFusion, BEVFusion | Handles geometry and multi-view/multi-sensor fusion. |
| SAM-style dense tokens | Add detection neck and object-query decoder | SAM mask decoder alone does not output detection boxes. |

## Related-work paragraph candidates

**Concise.** DETR introduced end-to-end object detection as set prediction with Transformer object queries and Hungarian matching. Deformable DETR made the formulation practical for multi-scale features by using sparse deformable attention, while DINO further improved convergence and AP through denoising and anchor-based query improvements. In multimodal perception, the same query mechanism underlies BEVFormer, DETR3D, PETR, FUTR3D, TransFusion, and BEVFusion-style 3D detection heads.

**Project-specific.** For [[26_MultimodalSeg]], DETR-family heads should be treated as detection branches rather than semantic segmentation heads. They are valuable if the project evaluates object AP or extends to 3D/BEV detection, but semantic mIoU still requires SegFormer/UPerNet/Mask2Former-style decoders. Deformable attention provides a strong architectural precedent for reliability-aware multimodal query sampling.

## References

- Carion, N. et al. (2020). *End-to-End Object Detection with Transformers*. ECCV. arXiv:2005.12872.
- Zhu, X. et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR. arXiv:2010.04159.
- Zhang, H. et al. (2023). *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection*. ICLR. arXiv:2203.03605.
- Li, Z. et al. (2022). *BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*. ECCV. arXiv:2203.17270.
- Wang, Y. et al. (2021). *DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries*. CoRL. arXiv:2110.06922.
- Liu, Y. et al. (2022). *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*. ECCV. arXiv:2203.05625.
