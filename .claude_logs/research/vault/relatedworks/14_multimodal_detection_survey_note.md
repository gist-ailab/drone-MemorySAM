---
title: Multimodal Object Detection Survey Note — camera, LiDAR, radar, thermal, event
tags: [related-work, survey-note, multimodal-object-detection, camera-lidar, radar, thermal, event, adverse-weather, rgb-t]
created: 2026-06-24
source: [[sources/02_source_map_multimodal_object_detection]], arXiv/OpenAlex verification run 2026-06-24
status: verified-draft
---

# Multimodal Object Detection Survey Note — camera, LiDAR, radar, thermal, event

## Scope and priority

This note summarizes the detection-side related-work package for [[00_MOC_26_MultimodalSeg]]. It prioritizes top venues and high-impact journals in the requested scope: CVPR, ICCV, ECCV, NeurIPS, ICML, WACV, IROS, ICRA, RA-L, TPAMI, IJCV, TITS, TGRS, TMM, and Information Fusion. The focus is multimodal object detection and 3D detection using camera/RGB, LiDAR/depth, radar, thermal, and event sensors, with explicit links back to semantic segmentation and RBMA.

## Core papers added in this package

| Group | Paper / method | Venue / source | Modalities | Fusion mechanism | Main project lesson |
|---|---|---|---|---|---|
| BEV fusion | [[relatedworks/10_bevfusion_relatedwork]] | ICRA 2023 / arXiv | Camera + LiDAR | Project both streams to unified BEV, then fuse. | Choose a fusion representation that preserves dense semantics and metric geometry. |
| Query fusion | [[relatedworks/11_transfusion_relatedwork]] | CVPR 2022 | LiDAR + camera | Object queries softly attend to image evidence. | Replace brittle hard matching with learned attention. |
| Modality interaction | [[relatedworks/12_deepinteraction_relatedwork]] | arXiv / top 3D detection lineage | LiDAR + camera | Preserve per-modality streams with deep interaction. | Avoid premature modality collapse and unimodal dominance. |
| Unified sensor set | [[relatedworks/13_futr3d_relatedwork]] | CVPRW 2023 / arXiv | Camera + LiDAR + radar | Modality-agnostic feature sampler + transformer decoder. | Support arbitrary/missing sensor combinations. |

## Required baseline and context coverage

### Camera-only / multi-view 3D detection foundations

| Method | Citation metadata | Role in related work | Link to segmentation/RBMA |
|---|---|---|---|
| DETR3D | “DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries,” commonly cited in the multi-view camera 3D detection lineage. | Introduces 3D object queries that sample multi-view image features by projecting 3D reference points to 2D views. | Shows that query-token attention can bridge spatial domains; RBMA can bias such attention when modalities/memories differ in reliability. |
| BEVFormer | Li et al., “BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers,” arXiv:2203.17270. | Learns BEV representations from multi-camera images using spatial and temporal transformers. | Strong bridge to segmentation because BEV supports detection and map/semantic tasks. |
| PETR / PETRv2 | Liu et al., “PETR: Position Embedding Transformation for Multi-View 3D Object Detection,” arXiv:2203.05625; PETRv2 arXiv:2206.01256. | Encodes 3D coordinate/position information into image features for multi-view 3D perception. | Positional encoding is an alternative to explicit modality reliability; useful but orthogonal to RBMA. |

### Early, middle, and late LiDAR-camera fusion baselines

| Method | Citation metadata | Fusion type | Key lesson |
|---|---|---|---|
| MVX-Net | Sindagi, Zhou, Tuzel, “MVX-Net: Multimodal VoxelNet for 3D Object Detection,” arXiv:1904.01649. | Early/mid voxel-level camera-LiDAR fusion. | Classic baseline showing how image features can be injected into voxel/point representations. |
| PointPainting | Vora, Lang, Helou, Beijbom, “PointPainting: Sequential Fusion for 3D Object Detection,” CVPR 2020 / arXiv:1911.10150. | Sequential fusion: 2D semantic segmentation scores are appended to LiDAR points. | Strong connection to semantic segmentation: segmentation outputs can supervise or enrich detection. However, errors in the 2D segmenter propagate to 3D. |
| CLOCs | Pang, Morris, Radha, “CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection,” arXiv:2009.00784. | Late candidate-level fusion of 2D and 3D detections. | Cheap and modular, but less able to learn dense cross-modal interactions. |
| CenterPoint | Yin, Zhou, Krahenbuhl, “CenterPoint: 3D Object Detection and Tracking,” CVPR 2021 lineage. | LiDAR-centered anchor/center-based detection baseline. | Important single-modality anchor when judging whether fusion truly helps. |

### Radar-camera-LiDAR fusion

Radar adds velocity and adverse-weather robustness but is sparse/noisy. FUTR3D is the most relevant required paper because it explicitly targets flexible sensor configurations including radar. Additional source-map hits include radar/vision BEV fusion and camera-radar networks such as CRN and UniBEVFusion. For the user's project, radar literature mainly supports the claim that **sensor reliability is condition-dependent**: radar may be more reliable in fog/rain for range/velocity, while RGB is more semantic in clear daylight.

### RGB-T / thermal detection baselines

RGB-T detection is the 2D detection analogue of RGB-T semantic segmentation. Typical baselines include multispectral pedestrian detection and visible-thermal fusion detectors, often evaluated under night, low-light, or difficult weather. In the existing source map, examples include “Thermal Object Detection in Difficult Weather Conditions Using YOLO,” “Robust Pedestrian Detection Based on Multi-Spectral Image Fusion and Convolutional Neural Networks,” and visible-thermal transfer-learning works. The conceptual relevance is direct: thermal is not merely an extra channel; it becomes the more reliable modality at night or under illumination failure.

### Event-sensor detection

Event cameras provide asynchronous brightness-change signals and are useful for high dynamic range, motion blur, and low-latency scenarios. The project source map includes high-impact event-vision survey literature, including TPAMI-level event-based vision surveys. Event detection work is less central than camera-LiDAR BEV detection for this package, but it is important for the general RBMA argument: modality reliability varies with dynamics and lighting, so attention should be able to prefer event memories/tokens when frame RGB is blurred or saturated.

### Adverse-weather / night object detection

Adverse-weather detection papers support the robustness axis: fog, rain, snow, night, sensor occlusion, and low illumination change which modality is trustworthy. Source-map examples include foggy-weather TransFusion variants, LossDistillNet for harsh-weather point clouds, thermal object detection in difficult weather, and nighttime pedestrian detection. These works should be cited as evidence that benchmark performance under clean conditions is insufficient for deployment-facing multimodal perception.

## Synthesis table — fusion mechanism vs. RBMA relevance

| Fusion family | Representative methods | Strength | Limitation | RBMA connection |
|---|---|---|---|---|
| Point/voxel-level fusion | MVX-Net, PointPainting | Simple geometric association; directly injects image semantics into LiDAR. | Projection errors and 2D segmentation errors propagate; sparse points discard dense image context. | RBMA can avoid blindly trusting projected evidence by weighting reliability. |
| Candidate-level late fusion | CLOCs | Modular; combines mature 2D and 3D detectors. | Limited deep feature interaction. | Useful baseline but weaker novelty threat to attention-logit bias. |
| BEV-level fusion | BEVFusion | Shared metric representation; strong for detection/map/segmentation tasks. | Reliability is often implicit; depends on calibration/lift quality. | Complementary spatial representation for reliability-aware attention. |
| Transformer query fusion | TransFusion, FUTR3D, DETR3D/PETR | Soft learned association across views/modalities. | Attention can still be dominated by unreliable features without explicit bias. | Direct architectural motivation for RBMA. |
| Modality-preserving interaction | DeepInteraction | Avoids premature collapse; preserves modality-specific evidence. | More complex; reliability still implicit. | Supports anti-collapse and reliability-weighted memory design. |
| RGB-T / radar / event robust fusion | Thermal, radar-camera, event detection | Sensor strengths change by condition. | Often benchmark/task-specific. | Strong empirical motivation for condition-aware reliability estimates. |

## Limitations and open verification items

- Some venue details for DETR3D, PETR/PETRv2, and CenterPoint should be rechecked against the final bibliography when the paper draft is assembled.
- Semantic Scholar and OpenAlex rate-limited during the scheduled run, so this note relies on existing project source maps plus arXiv API verification for the central papers.
- Exact benchmark numbers are intentionally not copied here unless verified from tables; use this note for related-work synthesis, not quantitative leaderboard claims.

## Recommended related-work paragraph for the user's paper

Multimodal object detection has explored several fusion locations and sensor configurations. Early LiDAR-camera methods such as MVX-Net and PointPainting inject image semantics into point or voxel representations, while CLOCs performs late candidate-level fusion. More recent 3D detectors move toward shared BEV or query-based representations: BEVFusion projects camera and LiDAR streams into a unified BEV space, TransFusion uses transformer queries to softly associate LiDAR proposals with image evidence, DeepInteraction preserves modality-specific streams while enabling cross-modal interaction, and FUTR3D generalizes query-based sampling to flexible camera/LiDAR/radar configurations. These works show that robust multimodal perception requires both an appropriate fusion representation and mechanisms that avoid brittle hard association. However, most methods treat reliability implicitly; our RBMA direction instead targets the attention operation itself by biasing memory/token selection according to modality reliability before feature aggregation.

## 2026-07-02 deep-research update (Track 5 — condition-adaptive detection + VFM)

Track 5 deep-research (adversarially verified) supersedes the vague "adverse-weather / night object detection" and "reliability is implicit" paragraphs above with concrete, mechanism-classified entrants. Full detail, taxonomy table, and verbatim numbers: [[relatedworks/15_condition_adaptive_detection]].

Key corrections/additions to this survey note:

1. **Explicit reliability handling now exists in detection** — but all published cases are learned + multiplicative post-softmax: ReliFusion (2502.01856, contrastive confidence × attention output, Eq.13–15 [VERIFIED-PDF]) and ModalPatch (2603.02481, NLL-trained variance head, `W̃ = W·[1−softmax(U)]` on deformable-attention weights [VERIFIED-PDF]). Neither is training-free nor pre-softmax.
2. **Additive pre-softmax logit biases ARE accepted machinery in det decoders** — MEFormer (2407.19156, Eq.9: box-center proximity bias `M = α·dist + β`) and, earlier, SMCA-DETR (2101.07448, ICCV 2021, log-Gaussian spatial priors). Both signals are geometric, not reliability → the RBMA cell (training-free entropy reliability → additive pre-softmax bias) remains unoccupied in detection as of 2026-07 (search-supported negative; one 403-blocked near-neighbor SeBFusion/BCAF doi:10.3390/app16062943 on the watch-list).
3. **Condition routing in detection is uniformly weather-supervised** (AW-MoE 2603.16261: GT-label-trained classifier, ~99% routing acc; WCBR 2604.05405: "weather-supervised" router) — supports P29 SDC's unsupervised-condition-latent novelty.
4. **VFM injection in multimodal detection** = robust-encoder feature fusion (RoboFusion, IJCAI 2024) or distillation (DINOv2→BEV 2510.10287; Thermal-Det 2605.10130); no detection work uses SAM2 memory attention for fusion (nearest: M4-SAM 2605.11760 — saliency segmentation, fusion pre-memory).
5. **Strongest adverse-weather motivation numbers**: SAMFusion (2508.16408) on SeeingThroughFog pedestrian 50–80 m: Fog 34.31 AP (+17.2), Snow 41.45 (+15.62) [VERIFIED-PDF]; ModalPatch +10~17 mAP under 50% modality drop on nuScenes val [VERIFIED-PDF].

## References

- Liu et al., “BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation,” ICRA 2023 / arXiv:2205.13542.
- Liang et al., “BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework,” arXiv:2205.13790.
- Bai et al., “TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers,” CVPR 2022 / arXiv:2203.11496.
- Yang et al., “DeepInteraction: 3D Object Detection via Modality Interaction,” arXiv:2208.11112.
- Chen et al., “FUTR3D: A Unified Sensor Fusion Framework for 3D Detection,” CVPRW 2023 / arXiv:2203.10642.
- Vora et al., “PointPainting: Sequential Fusion for 3D Object Detection,” CVPR 2020 / arXiv:1911.10150.
- Sindagi et al., “MVX-Net: Multimodal VoxelNet for 3D Object Detection,” arXiv:1904.01649.
- Pang et al., “CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection,” arXiv:2009.00784.
- Li et al., “BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers,” arXiv:2203.17270.
- Liu et al., “PETR: Position Embedding Transformation for Multi-View 3D Object Detection,” arXiv:2203.05625.
