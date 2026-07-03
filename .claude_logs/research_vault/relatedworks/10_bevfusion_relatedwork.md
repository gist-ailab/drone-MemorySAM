---
title: BEVFusion Related Work — unified BEV multi-sensor fusion
tags: [related-work, key-paper, multimodal-object-detection, camera-lidar, bev, fusion, semantic-segmentation-transfer]
created: 2026-06-24
source: [arXiv:2205.13542](https://arxiv.org/abs/2205.13542), [DOI:10.1109/ICRA48891.2023.10160968](https://doi.org/10.1109/ICRA48891.2023.10160968), [arXiv:2205.13790](https://arxiv.org/abs/2205.13790)
status: verified-draft
---

# BEVFusion Related Work — unified BEV multi-sensor fusion

## Citation metadata

| Item | Metadata |
|---|---|
| Primary paper A | Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela Rus, Song Han. **“BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.”** arXiv:2205.13542v3; ICRA 2023 DOI:10.1109/ICRA48891.2023.10160968. |
| Primary paper B | Tingting Liang, Hongwei Xie, Kaicheng Yu, Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Tieniu Tan. **“BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework.”** arXiv:2205.13790v3. |
| Venue priority | ICRA / major autonomous-driving 3D perception; source map records show high citation count for the ICRA BEVFusion entry. |
| Wikilinks | [[relatedworks/11_transfusion_relatedwork]], [[relatedworks/12_deepinteraction_relatedwork]], [[relatedworks/13_futr3d_relatedwork]], [[relatedworks/14_multimodal_detection_survey_note]], [[relatedworks/42_attention_logit_bias_novelty_defense]] |

## Task / setup

BEVFusion addresses autonomous-driving 3D perception with synchronized camera and LiDAR inputs. The ICRA version emphasizes a **unified bird's-eye-view (BEV) representation** for multiple tasks, including 3D object detection and semantic-oriented BEV/scene understanding. The “simple and robust” variant focuses more narrowly on LiDAR-camera 3D object detection and robustness when one sensor, especially LiDAR, is degraded or absent.

Typical evaluation setup is nuScenes-style multi-camera + LiDAR perception. The important project-level takeaway is not only the detector score, but the architectural decision to fuse modalities after each has been projected into a shared spatial coordinate system rather than forcing dense camera semantics onto sparse points.

## Modality fusion mechanism

| Aspect | BEVFusion design choice | Why it matters |
|---|---|---|
| Fusion space | Transform camera features and LiDAR features into BEV, then fuse. | Avoids point-level projection that discards dense image semantics. |
| Camera branch | Image features are lifted/projected into BEV. | Keeps semantic density useful for segmentation-like tasks. |
| LiDAR branch | Voxel/point-cloud features are encoded into BEV. | Keeps metric geometry and range accuracy. |
| Fusion operator | BEV-level feature fusion; downstream heads operate on unified BEV. | Allows shared heads for detection and map/segmentation tasks. |
| Robustness lesson | The robust variant explicitly motivates designs that do not collapse under LiDAR malfunction. | Closely related to missing/corrupted-modality robustness in [[AnySeg]] and RBMA. |

## Main claims / results

- BEVFusion argues that **point-level camera-to-LiDAR fusion is structurally lossy** for semantic tasks because the projection keeps only image evidence attached to sparse LiDAR points.
- A common BEV coordinate frame enables multi-task, multi-sensor learning and provides a natural interface for 3D boxes, occupancy, map segmentation, and planning-facing perception.
- The robust BEVFusion line shows that fusion should be designed with **sensor failure and degraded modality conditions** in mind rather than assuming all modalities are always reliable.
- For this project, BEVFusion is a top-priority detection analogue of RGB-X semantic segmentation: it changes the *fusion domain* to make cross-modal alignment easier.

## Limitations / caveats

- BEV construction depends on accurate calibration, ego-pose, depth/lift-splat assumptions, and camera/LiDAR synchronization.
- BEV fusion can hide modality-specific uncertainty unless the model preserves explicit reliability signals.
- LiDAR-camera BEV fusion is not directly equivalent to RGB-D/RGB-T/event semantic segmentation because BEV detection uses geometric alignment and 3D supervision.
- Robustness is mostly architectural or empirical; it is not the same as an explicit pre-softmax reliability bias such as the intended RBMA mechanism.

## Relevance to user's multimodal segmentation / detection project

BEVFusion supports the project argument that **fusion should occur where modalities become geometrically comparable**. For semantic segmentation, that may be a shared image/token/memory coordinate system rather than BEV. The direct conceptual transfer is: choose a fusion representation that preserves each modality’s useful evidence, then inject reliability before irreversible aggregation. RBMA can be positioned as complementary: BEVFusion unifies features spatially; RBMA would bias attention logits toward reliable memories/tokens before the attention softmax.

## Related-work paragraph candidates

**Short paragraph.** BEVFusion moved LiDAR-camera fusion from point-level association to a unified bird's-eye-view representation, arguing that projecting image evidence only onto sparse LiDAR points discards dense semantic cues. This lesson is relevant beyond 3D detection: multimodal segmentation also requires a fusion space that preserves modality-specific strengths before aggregation, especially under missing or corrupted sensors.

**Novelty-defense paragraph.** BEVFusion demonstrates the value of a shared spatial representation for multi-sensor perception, but it does not directly solve reliability weighting inside attention. In contrast, an RBMA-style module can be framed as a complementary mechanism that operates at the attention-logit level, biasing memory/token selection according to estimated modality reliability rather than only changing the coordinate frame of fusion.

## References

- Liu et al., “BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation,” arXiv:2205.13542; ICRA 2023.
- Liang et al., “BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework,” arXiv:2205.13790.
