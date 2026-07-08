---
title: TransFusion Related Work — robust LiDAR-camera transformer fusion
tags: [related-work, key-paper, multimodal-object-detection, camera-lidar, transformer, robust-fusion]
created: 2026-06-24
source: [arXiv:2203.11496](https://arxiv.org/abs/2203.11496), [DOI:10.1109/CVPR52688.2022.00116](https://doi.org/10.1109/CVPR52688.2022.00116)
status: verified-draft
---

# TransFusion Related Work — robust LiDAR-camera transformer fusion

## Citation metadata

| Item | Metadata |
|---|---|
| Paper | Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, Chiew-Lan Tai. **“TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers.”** CVPR 2022; arXiv:2203.11496v1; DOI:10.1109/CVPR52688.2022.00116. |
| Venue priority | CVPR 2022; top-venue detection/fusion paper in the project source map. |
| Modalities | LiDAR + multi-view camera/RGB. |
| Wikilinks | [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/12_deepinteraction_relatedwork]], [[relatedworks/13_futr3d_relatedwork]], [[relatedworks/42_attention_logit_bias_novelty_defense]] |

## Task / setup

TransFusion studies 3D object detection for autonomous driving using LiDAR and camera inputs. Its stated motivation is robustness to inferior image conditions and calibration/sensor misalignment, because many earlier fusion methods rely on hard LiDAR-point-to-image-pixel associations.

## Modality fusion mechanism

| Component | Description | Project lesson |
|---|---|---|
| LiDAR-first detection | LiDAR features/proposals provide reliable geometric object hypotheses. | Strong modality can anchor fusion without making image evidence mandatory. |
| Transformer decoder | Object queries attend to image features in a learned, soft way. | Soft attention is less brittle than fixed point-pixel pairing. |
| Image-guided refinement | Camera semantics supplement LiDAR proposals. | Use RGB semantics as enhancement, not as a hard dependency. |
| Robustness target | Designed for bad illumination and calibration errors. | Important precedent for adverse-weather/night detection and reliability-aware segmentation. |

## Main claims / results

- The paper claims that hard geometric association is a major weakness of earlier LiDAR-camera fusion under poor illumination and misalignment.
- Transformer-based soft association improves robustness by allowing object queries to gather image evidence adaptively.
- It demonstrates that high-performing multimodal detection can be built around **object-level queries** rather than only dense early feature concatenation.
- In the source map, TransFusion appears as a high-priority CVPR 2022 paper for multimodal object detection.

## Limitations / caveats

- Fusion remains calibrated-sensor dependent, even if less brittle than point-level association.
- Robustness is evaluated for specific corruptions; it does not provide a general uncertainty calculus for arbitrary sensor failure.
- The method is object-query/detection specific; direct semantic segmentation transfer requires adapting query-level fusion to dense pixel/token predictions.

## Relevance to user's multimodal segmentation / detection project

TransFusion is directly relevant to RBMA because it validates the principle that **learned attention can replace rigid cross-modal matching**. The user’s project can cite TransFusion when arguing that reliability-aware attention should be inserted before fusion, especially under degraded RGB/night/fog or imperfect depth/LiDAR alignment. Unlike TransFusion, RBMA can target dense segmentation memory attention rather than 3D object queries.

## Related-work paragraph candidates

**Detection paragraph.** TransFusion proposed a transformer-based LiDAR-camera detector that avoids brittle hard point-pixel fusion by allowing object queries to softly attend to image evidence. This made robustness to bad illumination and sensor misalignment a central design criterion for multimodal 3D detection.

**Bridge to RBMA.** The TransFusion design supports a broader view of multimodal perception in which cross-modal association should be learned and reliability-aware, not fixed by calibration alone. Our segmentation-oriented RBMA differs by biasing attention logits using modality reliability before softmax, targeting dense multimodal memory/token fusion rather than object-query refinement.

## References

- Bai et al., “TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers,” CVPR 2022, arXiv:2203.11496.
