---
title: DeepInteraction Related Work — preserving modality-specific representations
tags: [related-work, key-paper, multimodal-object-detection, modality-interaction, camera-lidar, transformer]
created: 2026-06-24
source: [arXiv:2208.11112](https://arxiv.org/abs/2208.11112)
status: verified-draft
---

# DeepInteraction Related Work — preserving modality-specific representations

## Citation metadata

| Item | Metadata |
|---|---|
| Paper | Zeyu Yang, Jiaqi Chen, Zhenwei Miao, Wei Li, Xiatian Zhu, Li Zhang. **“DeepInteraction: 3D Object Detection via Modality Interaction.”** arXiv:2208.11112v4; widely cited as a top multimodal 3D detection method. |
| Modalities | LiDAR point cloud + camera/RGB. |
| Core idea | Maintain per-modality representations while enabling deep interaction in encoder and decoder. |
| Wikilinks | [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/11_transfusion_relatedwork]], [[relatedworks/13_futr3d_relatedwork]], [[relatedworks/41_unimodal_bias_and_modality_collapse]] |

## Task / setup

DeepInteraction targets nuScenes-style multimodal 3D object detection. Its central criticism is that many fusion designs merge modalities too aggressively, overlooking useful modality-specific information. The proposed solution is to preserve separate LiDAR and camera representations while allowing repeated cross-modal interaction.

## Modality fusion mechanism

| Stage | Mechanism | Interpretation |
|---|---|---|
| Per-modality streams | Individual modality representations are learned and maintained. | Avoids premature modality collapse. |
| Multi-modal representational interaction encoder | Encoder exchanges information across modalities while retaining each stream. | Similar in spirit to cross-modal feature refinement rather than simple concatenation. |
| Multi-modal predictive interaction decoder | Detection prediction is refined with interacting modality evidence. | Late decision stage still benefits from both modality-specific and shared cues. |
| Overall strategy | “Interaction” rather than one-shot fusion. | Strong precedent for iterative reliability-aware segmentation fusion. |

## Main claims / results

- DeepInteraction claims that top 3D detectors are restricted by overlooking modality-specific useful information.
- It proposes modality interaction as an alternative to fully collapsing modalities into one representation early.
- The abstract reports that it surpasses prior methods on nuScenes at the time of publication.
- The key reusable claim for this project is that **preserving modality identity can improve multimodal detection**, especially when modalities have different failure modes.

## Limitations / caveats

- Modality preservation increases architectural complexity and may raise training/inference cost.
- The method does not explicitly estimate reliability or uncertainty for each modality; interaction is learned implicitly from supervision.
- It is designed for camera-LiDAR detection, not arbitrary RGB-X semantic segmentation with thermal/depth/event inputs.

## Relevance to user's multimodal segmentation / detection project

DeepInteraction is one of the strongest related-work supports for avoiding unimodal dominance and modality collapse. It aligns with [[relatedworks/41_unimodal_bias_and_modality_collapse]]: rather than letting RGB or LiDAR dominate, the model preserves separate streams and introduces structured interaction. RBMA can be framed as a lighter or more targeted alternative: retain modality-specific memory/token evidence, then bias attention according to reliability instead of fully entangling all streams at every layer.

## Related-work paragraph candidates

**Related-work paragraph.** DeepInteraction argued that multimodal 3D detectors lose useful information when camera and LiDAR features are fused too early or too completely. It therefore maintains modality-specific representations and introduces interaction modules in both representation learning and prediction, showing that preserving modality identity can be beneficial for high-performance 3D detection.

**Novelty bridge.** This principle transfers to multimodal segmentation: RGB, depth, thermal, event, and LiDAR cues should not be forced into a single undifferentiated feature tensor before their reliability is assessed. RBMA can be positioned as a reliability-conditioned interaction mechanism at the attention-logit level, whereas DeepInteraction provides architectural interaction without explicit reliability biasing.

## References

- Yang et al., “DeepInteraction: 3D Object Detection via Modality Interaction,” arXiv:2208.11112.
