---
title: DGFusion — Depth-Guided Sensor Fusion for Robust Semantic Perception
tags: [related-work, key-paper, multimodal-segmentation, lidar, depth, ra-l]
created: 2026-06-24
source: arXiv:2509.09828 / IEEE RA-L 2026; downloaded PDF text in [[sources/pdfs/priority_a/text/2509.09828.txt]]
status: verified-draft
---

# DGFusion — Depth-Guided Sensor Fusion for Robust Semantic Perception

## Citation

Brödermann et al., **“DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception,”** arXiv:2509.09828; accepted in IEEE Robotics and Automation Letters, 2026. PDF archived at [[sources/pdfs/priority_a/2509.09828.pdf]].

## Problem setting

DGFusion targets robust semantic perception for autonomous driving under challenging conditions. Its central observation is that sensor reliability is not only condition-dependent but also **spatially varying**: e.g., camera, event, and LiDAR reliability can differ by range, weather, and local scene region.

## Verified method summary

The PDF abstract states that DGFusion:

- treats multimodal segmentation as a **multi-task problem**;
- uses LiDAR measurements both as input and as ground truth for learning depth;
- adds an **auxiliary depth head**;
- encodes depth-aware features into **local depth tokens**;
- combines local depth tokens with a global condition token to condition attentive cross-modal fusion;
- proposes a robust depth loss for sparse/noisy LiDAR in adverse conditions;
- reports state-of-the-art panoptic and semantic segmentation on **MUSES** and **DeLiVER**.

## Novelty

DGFusion’s novelty is the use of **depth-guided, spatially local reliability conditioning** for multimodal fusion. Compared with global condition-only fusion, it adds local depth tokens so fusion can vary by image region.

## Main claims

- Local depth is a useful proxy for spatially varying sensor reliability.
- LiDAR should be used not only as an input modality but also as depth supervision.
- Robust depth supervision is needed because LiDAR returns can be sparse and noisy in adverse weather.

## Important implementation ideas

- Global condition token: captures scene-level environment / weather / illumination.
- Local depth token: captures per-region depth-dependent reliability.
- Cross-modal attention uses those tokens to adapt fusion.
- Robust log-depth loss masks extreme residuals from noisy LiDAR returns.

## Limitations / gaps for our project

- Reliability is tied strongly to depth/LiDAR availability.
- It does not directly use decoder predictive uncertainty as a general reliability signal.
- It is not SAM2/MemorySAM-based.
- The mechanism is attention conditioning, but not necessarily pre-softmax reliability logit bias in SAM2 memory attention.

## Comparison to our project

| Axis | DGFusion | Our direction |
|---|---|---|
| Reliability cue | Depth/LiDAR + condition token | Decoder uncertainty / confidence, optionally depth |
| Spatial adaptivity | Local depth tokens | Per-region reliability bias |
| Fusion target | Cross-modal fusion module | SAM2 memory attention logits |
| Dataset relevance | MUSES, DeLiVER | Same + RGB-T/event/LiDAR settings |

## Related-work paragraph candidate

DGFusion improves robust semantic perception by using LiDAR not only as an input modality but also as supervisory depth information. Its auxiliary depth head produces local depth tokens that condition attentive multimodal fusion together with a global condition token, enabling spatially varying sensor weighting on MUSES and DeLiVER. This supports the broader view that reliable multimodal segmentation requires local, input-dependent fusion. Unlike DGFusion, our approach does not require depth to be the sole reliability cue; instead, it estimates modality reliability from predictive uncertainty and injects it directly into SAM2-style memory attention logits.
