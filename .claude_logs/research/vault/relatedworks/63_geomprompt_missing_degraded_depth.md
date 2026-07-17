---
title: GeomPrompt — Geometric Prompt Learning for Missing and Degraded Depth
tags: [relatedwork, rgb-d, semantic-segmentation, missing-modality, geometric-prompt, gap-fill]
created: 2026-07-02
source: "[arXiv:2604.11585, project: https://geomprompt.github.io, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# GeomPrompt — Geometric Prompt Learning for Missing and Degraded Depth

## Citation / verification

- **Paper:** *GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth*.
- **arXiv:** `2604.11585v1`.
- **Venue:** CVPR 2026 URVIS Workshop.
- **Project page:** https://geomprompt.github.io
- **Code:** project page found; no GitHub URL verified in PDF.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task

RGB-D semantic segmentation when depth is missing or degraded.

## Methodology

GeomPrompt freezes an RGB-D segmenter and learns a fourth-channel geometric prompt from RGB. It does **not** reconstruct metric depth. Instead, it learns task-driven geometry sufficient for segmentation.

Conceptual equation:

```text
G_hat = g_theta(I_RGB)
Y_hat = F_frozen([I_RGB, G_hat])
```

A second variant, **GeomPrompt-Recovery**, predicts a correction for degraded depth. The aim is not depth denoising but producing a fourth-channel signal that improves the frozen segmenter.

## Reported facts

- SUN RGB-D: DFormer +6.1 mIoU over RGB-only.
- SUN RGB-D: GeminiFusion +3.0 mIoU over RGB-only.
- Severe depth corruptions: up to +3.6 mIoU.
- Latency: GeomPrompt 7.8ms vs monocular depth baselines 38.3ms / 71.9ms.

## Novelty relative to RBMA / P29 / P30

GeomPrompt solves missing/degraded depth by synthesizing or correcting a geometry prompt. RBMA solves modality unreliability by lowering its attention contribution through entropy bias. These are complementary.

P29 can treat a GeomPrompt channel/token as a condition-specific expert. P30 can consume geometric prompt features as additional memory features before class-token decoding.

## Limitations

- RGB-D specific.
- Requires a frozen RGB-D model expecting a depth/fourth channel.
- Requires segmentation supervision for prompt learning.

## Ours application direction

For missing-depth scenarios, RBMA can decide when to trust real depth versus a GeomPrompt-like synthetic geometry token. Proposed ablation:

```text
RGB + degraded depth
RGB + GeomPrompt
RGB + degraded depth + GeomPrompt + RBMA reliability bias
```
