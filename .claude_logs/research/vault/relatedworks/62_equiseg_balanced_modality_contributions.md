---
title: EQUISeg — Robust Multimodal Semantic Segmentation with Balanced Modality Contributions
tags: [relatedwork, multimodal-segmentation, missing-modality, prototypes, robustness, gap-fill]
created: 2026-07-02
source: "[arXiv:2509.24505, status: [VERIFIED-PDF]]"
status: gap-fill-verified
---

# EQUISeg — Robust Multimodal Semantic Segmentation with Balanced Modality Contributions

## Citation / verification

- **Paper:** *Robust Multimodal Semantic Segmentation with Balanced Modality Contributions* / **EQUISeg**.
- **arXiv:** `2509.24505v1`.
- **Status:** arXiv only in checked sources.
- **Code:** no official code URL found in PDF.
- **Verification tag:** `[VERIFIED-PDF]`.

## Task and protocol

EQUISeg studies arbitrary/multimodal semantic segmentation on DeLiVER and MUSES, especially robustness when modalities are missing or degraded.

It evaluates:

| Protocol | Meaning |
|---|---|
| EMM | Entire-Missing Modality |
| RMM | Random-Missing Modality |
| NM | Noisy Modality |

## Methodology

The problem is **dominant-modality collapse**: a model may over-use one modality and fail when it is degraded or missing. EQUISeg uses a SegFormer-based architecture with four-stage **Cross-modal Transformer Blocks (CMTB)**.

The key balancing component is **Self-Guided Module (SGM)**. It computes class-wise prototypes, then randomly splits modality prototypes into teacher/student sets to enforce balanced semantic guidance.

Prototype formula reported by the subagent:

```text
p_c = sum_j f_j 1[l_j=c] / sum_j 1[l_j=c]
```

Teacher-student guidance uses KL divergence:

```text
L_KL(f_t, f_s) = sum_c f_t(c) log(f_t(c)/f_s(c))
```

## Implementation details verified

- DeLiVER: 4× NVIDIA A6000.
- MUSES: 4× RTX 3090.
- Optimizer: AdamW.
- LR: `6e-5`.
- Warm-up: 10 epochs.
- Polynomial decay exponent: 0.9.
- Training: 200 epochs.
- Batch size: 2/GPU.
- Crop: 1024×1024.

## Novelty relative to RBMA / P29 / P30

EQUISeg is important for P29/P30 because it uses class-wise prototypes to balance modalities. However, EQUISeg prototypes are supervised and class-label dependent. P29's novelty must be scoped as **unsupervised condition prototype routing**, not merely prototype-based multimodal balance.

RBMA differs because it is inference-time predictive-entropy reliability bias rather than training-time prototype distillation.

## Limitations

- Requires semantic labels for prototype guidance.
- Mainly training-time balancing; less direct inference-time per-frame reliability correction.
- Not SAM/SAM2 based.

## Ours application direction

Use EQUISeg as a baseline/regularizer reference for avoiding modality collapse. In experiments, compare:

1. RBMA entropy bias only.
2. EQUISeg-style class prototype balancing regularizer.
3. RBMA + class-prototype balance.

For P29, explicitly state that the condition prototype is unsupervised and image-derived, unlike EQUISeg's supervised class prototype.
