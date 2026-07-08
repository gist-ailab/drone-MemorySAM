---
title: M4-SAM — SAM2 Memory and MoE-LoRA Implementation Supplement
tags: [relatedwork, sam2, moe-lora, video-segmentation, salient-object-detection, implementation, threat, gap-fill]
created: 2026-07-02
source: "[arXiv:2605.11760, code: https://github.com/HankLiu2020/M4-SAM, status: [VERIFIED-PDF+CODE]]"
status: gap-fill-verified
---

# M4-SAM — SAM2 Memory and MoE-LoRA Implementation Supplement

## Citation / verification

- **Paper:** *M⁴-SAM: Multi-Modal Mixture-of-Experts with Memory-Augmented SAM for RGB-D Video Salient Object Detection*.
- **arXiv:** `2605.11760v1`.
- **Code:** https://github.com/HankLiu2020/M4-SAM
- **Task:** RGB-D video salient object detection, not multi-class semantic segmentation.
- **Verification tag:** `[VERIFIED-PDF+CODE]`.

## Why this supplement exists

The vault already contains [[48_m4sam_moe_lora_sam2_threat]]. This note adds implementation-level detail from the gap-fill run and should be treated as a supplement, not a replacement.

## Methodology

M4-SAM adapts SAM2.1 Hiera-L using a prompt-free U-shaped RGB-D video segmentation architecture.

Key modules:

| Module | Role |
|---|---|
| Modality-Aware MoE-LoRA | RGB/depth/fusion expert groups for Q/V adaptation |
| Gated Multi-Level Feature Fusion | adaptive hierarchical aggregation |
| Pseudo-Guided Temporal Memory | coarse mask initializes first-frame memory to avoid cold start |

MoE-LoRA equation from the paper:

```text
h = W0 x + ΔW x = W0 x + B A x + B D(Ax)
```

Loss:

```text
L_total = L_pred + L_aux + L_moe
```

## Code-level verification

Important paths:

```text
M4SAM_Code/M4SAM.py
M4SAM_Code/Network/finetune_utils.py
M4SAM_Code/Network/adaptation_layers.py
M4SAM_Code/Network/xmem4sam.py
```

`finetune_utils.py`, `LoRA_moe.forward` confirms Q/V injection into packed QKV:

```python
qkv = self.attn_qkv(x)
new_q, moe_loss_q = self.lora_q(x)
new_v, moe_loss_v = self.lora_v(x)
qkv[:, :, :, : self.dim] += new_q
qkv[:, :, :, -self.dim:] += new_v
```

`adaptation_layers.py`, `ModalitySpecificMoE` confirms modality-specific expert groups: RGB, depth, fusion. The model requires `set_modality_type('rgb' or 'depth')` before forward.

`ConvLoRALinear.forward` uses SparseDispatcher and scales the LoRA residual by `* 0.1` to prevent NaNs.

## Novelty implication

M4-SAM strongly occupies **MoE-LoRA-in-SAM2** space. P29 must not claim generic MoE-LoRA novelty. The defensible novelty is narrower:

```text
unsupervised condition prototype / entropy-derived reliability -> FiLM Soft-MoE LoRA routing
inside multimodal semantic segmentation / SAM2 memory workflow
```

## Limitations

- Binary salient object detection, not semantic segmentation.
- Coarse pseudo mask quality affects temporal memory.
- Memory is temporal video memory, not necessarily cross-modal semantic memory.

## Ours application direction

Use M4-SAM as the required threat citation for P29. Then show that our method differs by routing according to scene condition/reliability rather than a hard modality dispatcher.
