---
title: MemorySAM — SAM2 Memory Attention Cross-Modal Implementation Supplement
tags: [relatedwork, memorysam, sam2, memory-attention, implementation, threat, gap-fill]
created: 2026-07-02
source: "[arXiv:2503.06700, code: https://github.com/Chenfei-Liao/MemorySAM, status: [VERIFIED-PDF+CODE]]"
status: gap-fill-verified
---

# MemorySAM — SAM2 Memory Attention Cross-Modal Implementation Supplement

## Citation / verification

- **Paper:** *MemorySAM: Memorize Modalities and Semantics with Segment Anything Model 2 for Multi-modal Semantic Segmentation*.
- **arXiv:** `2503.06700`.
- **Code:** https://github.com/Chenfei-Liao/MemorySAM
- **Verification tag:** `[VERIFIED-PDF+CODE]`.

## Why this supplement exists

The vault already has [[01_memorysam_relatedwork]]. This note adds implementation-level details relevant to RBMA novelty defense.

## Core mechanics

MemorySAM treats modalities as pseudo-frames of the same scene:

```text
x = {x_1, x_2, ..., x_M}
```

Previous modality memories are concatenated:

```text
Vp_i^f = Concat(Vfea_1, ..., Vfea_{i-1})
Vp_i^p = Concat(Vpos_1, ..., Vpos_{i-1})
```

Current feature attends to memory:

```text
Fc_i = Att_m(Fe_i, (Vp_i^f, Vp_i^p))
```

Final mask is averaged over modality masks:

```text
Mask = (1/M) sum_i Mask_i
```

SPMM is training-only semantic prototype supervision:

```text
L = lambda L_proto + L_Ohem(GT, Mask)
```

## Code-level verification

Important paths:

```text
semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py
semseg/models/sam2/sam2/modeling/sam2_base_mem.py
semseg/models/sam2/sam2/modeling/memory_attention.py
semseg/models/sam2/sam2/modeling/memory_encoder.py
```

`_LoRA_qkv.forward` confirms Q/V LoRA injection only:

```python
qkv = self.qkv(x)
new_q = self.linear_b_q(self.linear_a_q(x))
new_v = self.linear_b_v(self.linear_a_v(x))
qkv[:, :, :, : self.dim] += new_q
qkv[:, :, :, -self.dim:] += new_v
```

The inspected `LoRA_Sam.forward` wrapper contains `m = 2`, suggesting a visible path may be hard-coded for two modalities even though the paper formulates general M modalities.

## Novelty implication for ours

MemorySAM is the strongest prior art for SAM2 memory attention used as cross-modal semantic fusion. RBMA novelty must be scoped as adding **training-free reliability bias inside the SAM2 memory attention logits**, not merely using memory attention for modalities.

## Limitations / caveats

- SPMM is training-only, not an inference-time memory module.
- Code path inspected may be 2-modality specific.
- LoRA and memory/mask modules still add trainable adaptation.

## Ours application direction

Use MemorySAM as the base architecture and explicitly ablate:

1. MemorySAM baseline.
2. MemorySAM + entropy reliability as post-fusion weighting.
3. MemorySAM + RBMA pre-softmax memory attention bias.
