---
title: Priority A Comparison Matrix — Multimodal Segmentation Core
tags: [related-work, comparison-matrix, multimodal-segmentation]
created: 2026-06-24
source: [[relatedworks/01_memorysam_relatedwork]], [[relatedworks/02_dgfusion_relatedwork]], [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
status: verified-draft
---

# Priority A Comparison Matrix — Multimodal Segmentation Core

| Group | Paper / method | Venue / source | Main idea | Fusion level | Reliability handling | Most relevant gap for our work |
|---|---|---|---|---|---|---|
| Foundation-model MMSS | [[MemorySAM]] | arXiv:2503.06700 | Treat modalities as SAM2 frames and use memory attention | SAM2 memory | implicit | no explicit reliability/uncertainty logit bias |
| Depth-guided fusion | [[DGFusion]] | RA-L 2026 / arXiv:2509.09828 | Global condition token + local depth tokens + depth auxiliary head | cross-modal attention | depth-guided local reliability | depth-specific, not SAM2 memory |
| Loss regularization | [[Reducing Unimodal Bias]] | arXiv:2505.06635 | multi-scale functional entropy/Fisher regularization | training objective | balances modality contribution | no input-specific memory-attention mechanism |
| Adapter weaving | [[StitchFusion]] | arXiv:2408.01343 | lightweight adapters weave pretrained encoders | encoder/multiscale | not primary focus | no uncertainty or memory attention |
| Anymodal distillation | [[AnySeg]] | arXiv:2411.17141 | teacher-student unimodal/cross-modal distillation | training/distillation | missing-modality robustness | not local reliability in available modalities |
| RGB-X baseline | [[CMX]] | TITS 2023 / arXiv:2203.04838 | cross-modal feature rectification and fusion | feature fusion | cross-modal calibration | fixed RGB-X fusion, not SAM2 memory |
| ViT token fusion | [[TokenFusion]] | CVPR 2022 / arXiv:2204.08721 | multimodal token fusion for ViTs | token fusion | not explicit uncertainty | no predictive reliability |
| Modality selection | [[MAGIC++]] | arXiv:2412.16876 | hierarchical arbitrary modality selection | modality selection + interaction | modality selection | selection rather than logit bias |
| Condition-aware fusion | [[CAFuser]] | RA-L 2025 / arXiv:2410.10791 | condition token + modality adapters | condition-aware fusion | global condition | less spatial/local than RBMA/DGFusion |

## Related-work structure recommended for the paper

1. **Multimodal semantic segmentation and RGB-X fusion**: CMX, TokenFusion, DeLiVER, MUSES, MAGIC++, CAFuser, DGFusion.
2. **Foundation-model and SAM-based multimodal segmentation**: MemorySAM, SAM adapters, SAM-FuseNet, SAM2/SAM3 adaptation.
3. **Reliability, uncertainty, and modality bias**: Reducing Unimodal Bias, UTFNet, HyperDUM, evidential fusion, DGFusion.
4. **Adapter and parameter-efficient adaptation**: StitchFusion, ViT-Adapter, LoRA, SAM adapters.
5. **Object detection transfer**: BEVFusion, TransFusion, DeepInteraction, FUTR3D, camera-LiDAR detection heads.
