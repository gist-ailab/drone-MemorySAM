---
title: LoRA, adapters, and visual prompt tuning for parameter-efficient dense prediction
tags: [related-work, peft, lora, adapter, visual-prompt-tuning, dense-prediction, key-paper]
created: 2026-06-24
source: arXiv:2106.09685; arXiv:2205.13535; arXiv:2203.12119; arXiv:2412.04220; [[sources/02_source_map_adapter_lora_foundation_seg_det]]
status: verified-draft
---

# LoRA, adapters, and visual prompt tuning for parameter-efficient dense prediction

## Scope and positioning

This note summarizes the general parameter-efficient fine-tuning (PEFT) literature needed for [[26_MultimodalSeg]]: **LoRA**, bottleneck **adapters / AdaptFormer**, and **Visual Prompt Tuning (VPT)**. The emphasis is not language modeling itself, but why these mechanisms are useful when adapting large ViT/SAM/SAM2-style foundation models to segmentation, detection, and multimodal sensor inputs.

## Citation metadata

| Method | Primary source | Venue / status | Core idea | Relevance |
|---|---|---|---|---|
| LoRA | Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685v2, 2021 | ICLR 2022 / arXiv | Freeze pretrained weights and learn low-rank update matrices for selected linear projections | Direct basis for SAMed, MemorySAM, and MoE-LoRA SAM adaptation |
| AdaptFormer | Chen et al., “AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition,” arXiv:2205.13535v3, 2022 | NeurIPS 2022 / arXiv | Add lightweight bottleneck modules to transformer blocks while freezing most pretrained weights | Canonical ViT adapter mechanism; transferable to sensor/domain adaptation |
| Visual Prompt Tuning | Jia et al., “Visual Prompt Tuning,” arXiv:2203.12119v2, 2022 | ECCV 2022 / arXiv | Insert a small number of learned prompt tokens in the input/token sequence | Useful when one wants no architectural modification to the backbone |
| MoE-LoRA for SAM MMSS | Zhu et al., “Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts,” arXiv:2412.04220v1, 2024 | arXiv | Train modality-specialized LoRA experts and route/fuse their outputs | Closest LoRA-specific SAM multimodal segmentation precedent |
| StitchFusion | Li et al., “StitchFusion,” arXiv:2408.01343v2, 2024/2025 | arXiv | Multi-directional adapter modules exchange multi-scale information among pretrained encoders | Closest adapter-based multimodal segmentation baseline; see [[relatedworks/04_stitchfusion_relatedwork]] |

## Method mechanism

### LoRA

LoRA assumes that the task-specific update to a large pretrained weight matrix is low-rank. Instead of updating a dense weight $W$, it freezes $W$ and learns a small residual update

$$
W' = W + \Delta W, \qquad \Delta W = B A,
$$

where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{k \times r}$, and $r \ll \min(d,k)$. In transformer vision backbones, this update is usually attached to attention projections such as $Q$, $K$, $V$, and/or output projection layers, and sometimes to MLP projections.

### Bottleneck adapters / AdaptFormer

Adapters insert a small trainable bottleneck branch into each transformer block. The pretrained block remains mostly frozen; the adapter projects features down, applies a nonlinearity, projects back up, and adds the result as a residual. AdaptFormer adapts this strategy to ViTs and reports that adding less than 2% extra parameters can adapt pretrained ViTs effectively across image/video recognition tasks.

### Visual Prompt Tuning

VPT learns prompt tokens that are concatenated with image patch tokens. The backbone remains frozen, but the prompts steer intermediate representations. VPT is extremely parameter-efficient—less than 1% of model parameters in the original paper—and reduces per-task storage. Its limitation for dense prediction is that learned global tokens may not inject enough spatial or sensor-specific inductive bias unless combined with dense heads or multi-scale mechanisms.

### Mixture of LoRA experts

MoE-LoRA extends LoRA by assigning different low-rank experts to different modalities or conditions, then routing/fusing expert outputs. Zhu et al. (arXiv:2412.04220) explicitly target SAM for multimodal semantic segmentation, keeping SAM weights frozen and training MoE-LoRA layers. This is especially relevant to RGB-D/RGB-T/Event segmentation because a single LoRA update may underfit modality-specific statistics.

## Parameter / update strategy

| Strategy | Frozen components | Trainable components | Typical update size | Strength | Weakness |
|---|---|---|---|---|---|
| Full fine-tuning | None or few | Entire backbone + task head | Very high | Highest capacity | Expensive, overfits small sensor datasets, stores a full model per domain |
| Linear probing / head-only | Backbone | Classifier / segmentation head | Low | Stable, cheap | Cannot correct backbone features for new modalities |
| VPT | Backbone | Prompt tokens, task head | Very low | Minimal storage, easy multi-task switching | Limited spatial/domain adaptation for dense prediction |
| LoRA | Backbone base weights | Low-rank matrices in attention/MLP projections | Low to moderate | Good capacity-efficiency tradeoff, no inference latency if merged | Rank/location choices matter; can be insufficient for severe domain shift |
| Bottleneck adapter | Backbone base weights | Adapter MLPs / residual branches | Low to moderate | Modular, can host modality-specific branches | Adds inference modules and may need careful placement |
| MoE-LoRA / multi-adapter | Backbone base weights | Several LoRA/adapters + router/fusion | Moderate | Handles modality-specific statistics and reliability variation | Router can collapse; expert count adds complexity |

## Why PEFT is useful for large foundation models and multimodal sensor adaptation

PEFT is attractive because segmentation foundation models have already learned strong generic shape, boundary, and object priors. Full fine-tuning all weights for every sensor combination wastes memory, risks catastrophic forgetting, and is often infeasible when dense labels are scarce. PEFT preserves the pretrained prior while adding a small domain- or modality-specific correction. For multimodal sensing, the same frozen foundation model can be reused across RGB, thermal, depth, LiDAR-projected, SAR, event, or medical imagery by swapping a small adapter/LoRA set rather than maintaining a separate full model.

The multimodal case adds a second reason: different sensors fail differently. A PEFT design can allocate separate adapter parameters to each modality and use a router or reliability module to select or weight them. This creates a natural interface to [[Reliability-Biased Memory Attention]]: LoRA/adapters adapt the feature basis, while reliability-biased attention controls which modality tokens are trusted at fusion time.

## Dense prediction relevance

For semantic segmentation and detection, PEFT must preserve spatial detail and support multi-scale features. Pure prompt tuning is usually the least invasive but may be too weak for dense pixel prediction. LoRA in attention layers can adapt global token interactions, while adapters can introduce local/spatial inductive biases or cross-scale feature exchange. The strongest dense-prediction design is therefore often a hybrid: frozen foundation backbone + LoRA/adapters in selected transformer blocks + multi-scale decoder/head + reliability-aware fusion.

## Limitations and open questions

- **Placement sensitivity:** LoRA on only $Q/V$ projections may not be enough for strong domain shifts; adapters near early/mid layers may be needed for sensor statistics.
- **Rank sensitivity:** too small a rank underfits new modalities; too large a rank loses the storage/computation advantage.
- **Fusion vs. adaptation ambiguity:** improved performance may come from stronger unimodal adaptation rather than better multimodal fusion.
- **Router collapse:** MoE-LoRA may overuse the dominant modality/expert unless regularized.
- **Dense labels:** PEFT reduces trainable parameters but does not remove the need for representative pixel labels or robust pseudo-labeling.

## Comparison table for project use

| Candidate | Best use in [[26_MultimodalSeg]] | Relation to SAM/SAM2 | Relation to multimodal sensors | Verdict |
|---|---|---|---|---|
| LoRA | Cheaply adapt SAM/SAM2 image encoder and attention projections | Already used by SAMed and MemorySAM | Can attach separate LoRA modules per modality | Strong default |
| AdaptFormer-style adapters | Add task/domain residual corrections inside ViT blocks | Useful when frozen backbone lacks local/domain bias | Can be modality-specific or shared | Strong when compute budget allows modules |
| VPT | Very low-cost task conditioning | Could prompt frozen ViT/SAM without weight updates | Weak for severe modality shift unless paired with decoder | Secondary baseline |
| MoE-LoRA | Modality-specific experts and routing | Direct SAM multimodal adaptation precedent | Closest to RGB-X expert selection | High-priority comparison |
| StitchFusion MultiAdapter | Cross-modal feature exchange during encoding | Foundation encoders rather than SAM memory | Strong multimodal segmentation baseline | Important architectural comparator |

## Paragraph candidates

**General PEFT paragraph.** Parameter-efficient fine-tuning methods adapt large pretrained models by updating only a small set of task-specific parameters. LoRA represents weight updates with low-rank matrices, adapters add lightweight residual bottlenecks to transformer blocks, and visual prompt tuning learns a small set of prompt tokens while freezing the backbone. These methods are especially attractive for segmentation foundation models because they preserve pretrained object and boundary priors while avoiding the cost and overfitting risk of full fine-tuning on small dense-label datasets.

**Multimodal adaptation paragraph.** In multimodal segmentation, PEFT also provides a modular way to specialize a shared foundation model to heterogeneous sensors. Modality-specific LoRA or adapter branches can model depth, thermal, event, SAR, or medical-image statistics without duplicating the full backbone. Recent SAM adaptation work, including SAMed, MemorySAM, StitchFusion, and MoE-LoRA for SAM, indicates that lightweight updates are sufficient to recover substantial downstream performance. However, these methods mainly adapt representations; they do not by themselves decide which modality should be trusted when a sensor is corrupted. This motivates combining PEFT with reliability-aware fusion.

## References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685v2.
- Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., and Luo, P. (2022). *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition*. arXiv:2205.13535v3.
- Jia, M., Tang, L., Chen, B.-C., Cardie, C., Belongie, S., Hariharan, B., and Lim, S.-N. (2022). *Visual Prompt Tuning*. arXiv:2203.12119v2.
- Zhu, C., Xiao, B., Shi, L., Xu, S., and Zheng, X. (2024). *Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts*. arXiv:2412.04220v1.
- Li, B., Zhang, D., Zhao, Z., Gao, J., and Li, X. (2024/2025). *StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation*. arXiv:2408.01343v2.
