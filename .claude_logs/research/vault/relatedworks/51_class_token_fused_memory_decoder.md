---
title: Class-Token Decoder on Fused Multimodal Memory Features — P30 Novelty Defense
tags: [related-work, novelty-defense, P30, class-token-decoder, mask2former, sam2, memory-attention, multimodal-segmentation]
created: 2026-07-02
source: Track 7 of [[sources/07_parallel_research_prompts_2026-07-02]]; adversarially verified 2026-07-02 (2 independent skeptic passes on all critical claims)
status: verified-draft
---

# Class-Token Decoder on Fused Multimodal Memory Features — P30 Novelty Defense

## Problem setting

**P30 claim under test:** learnable per-class query tokens cross-attending to SAM2 MULTIMODAL MEMORY features (post-RBMA reliability-biased fusion), decoded to masks, with a learned modality router anchored by RBMA reliability — vs MaskFormer/Mask2Former-style heads on single-modality or pre-fused backbones.

Four sub-questions (Track 7 must-answers):
1. Query/mask-token decoders on top of FUSED multimodal features — who occupies this cell, and where does fusion happen relative to the queries?
2. Class/query tokens interacting with SAM/SAM2 features — does anyone attach class queries to SAM2 **memory** features?
3. Rare-class collapse (our P28 failure mode): do query decoders help or hurt tail classes?
4. High-resolution query decoding on a single-scale ViT (SAM3/PE) — precedents for our SAM3 branch.

## Novelty — VERDICT

**Partially-occupied (broad cell) / unoccupied-with-one-gray-zone (exact cell).**

- The broad cell "query/mask-token decoder on fused multi-sensor features" **IS occupied**: CAFuser & DGFusion (OneFormer head on fused features), BiXFormer (mask-classification × multi-sensor, per-modality queries), DF2RQ (region-wise queries driving fusion, remote sensing), RoadFormer+ (MaskFormer-paradigm decoder on fused RGB-X features).
- The exact cell "learnable class-token decoding on **SAM2 memory-attention features** of a multimodal (modalities-as-frames) stream, with **reliability-anchored** routing" is, as of 2026-07-02, **unoccupied by any external paper** — the universal negative survived two independent adversarial search passes (SAM2+memory+queries; mask-classification+SAM2+DELIVER/MCubeS; RGB-T/RGB-D SAM2 2025-26; MemorySAM citing-paper sweep). The closest, SAM-DAQ (2511.09870, AAAI), **REPLACES** the memory bank with queries for **binary** RGB-D video saliency.
- ⚠️ **Adversarial-verification gray zone (must be addressed in the paper):** MemorySAM (2503.06700) itself — our own baseline — decodes multi-class masks directly from SAM2 memory-attended multimodal features using SAM2's **stock mask decoder** with "the number of the output classes adapted to the targeted scene." SAM2's mask decoder internally uses learnable output tokens that cross-attend to those features. Whether stock per-class output tokens count as "learnable class tokens on memory features" is a **definitional gray zone**; the MemorySAM paper/repo do not document the token mechanism, and verification could not settle it. **P30 must explicitly differentiate** from MemorySAM's stock decoder: (i) a full multi-layer query decoder with mask-classification training (per-class mask+score decoupling), not SAM2's shallow two-way transformer producing per-pixel logits; (ii) queries participating in reliability-anchored modality routing; (iii) auxiliary per-pixel head for rare classes. Do NOT phrase novelty as "first tokens attending to memory features" — phrase it at the mask-classification-framework + reliability-routing level.
- Consequence: P30's novelty statement must NOT be "first query decoder on fused multimodal features" (false — CAFuser/BiXFormer). Anchor it on (i) mask-classification queries on SAM2 *memory* features of a modalities-as-frames stream, (ii) RBMA-anchored (training-free-signal-anchored) router, (iii) explicit differentiation from MemorySAM's stock mask decoder.

## Method (mechanisms of the occupants, with equations)

### CAFuser (arXiv:2410.10791, RA-L 2025) — queries strictly AFTER fusion [VERIFIED-PDF]
Fusion happens strictly before the queries. Verbatim (Sec. III-B): *"The resulting fused multi-scale feature maps are then passed to the pixel decoder and the OneFormer head to produce the prediction."* OneFormer's queries never participate in fusion.
- Condition Token (CT): from the highest-level RGB feature map, supervised with a verbo-visual contrastive loss (text prompts).
- **CA² (condition-aware cross-attention):** CT is passed through an FC layer and *appended to the RGB queries* (concatenated with the 49 RGB window tokens) inside the cross-modal fusion attention, then removed after attention → mechanism-class: **condition-token (query-append)**. NOT a pre-softmax logit bias — it extends the token set, it does not additively bias existing logits. [adversarially confirmed from primary source]
- **CAA (Condition-Aware Addition** — note: "Addition", not "Attention"): one weight per modality via FC + softmax over the flattened CT, then per-modality feature multiply:
$$ w = \mathrm{softmax}(\mathrm{FC}(\mathrm{CT})) \in \Delta^{M-1}, \qquad F_{fused} = \sum_m w_m F_m $$
→ mechanism-class: **feature-multiply / learned-gate**. No additive pre-softmax attention-logit bias anywhere in CAFuser. [adversarially confirmed]

### DGFusion (arXiv:2509.09828, RA-L 2026) — token-concat conditioning, still no logit bias [VERIFIED-PDF via adversarial pass]
Local depth tokens $t_d$ (from a LiDAR-supervised auxiliary depth head) + global condition token $t_c$ are **CONCATENATED** into the query features of the attentive fusion:
$$ F_q = [\,F_{rgb},\; t_c,\; t_d\,], \qquad \mathrm{Attn} = \mathrm{softmax}\!\left(\frac{F_q K^\top}{\sqrt{d}}\right)V $$
Standard softmax attention — no additive logit bias; the OneFormer head sees only fused features; fully learned (depth-GT-supervised), not training-free. ⚠️ DGFusion's "spatially varying sensor reliability" motivation **overlaps RBMA's rhetoric** — differentiate by mechanism (learned depth tokens vs training-free entropy logit bias) and locus (backbone fusion vs SAM2 memory attention).

### BiXFormer (arXiv:2506.03675) — mask-classification × multi-sensor, queries PER-modality [VERIFIED-PDF]
Verbatim (Sec. II-A): *"To our knowledge, this is the first attempt to marry the mask-classification segmentation framework with multi-sensor input."*
- Modality-specific queries $Q = \{Q_r, Q_x\}$; each attends ONLY to its own modality's features (Eq. 1: RGB queries ↔ RGB features, X queries ↔ X features) — never to fused features.
- Unified Modality Matching (UMM) = Modality-Agnostic Matching over all $2L$ predictions + Complementary Matching reassigning unmatched labels per modality; final prediction $y^{*i} = (\max(c^i_a, c^i_{sr}, c^i_{sx}), \max(m^i_a, m^i_{sr}, m^i_{sx}))$ — fusion only at matching/prediction level.
- Dual ImageNet-pretrained backbones (ResNet-34); no SAM/SAM2, no memory, no reliability signal.
- ⚠️ Venue caveat: authors state "accepted by TMM'26" — cite as **TMM'26 / arXiv:2506.03675**, not TMM'25; independent acceptance record not found, so keep the arXiv ID primary.
- ⚠️ BiXFormer's own "first" claim is itself dubious (CAFuser/OneFormer RA-L'25 and the MUSES baseline's fused-feature Mask2Former predate it) — this strengthens the "broad cell partially-occupied" verdict and is a caution against absolute "first" phrasing in P30 too.

### SAM-DAQ (arXiv:2511.09870, AAAI) — queries REPLACE SAM2 memory [VERIFIED-PDF]
Nearest memory-side neighbor. Depth fused BEFORE queries via depth-guided parallel adapters inside the Hiera encoder; frame-level + video-level learnable queries; the Query-driven Temporal Memory (QTM) module verbatim *"replaces the memory bank and prompt embedding with learnable queries."* Task is **binary** RGB-D video salient object detection (E/S/F-measure, MAE — no classes). Occupies "queries ↔ SAM2-memory-role" but: (i) binary, (ii) queries substitute memory rather than decode fused memory features, (iii) no reliability signal, (iv) temporal video, not modalities-as-frames. [adversarially confirmed from primary source, both passes]

### SHIFNet (arXiv:2503.02581) — frozen text class-embeddings on fused SAM2 features [VERIFIED-PDF]
RGB-T on SAM2-Hiera-L (image mode, FPN neck — **no memory attention involved**). SACF text-guided affinity fusion at 4 pyramid levels, then a Heterogeneous Prompting Decoder scoring frozen LanguageBind category embeddings against fused features by inner product:
$$ S_{ij} = F_{final}^{(i,j)} \cdot E_{cls}^\top $$
Second-nearest neighbor: class-embedding decoding on fused SAM2 features, BUT embeddings are frozen text vectors (not learnable queries with cross-attention layers), and no memory attention, no reliability signal.

### OpenWorldSAM (arXiv:2507.05427) — language queries into SAM2's prompt path [VERIFIED-PDF]
Queries $q_i = u + t_i$ (projected language embedding + K learnable positional tie-breakers) cross-attend to SAM2 level-3 image features via a soft-prompting transformer, then feed SAM2's mask decoder **as prompt embeddings**. No memory attention, RGB+language only.

### Rare-class evidence (must-answer 3) [both sources adversarially confirmed]
- **Frequency-based Matcher (arXiv:2406.03917, TMM):** one-to-one Hungarian matching starves tail classes — verbatim: low-frequency supervision *"is diminished to a great extent due to the large number of no-object targets and other higher-frequency category targets."*
- **GOOSE-M2F (arXiv:2606.15937** — NOT the challenge report 2606.21456, which has no aux head): rare classes with <50 pixels/crop *"receive zero gradient"* from the primary Mask2Former loss; a training-only Auxiliary Supervision Head (per-pixel CE at H/4, removed at inference) *"contributes +5–8% on rare-class categories."* ⚠️ Caveat: the +5–8% is stated **qualitatively**; the isolated ablation row shows **+3.4% composite mIoU**, not a rare-class-only breakdown — quote both if cited.
- Implication (our **inference**, not a source fact): fixed per-class tokens (one query per class, fixed assignment — semantic seg needs no Hungarian matching, sidestepping the documented starvation mode) + an auxiliary per-pixel head on the fused memory features.

### Single-scale ViT query decoding (must-answer 4)
- **ViTDet (arXiv:2203.16527, ECCV 2022):** simple feature pyramid from a single-scale plain-ViT map suffices for dense prediction (Mask R-CNN head).
- **EoMT (arXiv:2503.19108, CVPR 2025 Highlight** — confirmed): a small set of learned queries injected into the final encoder blocks of a plain DINOv2 (register-token) ViT; joint self-attention of patches+queries; masked attention annealed away → decoder-free inference, no adapter/pixel decoder/transformer decoder, *"up to 4× faster with ViT-L"* at accuracy similar to task-specific SOTA. **RGB-only** (COCO/ADE20K/Cityscapes); the only extension found is VidEoMT (temporal video, still single-sensor RGB) — no multimodal EoMT through 2026-07 [adversarially confirmed empty]. ⚠️ The "EoMT-style queries viable on SAM3/PE single-scale" step is our **extrapolation/inference**, not a source fact.

## Quantitative results (verbatim rows)

| Method | Dataset / metric | Value | Backbone / config | Tag | Split |
|---|---|---|---|---|---|
| CAFuser CA² | MUSES panoptic, Table I | **59.7 PQ** (Day 59.5, Night 57.3, Clear 61.4, Fog 57.5, Rain 59.6, Snow 57.2) | Swin-T + OneFormer | [VERIFIED-PDF] | [unknown] |
| CAFuser CA² | MUSES semantic, Table II | **78.2 mIoU** | Swin-T + OneFormer | [VERIFIED-PDF] | [unknown] |
| CAFuser CA² | DELIVER, Table III | **67.8 mIoU** | Swin-T, RGB-D-E-L | [VERIFIED-PDF] | [val] |
| CAFuser CA² | DELIVER, Table III | **55.6 mIoU** | Swin-T, RGB-D-E-L | [VERIFIED-PDF] | [test] |
| DGFusion | MUSES PQ / mIoU | **61.03 / 79.5** | CAFuser lineage | [vault-verified, not re-checked] | [test] |
| DGFusion | DELIVER CLDE | **56.7 mIoU** | CAFuser lineage | [vault-verified, not re-checked] | [test] |
| BiXFormer | DELIVER, Table II | R 53.32, D 50.18, E 1.03, L 1.49, **RDEL 58.29**, Mean 43.24 (+2.75) | ResNet-34, modality-dropout protocol | [VERIFIED-PDF] | [unknown] |
| SHIFNet | PST900 / FMB / MFNet, Tables II–IV | **89.8 / 67.8 / 59.2 mIoU** | SAM2-L, 32.27M trainable | [VERIFIED-PDF] | [unknown] |
| OpenWorldSAM | ADE20K-857 / VOC-20 / ScanNet-40, Table 1 | **60.4 / 73.7 / 55.6 mIoU**; RefCOCOg 74.0 cIoU | SAM2 + language, 4.5M trainable | [VERIFIED-PDF] | [unknown] |
| RoadFormer+ | MFNet Tab. VI / FMB Tab. VII / ZJU Tab. VIII | **62.7 / 74.1 / 93.0 mIoU** | RGB-X, fused-features-then-decoder | [VERIFIED-PDF] | [unknown] |
| Freq-based Matcher | ADE20K-Full, Table II (image-level) | Mask2Former **18.8 overall / 4.8 rare** → theirs **20.3 / 8.3** (+1.5 / +3.5) | Mask2Former base | [VERIFIED-PDF] | [unknown] |
| GOOSE-M2F | GOOSE (ICRA'26 challenge, 3rd) | **70.08% composite mIoU**; aux head "+5–8% rare" (qualitative), isolated ablation **+3.4% composite** | RGB+NIR Mask2Former, 200 queries | [VERIFIED-PDF] | [unknown] |
| MaskFormer | ADE20K-Full (847 cls) | ~+3.5 mIoU vs per-pixel | — | [ABSTRACT-ONLY — re-verify Table 3 before quoting] | [unknown] |

⚠️ Protocol warning: BiXFormer's DELIVER RDEL 58.29 is far below MemorySAM 65.38 / CMNeXt 66.30 — modality-dropout-robust training + likely different split/protocol (its per-modality E=1.03/L=1.49 shows the dropout regime). Never mix into one SOTA table without protocol tags (see [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]]).

## Limitations (of the occupants)

- **CAFuser:** condition signal is global (one token/image, no spatial variation); CT supervision requires text prompts at training; queries are fusion-blind.
- **DGFusion:** reliability tied to depth/LiDAR availability and depth-GT supervision; learned, not training-free; queries still post-fusion.
- **BiXFormer:** no cross-modal feature interaction at all (matching/prediction-level only); no reliability signal; weak absolute mIoU; ResNet backbone, no VFM.
- **SAM-DAQ:** binary task, no classes; discards the memory bank rather than decoding from it.
- **SHIFNet:** frozen text embeddings + dot product, no learnable query decoder; image-mode SAM2, memory attention unused.
- **EoMT:** RGB single-sensor only; requires large-scale pretrained ViT for the inductive-bias-free claim to hold.
- **DF2RQ (TGRS 2025, DOI 10.1109/TGRS.2025.3526247):** [ABSTRACT-ONLY / UNVERIFIED] region-wise queries drive dynamic fusion (queries AS fusion agents — nearest "queries-in-fusion" occupant) but remote-sensing modalities, no VFM/memory/reliability. IEEE paywall; **obtain PDF before camera-ready citation.**

## Improvement directions

1. **Let queries see reliability, not just fused features:** every occupant either fuses before queries (CAFuser/DGFusion/RoadFormer+) or never fuses features (BiXFormer). A query decoder whose cross-attention substrate already carries an additive reliability bias (RBMA) — and whose query→modality routing is anchored by the same training-free signal — is the open combination.
2. **Fixed per-class assignment + auxiliary per-pixel head** to avoid Hungarian tail-class starvation (2406.03917) and zero-gradient rare classes (GOOSE-M2F) — both confirmed failure modes of vanilla Mask2Former training.
3. **Decouple RBMA's entropy source from the final head:** compute B_i from an auxiliary per-modality decode so the reliability signal is not entangled with the class-token head's own softmax.
4. **SAM3/PE branch:** ViTDet simple-FPN + query head, or EoMT-style query injection into the final PE blocks (no adapter, leverages pretraining scale) — combining EoMT-style in-encoder queries with a multimodal memory stream is unoccupied (our extrapolation; precedent facts verified, the combination is not published).

## Comparison to RBMA-P29-P30 (mechanism-class)

| Work | Signal source | Injection location | Mechanism class | Query↔fusion relation |
|---|---|---|---|---|
| CAFuser CA² | text-supervised RGB condition token | token appended to fusion cross-attn queries | condition-token (query-append) | queries AFTER fusion |
| CAFuser CAA | same CT → FC+softmax | per-modality feature multiply | feature-multiply / learned-gate | queries AFTER fusion |
| DGFusion | depth-GT-trained local depth tokens + global CT | concatenated into fusion-attention query features $F_q=[F_{rgb},t_c,t_d]$, standard softmax | condition-token (concat) | queries AFTER fusion |
| BiXFormer | none (matching) | prediction-level max/matching (UMM) | output-scale (max-fusion of logits) | queries PER-modality, never fused |
| DF2RQ | region-wise queries themselves | queries drive dynamic fusion (details unverified) | learned-gate? [UNVERIFIED] | queries AS fusion agents |
| SHIFNet | text affinity (SACF) | feature reweighting pre-decoder; frozen-embedding dot-product head | feature-multiply + fixed class embeddings | class embeddings AFTER fusion |
| SAM-DAQ | depth adapters | encoder-level; queries replace SAM2 memory bank + prompts | condition-token (memory substitution) | queries REPLACE memory |
| EoMT | none | learned queries in final ViT encoder blocks | in-encoder query tokens | RGB-only, no fusion |
| MemorySAM (baseline, gray zone) | none | SAM2 stock mask decoder (internal output tokens) on memory-attended features | per-pixel decode via stock decoder — token mechanism undocumented | shallow stock tokens AFTER memory fusion |
| **P30 (ours)** | **RBMA training-free entropy B_i** | **additive pre-softmax bias in SAM2 memory cross-attn + reliability-anchored router** | **logit-additive-bias + mask-classification queries** | **class tokens cross-attend to reliability-biased fused MEMORY features** |

Key adversarially-confirmed facts protecting the RBMA cell from this track's side: CAFuser has **no additive pre-softmax attention-logit bias anywhere** (CA² = token append; CAA = softmax feature weights); DGFusion conditions attention by **token concatenation**, not logit bias. The logit-bias cell (see [[relatedworks/42_attention_logit_bias_novelty_defense]]) remains unoccupied from the query-decoder literature too.

## Application to ours (RBMA/P29/P30 적용방향)

1. **Novelty phrasing (mandatory rewrite):** never "first query decoder on fused multimodal features." Anchor: (i) mask-classification class tokens on SAM2 **memory-attention** features of a modalities-as-frames stream, (ii) query→modality routing anchored by the **training-free** RBMA reliability B_i, (iii) explicit differentiation from **MemorySAM's stock SAM2 mask decoder** (shallow two-way transformer + per-pixel logits, undocumented output tokens) — state that we replace it with a multi-layer mask-classification decoder with per-class queries and decoupled mask/score prediction. This paragraph is the direct answer to the gray zone flagged in verification.
2. **Must-cite set:** BiXFormer (TMM'26/arXiv:2506.03675 — owns the "first mask-classification × multi-sensor" phrase; note its claim is itself preceded by CAFuser), CAFuser, DGFusion, SAM-DAQ, SHIFNet, MemorySAM, DF2RQ (after PDF obtained), EoMT (SAM3 branch).
3. **P28 rare-class fix, evidence-backed design:** fixed per-class tokens (no Hungarian matching → sidesteps 2406.03917's starvation) + auxiliary per-pixel CE head on fused memory features (GOOSE-M2F; cite +3.4% composite ablation alongside the qualitative +5–8% rare claim). This design step is our inference from confirmed premises — present it as motivated design, not as a literature result.
4. **Ablation baselines this track hands us:** (a) queries-per-modality + UMM max-fusion (BiXFormer-style) vs our queries-on-fused-memory; (b) CAFuser-style CT-append vs RBMA logit bias at the same attention; (c) post-softmax vs pre-softmax reliability injection; (d) stock SAM2 decoder (MemorySAM) vs class-token decoder on identical RBMA features — ablation (d) simultaneously resolves the MemorySAM gray zone empirically.
5. **DGFusion rhetoric collision:** its "spatially varying sensor reliability" motivation overlaps ours — pre-empt in related work by contrasting mechanism (learned depth tokens, depth-GT supervision vs training-free entropy) and locus (backbone fusion attention vs VFM memory attention).
6. **SAM3 fallback:** if the memory-stream branch stalls, EoMT-style query injection into final PE blocks on a multimodal stream is a clean unoccupied design (flag as extrapolation until prototyped; the ~24 mIoU SAM3-RBMA plateau is consistent with missing multi-scale — ViTDet simple-FPN is the second precedented fix).

## Related-work paragraph candidate (English)

Query-based mask classification has recently been extended to multi-sensor segmentation. BiXFormer [arXiv:2506.03675] couples modality-specific queries with unified modality matching, combining per-modality predictions rather than features; CAFuser [RA-L'25] and DGFusion [RA-L'26] instead fuse multimodal features under a condition or depth token and pass the fused maps to an OneFormer head, so their queries never participate in fusion; RoadFormer+ likewise feeds fused RGB-X features to a MaskFormer-paradigm decoder. On the foundation-model side, SHIFNet scores frozen text-derived category embeddings against fused SAM2 image features, and SAM-DAQ replaces SAM2's memory bank with learnable queries for binary RGB-D saliency, while MemorySAM decodes multimodal memory-attended features with SAM2's stock per-pixel mask decoder. To our knowledge, no prior work decodes semantic masks by cross-attending learnable per-class tokens to the memory-attention features of a video foundation model whose "frames" are sensor modalities, nor anchors the resulting query-to-modality routing with a training-free reliability signal: prior query-based multimodal heads either consume features fused before the queries, keep queries strictly per-modality and merge only predictions, or substitute the memory mechanism itself with queries. Our decoder differs from MemorySAM's stock head in both training framework (mask classification with fixed per-class queries and an auxiliary per-pixel head, mitigating the tail-class starvation documented for one-to-one matching [arXiv:2406.03917; GOOSE-M2F]) and in that its cross-attention substrate is explicitly reliability-biased by RBMA.

## Gaps / could-not-verify (carried from findings + verification)

- DF2RQ full method + numbers (IEEE paywall) — PDF required before citation.
- EoMT exact table rows (COCO PQ / ADE20K mIoU / FPS) — architecture and 4× claim verified via official repo/HF doc; numbers pending.
- BiXFormer TMM acceptance — only the authors' "accepted by TMM26" statement; no independent record.
- MemorySAM 65.38 split (val/test) still unresolved (Track 3 item); MemorySAM stock-decoder token mechanism undocumented (the gray zone) — settle empirically via ablation (d) or by reading SAM2 decoder code paths in the MemorySAM fork.
- MaskFormer ADE20K-Full +3.5 figure — re-verify Table 3 before quoting verbatim.
- MemorySAM citation sweep limited to 13 Semantic Scholar entries; residual scoop risk only from unindexed very-recent work.

## Links

- [[relatedworks/42_attention_logit_bias_novelty_defense]] — logit-bias cell defense (RBMA side)
- [[relatedworks/31_mask2former_relatedwork]] / [[relatedworks/32_oneformer_relatedwork]] — head baselines
- [[relatedworks/01_memorysam_relatedwork]] — baseline + gray-zone target
- [[relatedworks/02_dgfusion_relatedwork]] / [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
- [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] — protocol tags for the numbers above
- [[sources/07_parallel_research_prompts_2026-07-02]] — Track 7 spec
