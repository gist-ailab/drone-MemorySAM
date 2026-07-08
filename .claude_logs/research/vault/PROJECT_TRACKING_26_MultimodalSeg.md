---
title: Project Tracking Board — 26_MultimodalSeg
tags: [project-tracking, status, moc, multimodal-segmentation]
created: 2026-06-24
source: [[00_MOC_26_MultimodalSeg]], [[relatedworks/00_relatedworks_index]]
status: active
---

# Project Tracking Board — 26_MultimodalSeg

> Purpose: 운동/외출 중에도 현재까지 무엇을 했고, 다음에 무엇을 해야 하는지 추적하기 위한 상태판.

## Legend

| Mark | Meaning |
|---|---|
| ✅ Done | 파일 생성/검증 완료 |
| 🟡 In progress | 일부 생성/초안 있음, 추가 추출 필요 |
| ⏭ Next | 바로 다음 작업 후보 |
| ⛔ Blocked | 외부 접근/API/원문 미확보 등으로 막힘 |

## 0. Project setup

| Status | Item | Output |
|---|---|---|
| ✅ | Obsidian project folder scaffold | `sources/`, `relatedworks/`, `material/`, `assets/` |
| ✅ | Project MOC | [[00_MOC_26_MultimodalSeg]] |
| ✅ | Imported Claude/NotebookLM seed memo | [[sources/00_imported_claude_related_work_2026-06-24]] |
| ✅ | First-pass Semantic Scholar source index | [[sources/01_source_index_multimodal_segmentation]] |
| ✅ | Large OpenAlex literature DB | [[sources/02_openalex_top_venue_literature_database]] |
| ✅ | Seed-paper candidate verification map | [[sources/03_seed_paper_verification_candidates]] |

## 1. Large source database

| Status | Category | Output | Count / note |
|---|---|---|---|
| ✅ | Full DB | [[sources/db/openalex_top_venue_multimodal_literature_db_2026-06-24.json]] / CSV / JSONL / SQLite | 3,010 records |
| ✅ | Top venue / vision journal flagged | [[sources/02_openalex_top_venue_literature_database]] | 408 records |
| ✅ | Multimodal semantic segmentation map | [[sources/02_source_map_multimodal_semantic_segmentation]] | 773 category hits |
| ✅ | Multimodal object detection map | [[sources/02_source_map_multimodal_object_detection]] | 1,106 category hits |
| ✅ | Adapter / LoRA / SAM adaptation map | [[sources/02_source_map_adapter_lora_foundation_seg_det]] | 1,142 category hits |
| ✅ | Segmentation/detection heads map | [[sources/02_source_map_segmentation_detection_heads]] | 758 category hits |
| ✅ | Uncertainty/reliability category present | inside main DB | 204 category hits |

## 2. Priority A — core multimodal semantic segmentation

| Status | Note | What was done | Remaining |
|---|---|---|---|
| ✅ | [[relatedworks/01_memorysam_relatedwork]] | PDF downloaded, method/novelty/gap summarized | Extract full table values into benchmark matrix |
| ✅ | [[relatedworks/02_dgfusion_relatedwork]] | RA-L/arXiv PDF downloaded, method/novelty/gap summarized | Add exact DeLiVER/MUSES tables |
| ✅ | [[relatedworks/03_unimodal_bias_entropy_relatedwork]] | Functional entropy regularization summarized | Extract all Tables 1–3 values cleanly |
| ✅ | [[relatedworks/04_stitchfusion_relatedwork]] | Adapter/MultiAdapter weaving summarized | Extract Table 1/2/7 values cleanly |
| ✅ | [[relatedworks/05_anyseg_relatedwork]] | Anymodal distillation summarized | Extract MUSES/DeLiVER tables cleanly |
| 🟡 | [[relatedworks/06_deliver_muses_mcubes_dataset_note]] | Dataset note scaffolded | Official MUSES/MCubeS source cards still needed |
| ✅ | [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]] | CMX/TokenFusion/MAGIC++/CAFuser grouped | Some exact tables still pending |
| ✅ | [[relatedworks/08_priority_a_comparison_matrix]] | High-level comparison matrix created | Add exact numeric benchmark table refs |
| ✅ | [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] | Benchmark extraction note overwritten with verified MemorySAM, DGFusion, Reducing Unimodal Bias, StitchFusion, AnySeg, MAGIC++, and CAFuser tables | Caveats remain: CAFuser Table III visual check recommended; StitchFusion Table 2 is non-target; official MUSES/MCubeS dataset cards still needed |

## 3. Downloaded/archived Priority A PDFs

| Status | Paper | PDF |
|---|---|---|
| ✅ | MemorySAM | [[sources/pdfs/priority_a/2503.06700.pdf]] |
| ✅ | DGFusion | [[sources/pdfs/priority_a/2509.09828.pdf]] |
| ✅ | Reducing Unimodal Bias | [[sources/pdfs/priority_a/2505.06635.pdf]] |
| ✅ | StitchFusion | [[sources/pdfs/priority_a/2408.01343.pdf]] |
| ✅ | AnySeg | [[sources/pdfs/priority_a/2411.17141.pdf]] |
| ✅ | DeLiVER | [[sources/pdfs/priority_a/2303.01480.pdf]] |
| ✅ | CMX | [[sources/pdfs/priority_a/2203.04838.pdf]] |
| ✅ | TokenFusion | [[sources/pdfs/priority_a/2204.08721.pdf]] |
| ✅ | MAGIC++ | [[sources/pdfs/priority_a/2412.16876.pdf]] |
| ✅ | CAFuser | [[sources/pdfs/priority_a/2410.10791.pdf]] |

## 4. Next work queue

### ⏭ Priority 1 — benchmark tables

- [x] Finish [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]].
- [x] Extract exact scores from MemorySAM Table 1/2.
- [x] Extract DGFusion MUSES Table I/II and DeLiVER table.
- [x] Extract StitchFusion Table 1/2/7. *(Table 2 verified as non-target FMB/PST900/MFNet.)*
- [x] Extract AnySeg Table 1/2.
- [x] Extract MAGIC++ Table I/II.
- [x] Extract CAFuser Table I/II/III. *(Table III should receive one final visual check because source text interleaves adjacent tables.)*
- [ ] Add official dataset card rows for MUSES and MCubeS.

### ⏭ Priority 2 — uncertainty / reliability / modality bias

Target output:

```text
relatedworks/40_uncertainty_reliability_fusion_relatedwork.md
relatedworks/41_unimodal_bias_and_modality_collapse.md
relatedworks/42_attention_logit_bias_novelty_defense.md
```

Must cover: UTFNet, HyperDUM, TMC/ETMC, conflict-guided evidential fusion, ReliFusion, READ, AG-Fusion, DGFusion, CAFuser, Reducing Unimodal Bias.

Status 2026-06-24: ✅ created [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/41_unimodal_bias_and_modality_collapse]], and [[relatedworks/42_attention_logit_bias_novelty_defense]]. Verified via OpenAlex/arXiv where possible: UTFNet, HyperDUM, TMC, CAFuser, DGFusion, MAGIC++, AnySeg, Any2Seg, RMMSS, Reducing Unimodal Bias, and Pattern Recognition semantic-conflict metadata. Marked ReliFusion/READ/AG-Fusion/EQUISeg as unresolved verification gaps because exact source matches were not found in this run.

### ⏭ Priority 3 — multimodal object detection

Target output:

```text
relatedworks/10_bevfusion_relatedwork.md
relatedworks/11_transfusion_relatedwork.md
relatedworks/12_deepinteraction_relatedwork.md
relatedworks/13_futr3d_relatedwork.md
relatedworks/14_multimodal_detection_survey_note.md
```

Must cover: BEVFusion, TransFusion, DeepInteraction, FUTR3D, BEVFormer/DETR3D/PETR, PointPainting/MVX-Net/CLOCs, RGB-T detection, radar-camera-LiDAR detection.

Status 2026-06-24: ✅ created [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/11_transfusion_relatedwork]], [[relatedworks/12_deepinteraction_relatedwork]], [[relatedworks/13_futr3d_relatedwork]], and [[relatedworks/14_multimodal_detection_survey_note]]. Verified central metadata via arXiv API and existing OpenAlex/source-map entries. Covered required methods: BEVFusion, TransFusion, DeepInteraction, FUTR3D, BEVFormer, DETR3D, PETR/PETRv2, PointPainting, MVX-Net, CLOCs, CenterPoint, RGB-T/thermal, radar-camera-LiDAR, event sensors, and adverse-weather/night detection. Caveat: exact benchmark table values and some secondary venue details should be checked during final bibliography/table extraction.

### ⏭ Priority 4 — adapters / LoRA / SAM adaptation

Target output:

```text
relatedworks/20_lora_adapter_relatedwork.md
relatedworks/21_vit_adapter_relatedwork.md
relatedworks/22_sam_adapter_relatedwork.md
relatedworks/23_multimodal_sam_adapter_matrix.md
```

Must cover: LoRA, ViT-Adapter, AdaptFormer, Visual Prompt Tuning, SAM-Adapter, MedSAM/SAMed, SAM-FuseNet, Mixture of LoRA Experts for multimodal semantic segmentation.

### ✅ Priority 5 — heads

Target output:

```text
relatedworks/30_segformer_relatedwork.md
relatedworks/31_mask2former_relatedwork.md
relatedworks/32_oneformer_relatedwork.md
relatedworks/33_detr_deformable_detr_dino_relatedwork.md
relatedworks/34_maskdino_yolo_maskrcnn_heads.md
```

Status 2026-06-24: ✅ created [[relatedworks/30_segformer_relatedwork]], [[relatedworks/31_mask2former_relatedwork]], [[relatedworks/32_oneformer_relatedwork]], [[relatedworks/33_detr_deformable_detr_dino_relatedwork]], and [[relatedworks/34_maskdino_yolo_maskrcnn_heads]]. Covered required segmentation heads/frameworks: SegFormer, Mask2Former, OneFormer, SETR, UPerNet, DeepLabv3+, and MaskDINO. Covered required detection heads/frameworks: DETR, Deformable DETR, DINO, MaskDINO, Mask R-CNN, YOLO family, and BEV/transformer 3D detection heads including BEVFormer/DETR3D/PETR with links to existing BEVFusion/TransFusion/FUTR3D notes. Each note records citation metadata, mechanism, supported tasks/metrics, strengths/limitations, relevance to semantic segmentation vs object detection, and reusable related-work paragraph candidates.

## 5. Current synthesis state

The project argument is currently converging to:

> Existing RGB-X / multimodal segmentation methods use feature fusion, token fusion, modality selection, condition tokens, depth tokens, adapters, or distillation. MemorySAM uniquely maps modalities into SAM2 memory attention, but it does not explicitly model reliability. RBMA should be positioned as reliability-aware SAM2 memory fusion: predictive uncertainty estimates reliability, and reliability is injected as an additive pre-softmax memory-attention logit bias.

## 6. Known caveats

- OpenAlex DB is a discovery DB; it contains noise and needs per-paper verification before final citation.
- Some PDF table extraction is messy because `pdftotext` collapses columns.
- MUSES and MCubeS official dataset cards are not fully extracted yet.
- Benchmark tables should include source table number/figure number for every numeric claim. **Current status:** [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] now includes source table IDs for extracted benchmark rows; remaining caveats are CAFuser Table III visual/PDF alignment and official MUSES/MCubeS dataset-card extraction.

## 7. Cron automation — scheduled research/document jobs

| Status | Job | Job ID | Schedule | Output target |
|---|---|---|---|---|
| ✅ Done | P1 benchmark tables | `052c5da1d3f3` | 2026-06-24 22:27 KST, once | [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] |
| ✅ Done | P2 reliability / uncertainty / novelty defense | `f85a0cf4ded2` | 2026-06-24 22:37 KST, once | [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/41_unimodal_bias_and_modality_collapse]], [[relatedworks/42_attention_logit_bias_novelty_defense]] |
| ✅ Done | P3 multimodal object detection | `a08e907d9127` | 2026-06-24 22:47 KST, once | [[relatedworks/10_bevfusion_relatedwork]] through [[relatedworks/14_multimodal_detection_survey_note]] |
| 🟡 Scheduled | P4 adapter / LoRA / SAM adaptation | `58ec3bc0b8d2` | 2026-06-24 22:57 KST, once | [[relatedworks/20_lora_adapter_relatedwork]] through [[relatedworks/23_multimodal_sam_adapter_matrix]] |
| 🟡 Scheduled | P5 segmentation / detection heads | `4c758ef85268` | 2026-06-24 23:07 KST, once | [[relatedworks/30_segformer_relatedwork]] through [[relatedworks/34_maskdino_yolo_maskrcnn_heads]] |
| 🟡 Recurring | Daily top-venue source sweep | `11306d3e510d` | daily 08:30 KST | [[sources/04_daily_source_sweep_log]] |

All cron jobs are pinned to `openai-codex / gpt-5.5`, with `file`, `terminal`, and `web` toolsets enabled, and `obsidian` + `arxiv` skills loaded.
## 5. Adapter / LoRA / SAM foundation-model adaptation — 2026-06-24 cron update

| Status | Item | Output |
|---|---|---|
| ✅ | LoRA / adapter / VPT synthesis | [[relatedworks/20_lora_adapter_relatedwork]] |
| ✅ | ViT-Adapter dense prediction synthesis | [[relatedworks/21_vit_adapter_relatedwork]] |
| ✅ | SAM/SAM2 adapter synthesis | [[relatedworks/22_sam_adapter_relatedwork]] |
| ✅ | Multimodal SAM adapter candidate matrix | [[relatedworks/23_multimodal_sam_adapter_matrix]] |

Best adapter candidates now identified: (1) MemorySAM + reliability-biased memory attention for SAM2-style modality-as-frame fusion, (2) MoE-LoRA for SAM as the closest modality-expert LoRA comparator, (3) StitchFusion MultiAdapter as the strongest non-SAM multimodal adapter baseline, and (4) ViT-Adapter for dense-prediction spatial/multi-scale adaptation. SAM3 remains an open verification item; no stable primary SAM3 source was verified in this run.

## Weekly collection + refinement/PDF jobs — 2026-06-25

| Status | Job | Job ID | Schedule | Output target |
|---|---|---|---|---|
| ✅ Scheduled | Weekly top-venue source sweep | `11306d3e510d` | every 7 days; next 2026-07-02 01:32 KST | [[sources/04_weekly_source_sweep_log]] |
| ✅ Scheduled | Related-work clustering and PDF material | `62c71609cfff` | 2026-06-25 02:20 KST | [[relatedworks/90_clustered_relatedwork_synthesis]], `material/01_multimodal_seg_clustered_relatedwork_{en,ko}.md/.pdf` |

Current policy: new-source collection is weekly; current collected sources should be refined into clustered related work and PDF-quality material.

## X trend watch status — 2026-06-25

- X trend watch query note prepared; weekly sweep can include X after xAI/Grok credits/subscription are available.

## Available-source / LinkedIn-aware policy — 2026-06-25

- X Search remains blocked by xAI/Grok spending-limit, so weekly sweeps should not call `x_search`.
- Public LinkedIn posts/pages are allowed as trend-discovery sources.
- Only linked/verified primary sources should be promoted into source DBs and related work.
- Weekly sweep `11306d3e510d` updated to include LinkedIn/lab/project/code discovery and keep new-source collection weekly.

## Related-work clustering and PDF material — 2026-06-25

| Status | Item | Output |
|---|---|---|
| ✅ | Clustered related-work synthesis | [[relatedworks/90_clustered_relatedwork_synthesis]] |
| ✅ | English PDF-ready review material | [[material/01_multimodal_seg_clustered_relatedwork_en]] |
| ✅ | Korean PDF-ready review material | [[material/01_multimodal_seg_clustered_relatedwork_ko]] |
| ✅ | PDF export | ReportLab export completed: [[material/01_multimodal_seg_clustered_relatedwork_en.pdf]] and [[material/01_multimodal_seg_clustered_relatedwork_ko.pdf]]. Note: pandoc/LaTeX/wkhtmltopdf were unavailable, so PDFs use the local ReportLab renderer. |

Synthesis clusters completed: direct multimodal semantic segmentation; multimodal object detection; adapters/LoRA/foundation-model adaptation; segmentation/detection heads; uncertainty/reliability/novelty; benchmarks/datasets. Main paper-writing conclusion: RBMA should be positioned as reliability-aware SAM2-style memory fusion where predictive reliability is injected as an additive pre-softmax attention-logit bias, distinct from feature scaling, late evidential fusion, modality selection, distillation, and condition/depth-token fusion.

## Material/PDF clarification — 2026-06-25 03:17

- User clarified that `material/` should prioritize **자료조사 및 정제** (source-heavy literature review/study material), not presentation-slide work.
- Existing/pending automation updated accordingly: touchNplug continuous loop now focuses on literature collection, verification, clustering, related-work synthesis, EN/KO study material, and PDF export where feasible.
- Local PDF export fallback created via ReportLab: `C:/Users/ailab-drone-nuc/md_to_pdf_reportlab.py`.

## Weekly source sweep — 2026-07-02

| Status | Item | Output | Count / note |
|---|---|---|---|
| ✅ | Weekly top-venue/source sweep | [[sources/04_weekly_source_sweep_log]] | 14 verified new/missing candidates promoted; 10 high-priority short source notes created/updated |
| 🟡 | API caveats | arXiv/GitHub/OpenAlex primary metadata usable; Semantic Scholar/Papers with Code partially rate-limited/unavailable; LinkedIn probe yielded no promoted primary links | X Search intentionally not used |

High-priority additions this run include RSGMamba, M4-SAM, GeomPrompt, ModalPatch, AW-MoE, MULTIAQUA, FS-SAM2, ClustViT, balanced-modality multimodal segmentation, OmniSegmentor, and selected SAM2 code candidates.
