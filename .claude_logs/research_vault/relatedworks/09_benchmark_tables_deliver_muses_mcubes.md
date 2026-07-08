---
title: Benchmark Tables — DeLiVER, MUSES, MCubeS
created: 2026-06-24
tags: [related-work, benchmark, dataset, multimodal-segmentation, deliver, muses, mcubes, key-paper]
source: [[PROJECT_TRACKING_26_MultimodalSeg]], [[relatedworks/01_memorysam_relatedwork]], [[relatedworks/02_dgfusion_relatedwork]], [[relatedworks/03_unimodal_bias_entropy_relatedwork]], [[relatedworks/04_stitchfusion_relatedwork]], [[relatedworks/05_anyseg_relatedwork]], [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
status: verified-with-caveats
---

# Benchmark Tables — DeLiVER, MUSES, MCubeS

> Scope: exact numeric benchmark values extracted from local Priority-A PDF text/PyMuPDF extraction for [[DeLiVER]], [[MUSES]], and [[MCubeS]]. Only values visible in extracted table text are recorded as verified. Where column alignment remains ambiguous, the row is marked **needs visual/PDF table check** rather than guessed.

## Source PDFs / text extracts

| Paper | Local PDF | Text extract | Requested tables | Status |
|---|---|---|---|---|
| MemorySAM | [[sources/pdfs/priority_a/2503.06700.pdf]] | [[sources/pdfs/priority_a/text/2503.06700.txt]] | Table 1, Table 2 | ✅ verified from extracted table text |
| DGFusion | [[sources/pdfs/priority_a/2509.09828.pdf]] | [[sources/pdfs/priority_a/text/2509.09828.txt]] | Table I, II, III | ✅ verified |
| Reducing Unimodal Bias | [[sources/pdfs/priority_a/2505.06635.pdf]] | [[sources/pdfs/priority_a/text/2505.06635.txt]] | Table 1, 2, 3 | ✅ values extracted; minor header typo preserved |
| StitchFusion | [[sources/pdfs/priority_a/2408.01343.pdf]] | [[sources/pdfs/priority_a/text/2408.01343.txt]] | Table 1, 2, 7 | ✅ Table 1/7 target rows verified; Table 2 is non-target FMB/PST900/MFNet |
| AnySeg | [[sources/pdfs/priority_a/2411.17141.pdf]] | [[sources/pdfs/priority_a/text/2411.17141.txt]] | Table 1, 2 | ✅ verified |
| MAGIC++ | [[sources/pdfs/priority_a/2412.16876.pdf]] | [[sources/pdfs/priority_a/text/2412.16876.txt]] | Table I, II | ✅ verified |
| CAFuser | [[sources/pdfs/priority_a/2410.10791.pdf]] | [[sources/pdfs/priority_a/text/2410.10791.txt]] | Table I, II, III | ✅ core values verified; Table III method alignment cross-checked with DGFusion Table III |

## 1. High-level benchmark rows for paper comparison

| Source table | Dataset | Method | Modality set | Backbone | Metric | Score | Verified comparison margin |
|---|---|---|---|---|---|---:|---:|
| MemorySAM Table 1 | DeLiVER | MemorySAM | R-D-E-L | Hiera-B+ w/ 1 LoRA | mIoU | 65.38 | +6.20 vs CMNeXt R-D-E-L |
| MemorySAM Table 2 | MCubeS | MemorySAM | R-A-D-N | Hiera-B+ w/ 1 LoRA | mIoU | 52.88 | +16.72 vs CMNeXt R-A-D-N |
| DGFusion Table I | MUSES | DGFusion | CLRE | Swin-T | PQ | 61.03 | +1.33 vs CAFuser |
| DGFusion Table II | MUSES | DGFusion | CLRE | Swin-T | mIoU | 79.5 | +1.00 vs CAFuser-CAA |
| DGFusion Table III | DeLiVER | DGFusion | CLDE | Swin-T | mIoU-test | 56.7 | +1.10 vs CAFuser |
| Reducing Unimodal Bias Table 2 | MUSES | Ours | F/E/L anymodal eval | source setting | Mean mIoU | 47.80 | +13.94 vs Any2Seg |
| Reducing Unimodal Bias Table 3 | DeLiVER | Ours | R/D/E/L anymodal eval | source setting | Mean mIoU | 48.29 | +3.25 vs Any2Seg |
| StitchFusion Table 1 | DeLiVER | StitchFusion | paper default | Swin-Tiny-1k | mIoU | 70.3 | +4.00 vs CMNeXt |
| StitchFusion Table 1 | MCubeS | StitchFusion | paper default | Swin-Large-22k | mIoU | 55.9 | +2.80 vs MMSFormer |
| AnySeg Table 1 | MUSES | AnySeg / Ours | F/E/L anymodal eval | SegFormer-B0 | Mean mIoU | 40.23 | +6.37 vs Any2Seg |
| AnySeg Table 2 | DeLiVER | AnySeg / Ours | R/D/E/L anymodal eval | SegFormer-B0 | Mean mIoU | 46.64 | +6.15 vs MAGIC |
| MAGIC++ Table I | MUSES | MAGIC++ | F/E/L MaSS | SegFormer-B0 | Mean mIoU | 35.53 | +2.19 vs MAGIC |
| MAGIC++ Table II | DeLiVER | MAGIC++ | R/D/E/L MaSS | SegFormer-B0 | Mean mIoU | 47.74 | +7.25 vs MAGIC |
| CAFuser Table I | MUSES | CAFuser-CA2 | CLRE | Swin-T | PQ / All | 59.7 | +6.1 vs MUSES baseline |
| CAFuser Table II | MUSES | CAFuser-CA2 | CLRE | Swin-T | mIoU | 78.2 | -0.3 vs CAFuser-CAA |
| CAFuser Table III | DeLiVER | CAFuser-CA2 | CLDE | Swin-T | mIoU-test | 55.6 | +1.1 vs GeminiFusion |

## 2. MemorySAM — Table 1 / Table 2

Source: [[relatedworks/01_memorysam_relatedwork]], `2503.06700.pdf`. Modalities: DeLiVER R=RGB, D=Depth, E=Event, L=LiDAR; MCubeS R=RGB, A=AoLP, D=DoLP, N=NIR.

### 2.1 Table 1 — DeLiVER

| Source table | Dataset | Method | Modalities | Backbone / adaptation | Metric | mIoU | Δ in source |
|---|---|---|---|---|---|---:|---:|
| MemorySAM Table 1 | DeLiVER | CMNeXt | RGB | MiT-B0 | mIoU | 51.29 | — |
| MemorySAM Table 1 | DeLiVER | CWSAM | RGB | ViT-B w/ Adapter | mIoU | 51.59 | +0.30 |
| MemorySAM Table 1 | DeLiVER | SAM-LoRA | RGB | ViT-B w/ 1 LoRA | mIoU | 51.84 | +0.55 |
| MemorySAM Table 1 | DeLiVER | MLE-SAM | RGB | Hiera-B+ w/ 4 LoRA | mIoU | 55.23 | +3.94 |
| MemorySAM Table 1 | DeLiVER | MemorySAM | RGB | Hiera-B+ w/ 1 LoRA | mIoU | 53.22 | +1.93 |
| MemorySAM Table 1 | DeLiVER | CMNeXt | R-D | MiT-B0 | mIoU | 59.61 | — |
| MemorySAM Table 1 | DeLiVER | CWSAM | R-D | ViT-B w/ Adapter | mIoU | 58.64 | -0.97 |
| MemorySAM Table 1 | DeLiVER | SAM-LoRA | R-D | ViT-B w/ 1 LoRA | mIoU | 60.25 | +0.64 |
| MemorySAM Table 1 | DeLiVER | MLE-SAM | R-D | Hiera-B+ w/ 4 LoRA | mIoU | 63.57 | +3.96 |
| MemorySAM Table 1 | DeLiVER | MemorySAM | R-D | Hiera-B+ w/ 1 LoRA | mIoU | 63.48 | +3.87 |
| MemorySAM Table 1 | DeLiVER | CMNeXt | R-D-E | MiT-B0 | mIoU | 59.84 | — |
| MemorySAM Table 1 | DeLiVER | CWSAM | R-D-E | ViT-B w/ Adapter | mIoU | 56.22 | -3.62 |
| MemorySAM Table 1 | DeLiVER | SAM-LoRA | R-D-E | ViT-B w/ 1 LoRA | mIoU | 60.08 | +0.24 |
| MemorySAM Table 1 | DeLiVER | MLE-SAM | R-D-E | Hiera-B+ w/ 4 LoRA | mIoU | 62.69 | +2.85 |
| MemorySAM Table 1 | DeLiVER | MemorySAM | R-D-E | Hiera-B+ w/ 1 LoRA | mIoU | 62.42 | +2.58 |
| MemorySAM Table 1 | DeLiVER | CMNeXt | R-D-E-L | MiT-B0 | mIoU | 59.18 | — |
| MemorySAM Table 1 | DeLiVER | CWSAM | R-D-E-L | ViT-B w/ Adapter | mIoU | 55.43 | -3.75 |
| MemorySAM Table 1 | DeLiVER | SAM-LoRA | R-D-E-L | ViT-B w/ 1 LoRA | mIoU | 59.54 | +0.36 |
| MemorySAM Table 1 | DeLiVER | MLE-SAM | R-D-E-L | Hiera-B+ w/ 4 LoRA | mIoU | 64.08 | +4.90 |
| MemorySAM Table 1 | DeLiVER | **MemorySAM** | **R-D-E-L** | **Hiera-B+ w/ 1 LoRA** | **mIoU** | **65.38** | **+6.20** |

### 2.2 Table 2 — MCubeS

| Source table | Dataset | Method | Modalities | Backbone / adaptation | Metric | mIoU | Δ in source |
|---|---|---|---|---|---|---:|---:|
| MemorySAM Table 2 | MCubeS | CMNeXt | R-A | MiT-B0 | mIoU | 37.21 | — |
| MemorySAM Table 2 | MCubeS | CWSAM | R-A | ViT-B w/ Adapter | mIoU | 49.78 | +12.57 |
| MemorySAM Table 2 | MCubeS | SAM-LoRA | R-A | ViT-B w/ 1 LoRA | mIoU | 48.74 | +11.53 |
| MemorySAM Table 2 | MCubeS | MLE-SAM | R-A | Hiera-B+ w/ 4 LoRA | mIoU | 50.61 | +13.40 |
| MemorySAM Table 2 | MCubeS | MemorySAM | R-A | Hiera-B+ w/ 1 LoRA | mIoU | 51.20 | +13.99 |
| MemorySAM Table 2 | MCubeS | CMNeXt | R-A-D | MiT-B0 | mIoU | 38.72 | — |
| MemorySAM Table 2 | MCubeS | CWSAM | R-A-D | ViT-B w/ Adapter | mIoU | 48.27 | +9.55 |
| MemorySAM Table 2 | MCubeS | SAM-LoRA | R-A-D | ViT-B w/ 1 LoRA | mIoU | 49.35 | +10.63 |
| MemorySAM Table 2 | MCubeS | MLE-SAM | R-A-D | Hiera-B+ w/ 4 LoRA | mIoU | 50.89 | +12.17 |
| MemorySAM Table 2 | MCubeS | MemorySAM | R-A-D | Hiera-B+ w/ 1 LoRA | mIoU | 52.20 | +13.48 |
| MemorySAM Table 2 | MCubeS | CMNeXt | R-A-D-N | MiT-B0 | mIoU | 36.16 | — |
| MemorySAM Table 2 | MCubeS | CWSAM | R-A-D-N | ViT-B w/ Adapter | mIoU | 50.59 | +14.43 |
| MemorySAM Table 2 | MCubeS | SAM-LoRA | R-A-D-N | ViT-B w/ 1 LoRA | mIoU | 49.46 | +13.30 |
| MemorySAM Table 2 | MCubeS | MLE-SAM | R-A-D-N | Hiera-B+ w/ 4 LoRA | mIoU | 51.02 | +14.86 |
| MemorySAM Table 2 | MCubeS | **MemorySAM** | **R-A-D-N** | **Hiera-B+ w/ 1 LoRA** | **mIoU** | **52.88** | **+16.72** |

## 3. DGFusion — Tables I / II / III

Source: [[relatedworks/02_dgfusion_relatedwork]], `2509.09828.pdf`. Modalities: C=Camera, L=LiDAR, R=Radar, E=Events.

### 3.1 Table I — MUSES panoptic segmentation test set

| Source table | Dataset | Method | Modalities | Backbone | Day | Night | Clear | Fog | Rain | Snow | Things | Stuff | SQ | RQ | PQ ↑ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DGFusion Table I | MUSES | Mask2Former | C | Swin-T | 49.35 | 39.38 | 48.84 | 46.48 | 45.39 | 45.10 | 31.29 | 58.23 | 78.34 | 58.01 | 46.89 |
| DGFusion Table I | MUSES | MaskDINO | C | Swin-T | 51.85 | 42.73 | 54.05 | 46.20 | 46.23 | 48.54 | 38.64 | 57.29 | 80.51 | 60.28 | 49.44 |
| DGFusion Table I | MUSES | OneFormer | C | Swin-T | 57.55 | 47.83 | 58.33 | 53.68 | 53.43 | 53.77 | 43.54 | 63.69 | 80.93 | 66.97 | 55.21 |
| DGFusion Table I | MUSES | HRFuser | CLRE | HRFuser-T | 44.60 | 40.01 | 47.03 | 43.59 | 42.69 | 40.62 | 28.33 | 55.23 | 78.41 | 54.34 | 43.90 |
| DGFusion Table I | MUSES | MUSES baseline | CLRE | 4xSwin-T | 54.06 | 49.73 | 55.28 | 50.34 | 53.77 | 50.51 | 39.94 | 63.53 | 81.06 | 65.02 | 53.60 |
| DGFusion Table I | MUSES | CAFuser-CAA | CLRE | Swin-T | 59.93 | 56.24 | 61.16 | 56.41 | 59.38 | 57.88 | 48.03 | 67.64 | 81.80 | 71.61 | 59.38 |
| DGFusion Table I | MUSES | CAFuser | CLRE | Swin-T | 59.49 | 57.34 | 61.36 | 57.52 | 59.63 | 57.20 | 48.42 | 67.90 | 82.03 | 71.75 | 59.70 |
| DGFusion Table I | MUSES | **DGFusion** | **CLRE** | **Swin-T** | **60.94** | **58.97** | **62.16** | **58.86** | **61.26** | **59.77** | **49.68** | **69.28** | **82.34** | **73.07** | **61.03** |

### 3.2 Table II — MUSES semantic segmentation test set

| Source table | Dataset | Method | Modalities | Backbone | Metric | mIoU ↑ |
|---|---|---|---|---|---|---:|
| DGFusion Table II | MUSES | Mask2Former | C | Swin-T | mIoU | 70.7 |
| DGFusion Table II | MUSES | SegFormer | C | MiT-B2 | mIoU | 72.5 |
| DGFusion Table II | MUSES | OneFormer | C | Swin-T | mIoU | 72.8 |
| DGFusion Table II | MUSES | CMNeXt | CLRE | MiT-B2 | mIoU | 72.1 |
| DGFusion Table II | MUSES | GeminiFusion | CLRE | MiT-B2 | mIoU | 75.3 |
| DGFusion Table II | MUSES | CAFuser-CAA | CLRE | Swin-T | mIoU | 78.5 |
| DGFusion Table II | MUSES | CAFuser | CLRE | Swin-T | mIoU | 78.2 |
| DGFusion Table II | MUSES | **DGFusion** | **CLRE** | **Swin-T** | **mIoU** | **79.5** |

### 3.3 Table III — DeLiVER semantic segmentation test set

| Source table | Dataset | Method | Modalities | Backbone | Metric | CLE | CLDE |
|---|---|---|---|---|---|---:|---:|
| DGFusion Table III | DeLiVER | CMNeXt | CLE / CLDE | MiT-B2 | mIoU-test | 50.3 | 53.0 |
| DGFusion Table III | DeLiVER | StitchFusion | CLE / CLDE | MiT-B2 | mIoU-test | 50.8 | 53.4 |
| DGFusion Table III | DeLiVER | GeminiFusion | CLE / CLDE | MiT-B2 | mIoU-test | 50.5 | 54.5 |
| DGFusion Table III | DeLiVER | CAFuser-CAA | CLE / CLDE | Swin-T | mIoU-test | 51.2 | 55.2 |
| DGFusion Table III | DeLiVER | CAFuser | CLE / CLDE | Swin-T | mIoU-test | 51.3 | 55.6 |
| DGFusion Table III | DeLiVER | **DGFusion** | **CLE / CLDE** | **Swin-T** | **mIoU-test** | **51.6** | **56.7** |

## 4. Reducing Unimodal Bias — Tables 1 / 2 / 3

Source: [[relatedworks/03_unimodal_bias_entropy_relatedwork]], `2505.06635.pdf`. Table 1 header extraction shows three RGB-Depth columns and the typo “RGB-Dpeth”; values preserve the visible order.

### 4.1 Table 1 — DeLiVER dual RGB-Depth modalities

| Source table | Dataset | Method | Modality set | Col. 1 | Col. 2 | Col. 3 | Mean |
|---|---|---|---|---:|---:|---:|---:|
| Reducing Unimodal Bias Table 1 | DeLiVER | CMNeXt | RGB-Depth | 1.60 | 1.44 | 63.58 | 22.81 |
| Reducing Unimodal Bias Table 1 | DeLiVER | CMNeXt‡ | RGB-Depth | 32.97 | 48.53 | 61.93 | 47.81 |
| Reducing Unimodal Bias Table 1 | DeLiVER | CMNeXt† | RGB-Depth | 53.39 | 53.73 | 62.24 | 56.45 |
| Reducing Unimodal Bias Table 1 | DeLiVER | MultiMAE | RGB-Depth | 24.60 | 38.55 | 58.94 | 40.70 |
| Reducing Unimodal Bias Table 1 | DeLiVER | MultiMAE‡ | RGB-Depth | 19.24 | 42.54 | 56.62 | 39.47 |
| Reducing Unimodal Bias Table 1 | DeLiVER | MultiMAE† | RGB-Depth | 52.46 | 45.62 | 58.68 | 52.25 |
| Reducing Unimodal Bias Table 1 | DeLiVER | FPT | RGB-Depth | 50.73 | 39.60 | 57.38 | 49.24 |
| Reducing Unimodal Bias Table 1 | DeLiVER | MAGIC | RGB-Depth | 37.26 | 59.02 | 66.89 | 54.39 |
| Reducing Unimodal Bias Table 1 | DeLiVER | **Ours** | **RGB-Depth** | **55.04** | **59.60** | **66.46** | **60.37** |

### 4.2 Table 2 — MUSES three-modality validation

| Source table | Dataset | Method | Modality set | F | E | L | FE | FL | EL | FEL | Mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reducing Unimodal Bias Table 2 | MUSES | CMX | F/E/L | 2.52 | 2.35 | 3.01 | 41.15 | 41.25 | 2.56 | 42.27 | 19.30 |
| Reducing Unimodal Bias Table 2 | MUSES | CMNeXt | F/E/L | 3.50 | 2.77 | 2.64 | 6.63 | 10.28 | 3.14 | 46.66 | 10.80 |
| Reducing Unimodal Bias Table 2 | MUSES | MAGIC | F/E/L | 43.22 | 2.68 | 22.95 | 43.51 | 49.05 | 22.98 | 49.02 | 33.34 |
| Reducing Unimodal Bias Table 2 | MUSES | Any2Seg | F/E/L | 44.40 | 3.17 | 22.33 | 44.51 | 49.96 | 22.63 | 50.00 | 33.86 |
| Reducing Unimodal Bias Table 2 | MUSES | **Ours** | **F/E/L** | **60.33** | **33.15** | **42.59** | **47.19** | **53.47** | **39.59** | **47.89** | **47.80** |

### 4.3 Table 3 — DeLiVER four-modality validation

| Source table | Dataset | Method | R | D | E | L | RD | RE | RL | DE | DL | EL | RDE | RDL | REL | DEL | RDEL | Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reducing Unimodal Bias Table 3 | DeLiVER | CMNeXt | 0.86 | 0.49 | 0.66 | 0.37 | 47.06 | 9.97 | 13.75 | 2.63 | 1.73 | 2.85 | 59.03 | 59.18 | 14.73 | 59.18 | 39.07 | 20.77 |
| Reducing Unimodal Bias Table 3 | DeLiVER | MAGIC | 32.60 | 55.06 | 0.52 | 0.39 | 63.32 | 33.02 | 33.12 | 55.16 | 55.17 | 0.26 | 63.37 | 63.36 | 33.32 | 55.26 | 63.40 | 40.49 |
| Reducing Unimodal Bias Table 3 | DeLiVER | Any2Seg | 39.02 | 60.11 | 2.07 | 0.31 | 68.21 | 39.11 | 39.04 | 60.92 | 60.15 | 1.99 | 68.24 | 68.22 | 39.06 | 60.95 | 68.25 | 45.04 |
| Reducing Unimodal Bias Table 3 | DeLiVER | **Ours** | **55.56** | **57.74** | **32.40** | **34.26** | **59.69** | **47.70** | **47.78** | **48.99** | **50.26** | **34.37** | **53.42** | **54.12** | **45.24** | **47.33** | **51.65** | **48.29** |

## 5. StitchFusion — Tables 1 / 2 / 7

Source: [[relatedworks/04_stitchfusion_relatedwork]], `2408.01343.pdf`.

### 5.1 Table 1 — Broad multimodal semantic segmentation comparison

| Source table | Method | Backbone | Publication | Additional strategies | NYUDv2 | DeLiVER | MCubeS | SUN |
|---|---|---|---|---|---:|---:|---:|---:|
| StitchFusion Table 1 | TokenFusion | MiT-B5 (MiT-B2) | CVPR 2022 | ✗ | 55.1 | 63.5 | — | 53.0 |
| StitchFusion Table 1 | CMNeXt | MiT-B4 (MiT-B2) | CVPR 2023 | ✗ | 56.9 | 66.3 | 51.5 | — |
| StitchFusion Table 1 | CMX | MiT-B5 | TITS 2023 | ✗ | 56.9 | 62.7 | — | 52.4 |
| StitchFusion Table 1 | GeminiFusion | MiT-B5 (MiT-B2) | ICML 2024 | ✗ | 57.7 | 66.9 | — | 53.3 |
| StitchFusion Table 1 | GeminiFusion | Swin-Large-22k | ICML 2024 | ✗ | 60.2 | — | — | 54.6 |
| StitchFusion Table 1 | MCubeSNet | MiT-B4 | CVPR 2022 | ✗ | — | — | 42.9 | — |
| StitchFusion Table 1 | ShareCMP | MiT-B2 | arXiv 2022 | ✗ | — | — | 50.3 | — |
| StitchFusion Table 1 | MMSFormer | MiT-B4 | IOJSP 2023 | ✗ | — | — | 53.1 | — |
| StitchFusion Table 1 | **StitchFusion** | **MiT-B2 (-B4,-B5)** | **ACMMM 2025** | **✗** | **57.8** | **68.2** | **53.9** | **53.4** |
| StitchFusion Table 1 | **StitchFusion** | **Swin-Tiny-1k (-22k)** | **ACMMM 2025** | **✗** | **53.8** | **70.3** | **52.3** | **50.3** |
| StitchFusion Table 1 | **StitchFusion** | **Swin-Large-22k** | **ACMMM 2025** | **✗** | **59.6** | **—** | **55.9** | **54.8** |

### 5.2 Table 2 — FMB / PST900 / MFNet

Requested for extraction completeness. This table does **not** contain DeLiVER, MUSES, or MCubeS, so it is not expanded into the project benchmark matrix. Use only if the project later compares FMB/PST900/MFNet.

### 5.3 Table 7 — Parameter efficiency on DeLiVER and MCubeS

| Source table | Dataset | Method | Backbone | Metric | RGB-D | RGB-DE | RGB-DEL |
|---|---|---|---|---|---:|---:|---:|
| StitchFusion Table 7 | DeLiVER | CMNeXt | MiT-B2 | Params (M) | 58.69 | 58.72 | 58.73 |
| StitchFusion Table 7 | DeLiVER | StitchFusion | MiT-B2 | Params (M) | 25.93 | 26.22 | 26.50 |
| StitchFusion Table 7 | DeLiVER | CMNeXt | MiT-B2 | mIoU (%) | 63.58 | 64.44 | 66.30 |
| StitchFusion Table 7 | DeLiVER | **StitchFusion** | **MiT-B2** | **mIoU (%)** | **65.75** | **66.03** | **68.18** |

| Source table | Dataset | Method | Backbone | Metric | RGB-A | RGB-AD | RGB-ADN |
|---|---|---|---|---|---:|---:|---:|
| StitchFusion Table 7 | MCubeS | MMSFormer | MiT-B4 | Params (M) | 64.88 | 65.27 | 65.65 |
| StitchFusion Table 7 | MCubeS | StitchFusion | MiT-B4 | Params (M) | 65.28 | 66.45 | 68.02 |
| StitchFusion Table 7 | MCubeS | MMSFormer | MiT-B4 | mIoU (%) | 51.30 | 52.03 | 53.11 |
| StitchFusion Table 7 | MCubeS | **StitchFusion** | **MiT-B4** | **mIoU (%)** | **52.68** | **53.26** | **53.92** |

## 6. AnySeg — Tables 1 / 2

Source: [[relatedworks/05_anyseg_relatedwork]], `2411.17141.pdf`.

### 6.1 Table 1 — MUSES anymodal semantic segmentation

Backbone: SegFormer-B0. Columns: F=frame camera, E=event cameras, L=LiDAR.

| Source table | Dataset | Method | Pub. | Training | F | E | L | FE | FL | EL | FEL | Mean |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AnySeg Table 1 | MUSES | CMX | T-ITS 2023 | — | 2.52 | 2.35 | 3.01 | 41.15 | 41.25 | 2.56 | 42.27 | 19.30 |
| AnySeg Table 1 | MUSES | CMNeXt | CVPR 2023 | FEL | 3.50 | 2.77 | 2.64 | 6.63 | 10.28 | 3.14 | 46.66 | 10.80 |
| AnySeg Table 1 | MUSES | MAGIC | ECCV 2024 | — | 43.22 | 2.68 | 22.95 | 43.51 | 49.05 | 22.98 | 49.02 | 33.34 |
| AnySeg Table 1 | MUSES | Any2Seg | ECCV 2024 | — | 44.40 | 3.17 | 22.33 | 44.51 | 49.96 | 22.63 | 50.00 | 33.86 |
| AnySeg Table 1 | MUSES | **Ours / AnySeg** | **—** | **—** | **46.01** | **19.57** | **32.13** | **46.29** | **51.25** | **35.21** | **51.14** | **40.23** |

### 6.2 Table 2 — DeLiVER anymodal semantic segmentation

| Source table | Dataset | Method | R | D | E | L | RD | RE | RL | DE | DL | EL | RDE | RDL | REL | DEL | RDEL | Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AnySeg Table 2 | DeLiVER | CMNeXt | 0.86 | 0.49 | 0.66 | 0.37 | 47.06 | 9.97 | 13.75 | 2.63 | 1.73 | 2.85 | 59.03 | 59.18 | 14.73 | 59.18 | 39.07 | 20.77 |
| AnySeg Table 2 | DeLiVER | MAGIC | 32.60 | 55.06 | 0.52 | 0.39 | 63.32 | 33.02 | 33.12 | 55.16 | 55.17 | 0.26 | 63.37 | 63.36 | 33.32 | 55.26 | 63.40 | 40.49 |
| AnySeg Table 2 | DeLiVER | **Ours / AnySeg** | **47.11** | **52.17** | **17.33** | **19.01** | **60.37** | **47.49** | **48.13** | **52.82** | **52.29** | **21.47** | **60.16** | **60.60** | **47.98** | **52.44** | **60.26** | **46.64** |

## 7. MAGIC++ — Tables I / II

Source: [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]], `2412.16876.pdf`.

### 7.1 Table I — MUSES MaSS validation

| Source table | Dataset | Method | Pub. | Training | F | E | L | FE | FL | EL | FEL | Mean |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAGIC++ Table I | MUSES | CMX | TITS 2023 | — | 2.52 | 2.35 | 3.01 | 41.15 | 41.25 | 2.56 | 42.27 | 19.30 |
| MAGIC++ Table I | MUSES | CMNeXt | CVPR 2023 | FEL | 3.50 | 2.77 | 2.64 | 6.63 | 10.28 | 3.14 | 46.66 | 10.80 |
| MAGIC++ Table I | MUSES | Any2Seg | ECCV 2024 | — | 44.40 | 3.17 | 22.33 | 44.51 | 49.96 | 22.63 | 50.00 | 33.86 |
| MAGIC++ Table I | MUSES | MAGIC | ECCV 2024 | — | 43.22 | 2.68 | 22.95 | 43.51 | 49.05 | 22.98 | 49.02 | 33.34 |
| MAGIC++ Table I | MUSES | **MAGIC++** | **—** | **—** | **45.56** | **17.93** | **29.92** | **40.58** | **46.07** | **28.10** | **40.58** | **35.53** |

### 7.2 Table II — DeLiVER anymodal semantic segmentation

| Source table | Dataset | Method | R | D | E | L | RD | RE | RL | DE | DL | EL | RDE | RDL | REL | DEL | RDEL | Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAGIC++ Table II | DeLiVER | CMNeXt | 0.86 | 0.49 | 0.66 | 0.37 | 47.06 | 9.97 | 13.75 | 2.63 | 1.73 | 2.85 | 59.03 | 59.18 | 14.73 | 59.18 | 39.07 | 20.77 |
| MAGIC++ Table II | DeLiVER | MAGIC | 32.60 | 55.06 | 0.52 | 0.39 | 63.32 | 33.02 | 33.12 | 55.16 | 55.17 | 0.26 | 63.37 | 63.36 | 33.32 | 55.26 | 63.40 | 40.49 |
| MAGIC++ Table II | DeLiVER | **MAGIC++** | **48.67** | **52.83** | **19.03** | **18.67** | **61.82** | **49.38** | **49.76** | **54.39** | **53.18** | **18.67** | **61.76** | **61.87** | **50.19** | **54.25** | **61.67** | **47.74** |

## 8. CAFuser — Tables I / II / III

Source: [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]], `2410.10791.pdf`.

### 8.1 Table I — MUSES panoptic segmentation, PQ

| Source table | Dataset | Method | Modalities | Backbone | Clear | Fog | Rain | Snow | Day | Night | All / PQ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CAFuser Table I | MUSES | Mask2Former | C | Swin-T | 48.8 | 46.5 | 45.4 | 45.1 | 49.4 | 39.4 | 46.9 |
| CAFuser Table I | MUSES | MaskDINO | C | Swin-T | 54.1 | 46.2 | 46.23 | 48.54 | 51.9 | 42.7 | 49.4 |
| CAFuser Table I | MUSES | OneFormer | C | Swin-T | 58.3 | 53.7 | 53.4 | 53.8 | 57.6 | 47.8 | 55.2 |
| CAFuser Table I | MUSES | HRFuser | CLRE | HRFuser-T | 47.0 | 43.6 | 42.7 | 40.6 | 44.6 | 40.0 | 43.9 |
| CAFuser Table I | MUSES | MUSES baseline | CLRE | 4xSwin-T | 55.3 | 50.3 | 53.8 | 50.5 | 54.1 | 49.7 | 53.6 |
| CAFuser Table I | MUSES | CAFuser-CAA | CLRE | Swin-T | 61.2 | 56.4 | 59.4 | 57.9 | 59.9 | 56.2 | 59.4 |
| CAFuser Table I | MUSES | **CAFuser-CA2** | **CLRE** | **Swin-T** | **61.4** | **57.5** | **59.6** | **57.2** | **59.5** | **57.3** | **59.7** |

### 8.2 Table II — MUSES semantic segmentation

The local PDF text collapses some method/modality/backbone columns. Numeric rows are cross-consistent with DGFusion Table II except for CMNeXt 72.4 vs DGFusion 72.1; use the source-specific value when citing.

| Source table | Dataset | Method | Modalities | Backbone | Metric | mIoU |
|---|---|---|---|---|---|---:|
| CAFuser Table II | MUSES | Mask2Former | C | Swin-T | mIoU | 70.7 |
| CAFuser Table II | MUSES | SegFormer | C | MiT-B2 | mIoU | 72.5 |
| CAFuser Table II | MUSES | OneFormer | C | Swin-T | mIoU | 72.8 |
| CAFuser Table II | MUSES | CMNeXt | CLRE | MiT-B2 | mIoU | 72.4 |
| CAFuser Table II | MUSES | GeminiFusion | CLRE | MiT-B2 | mIoU | 75.3 |
| CAFuser Table II | MUSES | CAFuser-CAA | CLRE | Swin-T | mIoU | 78.5 |
| CAFuser Table II | MUSES | **CAFuser-CA2** | **CLRE** | **Swin-T** | **mIoU** | **78.2** |

### 8.3 Table III — DeLiVER semantic segmentation

C=RGB camera, D=Depth, E=Events, L=LiDAR. The source extraction interleaves Table VI on the same page; method alignment below follows the visible sequence and is cross-checked against DGFusion Table III for mIoU-test values.

| Source table | Dataset | Method | Modalities | Backbone | Metric | mIoU-val | mIoU-test | Status |
|---|---|---|---|---|---|---:|---:|---|
| CAFuser Table III | DeLiVER | CMNeXt | CLDE | MiT-B2 | mIoU | 66.3 | 53.0 | verified |
| CAFuser Table III | DeLiVER | StitchFusion | CLDE | MiT-B2 | mIoU | 68.2 | 53.4 | verified |
| CAFuser Table III | DeLiVER | GeminiFusion | CLDE | MiT-B2 | mIoU | 66.9 | 54.5 | verified |
| CAFuser Table III | DeLiVER | CAFuser-CAA | CLDE | Swin-T | mIoU | 68.6 | 55.2 | verified |
| CAFuser Table III | DeLiVER | **CAFuser-CA2** | **CLDE** | **Swin-T** | **mIoU** | **67.8** | **55.6** | verified |

## 9. Reusable synthesis notes

- [[DGFusion]] is the cleanest MUSES all-sensor panoptic benchmark in this set: **61.03 PQ** on CLRE, +1.33 over CAFuser in DGFusion Table I.
- [[MemorySAM]] reports **65.38 mIoU** on DeLiVER R-D-E-L and **52.88 mIoU** on MCubeS R-A-D-N in its own SAM-adaptation setting.
- [[StitchFusion]] reports strong broader semantic-segmentation values: **70.3 mIoU** on DeLiVER with Swin-Tiny-1k and **55.9 mIoU** on MCubeS with Swin-Large-22k in Table 1; these should not be directly mixed with MemorySAM rows without noting protocol/backbone differences.
- [[AnySeg]], [[MAGIC++]], and Reducing Unimodal Bias share similar anymodal / MaSS-style combination tables, but these are distinct from standard full-modality semantic segmentation and from panoptic PQ.

## 10. Remaining open questions / caveats

- [ ] **Visual/PDF table check recommended** before camera-ready use of CAFuser Table III because extracted text interleaves Table VI, even though values align with DGFusion Table III.
- [ ] **StitchFusion Table 2** is verified as non-target for this DeLiVER/MUSES/MCubeS benchmark note; expand only if the project later compares FMB/PST900/MFNet.
- [ ] **Dataset-card details** for official MUSES and MCubeS still need separate extraction in [[relatedworks/06_deliver_muses_mcubes_dataset_note]].
- [ ] **Protocol comparability** remains a caveat: standard full-modality semantic segmentation, panoptic segmentation, MaSS, and anymodal evaluation should stay separated in final paper tables.

---

## 2026-07-02 deep-research update (Track 3 — protocol resolution + SOTA table 확정판)

Source: parallel deep-research Track 3 (see [[sources/07_parallel_research_prompts_2026-07-02]]) + adversarial verification (2 independent skeptic passes, all 5 critical claims **confirmed**). Detailed protocol analysis in [[relatedworks/93_benchmark_protocol_split_resolution]]. Everything below re-extracted from arXiv HTML / official GitHub on 2026-07-02.

### U1. DELIVER "two-cluster" problem — RESOLVED (actually three clusters) [skeptic-confirmed ×2]

| Reported CMNeXt number | True config | Evidence | Tag |
|---|---|---|---|
| **66.30** | MiT-**B2**, **val**, R-D-E-L, 1024×1024 crop | CMNeXt Tab.1(a); DELIVER repo README; CAFuser Tab.III "mIoU-val 66.3" | [VERIFIED-PDF]+[REPO] [val] |
| **53.0** | MiT-**B2**, **test**, CLDE | CAFuser Tab.III "mIoU-test 53.0"; DGFusion Tab.III | [VERIFIED-PDF]×2 [test] |
| **59.18** | MiT-**B0**, **val**, R-D-E-L | MemorySAM Tab.1, MAGIC, AnySeg Tab.2, EGFormer Tab.2, MLE-SAM Tab.I | [VERIFIED-PDF]×4 [val] |

- Cause = **split (val vs test) × backbone (B0 vs B2)**. **Resolution is NOT a factor** — proven not by matching crop sizes (CAFuser/DGFusion never state DELIVER input resolution) but by a same-model control: CAFuser Tab.III lists the SAME CMNeXt at 66.3 val / 53.0 test, and the same CAFuser-CA² at 67.8 val / 55.6 test, so the ~12–13 pt gap is the split.
- Code evidence: DELIVER repo `tools/val_mm.py` line 141 hardcodes the `'val'` split (test line commented out); README says "Please check tools/val_mm.py to modify the dataset for validation and test sets" — no split flag exists, README numbers = val protocol. [REPO]
- Splits: 3,983 train / 2,005 val / 1,897 test, 25 classes, 1042×1042 images. GeminiFusion's paper even describes DELIVER as "3983 training and 2005 testing" — i.e., silently uses val as test. [VERIFIED-PDF]

### U2. MemorySAM 65.38 = DELIVER **val** [code-inferred, skeptic-confirmed ×2]

- The paper (2503.06700) **never states the split**; no author confirmation exists (repo has 0 issues). Verdict rests on code: repo `val_mm_sam.py` line 146 instantiates the dataset with hardcoded `split='val'` (the test-set line 148 is commented out), and `deliver.py` `__main__` also uses `split='val'`. Precision note (skeptic correction to the original finding): the operative hardcoding is in **`val_mm_sam.py`**, not the `deliver.py` dataset class (which accepts train/val/test, default 'train').
- Its Table-1 baselines (CMNeXt MiT-B0 = 59.18) all belong to the val/B0 cluster → 65.38 sits in the **val** protocol. **Never compare 65.38 to test-cluster numbers (CAFuser 55.6, DGFusion 56.7)** — this correction applies to §1/§2 of this note: the MemorySAM rows above should be read as [val, code-inferred].
- Fairness caveat: MemorySAM backbone = SAM2 Hiera-B+ vs MiT-B0 baselines (much larger encoder). Our tables must show B2-val rows alongside.

### U3. DELIVER SOTA — consolidated, split-tagged [skeptic-confirmed]

**(a) VAL split, B2/Swin cluster (the "high" cluster):**

| Method | arXiv/venue | Backbone | mIoU | Split | Tag |
|---|---|---|---:|---|---|
| StitchFusion | 2408.01343, ACM MM'25 | Swin-Tiny-1k | **70.34** | [val]* split not stated in caption; inferred via CAFuser Tab.III listing StitchFusion-B2 68.2 as mIoU-val | [VERIFIED-PDF]* |
| Mul-VMamba | KBS 334:115119 | VMamba 55.33M | 68.98 | [unknown, likely val] | [ABSTRACT-ONLY] |
| CAFuser-CAA | 2410.10791, RA-L'25 | Swin-T | 68.6 | [val] (its test = 55.2, below CA²'s 55.6) | [VERIFIED-PDF] |
| StitchFusion | 2408.01343 | MiT-B2 | 68.18 | [val] | [VERIFIED-PDF] |
| OmniSegmentor | 2509.15096, NeurIPS'25 | DFormer-L | 68.0 | [val] Tab.1(f) | [VERIFIED-PDF] |
| EQUISeg | 2509.24505 | n/s | 67.90 | [unknown; values match val cluster] | [ABSTRACT-ONLY] |
| CAFuser-CA² | 2410.10791 | Swin-T | 67.8 | [val] | [VERIFIED-PDF] |
| MAGIC | 2407.11344, ECCV'24 | SegFormer-B2 | 67.66 | [val, cluster-inferred] | [VERIFIED-PDF] |
| HyperDUM | 2503.20011, CVPR'25 | CMNeXt-B2+UQ | 67.59 (10-case mean) | [val] | [VERIFIED-PDF] |
| GeminiFusion | 2406.01210, ICML'24 | MiT-B2 | 66.9 | [val, uses val-as-test] | [VERIFIED-PDF] |
| CMNeXt | 2303.01480, CVPR'23 | MiT-B2 | 66.30 | [val] | [VERIFIED-PDF] |

**(b) VAL split, B0/SAM cluster (modality-agnostic line, MemorySAM's comparison universe):**

| Method | arXiv | Backbone | RDEL mIoU | Anymodal mean | Split | Tag |
|---|---|---|---:|---:|---|---|
| MemorySAM | 2503.06700 | SAM2 Hiera-B+ (1 LoRA) | **65.38** | — | [val, code-inferred] | [VERIFIED-PDF] |
| MLE-SAM | 2412.04220 | Hiera-B+ MoE-LoRA | 64.08 | — | [val, same protocol] | [VERIFIED-PDF] |
| MAGIC | 2407.11344 | SegFormer-B0 | 63.40 | 40.49 | [val] | [VERIFIED-PDF] |
| AnySeg | 2411.17141 | SegFormer-B0 | 60.26 | **46.64** | [val] | [VERIFIED-PDF] |
| RobustSeg/RMMSS | 2505.12861 | MiT-B0 | 60.16 | 49.89 (robustness mean) | [val] | [VERIFIED-PDF] |
| EGFormer | 2505.14014 | SegFormer-B0 | 59.53 | — | [val] | [VERIFIED-PDF] |
| CMNeXt (repro) | — | SegFormer-B0 | 59.18 | 20.77 | [val] | [VERIFIED-PDF] |
| MAGIC++ | 2412.16876 | SegFormer-B0 | 47.74 (anymodal-trained) | 48.67 MaSS | [val] | [VERIFIED-PDF] |
| FunEntropy-Reg | 2505.06635 | (B0 line) | — | 48.29 | [val] | [VERIFIED-PDF] |

**(c) TEST split (CAFuser/DGFusion protocol) — extends §3.3 above with the new SAM entrant:**

| Method | Backbone | CLE test | CLDE test | Tag |
|---|---|---:|---:|---|
| DGFusion (4-mod) | Swin-T | 51.6 | 56.7 | [VERIFIED-PDF] [test] |
| **MM-SAM-adapter (2509.10408)** | SAM ViT-L + ConvNeXt-S side-adapter, 1024×1024 | — | **RGB-D 57.35 / RGB-L 57.14 / RGB-E 55.70** (RGB-L Hard 45.46) | [VERIFIED-PDF] [test] |

⚠ **SCOOP ALERT [skeptic-confirmed, slightly understated in original finding]**: MM-SAM-adapter with only **2 modalities** beats 4-modality DGFusion on DELIVER test (57.35 RGB-D is the actual ceiling, not 57.14) **and** on MUSES test (81.07 RGB-L > 79.5). Concurrent Sept-2025 papers; neither cites the other. Any "VFM-based multimodal SOTA" headline claim by us is **false unless we beat 81.07 on MUSES test** or scope claims to arbitrary-modal / robustness / condition-adaptive settings.

### U4. MUSES update — corrections and additions to §3.1/§3.2

- Dataset card [VERIFIED-PDF, 2401.12761 ECCV'24]: 1,500 train / 250 val / 750 test; **test labels withheld — public eval server** with 3 tracks × {RGB-only, multimodal}: semantic, panoptic, **uncertainty-aware panoptic (AUPQ)**. Sensors: frame cam, HD event cam, MEMS lidar, FMCW radar; 19 Cityscapes classes. Conditions: 500 clear-day / 1,000 adverse-day / 1,000 night.
- GeminiFusion MUSES 75.3 (vault open question resolved): the number exists **only in CAFuser Tab.II** (= CAFuser's benchmark-server run); GeminiFusion's own paper contains no MUSES experiments. CMNeXt CLRE: CAFuser Tab.II says 72.4, DGFusion Tab.II says 72.1 — keep source-specific.
- New test-set rows (semantic, all [VERIFIED-PDF] [test]): DGFusion CLRE **79.5**; **MM-SAM-adapter RGB-L 81.07 / RGB-E 79.92** (Tab.6/8, "vs CAFuser 78.18 (all 4)"). Per-condition mIoU (MM-SAM-adapter Tab.6, RGB-L|RGB-E): Day 83.34|83.39, Night 74.97|72.38, Fog 74.12|68.97.
- Per-condition PQ (test): DGFusion's weakest conditions are **Night 58.97 and Fog 58.86** (vs Rain 61.26, Snow 59.77, Clear 62.16) — our condition-adaptive gains story targets exactly these columns. [skeptic-confirmed]
- B0/anymodal cluster (F-E-L, no radar): MAGIC 33.34 → MAGIC++ 35.53; AnySeg 40.23; MLE-SAM full F-E-L 74.8 (⚠ split unstated; MUSES test requires the server, so 74.8 is presumably val/local).

### U5. MCubeS update — additions to prior sections

Single-cluster protocol (everyone reports **test**, 102 images; 302/96/102 splits, 20 material classes; image size 1224×1024 per Kyoto/U3M — ⚠ MemorySAM HTML fetch claimed 1920×1080, re-check its §4.1). New verified rows: StitchFusion Swin-L-22k **55.9** > StitchFusion B4 53.92 > MMSFormer B4 53.11 > **MemorySAM 52.88** > U3M 51.69 > CMNeXt-B2 51.54 > MLE-SAM 51.02; B0 line: EGFormer 43.40 vs CMNeXt-B0 36.16. Mul-VMamba 54.65 [ABSTRACT-ONLY].

### U6. MULTIAQUA (new, public) [VERIFIED-PDF]

arXiv 2512.17450, dataset public (lmi.fe.uni-lj.si/en/multiaqua/), used by a MaCVi @ CVPR 2026 challenge. 3,293 frames; RGB + thermal LWIR + NIR + polarization + LiDAR + radar; 4 classes; **day train/val, night-only test**. Table III (val-day/test-night mIoU): CMNeXt-DH **93.58/74.25**, StitchFusion-D 89.81/74.23, CMNeXt-D 92.95/72.24. Their robustness tricks are training-time (RGB-zeroed double forward, modality-specific heads); ours is inference-time — clean day→night reliability-shift showcase.

### U7. Per-condition DELIVER + robustness protocol

- Per-condition DELIVER breakdowns exist **only on the val split** in every paper checked (CMNeXt Tab.2, HyperDUM Tab.4, and EQUISeg 2509.24505 — split unstated but val-cluster values). **DGFusion publishes NO per-condition DELIVER table** (aggregate test only). [skeptic-confirmed; caveat: universal negative verified over checked papers — test labels are public so an unchecked paper could in principle report per-condition test.]
- Headline val per-condition rows: CMNeXt-B2 Night 62.46 → HyperDUM 64.21 (+1.75, its biggest gain); Cloudy 68.70→69.76; mean 66.30→67.59. ⚠ The five sensor-failure-case cells (MB/OE/UE/LJ/EL) had inconsistent row alignment between the two automated extractions — **must re-read CMNeXt Tab.2 and HyperDUM Tab.4 visually before LaTeX**.
- Robustness protocol to adopt: "Benchmarking MMSS under Sensor Failures" (2503.18445, same group as MemorySAM): EMM (15 combos) / RMM (r ∈ {0.25,0.5,0.75}) / NM (noise ×3). Key rows: MAGIC++ 44.85 EMM-avg, MAGIC 44.97, StitchFusion 41.98, CMNeXt 37.90 (collapses to 2.31% under high noise).

### U8. Recommended reporting protocol for our paper

1. DELIVER: **dual-column val+test** (CAFuser Tab.III format), CLDE + CLE ablation, backbone size beside every baseline; 10-case val breakdown; EMM/RMM/NM robustness appendix.
2. MUSES: submit to the official test server — semantic + panoptic + **AUPQ (uncertainty track: natural fit for RBMA's B_i maps; no fusion-cluster paper has led with it)**.
3. MCubeS: standard test split with B0/B2/Hiera columns. 4. MULTIAQUA: day→night condition-shift showcase vs CMNeXt-DH 74.25.
5. **Never place 65.38 (val, Hiera) in a column with 56.7 (test, Swin-T).** Publication targets: val ceiling 70.34 (StitchFusion Swin-T, split-inferred), test CLDE 57.35 (MM-SAM-adapter RGB-D), MUSES test 81.07 mIoU / 61.03 PQ.

### U9. Remaining gaps (post-verification)

- StitchFusion 70.34 split not confirmed from its own caption (repo 404); val inferred via CAFuser cross-check. EQUISeg 67.90 table not rendered; Mul-VMamba paywalled.
- CMNeXt Tab.2 / HyperDUM Tab.4 failure-case row alignment → visual PDF check pending.
- MLE-SAM MUSES 74.8 split unstated; MCubeS resolution discrepancy (1224×1024 vs "1920×1080"); MUSES leaderboard login-gated (possible unseen server entries).
- MAGIC/GeminiFusion/MemorySAM captions never literally print "val" — assignments rest on number-cluster consistency + repo code.
