---
title: Related Works Index — 26_MultimodalSeg
tags: [related-work, index, multimodal-segmentation, semantic-segmentation]
created: 2026-06-24
source: [[sources/00_imported_claude_related_work_2026-06-24]], [[sources/01_source_index_multimodal_segmentation]]
status: scaffold
---

# Related Works Index — 26_MultimodalSeg

This folder will contain one synthesis note per important paper or method. Each note should be written so it can be reused directly in a paper's Related Work section.

## Per-paper note template

```markdown
---
title: <Paper Title>
tags: [related-work, key-paper, method]
created: YYYY-MM-DD
source: <URL / DOI / arXiv>
status: draft
---

# <Paper Title>

## Citation

## Problem setting

## Novelty

## Main claims

## Method

## Figures / tables to remember

## Limitations

## Comparison to our project

## Related-work paragraph candidate
```

## Priority 1 — must synthesize first

| Paper / method | Why it matters | Target note |
|---|---|---|
| MemorySAM | Direct baseline: SAM2 modality-as-frame memory attention | `01_memorysam_relatedwork.md` |
| DGFusion | Depth/LiDAR-guided robust multimodal segmentation | `02_dgfusion_relatedwork.md` |
| Reducing Unimodal Bias | Optimization-level anti-unimodal-dominance regularizer | `03_unimodal_bias_entropy_relatedwork.md` |
| StitchFusion | MultiAdapter early encoder fusion with pretrained encoders | `04_stitchfusion_relatedwork.md` |
| AnySeg | Anymodal multimodal segmentation / distillation | `05_anyseg_relatedwork.md` |
| DELIVER / MUSES / MCubeS | Dataset and benchmark positioning | `06_datasets_deliver_muses_mcubes.md` |

## Priority 2 — novelty defense around reliability and uncertainty

- UTFNet
- HyperDUM
- Conflict-guided evidential multimodal fusion
- TMC / ETMC
- ReliFusion
- READ
- AG-Fusion
- MAGIC / MAGIC++
- Any2Seg
- RMMSS
- EQUISeg


## Priority 2 verified-draft notes — reliability / bias / RBMA novelty

- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]] — uncertainty, evidential fusion, reliability-aware feature/condition fusion; covers UTFNet, HyperDUM, TMC, CAFuser, DGFusion, conflict/evidential candidates. **2026-07-02 update (Track 2)**: CAFuser CA²=query-concat / CAA=scalar-multiply verified, DGFusion MUSES 61.03 PQ / 79.5 mIoU as best published, DELIVER two-cluster resolved as val/test protocol, full mechanism taxonomy table.
- [[relatedworks/41_unimodal_bias_and_modality_collapse]] — unimodal bias, RGB-centered collapse, modality selection, anymodal distillation; covers Reducing Unimodal Bias, MAGIC++, AnySeg, Any2Seg, RMMSS.
- [[relatedworks/42_attention_logit_bias_novelty_defense]] — novelty defense for RBMA as additive pre-softmax SAM2 memory-attention logit bias; distinguishes feature modulation, output scaling, learned gating, modality selection, loss weighting, and evidential uncertainty. **2026-07-02 updates ×2 (Tracks 2+8)**: logit-bias cell verdict downgraded to "unrefuted but uncertain" then OCCUPIED-adjacent (PRIMED 2605.07154, SAE 2603.16558); 9-entry near-miss ranking; fenced 4-axis novelty claim wording ("the methods we examined" / "to our knowledge" hedges mandatory).
- [[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]] — (2026-07-02, Track 2, NEW) HyperDUM (CVPR'25) deep-dive: hyperdimensional deterministic uncertainty for multimodal fusion; DELIVER val Table 4 verified (66.30→67.59 [val]); mandatory comparison row for RBMA.
- [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]] — (2026-07-02, Track 2, NEW) SAE (2603.16558): nearest mechanism match to RBMA — training-free entropy-derived additive attention-logit term (Eq.7), but in LVLM hallucination/LLM-decoder domain, not multimodal dense seg.
- [[relatedworks/46_attention_reweighting_detection_nearmisses]] — (2026-07-02, Track 2, NEW) attention-reweighting near-misses: ModalPatch, ReliFusion (post-softmax characterization flagged, not re-verified), SAM2Long (training-free multiplicative temporal).
- [[relatedworks/47_reliability_fusion_2025_2026_new_entrants]] — (2026-07-02, Track 2, NEW) 2025–26 entrants sweep: RSGMamba, UP-Fuse, AW-MoE, EQUISeg, MULTIAQUA, AECF, READ, SGMA — mechanism-class tags per entrant.
- [[relatedworks/43_a_signal_entropy_priorart]] — (2026-07-02, Track 4 kill-check, adversarially verified) A-signal prior-art verdict: output-level cell OCCUPIED (UNO ICRA'20 — must-cite), learned-evidence feature cell OCCUPIED (UTFNet/HyperDUM/2309.05919), loss/TTA cell OCCUPIED (Latte/READ), attention-logit-additive-bias cell for multimodal dense seg UNOCCUPIED (unfalsified, not proven — hedge required; nearest mechanism match SAE 2603.16558 in LVLM domain). Includes signal×injection taxonomy table and English related-work paragraph.

## Priority 3 — ViT, SAM, LoRA, adapter, token-efficiency axis

- SegFormer
- Mask2Former
- OneFormer
- SAM / SAM2 / SAM3
- LoRA / adapters / prompt tuning for segmentation
- Expedit / Expedit-SAM
- DToP
- ToMe / PiToMe / Token Transforming
- Vision Transformers Need Registers

## Priority 4 — object detection transfer

- Camera-LiDAR fusion 3D object detection
- BEVFusion / TransFusion-style LiDAR-camera detection
- RGB-T object detection
- Robust detection under night/fog/rain/adverse conditions

## Priority 4 verified-draft notes — multimodal object detection / 3D detection

- [[relatedworks/10_bevfusion_relatedwork]] — BEVFusion family; unified BEV camera-LiDAR fusion, multi-task BEV perception, robustness to sensor failure.
- [[relatedworks/11_transfusion_relatedwork]] — CVPR 2022 TransFusion; transformer object-query LiDAR-camera fusion robust to poor illumination and misalignment.
- [[relatedworks/12_deepinteraction_relatedwork]] — modality-preserving LiDAR-camera interaction; anti-collapse lesson for segmentation/RBMA.
- [[relatedworks/13_futr3d_relatedwork]] — query-based unified camera/LiDAR/radar feature sampling for flexible sensor configurations.
- [[relatedworks/14_multimodal_detection_survey_note]] — synthesis note covering BEVFormer, DETR3D, PETR/PETRv2, PointPainting, MVX-Net, CLOCs, CenterPoint, RGB-T, radar, event, and adverse-weather detection. **2026-07-02 update appended (Track 5)**: condition-adaptive detection deep-research delta.
- [[relatedworks/15_condition_adaptive_detection]] — (2026-07-02, Track 5, NEW, adversarially verified) condition-adaptive multimodal detection synthesis: per-paper mechanisms with verified equations, verbatim number table ([VERIFIED-PDF]/[ABSTRACT-ONLY] + [val]/[test] tags), signal×injection taxonomy for detection. Verdict: "training-free entropy → additive pre-softmax attention-logit bias" unoccupied in detection ("to our knowledge" hedge required — SMCA-DETR refutes "MEFormer only pre-softmax additive" phrasing; SeBFusion/BCAF 403-blocked, watch-listed). Recommended minimal det experiment: inference-only λ·B injection into MEFormer on nuScenes-C.

## Expanded source database links — 2026-06-24

- [[sources/02_openalex_top_venue_literature_database]] — 3,010-record OpenAlex discovery database, 408 top-conference / major-vision-journal flagged records.
- [[sources/03_seed_paper_verification_candidates]] — seed-paper candidate matches for immediate paper-level verification.
- [[sources/02_source_map_multimodal_semantic_segmentation]] — direct multimodal semantic segmentation source map.
- [[sources/02_source_map_multimodal_object_detection]] — direct multimodal object detection source map.
- [[sources/02_source_map_adapter_lora_foundation_seg_det]] — adapters, LoRA, prompt tuning, SAM/foundation-model adaptation source map.
- [[sources/02_source_map_segmentation_detection_heads]] — segmentation-head and detection-head source map.

### Related-work grouping required by the project

1. **Direct multimodal semantic segmentation** — RGB-D, RGB-T, event, LiDAR-camera, DeLiVER/MUSES/MCubeS, uncertainty/reliability fusion.
2. **Direct multimodal object detection** — camera-LiDAR, BEV, radar-camera-LiDAR, RGB-T detection, adverse-condition detection.
3. **Adapter / LoRA / foundation-model adaptation** — parameter-efficient adaptation for SAM/SAM2/SAM3, ViT, segmentation and detection.
4. **Segmentation and detection heads** — SegFormer, Mask2Former, OneFormer, DETR/Deformable-DETR/DINO/MaskDINO/YOLO/Mask R-CNN style heads.

## Priority A verified-draft notes

- [[relatedworks/01_memorysam_relatedwork]]
- [[relatedworks/02_dgfusion_relatedwork]]
- [[relatedworks/03_unimodal_bias_entropy_relatedwork]]
- [[relatedworks/04_stitchfusion_relatedwork]]
- [[relatedworks/05_anyseg_relatedwork]]
- [[relatedworks/06_deliver_muses_mcubes_dataset_note]]
- [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
- [[relatedworks/08_priority_a_comparison_matrix]]

Source PDFs and extracted text are archived under [[sources/pdfs/priority_a]].
- [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] — in-progress benchmark table extraction for DeLiVER/MUSES/MCubeS. **2026-07-02 update (§U1–U9)**: two-cluster resolved (split×backbone), split-tagged SOTA tables, MM-SAM-adapter scoop alert (DELIVER test 57.35 RGB-D / MUSES test 81.07 RGB-L with only 2 modalities), MULTIAQUA, per-condition + EMM/RMM/NM robustness protocol.
- [[relatedworks/93_benchmark_protocol_split_resolution]] — 2026-07-02, verified-draft (adversarially verified ×2): DELIVER protocol forensics (66.30 = B2-val / 53.0 = B2-test / 59.18 = B0-val; official-code line evidence), MemorySAM 65.38 = val [code-inferred, val_mm_sam.py L146], dual-split reporting rules, English related-work paragraph candidate. (2026-07-08: 구 `46_benchmark_protocol_split_resolution` — Track 2 [[relatedworks/46_attention_reweighting_detection_nearmisses]]와의 번호 충돌 해소를 위해 `93_`으로 rename.)
## Adapter / LoRA / foundation-model adaptation notes — added 2026-06-24

| Note | Coverage | Status |
|---|---|---|
| [[relatedworks/20_lora_adapter_relatedwork]] | LoRA, AdaptFormer, Visual Prompt Tuning, MoE-LoRA, PEFT motivation for foundation segmentation models | ✅ verified-draft |
| [[relatedworks/21_vit_adapter_relatedwork]] | ViT-Adapter, dense prediction adapters, detection/segmentation relevance | ✅ verified-draft |
| [[relatedworks/22_sam_adapter_relatedwork]] | SAM-Adapter, MedSAM, SAMed, MemorySAM, MoE-LoRA SAM, SAM-FuseNet, ClassWise-SAM-Adapter, SAM3 caveat | ✅ verified-draft |
| [[relatedworks/23_multimodal_sam_adapter_matrix]] | Practical comparison/ranking matrix for multimodal SAM/SAM2 adapter choices | ✅ verified-draft |

Key synthesis: PEFT is useful because it preserves foundation-model shape/boundary priors while adding small domain/modality-specific corrections. For [[26_MultimodalSeg]], the best candidates are MemorySAM-style SAM2 memory adaptation, MoE-LoRA modality experts, StitchFusion-style MultiAdapter feature weaving, and ViT-Adapter dense-prediction adapters. The identified gap is explicit reliability-aware fusion: existing PEFT/adaptation methods adapt representations but rarely inject calibrated modality reliability into fusion attention.

## Segmentation / detection heads verified-draft notes — added 2026-06-24

| Note | Coverage | Status |
|---|---|---|
| [[relatedworks/30_segformer_relatedwork]] | SegFormer plus SETR, UPerNet, DeepLabv3+ semantic heads; mIoU-oriented attachment to multimodal/SAM encoders | ✅ verified-draft |
| [[relatedworks/31_mask2former_relatedwork]] | Mask2Former universal mask-classification head for semantic/instance/panoptic segmentation | ✅ verified-draft |
| [[relatedworks/32_oneformer_relatedwork]] | OneFormer task-conditioned universal segmentation and prompt-like task tokens | ✅ verified-draft |
| [[relatedworks/33_detr_deformable_detr_dino_relatedwork]] | DETR, Deformable DETR, DINO, and 3D/BEV transformer detection heads including BEVFormer/DETR3D/PETR | ✅ verified-draft |
| [[relatedworks/34_maskdino_yolo_maskrcnn_heads]] | MaskDINO, Mask R-CNN, YOLO family, and practical 2D/3D detection/instance-segmentation heads | ✅ verified-draft |

Key synthesis: semantic heads such as SegFormer/UPerNet/DeepLabv3+ are the cleanest mIoU baselines for fused RGB-X/SAM-style features; Mask2Former/OneFormer/MaskDINO add mask-query universality for instance and panoptic outputs; DETR/Deformable DETR/DINO/YOLO/Mask R-CNN and BEV heads are detection branches evaluated by AP/NDS rather than semantic mIoU. For [[26_MultimodalSeg]], head choice should be decoupled from the reliability-aware multimodal encoder so RBMA/fusion gains can be attributed cleanly.


- [[91_jepa_predictive_representations_for_multimodal_seg]] — JEPA/I-JEPA/V-JEPA implications for multimodal segmentation representation learning. (2026-07-08: 구 `90_jepa_*` — [[relatedworks/90_clustered_relatedwork_synthesis]]와의 번호 충돌 해소를 위해 `91_`로 rename.)
- [[92_rf_detr_detection_segmentation_head]] — RF-DETR as real-time DETR/NAS detection and instance-segmentation head baseline.

## 2026-07-02 deep-research update — Track 1 VFM multimodal landscape (52–57)

Adversarially verified (dual-skeptic) notes; supersede any earlier "MemorySAM is only / no prior logit-bias" phrasing elsewhere in the vault:

| Note | Coverage | Status |
|---|---|---|
| [[relatedworks/52_vfm_multimodal_landscape_synthesis]] | Synthesis + mechanism-class taxonomy; corrected novelty cells (SAM4D, DAMM-Diffusion, SAM2Long, MMMS, SAM3-UNet counter-evidence); DELIVER two-cluster top-5 (StitchFusion 68.18 leads clean-val — 70.3 Swin-T UNSUBSTANTIATED; DGFusion 56.7 leads test-CLDE) | ✅ verified-draft |
| [[relatedworks/53_mm_sam_adapter_relatedwork]] | MM SAM-adapter (2509.10408): SAM1 + deformable cross-attn injectors, DELIVER-test 57.35 RGB-D, third protocol variant, RGB-easy/hard splits | ✅ verified-draft |
| [[relatedworks/54_omnisegmentor_relatedwork]] | OmniSegmentor (2509.15096, NeurIPS'25): ImageNeXt 5-modality pretraining, DELIVER 68.0 — orthogonal pretraining axis, composable with RBMA | ✅ verified-draft |
| [[relatedworks/55_sam2_memory_attention_occupants]] | RBMA base-claim check: SAM4D (ICCV'25) = second cross-modal memory-attention occupant (promptable); MemorySAM still alone in the semantic-seg cell; M⁴-SAM/OmniSAM memory = temporal/spatial only | ✅ verified-draft |
| [[relatedworks/56_sam_family_multimodal_periphery]] | SHIFNet, MLE-SAM, FusionSAM, MM-SAM, MMMS (DINOv2 on DELIVER but NoC-metric only), SAM-DAQ, X-SAM caveat, MemorySAM citing sweep (13 hits, none touch memory attention) | ✅ verified-draft |
| [[relatedworks/57_sam3_pe_multiscale_gap]] | SAM3/PE single-scale limitation; ViTDet simple-FPN recipe; SAM3-UNet + DART precedents (neck mechanism published — only multimodal-semantic-seg application unoccupied) | ✅ verified-draft |

Key corrected claims (do not cite the pre-verification versions): (1) SAM4D (2506.21547) also uses SAM2-style memory attention across modalities — "MemorySAM only" holds solely for multimodal *semantic* segmentation; (2) pre-softmax reliability injection has near-occupants — DAMM-Diffusion (learned, multiplicative, modality) and SAM2Long (training-free, multiplicative, temporal) — RBMA's cell is the additive × training-free × cross-modality × semantic-seg intersection; (3) StitchFusion Swin-T 70.3 is unsubstantiated (GitHub lists Swin as TODO), cite 68.18–68.20; (4) MMMS (2509.12963) is a DINOv2-based method reporting DELIVER numbers (interactive NoC, not mIoU); (5) multi-scale necks on frozen SAM3 are already published (SAM3-UNet).

## 2026-07-02 deep-research update — Track 6 (P29 SDC defense)

- [[relatedworks/50_moe_lora_condition_routing]] — **Track 6, P29 SDC novelty defense** (adversarially verified, 2 skeptic passes): MoE-LoRA condition-routing prior art. Verdict: P29 cell (unsupervised image-derived condition prototype → FiLM on Soft-MoE-LoRA gate, multimodal dense seg) UNOCCUPIED; nearest 3 = MoCLE (text-cluster gate input), AW-MoE (supervised weather routing), MoFME (FiLM-as-expert). Near-occupant caveats folded in: MLE-SAM (2412.04220, unsupervised modality-stat LoRA gate on SAM2/DELIVER) and DAMP (2512.20251, training-free degradation stats as restoration gate input) — universal "no training-free stat routing" claims are now forbidden; scoped claim wording inside the note. Also covers P30 router anchoring (Loss-Free Balancing = mechanism-nearest, load-anchored; LER-YOLO = learned reliability feature-multiply).

## 2026-07-02 deep-research update — Track 7 (P30 class-token decoder defense)

- [[relatedworks/51_class_token_fused_memory_decoder]] — **Track 7, P30 novelty defense** (adversarially verified): query/class-token decoders on fused multimodal features. Verdict: broad cell OCCUPIED (CAFuser/DGFusion/BiXFormer/DF2RQ use queries on fused features); exact cell "class tokens on SAM2 memory features + reliability-anchored routing" unoccupied externally — but gray zone flagged: MemorySAM's stock SAM2 decoder already runs internal output tokens on memory features → note mandates explicit differentiation (mask-classification framework, fixed per-class queries, RBMA-anchored routing) + ablation vs stock decoder. Corrections folded in: BiXFormer = TMM'26, GOOSE-M2F rare-class +5–8% is qualitative (isolated ablation +3.4%), EoMT-on-SAM3 = extrapolation, CAFuser/DGFusion confirmed no pre-softmax logit bias.

## 2026-07-02 deep-research update — Track 8 threat watch (scoop alerts; 2 claims REFUTED)

Triage table + verdicts: [[sources/08_threat_watch_2026H2]] (arXiv 2026-01→07 sweep, MemorySAM 13-citer audit, 2-skeptic adversarial verification). **Two pre-verification claims were REFUTED — cite these before any novelty sentence:**

| Note | Coverage | Threat | Status |
|---|---|---|---|
| [[relatedworks/60_primed_attention_logit_bias_threat]] | PRIMED (2605.07154): learned modality-prior additive PRE-SOFTMAX attention-logit bias in referring audio-visual **segmentation** — occupies the dense-prediction logit-bias cell; RBMA novelty narrows to training-free entropy × SAM2 memory attention × RGB-X seg | RBMA **HIGH** | ✅ verified-draft (equations verified; full tables pending) |
| [[relatedworks/58_sae_entropy_logit_bias_threat]] | SAE (2603.16558): **training-free entropy-derived** additive pre-softmax attention bias (LVLM hallucination, LLM-decoder site) — mechanism-level occupant; "first additive entropy-reliability logit bias" not claimable | RBMA **HIGH** | ✅ verified-draft |
| [[relatedworks/59_bixformer_mask_classification_threat]] | BiXFormer (2506.03675, IEEE TMM): MMSS as mask-level classification (UMM+CMA) — generic "query decoder for MMSS" occupied; P30 = class tokens on reliability-biased SAM2 memory-fused features only | P30 **HIGH** | ✅ verified-draft (abstract-level; tables pending) |
| [[relatedworks/48_m4sam_moe_lora_sam2_threat]] | M⁴-SAM (2605.11760): Modality-Aware MoE-LoRA + modality dispatcher in SAM2 encoder (RGB-D VSOD; memory = init only) — "MoE-LoRA-in-SAM2" occupied (after 2412.04220); P29 rests on unsupervised condition signal | P29 **MED-HIGH** | ✅ verified-draft (abstract-only) |

Other verdicts: SAM2-memory-attention-for-cross-modal-fusion base claim CONFIRMED ×2 (13/13 MemorySAM citers clean; SAM4D 2506.21547 cite-and-distinguish); SAM3-RBMA "first multimodal SAM3" REFUTED by SAMCM-SR (MDPI Appl. Sci. 16(5):2351 — prompt-level cross-modal SAM3) → re-scope to feature-level fusion inside SAM3. Blocking reads before submission: PRIMED, SAE, ICRCV underwater condition-aware paper (P29). [[relatedworks/42_attention_logit_bias_novelty_defense]] carries the corresponding claim-scope downgrades (see its two 2026-07-02 update sections).

## Gap-fill deep research additions — 2026-07-02

- [[61_rsgmamba_reliability_self_gated_mamba]] — gap-fill note: rsgmamba reliability self gated mamba.
- [[62_equiseg_balanced_modality_contributions]] — gap-fill note: equiseg balanced modality contributions.
- [[63_geomprompt_missing_degraded_depth]] — gap-fill note: geomprompt missing degraded depth.
- [[64_multiaqua_maritime_robust_training]] — gap-fill note: multiaqua maritime robust training.
- [[65_crossweaver_arbitrary_modality_segmentation]] — gap-fill note: crossweaver arbitrary modality segmentation.
- [[66_rtfdnet_fusion_decoupling_rgbt]] — gap-fill note: rtfdnet fusion decoupling rgbt.
- [[67_sgma_semantic_guided_modality_aware]] — gap-fill note: sgma semantic guided modality aware.
- [[68_tuni_rgbt_pretraining_rectification]] — gap-fill note: tuni rgbt pretraining rectification.
- [[69_taseg_text_aware_rgbt_sam_clip]] — gap-fill note: taseg text aware rgbt sam clip.
- [[70_m4sam_memory_moe_lora_impl_supplement]] — gap-fill note: m4sam memory moe lora impl supplement.
- [[71_bedsam2_boundary_depth_sam2]] — gap-fill note: bedsam2 boundary depth sam2.
- [[72_primed_implementation_supplement]] — gap-fill note: primed implementation supplement.
- [[73_sae_entropy_logit_bias_impl_supplement]] — gap-fill note: sae entropy logit bias impl supplement.
- [[74_memorysam_implementation_supplement]] — gap-fill note: memorysam implementation supplement.
- [[75_detector_gapfill_adaptive_fusion_2026]] — gap-fill note: detector gapfill adaptive fusion 2026.

## Gap-fill per-paper detection and abstract-only additions — 2026-07-02

- [[76_samfusion_sensor_adaptive_adverse_weather_detection]] — per-paper gap-fill note.
- [[77_wcbr_weather_conditioned_branch_routing]] — per-paper gap-fill note.
- [[78_decouple_recouple_robust_3d_detection]] — per-paper gap-fill note.
- [[79_ccf_domain_generalized_multimodal_3d_detection]] — per-paper gap-fill note.
- [[80_mambafusion_adaptive_state_space_detection]] — per-paper gap-fill note.
- [[81_difffusion_restoration_multimodal_detection]] — per-paper gap-fill note.
- [[82_radarxformer_cross_dimension_radar_camera]] — per-paper gap-fill note.
- [[83_cllap_lidar_augmented_pretraining_radar_camera]] — per-paper gap-fill note.
- [[84_dgfusion_dual_guided_hard_instance_detection]] — per-paper gap-fill note.
- [[85_latent_scenario_sampling_missing_modalities]] — per-paper gap-fill note.
- [[86_cbc_slp_structured_latent_projection]] — per-paper gap-fill note.
- [[87_unimrseg_modality_relax_segmentation]] — per-paper gap-fill note.
- [[88_mle_sam_moe_lora_prior]] — per-paper gap-fill note.

## Index completeness additions — 2026-07-08

이 인덱스에 누락되어 있던 기존 노트 (파일 목록 대조로 추가):

- [[relatedworks/49_corb_novelty_defense]] — CoRB (P32-B) novelty defense: 4-pillar claim, RSGMamba/MAGIC++ near-misses, posterior-Bhattacharyya discriminator. See also [[P32_CoRB_novelty_risk_register]] and [[00_P32_CoRB_index]].
- [[relatedworks/90_clustered_relatedwork_synthesis]] — 6-cluster related-work synthesis + related-work paragraph candidates; exported to `material/01_multimodal_seg_clustered_relatedwork_{en,ko}.md/.pdf`.

번호 rename 기록 (2026-07-08, 충돌 해소): `46_benchmark_protocol_split_resolution` → `93_`, `90_jepa_predictive_representations_for_multimodal_seg` → `91_`. 상세는 [[VAULT_CHANGELOG_2026-07-08]].
