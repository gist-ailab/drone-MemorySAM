# RELATED-WORK fact sheet — ReliaDINO (RA-L)

Compiled 2026-07-15 from: repo `.claude_logs/research/novelty-and-related-work.md` (canonical positioning),
vault `relatedworks/` notes (00, 01, 02, 04, 05, 06, 07, 08, 20–23, 30, 31, 42, 44, 53, 54, 88),
vault `architecture/P34_full_architecture_P35_changes_20260713.md` §7–8.
Model naming: paper text uses **ReliaDINO** only (full model = with PRR router; "ReliaDINO (no router)" = ablation variant). No internal codes.

---

## (i) Final bib key list (latex/references.bib)

Benchmarks/datasets:
- `cmnext2023` — Zhang et al., CVPR'23, arXiv 2303.01480 (DELIVER benchmark + CMNeXt). Verified in vault.
- `muses2024` — Brödermann et al., ECCV'24 (MUSES). % TODO verify (id 2401.12761 from knowledge)
- `mcubes2022` — Liang et al., CVPR'22 (MCubeS). % TODO verify

Multimodal seg methods:
- `memorysam2025` — Liao et al., arXiv 2503.06700. Verified.
- `dgfusion2026` — Brödermann et al., RA-L 2026, arXiv 2509.09828. Verified (authors partial % TODO).
- `cafuser2025` — Brödermann et al., RA-L 2025, arXiv 2410.10791. Verified (authors partial % TODO).
- `stitchfusion2024` — Li et al., arXiv 2408.01343. Verified.
- `geminifusion2024` — ICML'24. % TODO verify (all fields from knowledge)
- `magic2024` (ECCV'24, % TODO verify) + `magicpp2024` (arXiv 2412.16876, verified id).
- `anyseg2024` — Zheng et al., arXiv 2411.17141. Verified id.
- `omnisegmentor2025` — Yin et al., NeurIPS'25, arXiv 2509.15096. Verified.
- `mmsamadapter2025` — Curti et al., arXiv 2509.10408. Verified id/authors; exact title % TODO.
- `mlesam` — Zhu et al., arXiv **2412.04220** (MLE-SAM, MoE-LoRA on SAM/SAM2). Verified.
- `cmx2023` — TITS 2023, arXiv 2203.04838. Verified.
- `tokenfusion2022` — CVPR'22, arXiv 2204.08721. Verified id; authors % TODO.

Reliability/condition-aware fusion:
- `hyperdum2025` — CVPR'25, arXiv 2503.20011. Verified id/venue/Table-4 numbers; **authors % TODO**.
- `utfnet2023` — GRSL'23. % TODO verify authors/article no.
- `tmc2021` — Han et al., ICLR'21. % TODO verify authors.
- `relifusion2025` — arXiv 2502.01856 (3D det, learned reliability × attn output). Authors/title % TODO.
- `read2024` — ICLR'24 (OpenReview TPZRq4FALB). Authors/title % TODO.
- `zheng2025entropy` — ICCV'25, arXiv 2505.06635 (functional-entropy loss regularization). Title/authors % TODO.

Near-occupants for negative-result framing (C4):
- `primed2026` — arXiv 2605.07154 (learned modality-prior additive pre-softmax bias, RAVS). Authors/title % TODO.
- `sae2026` — arXiv 2603.16558 (training-free entropy-derived additive attn bias, LVLM decoder). Authors/title % TODO.
- `sam2long2024` — arXiv 2410.16268 (training-free multiplicative key scaling in SAM2 memory attn). Authors % TODO.

Foundation models / PEFT / heads / misc:
- `sam2023` (2304.02643), `sam2_2024` (2408.00714), `dinov3_2025` (% TODO verify id 2508.10104), `clip2021`,
- `lora2022` (2106.09685), `vitadapter2023` (2205.08534), `samed2023` (2304.13785),
- `segformer2021` (2105.15203), `maskformer2021` (2107.06278), `mask2former2022` (2112.01527),
- `ohem2016` (only if OHEM appears in training details; % TODO confirm usage).

---

## (ii) Taxonomy paragraph plan for Related Work (4 paragraphs)

### Para 1 — Multimodal semantic segmentation: benchmarks + arbitrary-modal methods
- Open with DELIVER `cmnext2023` (RGB-D-E-L, adverse conditions) and MUSES `muses2024`; MCubeS `mcubes2022` optional breadth.
- Fusion-architecture line: CMX `cmx2023` (feature rectification+fusion), TokenFusion `tokenfusion2022` (token-level ViT fusion), GeminiFusion `geminifusion2024` (pixel-wise intra/inter-modal attn), StitchFusion `stitchfusion2024` (adapter feature weaving between pretrained encoders).
- Arbitrary/any-modal line: MAGIC/MAGIC++ `magic2024`/`magicpp2024` (hierarchical modality selection, anti-RGB-centrism), AnySeg `anyseg2024` (uni-/cross-modal distillation for missing modalities), OmniSegmentor `omnisegmentor2025` (ImageNeXt 5-modality pretraining, DELIVER val 68.0 — pretraining axis, orthogonal/composable with ours).
- Positioning hook: these methods adapt *representations or architecture*; none decides at inference which sensor to trust — sets up para 2.
- ⚠ Protocol landmine (from novelty doc §2.5 / vault 09/93): DELIVER numbers split into **val cluster** (CMNeXt-B2 66.30 val; HyperDUM 67.59 val; OmniSegmentor 68.0; CAFuser-CAA 68.6 val) vs **test-CLDE cluster** (CMNeXt 53.0 < StitchFusion 53.4 < GeminiFusion 54.5 < CAFuser 55.6 < DGFusion 56.71). Never mix rows; every quoted number carries a split tag. MM SAM-adapter 57.35 = third protocol (2-modal, own easy/hard partition) — footnote only.

### Para 2 — Condition/reliability-aware fusion (our cell)
- CAFuser `cafuser2025`: RGB-derived CLIP-space condition token guides fusion (CA² = query concat / CAA = scalar multiply); requires verbo-visual contrastive supervision + condition annotation.
- DGFusion `dgfusion2026`: LiDAR as input **and** GT depth supervision (robust log-depth loss) → local depth tokens + global condition token conditioning cross-modal fusion; DELIVER test SOTA 56.71 (CLDE).
- HyperDUM `hyperdum2025`: hyperdimensional prototype distance (label-built prototypes + fine-tuned weighting layer) → learned feature reweighting pre-fusion; DELIVER val 66.30→67.59.
- UTFNet `utfnet2023` + TMC `tmc2021`: learned evidential/Dirichlet uncertainty, feature-weighting / opinion aggregation (classification for TMC).
- Loss-level line: zheng2025entropy (functional entropy regularization = training-time only); READ `read2024` (TTA loss weighting against modality reliability bias); ReliFusion `relifusion2025` (learned reliability scales cross-attn *output*, 3D det).
- Common denominator to state: every prior method injects the reliability/condition signal via extra supervision (depth GT, text, labels for prototypes) and/or at feature/output/loss level.

### Para 3 — Foundation-model-based multimodal seg + adapter/LoRA
- PEFT basis: LoRA `lora2022`; ViT-Adapter `vitadapter2023` (dense-prediction adaptation of plain ViTs); SAMed `samed2023` (LoRA-on-SAM precedent).
- SAM/SAM2 line: SAM `sam2023`, SAM2 `sam2_2024`; MemorySAM `memorysam2025` (modalities-as-frames into SAM2 memory attention — architectural ancestor of our earlier reliability-bias attempts); MLE-SAM `mlesam` (modality-specific MoE-LoRA experts + learned routing on frozen SAM); MM SAM-adapter `mmsamadapter2025` (deformable cross-attn injection of fused features into SAM ViT-L; 2-modality only).
- DINO line: DINOv3 `dinov3_2025` as frozen VFM; state plainly that **to our knowledge no prior multimodal segmentation method builds on DINOv3** (C1 claim; hedge "to our knowledge") — nearest is MMMS (DINOv2, NoC metric, not mIoU — cite only if reviewer-proofing needed, no bib key yet \todo).
- Gap sentence (from vault 23 synthesis): PEFT/adapter methods answer "how to represent each modality" but not "how much to trust it now".

### Para 4 — Where ReliaDINO sits
- One frozen DINOv3 ViT-L multiplexed by per-modality LoRA (C1) + 2-layer cross-modal attention; reliability enters **only at the output gate (calibrated competence-gated fusion) and the per-class reliability-anchored router** — *not* as an attention-logit bias (C2).
- Supervision minimality (C3): segmentation labels only — vs DGFusion (GT log-depth + condition meta-text/CLIP), CAFuser (condition supervision), HyperDUM (label-built prototypes).
- Negative finding (C4): five generations of attention-logit reliability biasing on SAM2/SAM3/DINOv3 memory/cross-attention were ineffective or harmful once per-modality LoRA absorbs modality adaptation (CKA 0.8–0.9 alignment removes the pathology routing used to rescue). **Inversion note**: the old novelty-doc §3 headline ("logit-additive bias unprecedented, value-preservation advantage") is now reported as the negative result — the unoccupied cell was unoccupied for a reason; cite near-occupants `primed2026` (learned, RAVS), `sae2026` (training-free entropy, LVLM), `sam2long2024` (multiplicative, temporal) to show the neighborhood and why our audit is informative.
- Keep the novelty-doc hedging discipline: never claim "first additive attention bias"; scope all "first/no prior" claims with "to our knowledge, as of mid-2026".

### Must-write contrasts (novelty doc §3, adapted to inverted story)
1. **ReliFusion/READ vs ours** — they place reliability at cross-attention *output scaling* / *loss weighting*; we place it at the *output gate and per-class router* after fusion — and we additionally show that placing it *inside* attention logits (which none of them do) does not help on DELIVER-scale multimodal seg. One sentence, pinned.
2. **DGFusion vs ours (sharpest)** — DGFusion: depth-supervised (LiDAR GT + robust log-depth loss) + condition meta-text, reliability as depth-token *conditioning of fusion*; ours: **no supervision beyond seg labels**, self-derived calibrated competence, applied at gate/router, modality-agnostic. Same venue (RA-L), direct test-SOTA baseline (56.71) — make the supervision-budget contrast explicit and include the fairness protocol (val-only checkpoint selection).
3. **Value-preservation argument — inverted**: the old defense said logit bias beats feature-zeroing because Values are preserved. Final story: with per-modality LoRA on a strong frozen VFM, logit-level suppression is *unnecessary* (no-op) — reliability information is only useful where class decisions are formed (gate/router). Frame as an empirically-grounded placement finding, not a retraction.

---

## (iii) Per-competitor one-line differentiation (from architecture note §7–8 + vault notes)

| Competitor | One-liner (ReliaDINO vs) |
|---|---|
| **DGFusion** | We exceed it on DELIVER test with segmentation labels only — no GT log-depth supervision, no condition meta-text, roughly half the training schedule — and with a frozen backbone. |
| **CAFuser** | No condition signal needed: no CLIP text embeddings, no condition annotation; competence is self-derived from the model's own calibrated predictions. |
| **MLE-SAM** | Demonstrates the learned modality-dispatch gate is not required for a strong frozen backbone; our routing is reliability-anchored and per-class, and generalizes across backbones. |
| **CMNeXt / AnySeg / MAGIC++** | A single frozen VFM with low-rank per-modality adaptation suffices — no bespoke multimodal architecture, no distillation stage, no modality-selection heuristics. |
| **OmniSegmentor** | Zero extra pretraining: no 5-modality ImageNeXt corpus; we match/exceed from off-the-shelf DINOv3 weights (axes are orthogonal/composable — say so). |
| **MemorySAM** | SAM2 memory-attention fusion is superseded by frozen-DINOv3 + LoRA multiplexing; our lineage audit shows the memory-attention site is the wrong place for reliability. |
| **MM SAM-adapter** | Not restricted to 2 modalities; no encoder fine-tuning; explicit interpretable competence signal instead of implicit learned selectivity. |
| **HyperDUM** | Training-free at the signal level: no label-built prototypes, no fine-tuned weighting layer; reliability applied per-class at the output gate, and we report on the *test* split. |
| **UTFNet / TMC** | No evidential heads or Dirichlet losses; calibration-forced competence from standard seg training. |
| **ReliFusion / READ** | Reliability at gate/router (dense seg), not attention-output scaling (det) or TTA loss weighting (classification). |
| **PRIMED / SAE / SAM2Long** | Cited as near-occupants of reliability-in-attention; our controlled audit explains why that placement family underperforms in multimodal dense seg with per-modality PEFT. |

Headline numbers to pair with claims (from task directive; final P36 numbers TODO until training finishes 2026-07-15):
- ReliaDINO full: val 67.74 @ep52 / test 57.14 @ep58 (best so far, \todo final).
- ReliaDINO (no router) ablation: val 68.19 / test 57.60.
- Baselines: DGFusion test 56.71 (test-SOTA, CLDE), CAFuser-CAA val 68.6 (val-SOTA). CMNeXt 66.51 = official target only, NOT SOTA (memory convention).

---

## Gaps

1. **Author lists / exact titles missing** (marked % TODO in bib): HyperDUM, ReliFusion, READ, zheng2025entropy, PRIMED, SAE, MUSES, MCubeS, GeminiFusion, TokenFusion, DINOv3, MAGIC, UTFNet, TMC, MM SAM-adapter exact title, DGFusion/CAFuser full author lists.
2. **DINOv3 arXiv id (2508.10104) is from model knowledge** — no vault note on DINOv3 bibliography exists; must verify before submission.
3. **"First DINOv3-based multimodal seg" (C1)** — vault has MMMS (2509.12963, DINOv2, NoC metric) as nearest; no dedicated DINOv3-sweep note. Needs a fresh arXiv sweep dated at submission (novelty doc §4 discipline: re-sweep last 6 months).
4. **MemorySAM 65.38 split** still only code-inferred as val (vault 93); if quoted, tag "[val, code-inferred]".
5. **PRIMED/SAE full-paper reads** are flagged as blocking follow-ups in vault (Track 8) — required before the C4 negative-result paragraph cites them as scoped near-occupants.
6. **P36 final numbers** (full model) TODO — training ends 2026-07-15; all headline sentences must carry \todo until then.
7. **ohem2016 inclusion conditional** — confirm whether OHEM is actually in the final training recipe.
8. **Per-class DELIVER test audit (C4) "first" claim** — no vault note verifies absence of prior per-class *test* breakdowns; needs a quick check of CMNeXt/CAFuser/DGFusion supplementary tables.
