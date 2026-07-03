---
title: Threat Watch 2026H2 — Fresh-Sweep Triage (RBMA / P29 SDC / P30)
tags: [threat-watch, scoop-check, rbma, sdc, p30, multimodal-segmentation, 2026H1, triage]
created: 2026-07-02
source: Track 8 of [[sources/07_parallel_research_prompts_2026-07-02]] — arXiv 2026-01→2026-07 sweep + MemorySAM citation audit + 2-skeptic adversarial verification
status: verified-draft
---

# Threat Watch 2026H2 — arXiv 2026-01 → 2026-07 sweep + adversarial verification

Executed 2026-07-02. Query battery: multimodal seg × (reliability | uncertainty | condition | adverse); SAM2 memory attention / modality-as-frame; attention logit bias / additive attention bias / ALiBi×multimodal; training-free uncertainty fusion / entropy-guided fusion; LoRA expert routing × (condition | weather | domain); DELIVER/MUSES/MCubeS new entries; MULTIAQUA citations; SAM3/PE adaptation; MemorySAM citing-paper audit (Semantic Scholar, 13 citations). **All five critical claims were then adversarially re-checked by two independent skeptic passes — two claims were REFUTED. The verdicts below are the post-verification versions; do not cite the pre-verification sweep conclusions.**

## 0. Bottom line (POST-VERIFICATION threat verdicts)

- **RBMA cell ("reliability/uncertainty/condition → additive PRE-SOFTMAX attention-logit bias in multimodal dense-prediction fusion"): ⚠️ REFUTED as stated — the cell is OCCUPIED.**
  - **PRIMED (arXiv:2605.07154, 2026-05, Referring Audio-Visual Segmentation)** adds a modality-prior bias directly to cross-attention logits before softmax: `MHCA(Q,K,V) = Softmax(QK^T/√d + b_M)V`, `b_M = γ_p·log(P/(1−P))` where P is a learned modality prior distilled from Qwen3-omni soft labels. Segmentation = dense prediction → the literal cell is taken. [VERIFIED-PDF quote via skeptic2]
  - **SAE (arXiv:2603.16558, 2026)** is *training-free* and adds an **entropy-derived reliability term** to pre-softmax cross-modal attention logits (`S̃ = S + λ·SAE·C`, λ=0.5) in LVLM visual attention (hallucination mitigation, LLM-decoder site). So "first additive entropy-reliability logit bias" is NOT claimable at mechanism level. [VERIFIED via skeptic1]
  - **Surviving RBMA novelty (must be restated this way):** the *conjunction* of (i) training-free per-modality decoder softmax-entropy signal, (ii) injection into **SAM2 memory attention** over modality memory tokens, (iii) RGB-X multi-sensor **semantic segmentation** under adverse conditions. PRIMED's signal is learned/distilled (not training-free) and its site is a bespoke MHCA in RAVS; SAE's task is hallucination mitigation in an LLM decoder, not dense prediction. Full notes: [[relatedworks/60_primed_attention_logit_bias_threat]], [[relatedworks/58_sae_entropy_logit_bias_threat]]. **Both MUST be cited or we risk a reviewer scoop-call.** [[relatedworks/42_attention_logit_bias_novelty_defense]] is updated accordingly.
- **SAM2-memory-attention-for-cross-modal-fusion base claim: ✅ CONFIRMED (2 independent skeptics).** MemorySAM's complete S2 citation set (13 papers, both skeptics reproduced the count) contains zero memory-attention modifications; M⁴-SAM uses SAM2 memory only for pseudo-guided *initialization*. Caveats to carry into the paper: (a) **SAM4D (2506.21547, ICCV 2025)** names a "Motion-aware Cross-modal Memory Attention" — its temporal memory attention stays per-modality (cross-modal exchange is a separate cross-attention stage in a from-scratch camera-LiDAR model), but the module name alone demands a citation-and-distinguish; (b) CRISP-SAM2 (2506.23121) does text-image interaction in the encoder, not memory attention; (c) one MemorySAM citer ("Condition Aware MMSS … Underwater", ICRCV 2025) had only a truncated abstract — residual risk.
- **P29 SDC cell (unsupervised image-derived condition latent → FiLM-modulated Soft-MoE LoRA routing): NARROWED but no exact occupant found (1 confirmed / 1 uncertain).** New fence posts: **M⁴-SAM (2605.11760)** = Modality-Aware MoE-LoRA + modality dispatcher inside SAM2's encoder (routing by modality *identity*, RGB-D VSOD); **MoE-LoRA SAM (2412.04220, Dec 2024)** = MoE-LoRA + adaptive routing on SAM v1 for MMSS **on DELIVER/MUSES/MCubeS** — it PREDATES M⁴-SAM and further shrinks the architecture-side novelty; **MoCLE (2312.12379)** already does UNSUPERVISED cluster-conditional LoRA-expert routing (instruction tuning, not vision-condition); **MoFME (2312.16610)** does FiLM-modulated experts + uncertainty-aware router (deweathering, no LoRA, no seg); **CAFuser (2410.10791)** does condition-token-guided fusion on MUSES/DELIVER but its token is *supervised* (verbo-visual contrastive on condition attributes). SDC novelty must rest ENTIRELY on the **unsupervised image-derived condition signal** (no labels, no text, no modality-ID, no external sensors) + FiLM-on-Soft-MoE-gate combination. ⚠️ **Blocking open threat:** the ICRCV 2025 underwater condition-aware paper must be read before claiming the cell — skeptic1 could not verify its mechanism (S2 rate-limited, no arXiv copy); skeptic2's record says its condition comes from *external environmental sensors* (which would leave "unsupervised image-derived" safe). Both reports logged; resolve from the primary PDF.
- **P30 cell: ✅ CONFIRMED narrowing required.** BiXFormer (2506.03675, IEEE TMM per S2) definitively occupies "query/mask-level classification for MMSS" (reformulates MMSS as mask-level classification; Unified Modality Matching + Cross-Modality Alignment; cites MemorySAM). No occupant found for the narrowed cell "class-token decoding on **reliability-biased SAM2 memory-fused features** with training-free-anchored router" — P30 must be framed exactly there, never as "query decoder for MMSS". Full note: [[relatedworks/59_bixformer_mask_classification_threat]].
- **SAM3-RBMA first-mover claim: ⚠️ REFUTED as literally stated.** **SAMCM-SR (MDPI Applied Sciences 16(5):2351, 2026)** is a published multimodal SAM3 segmentation paper: low-light-enhanced visible-light detections are geometrically transferred as **cross-modal spatial priors that prompt SAM3** on super-resolved infrared (power-equipment inspection). Its cross-modality is prompt/prior-level — SAM3's input stays single-modality IR — so the narrower cell "**feature-level RGB-X fusion inside SAM3** (reliability-biased attention)" still appears open, and first-mover framing must be re-scoped to that. Also: "at least six RGB-only SAM3 dense-adaptation papers" could only be verified as ~5 (SAM3-Adapter 2511.19425, SAM3-UNet 2512.01789, lesion 2603.25945, few-shot 2604.05433, SAM3-I) — do not print "six".
- **Benchmark pressure (unchanged):** MaCVi @ CVPR 2026 runs a live MULTIAQUA multimodal seg challenge (metric M = avg(val mIoU, nighttime-test mIoU); all 3 modalities RGB-T-L mandatory) — expect a wave of entries in the workshop report (2604.13244). Nighttime-test emphasis matches the RBMA story.

## 0.1 Adversarial-verification digest (claim → verdict)

| # | Critical claim (pre-verification) | skeptic1 | skeptic2 | Net verdict |
|---|---|---|---|---|
| C1 | No work injects reliability/uncertainty/condition as additive pre-softmax attention-logit bias in multimodal dense-prediction fusion | uncertain (found **SAE** 2603.16558 — training-free entropy logit bias, LVLM) | **refuted** (found **PRIMED** 2605.07154 — logit bias in referring AV *segmentation*) | **REFUTED** — cell occupied; RBMA novelty = signal (training-free entropy) × site (SAM2 memory attn) × task (RGB-X seg) only |
| C2 | No paper besides MemorySAM uses SAM2 memory attention for cross-modal fusion (13 S2 citers audited; M⁴-SAM = init only) | confirmed | confirmed (adds SAM4D + CRISP-SAM2 dismissals) | **CONFIRMED** — cite SAM4D as adjacent prior art; underwater citer abstract truncated (residual risk) |
| C3 | P29 exact cell unoccupied; M⁴-SAM occupies MoE-LoRA-in-SAM2; novelty must rest on unsupervised condition signal | uncertain (MoCLE, MoFME, CAFuser near-misses; underwater paper unread) | confirmed (adds 2412.04220 predating M⁴-SAM; underwater = external sensors) | **CONFIRMED-NARROWED** — claim only the unsupervised-image-latent + FiLM-Soft-MoE conjunction; read underwater paper first (blocking) |
| C4 | BiXFormer occupies generic query/mask-classification MMSS; P30 must be restated | confirmed | confirmed | **CONFIRMED** — P30 = class tokens on reliability-biased SAM2 memory-fused features |
| C5 | No multimodal (RGB-X) SAM3 seg paper exists; ≥6 RGB-only SAM3 dense papers; SAM3-RBMA first-mover | **refuted** (SAMCM-SR) | **refuted** (SAMCM-SR; only ~5 RGB-only verified) | **REFUTED** — re-scope to "feature-level fusion inside SAM3"; cite SAMCM-SR |

## 1. Threat-triage table

Threat columns re-scored after verification. New rows from the adversarial passes are marked ★.

| # | Paper | arXiv / ID | Venue+year | Mechanism class | Dataset + number (as sourced) | Threat RBMA | Threat P29 | Threat P30 | Tag |
|---|-------|-------|-----------|-----------------|-------------------------------|------------|-----------|-----------|-----|
| ★1 | PRIMED (referring audio-visual segmentation, modality-prior attention bias) | 2605.07154 | arXiv 2026-05 | **logit-additive-bias** (b_M = γ_p·log(P/(1−P)) added pre-softmax in MHCA; P = learned modality prior distilled from Qwen3-omni) | RAVS benchmarks (numbers not yet extracted) | **HIGH** (occupies the dense-prediction logit-bias cell; signal learned, site non-SAM2) | LOW | LOW | VERIFIED-PDF (equation quoted by skeptic2; full tables pending) |
| ★2 | SAE (entropy-reliability attention bias for LVLM hallucination) | 2603.16558 | arXiv 2026 | **logit-additive-bias**, training-free (S̃ = S + λ·SAE·C, λ=0.5; entropy-derived reliability into pre-softmax cross-modal attention, LLM decoder) | LVLM hallucination benchmarks (numbers not extracted) | **HIGH** (mechanism-level occupant: training-free entropy → logit bias; task/site differ) | LOW | LOW | VERIFIED (skeptic1 mechanism check; full read pending) |
| 3 | RSGMamba: Reliability-Aware Self-Gated SSM for MMSS | 2604.12319 | arXiv 2026-04 (v2 04-15) | learned-gate (sigmoid MLP gates on SSM output matrix C) | NYUDv2 58.8 / SUN-RGBD 54.0 / MFNet 61.1 / PST900 88.9 (RSGMamba-B, SegMAN-B, 48.6M; Tables I–III) [val/test split not restated — see §3.1] — no DELIVER/MUSES/MCubeS | MED | LOW | LOW | VERIFIED-PDF (arXiv HTML v2) |
| 4 | EQUISeg: Balanced Modality Contributions | 2509.24505 | arXiv 2025-09 (preprint) | loss-level (training-only prototype KL distillation, SGM) + cross-attn CMTB | DELIVER MiT-B2 **67.90** [val] (their table; StitchFusion 68.20 same table); MUSES MiT-B0 **50.26** (MAGIC++ 50.14) | LOW-MED | LOW | LOW | VERIFIED-PDF (arXiv HTML v1) |
| 5 | M⁴-SAM: MoE + Memory-Augmented SAM (RGB-D VSOD) | 2605.11760 | arXiv 2026-05-12 | Modality-Aware MoE-LoRA (conv experts + **modality dispatcher**) in SAM2 encoder; gated multi-level fusion; pseudo-guided memory init | 3 RGB-D VSOD datasets, "SOTA all metrics", no numbers in abstract [unknown] | LOW | **MED-HIGH** (MoE-LoRA-on-SAM2 occupied; condition-routing cell still free) | LOW | ABSTRACT-ONLY (dispatcher wording confirmed by both skeptics) |
| ★6 | MoE-LoRA SAM (adaptive-routing MoE-LoRA on SAM v1 for MMSS) | 2412.04220 | arXiv 2024-12 | MoE-LoRA + adaptive routing on SAM v1 | **DELIVER / MUSES / MCubeS** (numbers not extracted here) | LOW | **MED** (predates M⁴-SAM; shrinks P29 architecture novelty — cite BEFORE M⁴-SAM) | LOW | ABSTRACT-ONLY (skeptic2) |
| ★7 | MoCLE: unsupervised cluster-conditional LoRA-expert routing | 2312.12379 | 2023-12 (instruction tuning) | MoE-LoRA routed by unsupervised instruction clusters | LVLM instruction benchmarks | LOW | **MED** ("unsupervised cluster → LoRA routing" precedent, non-visual condition) | LOW | ABSTRACT-ONLY (skeptic1) |
| ★8 | MoFME: FiLM-modulated experts + uncertainty-aware router | 2312.16610 | 2023-12 (deweathering) | FiLM-instantiated experts, uncertainty router (no LoRA, no seg) | deweathering benchmarks | LOW | **MED** (FiLM-on-experts precedent — cite in SDC related work) | LOW | ABSTRACT-ONLY (skeptic1/2) |
| 9 | SGMA: Semantic-Guided Modality-Aware Seg (RS, incomplete modalities) | 2603.02505 | arXiv 2026-03-03 | learned prototype-feature-alignment robustness score → adaptive fusion weight + sample reweighting | RS datasets, no numbers in abstract [unknown] | MED | LOW | LOW | ABSTRACT-ONLY |
| 10 | ModalPatch: plug-and-play modality-drop robustness (3D det) | 2603.02481 | arXiv 2026-03-03 | history-based feature prediction + uncertainty-guided cross-modality fusion (feature-level) | nuScenes-class 3D det [unknown] | LOW (det; Track 5) | LOW | LOW | ABSTRACT-ONLY |
| 11 | GeomPrompt: geometric prompts under missing/degraded depth | 2604.11585 | CVPR 2026 URVIS WS | RGB→synthesized 4th-channel geometric prompt (input-level) | SUN RGB-D: +6.1 mIoU on DFormer, +3.0 on GeminiFusion; +3.6 under severe depth corruption; 7.8 ms [unknown] | LOW | LOW | LOW | ABSTRACT-ONLY |
| 12 | OmniSegmentor (flexible multimodal pretraining, ImageNeXt) | 2509.15096 | **NeurIPS 2025** | pretraining-level (5-modality ImageNet-scale) | NYUDv2, EventScape, MFNet, DELIVER, SUNRGBD, KITTI-360 — "new SOTA records", numbers not in abstract [unknown] | MED (leaderboard, not mechanism) | LOW | LOW | ABSTRACT-ONLY (tables = Track 1/3 job) |
| 13 | VLMFusionOcc3D (WeathFusion) | 2603.02609 | arXiv 2026-03-03 | learned-gate driven by vehicle metadata + weather prompts (supervised condition signal); occupancy | nuScenes, SemanticKITTI [unknown] | LOW | MED (condition gating, but supervised/textual) | LOW | ABSTRACT-ONLY |
| 14 | MaCVi 2026 challenge overview (MULTIAQUA multimodal seg) | 2604.13244 | CVPRW 2026 | challenge report; M = avg(val mIoU, night-test mIoU); all 3 modalities mandatory | leaderboard at macvi.org (PDF >10MB, not parsed) | MED (benchmark race) | — | — | ABSTRACT-ONLY |
| 15 | BiXFormer: mask-level classification MMSS | 2506.03675 | IEEE TMM (per S2) | **query-based** (Unified Modality Matching: modality-agnostic + complementary label assignment; Cross-Modality Alignment on queries) | "+2.75% and +22.74% mIoU over prior arts" (benchmarks unnamed in abstract) [unknown] | LOW | LOW | **HIGH** (generic query-decoder-for-MMSS occupied) | ABSTRACT-ONLY (both skeptics confirm abstract wording) |
| 16 | MRAF: reliability-aware fusion (polyglot speaker ID) | 2606.12495 | arXiv 2026-06-10 (cs.SD) | feature-multiply (reliability weights on token representations BEFORE cross-attention) | POLY-SIM 2026 | LOW (near-miss; non-vision) | LOW | LOW | ABSTRACT-ONLY |
| 17 | ViSymRe: Biased Cross-Attention | 2412.11139 | arXiv 2024-12 | logit-additive-bias, **learned** (contrastive bias B in attention scores; similarity signal, not reliability) | symbolic regression | MED (structural precedent — cite) | LOW | LOW | ABSTRACT-ONLY (module ownership confirmed by skeptic2) |
| ★18 | SAM4D: camera+LiDAR SAM2-gen with "Motion-aware Cross-modal Memory Attention" | 2506.21547 | ICCV 2025 | temporal memory attention **per-modality**; cross-modal exchange is a separate cross-attn stage; from-scratch model, not SAM2 adaptation | camera-LiDAR 4D benchmarks | **MED** (module NAME collides with our base claim — cite & distinguish) | LOW | LOW | VERIFIED (skeptic2 architecture check) |
| ★19 | SAMCM-SR: cross-modal SAM3 segmentation of power-equipment IR | MDPI Appl. Sci. 16(5):2351 | 2026 (journal, pre-July) | prompt/prior-level cross-modality: visible-light detections → geometric transform → spatial priors prompting SAM3 on super-resolved IR (SAM3 input stays single-modality) | power-equipment IR dataset | **MED** (kills "no multimodal SAM3" as literal claim; feature-level-fusion-inside-SAM3 cell still open) | LOW | LOW | VERIFIED (both skeptics; full read pending) |
| 20 | AECF: Adaptive Entropy-Gated Contrastive Fusion | 2505.15417 | arXiv 2025-05 (stat.ML) | loss/training-level (entropy as adaptive coefficient + curriculum mask) | AV-MNIST, MS-COCO; masked-input mAP +18pp @50% drop | LOW-MED (entropy signal, training-time only; Track 4) | LOW | LOW | ABSTRACT-ONLY |
| 21 | EGFormer: Any-modal Scoring + Modal Dropping | 2505.14014 | arXiv 2025-05 | per-modality importance score → ranking + hard dropping (selection) | −88% params, −50% GFLOPs, SOTA UDA transfer | MED (scoring exists; mechanism = hard selection) | LOW | LOW | ABSTRACT-ONLY |
| 22 | RMMSS: hybrid prototype distillation + feature selection | 2505.12861 | arXiv 2025-05 (v2 08-18) | distillation + trainable feature-score selection | +2.80/3.89/0.89% missing-modality on 3 datasets; −0.1% full-modality | LOW | LOW | LOW | ABSTRACT-ONLY |
| 23 | Reducing Unimodal Bias w/ Functional Entropy Reg. | 2505.06635 | S2: ICCV 2025 (conflicting "accepted 2026-03-01" snippet — report both) | loss-level entropy regularization, parameter-free | synthetic+real MMSS (numbers = Track 3) | MED (entropy signal, LOSS-level) | LOW | LOW | ABSTRACT-ONLY |
| 24 | Condition-Aware MMSS Adapted to Underwater | DOI 10.1109/ICRCV67407.2025.11349188 | ICRCV 2025 | condition tokens (CAFuser-style) for RGB+sonar+depth; condition source CONFLICTING: skeptic1 = unverifiable, skeptic2 record = external environmental sensors | n/a | LOW | **MED — BLOCKING READ** (must read primary PDF before claiming P29 cell) | LOW | ABSTRACT-ONLY/truncated (cites MemorySAM) |
| 25 | UG2+ 2026 Track 2 pipeline (adverse-weather seg) | 2605.22216 | CVPRW 2026 | RGB-only semi-supervised (UniMatch V2) | challenge numbers | LOW | LOW | LOW | ABSTRACT-ONLY |
| 26 | SAM3-UNet | 2512.01789 | arXiv 2025-12 | adapter-enhanced SAM3 + U-Net decoder — RGB only | med/natural benchmarks | LOW | LOW | LOW | ABSTRACT-ONLY |
| 27 | SAM3-Adapter | 2511.19425 | arXiv 2025-11 | adapter tuning of SAM3 — RGB only | task benchmarks | LOW | LOW | LOW | ABSTRACT-ONLY |
| 28 | SegEarth-OV3 / Prompt-Calibrated SAM3 (RS OV seg) | 2512.08730 / 2606.21863 | arXiv 2025-12 / 2026-06 | SAM3 open-vocab adaptation, RGB RS | RS OV-seg benchmarks | LOW | LOW | LOW | ABSTRACT-ONLY |
| 29 | SAM-FuseNet (RGB-T aerial, SAM2 ViT + LoRA modality-specific encoders) | IEEE (no arXiv) | IEEE journal 2026 | SAM2-backbone dual encoders + per-modality LoRA | RGB-T aerial | LOW-MED (SAM2-multimodal family grows; no memory-attention use) | LOW | LOW | UNVERIFIED-BLOG (paywalled) |

Corrected SAM3 context line (replaces the pre-verification claim): SAM2→SAM3 gap analysis (2512.06032), eye-image SAM3 (2603.17715), lesion study (2603.25945), few-shot (2604.05433), SAM3-I are RGB-only (~5 verified, not "six"); **SAMCM-SR is multimodal at the prompt level** → the open cell is specifically *feature-level RGB-X fusion inside SAM3* (SAM3-RBMA's actual target; its ~24 mIoU plateau remains an engineering, not a scooping, problem).

## 2. MemorySAM citing-paper audit (Semantic Scholar, arXiv:2503.06700, retrieved 2026-07-02; count=13 independently reproduced by both skeptics)

None of the 13 modifies SAM2 memory attention:

1. MaCVi 2026 challenge overview (2604.13244) — cites as related work.
2. PanoEnv (2602.21992) — RL panoramic spatial reasoning; irrelevant.
3. Condition-Aware MMSS Underwater (ICRCV 2025) — condition tokens, RGB+sonar+depth. Row 24; **blocking read for P29**.
4. Med-K2N (2510.02815) — medical modality translation; irrelevant mechanism.
5. MedAlmighty (Frontiers AI 2025) — DINOv2 distillation; irrelevant.
6. BiXFormer (2506.03675) — mask-level classification MMSS. **P30 HIGH** (row 15).
7. "MLLMs are Deeply Affected by Modality Bias" (2505.18657) — position paper.
8. EGFormer (2505.14014) — modality scoring + dropping (row 21).
9. Reducing Unimodal Bias (2505.06635) — functional-entropy loss (row 23).
10. Split Matching (2505.05023) — zero-shot seg; irrelevant.
11. Benchmarking MMSS Under Sensor Failures (2503.18445, CVPRW 2025) — adopt as robustness eval protocol.
12. AnySeg (2411.17141) — distillation; known.
13. Adversarial Robustness for Unified Multi-Modal Encoders (2505.11895) — irrelevant.

Adjacent non-citing candidates checked and dismissed by skeptic2: **SAM4D** (2506.21547 — per-modality temporal memory attention, from-scratch model; cite for the name), **CRISP-SAM2** (2506.23121 — text-image interaction in encoder, not memory attention), **MSM-Seg** (2510.10679 — custom dual-memory framework, not SAM2).

Caveat: S2 lags Google Scholar; re-run pre-submission.

## 3. Near-miss record for the logit-bias cell (updated post-verification)

Ordered by proximity to RBMA:

1. **PRIMED (2605.07154)** — OCCUPANT of "additive pre-softmax bias in dense-prediction cross-modal attention". Differs on signal (learned prior distilled from Qwen3-omni vs our training-free entropy) and site (bespoke MHCA in RAVS vs SAM2 memory attention). → [[relatedworks/60_primed_attention_logit_bias_threat]]
2. **SAE (2603.16558)** — OCCUPANT of "training-free entropy-derived additive pre-softmax bias" at mechanism level. Differs on task (LVLM hallucination) and site (LLM decoder visual attention). → [[relatedworks/58_sae_entropy_logit_bias_threat]]
3. **ViSymRe (2412.11139)** — "Biased Cross-Attention": learned contrastive bias B added to attention scores; signal = matched-pair similarity, task = symbolic regression. Cite and distinguish.
4. **MRAF (2606.12495, cs.SD)** — reliability weights on token representations *before* cross-attention (feature-multiply). Confirms the field default is pre-attention feature weighting.
5. **SwinTF3D (2512.22878)** — text-derived class-wise bias injected into segmentation OUTPUT logits (not attention logits). [UNVERIFIED-BLOG — snippet only.]
6. A search-summary phrase "NSR-dependent pre-softmax bias … reliability-aware mixture-of-experts controller" could NOT be traced to any real paper — synthesis noise; logged so future sweeps don't re-chase it.

**Consequence for the paper:** the RBMA novelty sentence may no longer say "no prior work adds a reliability bias to attention logits". It must say: prior additive-logit-bias work exists in adjacent settings (PRIMED: learned modality prior in referring AV segmentation; SAE: training-free entropy bias in LVLM decoding; ViSymRe: learned similarity bias in symbolic regression) — RBMA is the first to inject a *training-free per-modality predictive-entropy* reliability prior into *SAM2 memory attention* for *multi-sensor semantic segmentation*, and the first such mechanism evaluated under adverse-condition RGB-X benchmarks. (Pending: full reads of PRIMED/SAE could shrink this further — see §5.)

## 4. Other 2026H1 context

- **DGFusion** officially RA-L 2026 (accepted 2026-01-26; github.com/timbroed/DGFusion) — main competitor status unchanged (DELIVER test CLDE 56.7 per vault; Track 2 owns detail).
- **OmniSegmentor = NeurIPS 2025** confirmed; pretraining contribution (ImageNeXt, 5 modalities); DELIVER number may reset the leaderboard bar — Track 1/3.
- **Sensor-failure benchmark (2503.18445)** — adopt for robustness evaluation; EQUISeg's EMM/RMM/NM corruption protocol likewise.
- LoRA-routing 2026 (LD-MoLE 2509.25684, DR-LoRA 2601.04823, LoRAuter 2601.21795, Task-Aware LoRA composition 2602.21222) — all LLM/task-representation routing; with MoCLE (2312.12379) as the unsupervised-cluster precedent from the LLM side, P29's claim survives only as *visual-condition*-driven unsupervised routing.

## 5. Residual gaps / follow-ups (ranked)

1. **BLOCKING: read PRIMED (2605.07154) full PDF** — extract tables, exact P definition, whether any variant is training-free; needed before writing the RBMA novelty paragraph.
2. **BLOCKING: read SAE (2603.16558) full PDF** — exact entropy definition and C term; check for any dense-prediction experiment.
3. **BLOCKING (P29): read "Condition Aware MMSS … Underwater" (ICRCV 2025)** — resolve the skeptic1/skeptic2 conflict on its condition source (unverifiable vs external sensors) before claiming the SDC cell.
4. SAMCM-SR (MDPI Appl. Sci. 16(5):2351) full read — confirm prompt-level-only cross-modality so the SAM3 re-scoped claim is safe.
5. MaCVi 2026 full PDF (2604.13244, >10 MB) + macvi.org MULTIAQUA leaderboard scrape.
6. BiXFormer full tables (which datasets give +2.75/+22.74; architecture of query-feature matching) — Track 7 overlap.
7. OmniSegmentor DELIVER number — Track 1/3.
8. Venue conflict on 2505.06635 (ICCV 2025 vs "accepted 2026-03-01") — verify before citing.
9. SAM-FuseNet (IEEE, paywalled) mechanism; M⁴-SAM / SGMA / ModalPatch / VLMFusionOcc3D full-text numbers; MoE-LoRA SAM (2412.04220) DELIVER/MUSES/MCubeS tables (feeds both P29 defense and Track 3 SOTA table).
10. Google Scholar recount of MemorySAM citations pre-submission (S2 lag).

## Links

- HIGH-threat full notes: [[relatedworks/60_primed_attention_logit_bias_threat]] · [[relatedworks/58_sae_entropy_logit_bias_threat]] · [[relatedworks/59_bixformer_mask_classification_threat]] · [[relatedworks/48_m4sam_moe_lora_sam2_threat]]
- Novelty-defense note updated in place: [[relatedworks/42_attention_logit_bias_novelty_defense]] (see its "2026-07-02 deep-research update")
