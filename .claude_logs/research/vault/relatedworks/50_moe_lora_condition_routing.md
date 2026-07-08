---
title: MoE-LoRA Condition-Routing Prior Art — P29 SDC Novelty Defense
tags: [related-work, novelty-defense, moe-lora, condition-routing, film-router, p29-sdc, p30-router, soft-moe]
created: 2026-07-02
source: parallel deep-research Track 6 (sources/07_parallel_research_prompts_2026-07-02.md) + adversarial verification (2 independent skeptic passes, 2026-07-02)
status: verified-draft
---

# MoE-LoRA Condition Routing — P29 SDC 방어 노트 (Track 6)

**P29 SDC claim under test:** UNSUPERVISED image-derived condition latent (global feature stats → prototype/cluster bank) used to ROUTE/MODULATE LoRA experts via **FiLM on a Soft-MoE gate** — no condition labels, no text/CLIP, no extra sensors, applied to multimodal semantic segmentation on a SAM2 backbone.

## Verdict (adversarially verified, 2026-07-02)

**UNOCCUPIED as a combination — but two caveats from adversarial verification must be carried into the paper's framing:**

1. The exact cell — *(unsupervised visual condition latent from global feature stats) × (FiLM on a Soft-MoE-LoRA gate) × (multimodal dense segmentation)* — has **no occupant** in any searched source as of 2026-07-02. Two independent skeptic passes confirmed claims 1, 3, 5 (see per-claim verdicts below).
2. **Caveat A (MLE-SAM crowding):** MLE-SAM (arXiv:2412.04220) routes per-modality LoRA experts on **unsupervised spatially-pooled feature statistics** (plain softmax gate) on the SAME backbone (SAM2) and SAME benchmarks (DELIVER/MUSES/MCubeS). Its signal is a *modality* feature stat, not a *condition* latent; no prototypes, no FiLM. The narrow claim holds, but differentiation from MLE-SAM is thin and **must be argued explicitly**, not assumed.
3. **Caveat B (DAMP breach at the domain boundary):** the universal claim "no perception work drives expert routing with training-free feature statistics" is **refuted if restoration counts as perception**. DAMP (arXiv:2512.20251, hyperspectral restoration, 2025-12) routes degradation experts using six **training-free hand-crafted input statistics** (high-freq energy ratio, texture uniformity, etc.) concatenated into a learned MoE gate (their Eq. 3) with no label supervision on the metrics. DAMP's stats enter as **gate INPUT**, not as FiLM modulation and not as a gate-logit bias — so P29/P30's mechanism cells stay open — but any sentence of the form "no prior work uses training-free input statistics for expert routing" is now FALSE and must not appear in the paper.

### Per-claim adversarial verdicts

| # | Claim | Verdict | Counter-evidence / caveat |
|---|---|---|---|
| 1 | No published work routes/modulates LoRA experts with an UNSUPERVISED image-derived condition latent; MoCLE clusters TEXT instructions, cluster embedding = gate input (not FiLM) | **confirmed** (2 skeptics) | MLE-SAM (2412.04220) is a strong near-occupant: unsupervised pooled per-modality feature stats → softmax LoRA gate, same benchmarks. Signal = modality stat, not condition latent; no FiLM/prototypes. ClusIR (2512.10948, learnable-cluster probabilistic expert routing, all-in-one restoration, non-LoRA) is the nearest adjacent occupant. |
| 2 | ALL perception condition/weather-expert routing uses supervised or contrastive condition signals; none uses training-free feature-stat prototypes | **uncertain / scope-dependent** | Positive sub-facts reproduce (AW-MoE = supervised CE weather classifier, ~99% acc; WM-MoE = weather-cluster contrastive WGF-CL). BUT **DAMP (2512.20251)** breaches the universal negative if degradation-expert routing in restoration counts as perception-domain condition routing. Scope the claim to *multimodal dense segmentation / detection* only. |
| 3 | No precedent for FiLM(condition latent) modulating an MoE GATE; MoFME's FiLM instantiates EXPERTS, MC-dropout only calibrates the router | **confirmed** (2 skeptics) | Nearest misses: M2IR conditional feature modulator (FiLM on features, arXiv:2603.14816); PCG-MoLE additive concept-attention on the gate vector (g̃=g+g′ — supervised prototypes, classification, no scale term, not FiLM). FiLM-on-gate cell open as of 2026-07. |
| 4 | Only gradient-free additive gate-logit bias = Loss-Free Balancing (load statistics); no work anchors a learned gate with a training-free input-derived reliability signal → P30 cell open | **uncertain / must narrow wording** | LFB (2408.15664) confirmed (load-statistic bias, pre-top-K, gradient-free, not input-derived). But DAMP concatenates training-free input-derived stats into a learned gate (gate INPUT, not logit bias), and entropy-of-router works exist (entropy-aware domain-routed MoE, arXiv:2606.10454 — entropy of the gate itself, not input reliability). The EXACT cell — training-free per-modality softmax-entropy as **ADDITIVE gate-logit bias** — remains unoccupied, but the broader framing "training-free input-derived signal anchoring a learned gate" has a concrete near-occupant (DAMP). |
| 5 | LER-YOLO reliability = LEARNED alignment-error subnetwork applied as feature-multiply on router input, not training-free entropy, not gate/logit bias | **confirmed** (2 skeptics, PDF-level: Eq. 4 F_in = C(F_ir, F̃_rgb ⊙ R); R from learned subnetwork φ supervised by self-supervised loss L_uta) | Strategic conclusion holds: P30's training-free-entropy gate-bias cell remains distinct. Watch for venue version. |

---

## Problem setting

Condition-adaptive parameter-efficient adaptation: a frozen VFM (SAM2) must specialize its LoRA experts to the current environmental condition (rain/snow/night/fog) of a multimodal scene, without condition labels, text encoders, or extra sensors. Prior art splits into (i) MoE-LoRA routing in LLM/VLM land (router driven by tokens, text semantics, or instruction clusters), (ii) condition/weather-expert routing in perception (router trained with weather labels or condition-guided contrastive losses), and (iii) gate-stabilization machinery (load-balancing losses, gradient-free biases). P29 sits at the empty intersection; P30's reliability-anchored router extends Q4.

## Novelty (what each nearest work actually does)

Three named nearest works (Track 6 deliverable), plus the two adversarial near-occupants:

1. **MoCLE (arXiv:2312.12379, ICLR 2024)** — unsupervised k-means clusters → LoRA-expert gate, but clusters **TEXT instruction embeddings** (all-MiniLM-L6-v2 sentence encoder) and feeds a learnable cluster embedding as **gate input**; visual input never influences routing. VLM instruction tuning, no dense prediction. [VERIFIED-PDF]
2. **AW-MoE (arXiv:2603.16261, 2026-03)** — weather-conditioned expert routing in multimodal 3D detection, but router = **supervised weather classifier** (CE on weather labels, Algorithm 1 Stage 2, ~99% acc), experts = **full branches** (backbone+fusion+head each), hard top-1 select. [VERIFIED-PDF]
3. **MoFME (arXiv:2312.16610, AAAI 2024)** — FiLM + MoE + uncertainty-aware router in adverse weather, but **FiLM instantiates the EXPERTS** on a single shared FFN ("implicitly instantiates multiple experts via learnable activation modulations"); MC-dropout UaR only **calibrates** gate weights. The FiLM arrow points at the opposite module from P29. [VERIFIED-PDF]

Near-occupants that the paper must cite and pre-empt:

4. **MLE-SAM (arXiv:2412.04220)** — per-modality LoRA experts on SAM2 Q,V projections; router wᵢᵐ = σ(Wᵢ·fᵢᵐ + bᵢ) on spatially averaged per-modality embeddings; unsupervised gate input but **modality-routing, not condition-routing**; no prototypes, no FiLM, no anchoring. Also a direct DELIVER competitor. [VERIFIED-PDF]
5. **DAMP (arXiv:2512.20251)** — six training-free hand-crafted degradation statistics concatenated into a learned MoE gate (Eq. 3), hyperspectral restoration. Training-free feature-stat **gate input**; not FiLM, not logit bias, not segmentation, not multimodal. [VERIFIED-PDF per skeptic pass]

## Method (equations of the occupied cells)

- **MoCLE gate** (Eq. 1): 𝐆 = top_k(softmax(1/τ(𝐖_gate 𝐂[𝐱ᵢ] + ϵ))) — router input is the learnable embedding 𝐂[𝐱] of the instruction's k-means cluster (K=64 InstructBLIP / K=4 LLaVA-1.5). Output (Eq. 2–3): 𝐲ᵢ = ∑ₑ Gₑ𝐖ₑ𝐱ᵢ + 𝐖₀𝐱ᵢ, with universal expert weighted (1−G_max). [VERIFIED-PDF]
- **AW-MoE router** (Eq. 1–3): z = 𝒞(ℐ_img) ∈ ℝ^{N_W} (supervised classifier); P = softmax(z); 𝒮 = TopK(P, K), default K=1. Expert (Eq. 4): B_w = ℋ_w(ℱ_w(ℰ_w({f}))) — full weather-specific branch. [VERIFIED-PDF]
- **MoFME FiLM-expert** (Eq. 5): FM(𝒙) = γ∘𝒙 + β; FME(𝒙|γ,β) = FFN{∑ᵢ rᵢ(𝒙)·[γ⁽ⁱ⁾∘𝒙 + β⁽ⁱ⁾]} — E FiLM modulations share one FFN; the FiLM parameters are per-expert learned constants (the condition does NOT generate them). UaR (Eq. 7): ř(𝒙) = Σ̌⁻¹[r(𝒙)−μ̌]/‖·‖₂ via MC-dropout. [VERIFIED-PDF]
- **LER-YOLO reliability** (Eq. 2–4): R = σ(ϕ([F_ir, F̃_rgb])) learned; ℒ_uta = 1/N ∑(R_ij‖F_ir−F̃_rgb‖₁ − λ log(R_ij+ϵ)); injection F_in = C(F_ir, F̃_rgb ⊙ R) — **feature-multiply BEFORE the router**, never a gate weight or logit bias. [VERIFIED-PDF]
- **MLE-SAM router** (Eq. 8–9): 𝐰ᵢᵐ = σ(𝐖ᵢ·𝐟ᵢᵐ + 𝐛ᵢ) on spatially averaged per-modality embeddings, softmax + top-k over modalities; LoRA on Q,V (Eq. 5: Q′ᵐ=Qᵐ+ΔQᵐ). No load-balancing reported. [VERIFIED-PDF]
- **Loss-Free Balancing** (arXiv:2408.15664): expert-wise bias added to routing scores **before top-K**, updated gradient-free from recent expert LOAD; "does not produce any interference gradients". The canonical non-gradient additive gate-logit-bias precedent — anchor signal is model-side load, not input-side reliability. [VERIFIED-abs]
- **DAMP gate**: gate(concat(learned features, six hand-crafted degradation statistics)) — training-free stats as gate INPUT (their Eq. 3). [VERIFIED per skeptic pass]
- **P29 (ours, for contrast)**: condition latent c = prototype-bank lookup on global feature stats (training-free clustering); gate logits g(x) FiLM-modulated: g̃(x) = γ(c)∘g(x) + β(c) on a Soft-MoE-LoRA gate. **P30 (ours)**: learned modality router anchored by additive training-free RBMA-entropy bias on gate logits (LFB-mechanism family, input-side signal).

## Quantitative results (verbatim rows)

| Work | Benchmark / table | Config | Number | Tags |
|---|---|---|---|---|
| AW-MoE | K-Radar Table II (train 17,486 / test 17,458 frames) | LiDAR+4D-Radar, IoU=0.3, Total AP_3D | L4DR baseline 78.0 → **83.9** (+5.9) | [VERIFIED-PDF] [test] |
| AW-MoE | K-Radar Table II, Light Snow AP_3D | same | 78.9 → **90.2** (+11.3) | [VERIFIED-PDF] [test] |
| MLE-SAM | DELIVER Table I | RGB-D-E-L, Hiera-B+ | mIoU **64.08** (+4.90 over CMNeXt ⇒ implies the 59.18 protocol cluster / MemorySAM-style protocol) | [VERIFIED-PDF] [unknown split — feed to Track 3 audit] |
| MLE-SAM | MCubeS Table III | RGB-A-D-N, Hiera-B+ | mIoU **51.02** (+14.86 over CMNeXt) | [VERIFIED-PDF] [unknown] |
| MLE-SAM | DELIVER per-condition (search snippet) | night 62.68 / rain 62.71 mIoU | — | [UNVERIFIED-BLOG] [unknown] |
| MoFME | All-Weather / RainCityscapes | multi-deweather | +0.1–0.2 dB over baselines, −72% params, −39% inference time; downstream seg mIoU comparable | [VERIFIED-PDF] [unknown] |
| LER-YOLO | MBU benchmark, YOLOv5s protocol | misaligned RGB-IR UAV det | **89.7±0.2** AP₅₀ (best 89.9) | [VERIFIED-PDF] [unknown] |
| DynMoLE | LLM fine-tuning | Tsallis-entropy routing | outperforms LoRA by 9.6%, MoLA by 2.3% | [ABSTRACT-ONLY] [unknown] |

## Limitations (of the prior art, and of our own claim)

Prior art:
- MoCLE: text-encoder cluster identity; hard cluster membership at inference; no dense prediction, no condition semantics.
- AW-MoE: needs weather LABELS; discrete N_W taxonomy; full expert branches (heavy); detection only.
- WM-MoE: weather feature learned with weather-cluster-guided contrastive supervision (WGF-CL); [ABSTRACT-ONLY — whether the clusters use labels is UNVERIFIED, arXiv HTML broken; treat as label-assisted until PDF read].
- MoFME: FiLM params are learned constants selected by the router — condition never generates them; M MC-dropout passes at inference; restoration task.
- LER-YOLO: reliability net is learned (extra params + training); pairwise RGB-IR only.
- MLE-SAM: modality-granularity routing only; no condition awareness; no gate stabilization.

Our claim (post-adversarial):
- The claim MUST be scoped: (i) "condition routing without labels" → scope to multimodal dense segmentation/detection, because DAMP occupies training-free-stat gating in restoration; (ii) "anchoring a learned gate with a training-free signal" → scope to **additive gate-logit bias from an input-derived reliability signal**, because DAMP occupies gate-INPUT concatenation and LFB occupies load-statistic bias; (iii) MLE-SAM must be cited and differentiated on signal semantics (modality stat vs condition prototype), mechanism (gate input vs FiLM-on-gate), and granularity (per-modality vs per-condition).

## Improvement directions

- MoCLE → replace text clusters with image-derived global-stat clusters; soft membership; per-layer condition injection (exactly P29's move — cite MoCLE as the text-domain analog).
- AW-MoE → unsupervised condition discovery; LoRA experts instead of full branches; continuous condition latent instead of discrete classes (P29 does all three).
- MoFME → let the condition latent GENERATE the FiLM parameters and point FiLM at the gate, not the experts (P29's inversion).
- Loss-Free Balancing → swap the load-statistic anchor for an input-derived, training-free reliability signal (P30's move; same injection family, different signal source and purpose — balance vs correctness).
- Mod-Squad's task↔expert mutual-information loss → adapt as a condition↔expert MI regularizer against gate collapse (candidate P29 ablation).
- Exploring Expert Specialization (2509.10025): unsupervised expert specialization correlates with semantic categories — supporting citation that routers CAN discover conditions unsupervised; P29 makes it explicit via prototypes.

## Mechanism-class taxonomy (signal source × injection point)

| Work | Venue | Signal source | Supervision of signal | Experts | Injection point | Tag |
|---|---|---|---|---|---|---|
| MoCLE (2312.12379) | ICLR 2024 | k-means clusters of instruction TEXT embeddings | unsupervised clusters (pretrained text encoder) | LoRA | cluster embedding = gate INPUT (soft topk) | [VERIFIED-PDF] |
| AW-MoE (2603.16261) | arXiv 2026-03 | image weather CLASSIFIER | **supervised (weather labels, CE)** | full branches | softmax→top-1 expert select | [VERIFIED-PDF] |
| WM-MoE (2303.13739) | arXiv 2023 (rev 2024) | learned weather feature (weather-cluster contrastive) | contrastive w/ weather clusters (label use unverified) | FFN (multi-scale) | token gate INPUT | [ABSTRACT-ONLY] |
| MoFME (2312.16610) | AAAI 2024 | token router; MC-dropout uncertainty | self (MC dropout) | FiLM-instantiated shared FFN | uncertainty CALIBRATES gate; FiLM = expert | [VERIFIED-PDF] |
| LER-YOLO (2605.20667) | arXiv 2026-05 | learned spatial reliability map (alignment error) | self-supervised recon loss | 3 fusion experts | reliability ⊙ router-INPUT features (feature-multiply) | [VERIFIED-PDF] |
| MLE-SAM (2412.04220) | arXiv 2024-12 | per-modality pooled feature stats | unsupervised input, learned linear gate | LoRA (SAM2 Q,V) | gate INPUT, softmax top-k over modalities | [VERIFIED-PDF] |
| **DAMP (2512.20251)** | arXiv 2025-12 | six hand-crafted degradation statistics | **training-free (no label supervision on stats)** | degradation experts | stats concatenated into gate INPUT (Eq. 3) | [VERIFIED per skeptic pass] |
| ClusIR (2512.10948) | arXiv 2025-12 | learnable clusters (probabilistic) | learned | non-LoRA restoration experts | probabilistic cluster routing | [ABSTRACT-ONLY] |
| PCG-MoLE (2506.04673) | arXiv 2025-06 | class-label prototypes | supervised (few-shot class labels) | LoRA | additive concept-attention on gate vector (g̃=g+g′) | [ABSTRACT-ONLY per skeptic pass] |
| M⁴-SAM (2605.11760) | arXiv 2026-05 | modality identity (dispatcher) | architectural | conv-LoRA (SAM2) | modality dispatch | [ABSTRACT-ONLY] |
| X-LoRA (2402.07148) | APL ML 2024 | hidden states | learned | LoRA | layerwise token softmax scalings | [ABSTRACT-ONLY] |
| MoE-Adapters4CL (2403.11549) | **CVPR 2024** (not NeurIPS) | image distribution discriminators (DDAS) | per-task trained | adapters | task-level dispatch (MoE vs zero-shot CLIP) | [ABSTRACT-ONLY] |
| Mod-Squad | CVPR 2023 | task identity + token, MI loss task↔expert | supervised tasks | ViT attn/FFN MoE | gate + MI regularizer | [ABSTRACT-ONLY] |
| DynMoLE (2504.00661) | arXiv 2025-04 | Tsallis entropy of ROUTER dist. | self | LoRA | sparsity switch (soft↔topk) | [ABSTRACT-ONLY] |
| AdaMoLE (2405.00361) | arXiv 2024-05 | threshold network on input context | learned | LoRA | adaptive top-k threshold | [ABSTRACT-ONLY] |
| LD-MoLE (2509.25684) | arXiv 2025-09 | token hidden state | learned, differentiable | LoRA | closed-form dynamic allocation | [ABSTRACT-ONLY] |
| SAMoRA (2604.19048) | ACL 2026 Findings | textual semantics ↔ expert alignment | learned + regularizer | LoRA | semantic-aware gate + task-adaptive scaling | [ABSTRACT-ONLY] |
| Loss-Free Balancing (2408.15664) | arXiv 2024-08 (DeepSeek) | expert LOAD statistics | **gradient-free update** | any MoE | **additive expert-wise bias on routing scores pre-topK** | [VERIFIED-abs] |
| Self-Routing (2604.00421) | arXiv 2026-04 | hidden-state subspace | — | MoE | REPLACES router (subspace = logits); no bias, no reliability | [ABSTRACT-ONLY per skeptic pass] |
| Entropy-aware domain-routed MoE (2606.10454) | arXiv 2026-06 | entropy of the GATE itself | self | MoE | routing control | [ABSTRACT-ONLY per skeptic pass] |
| c-BTM (2303.14177) | arXiv 2023 | unsupervised document clusters | unsupervised | whole LMs | sparse ensemble at inference | [ABSTRACT-ONLY] |
| SkillMoV (2606.17615) | arXiv 2026-06 | LEARNABLE class prototypes | learned (not unsupervised) | view-MLPs | prototype concat as gate input g=σ(W_g[h;s]+b_g) | [ABSTRACT-ONLY] |
| **P29 SDC (ours)** | — | **unsupervised image-derived condition prototype (global feature stats)** | **training-free clustering** | **LoRA (Soft-MoE)** | **FiLM ON THE GATE** | — |
| **P30 router (ours)** | — | **training-free per-modality decoder softmax-entropy (RBMA)** | **training-free** | **modality experts** | **additive gate-logit bias (input-derived)** | — |

## Comparison to RBMA-P29-P30 (mechanism-class)

- MoCLE: **condition-token → gate-input** (hard text-cluster ID). P29 = condition-latent → **FiLM-on-gate** (condition changes the gate's computation, does not replace its input).
- AW-MoE: **supervised condition-token → hard expert select**. P29 removes labels (prototype bank from unlabeled global stats), full branches → LoRA, hard top-1 → FiLM-modulated Soft-MoE.
- WM-MoE: **learned condition feature → gate-input** (contrastive). P29 signal is training-free.
- MoFME: **FiLM-as-expert + uncertainty-calibrated gate**. P29's FiLM arrow points at the GATE; reviewers skimming both WILL conflate them — state the inversion explicitly.
- LER-YOLO: **learned-gate / feature-multiply on router input**. RBMA/P30 = training-free entropy as **logit-additive-bias**; different signal supervision AND different injection point.
- MLE-SAM: **learned-gate on unsupervised modality stats (gate-input)**. P29 = condition granularity + FiLM mechanism + prototype bank; orthogonal and composable with modality routing.
- DAMP: **training-free stats → gate-input (concat)**. P29 = FiLM modulation; P30 = additive logit bias. Same signal philosophy, different injection class — cite to pre-empt, differentiate on injection point.
- Loss-Free Balancing: **logit-additive-bias, gradient-free, load-anchored (balance)**. P30 = same injection family, **input-derived reliability anchor (correctness)**. Beautiful rhetorical parallel to RBMA itself (additive pre-softmax bias in attention) — see [[relatedworks/42_attention_logit_bias_novelty_defense]].

## Application to ours (RBMA/P29/P30 적용방향)

1. **P29 related work 구성**: MoCLE(텍스트 클러스터 게이트), AW-MoE(supervised weather routing), MoFME(FiLM-as-expert)를 3대 nearest로 전면 배치하고, 각각 signal source / supervision / injection point 3축에서 차별화. MLE-SAM과 DAMP는 "near-occupant" 문단에서 선제 인용 — 특히 MLE-SAM은 같은 SAM2+DELIVER 공간이므로 실험 비교 대상으로도 포함해야 함 (DELIVER 64.08, split 미확정 → Track 3 감사에 전달됨).
2. **클레임 문장 수위 조정 (필수)**: "no prior work uses training-free input statistics for expert routing" 류의 보편 부정문 금지 (DAMP 반례). 허용되는 클레임: (a) "no prior work FiLM-modulates an MoE gate with a condition latent" [2-skeptic confirmed], (b) "no prior work derives an unsupervised image-only condition prototype from global feature stats to modulate LoRA-expert routing **in multimodal dense prediction**" [scoped], (c) "no prior work anchors a learned gate with a training-free input-derived reliability signal **as an additive gate-logit bias**" [scoped; LFB=load bias, DAMP=gate input].
3. **P29 ablation 설계**: (i) FiLM-on-gate vs cluster-embedding-as-gate-input (MoCLE-style) vs stat-concat-gate-input (DAMP/MLE-SAM-style) — injection-point ablation이 novelty의 실증 근거; (ii) 조건↔expert MI regularizer (Mod-Squad 차용) vs load-balancing loss vs LFB-style bias — gate-collapse 대책 비교.
4. **P30 router**: Loss-Free Balancing을 mechanism-nearest로 인용하고 "same injection family (additive pre-topK score bias, gradient-free), different signal (input-derived per-modality entropy vs expert load) and purpose (correctness vs balance)"로 1문장 차별화. LER-YOLO는 reliability×routing 결합의 최초 발견 사례(2026-05)로 인용 + 학회 버전 감시.
5. **스쿠프 감시**: LER-YOLO(2605.20667), SkillMoV(2606.17615, prototype-conditioned gating), WM-MoE PDF(클러스터 라벨 사용 여부 미결 — pdftotext /pdf/2303.13739 시도), ClusIR/DAMP 계열 restoration→segmentation 전이 여부.

## Related-work paragraph candidate (English)

> Condition-specialized experts have so far required an externally supervised or learned condition signal. AW-MoE trains its router as a weather classifier with cross-entropy on weather labels [AW-MoE], WM-MoE learns weather features with condition-cluster-guided contrastive objectives [WM-MoE], and MoCLE conditions LoRA routing on clusters of textual instruction embeddings [MoCLE]. Closer to our setting, MLE-SAM routes per-modality LoRA experts in SAM2 from pooled modality features [MLE-SAM], and hand-crafted training-free degradation statistics have recently been concatenated into a restoration MoE gate [DAMP]; both, however, feed the signal as plain gate input at modality or degradation granularity. FiLM has been used inside MoE only to instantiate the experts themselves under a token-level router [MoFME], and the only gradient-free additive bias on routing scores anchors the gate to expert-load statistics for balancing [Loss-Free Balancing], while reliability-guided routing relies on a learned alignment-error subnetwork multiplied into router-input features [LER-YOLO]. To our knowledge, no prior work derives an unsupervised, image-only condition latent from global feature statistics and uses it to FiLM-modulate the gate of a Soft-MoE LoRA router for multimodal dense prediction, nor anchors such a learned gate with a training-free per-modality reliability signal injected as an additive gate-logit bias.

## Watchlist / follow-ups

- LER-YOLO (2605.20667): watch for acceptance; closest "reliability→routing" work; equation-level detail beyond abstract confirmed consistent by skeptic pass.
- WM-MoE: obtain PDF to settle whether weather clusters use labels (arXiv HTML broken; try pdftotext on /pdf/2303.13739).
- DAMP (2512.20251) / ClusIR (2512.10948): monitor for transfer of training-free-stat / cluster routing from restoration into segmentation — would close Caveat B's scope protection.
- SkillMoV (2606.17615): concurrent "prototype-conditioned gating" — monitor.
- MLE-SAM DELIVER 64.08: split unstated; +4.90-vs-CMNeXt phrasing implies the 59.18 protocol cluster — handed to Track 3 audit.
- AW-MoE code repo (windlinsherlock/AW-MoE) — check release for router-training details.
- Not exhaustively covered: GitHub-only projects; VLMo/BEiT-3 MoME and Soft-MoE (Puigcerver, ICLR 2024) cited from background knowledge, not re-verified.

## Links

- [[relatedworks/42_attention_logit_bias_novelty_defense]] — RBMA logit-bias defense (mechanism-family parallel to P30)
- [[relatedworks/20_lora_adapter_relatedwork]] / [[relatedworks/22_sam_adapter_relatedwork]] — PEFT background incl. MoE-LoRA SAM
- [[relatedworks/23_multimodal_sam_adapter_matrix]]
- [[sources/07_parallel_research_prompts_2026-07-02]] — Track 6 prompt + placement rule
