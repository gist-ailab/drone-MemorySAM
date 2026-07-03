---
title: A-Signal Prior-Art Kill-Check — Decoder Predictive Entropy in Dense-Seg Fusion (Track 4)
tags: [related-work, priorart, novelty-defense, a-signal, entropy, uncertainty-fusion, rbma, kill-check]
created: 2026-07-02
source: parallel deep-research Track 4 ([[sources/07_parallel_research_prompts_2026-07-02]]); adversarially verified (2 independent skeptic passes)
status: verified-draft
---

# A-Signal Prior-Art Kill-Check — Decoder Predictive Entropy in Dense-Seg Fusion

## Problem setting

우리 A-신호 claim under test: **training-free, GT-free, per-modality DECODER PREDICTIVE ENTROPY**

$$B_i = 1 - \frac{H(\mathrm{softmax}(\mathrm{Dec}_i(f_i)))}{\log C}$$

를 multimodal DENSE segmentation fusion의 weight/bias로 사용 — 구체적으로 SAM2 memory cross-attention logits에 대한 **ADDITIVE PRE-SOFTMAX BIAS**:

$$\mathrm{Attention} = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \lambda \cdot B\right)V \quad \text{(RBMA)}$$

점검 대상 cell: "raw per-modality decoder softmax-entropy → dense-seg fusion weight", 그리고 sub-cell "→ attention-logit additive bias".

## Verdict (adversarially verified 2026-07-02)

**PARTIALLY-OCCUPIED overall / UNOCCUPIED (unfalsified) for the exact RBMA cell.**

| Cell | Status | Occupant(s) |
|---|---|---|
| Softmax-derived per-modality uncertainty → **output-level** dense-seg fusion | **OCCUPIED** | UNO (ICRA'20) [VERIFIED-PDF], Blum et al. (IROS'18) [ABSTRACT-ONLY] |
| Learned uncertainty/evidence → **feature-level** dense-seg fusion | OCCUPIED | UTFNet, HyperDUM [VERIFIED-HTML], USNet, 2309.05919, UDML, RSGMamba |
| Entropy/confidence → **loss / pseudo-label** level (TTA) | OCCUPIED | Tent/EATA/SAR, MM-TTA, Latte, READ [VERIFIED-PDF], Night-TTA |
| Training-free reliability signal → **additive attention-logit bias**, multimodal dense seg | **UNOCCUPIED — unfalsified, not proven** | none found (~15 targeted searches + 2 independent adversarial refutation passes, ~10+3 angles) |

⚠️ **Epistemic status of the last row (both skeptic passes: "uncertain")**: 이것은 universal negative라 원리적으로 확증 불가. 두 독립 adversarial 검증 pass 모두 counterexample을 찾지 못했으나, 결론은 "**unfalsified after adversarial search**"로 표기해야 하며 "proven unoccupied"로 쓰면 안 됨. 논문에서는 "to our knowledge" 헤지 필수. 가장 가까운 후보들이 cell을 점유하지 못하는 이유:

- **SAE (arXiv:2603.16558)** — training-free entropy-guided **attention-logit modulation은 존재**하지만 도메인이 LVLM hallucination mitigation이지 multimodal dense-seg fusion이 아님. **메커니즘 관점에서 가장 가까운 발견물** — 반드시 인용하고 도메인 차이를 명시할 것.
- **SAM2 memory lineage** (SAMURAI arXiv:2411.11922, SAM2Long arXiv:2410.16268, SAMed-2 MICCAI'25) — memory-frame **SELECTION/filtering**이지 logit bias가 아님; MemorySAM (arXiv:2503.06700)은 uncertainty weighting 자체가 없음.
- **UMFNet (CVPR 2026, RGB-T SOD, PDF verified by skeptic)** — **learned** Gaussian latent uncertainty + **MULTIPLICATIVE** confidence gating (signal도 injection도 다름).
- **SGMA (arXiv:2603.02505)** — learned prototype-alignment reliability; 방향도 반대 (attention→reliability).
- **AFRDA (2507.17957), UGDD-Net (2605.09600)** — uncertainty-in-attention이지만 **single-modality**.

## Novelty (what survives, what does not)

- ❌ **주장 불가**: "first to use per-modality prediction entropy for multimodal seg fusion" — UNO(출력 레벨, 분석 변형으로 deterministic entropy 포함), Latte(pseudo-label filter), Seeing-Through-Fog(측정 entropy, detection)가 존재.
- ❌ **주장 불가**: "pre-softmax" 단독 표현 — UNO의 Eq.6 `p_i = Softmax(l_i · δ_min)`도 기술적으로 decoder **output logits에 대한 pre-softmax multiplicative scaling**임 (skeptic2 확인). RBMA 표현은 반드시 **"additive"+"attention-logit"** 두 한정어를 모두 포함해야 함.
- ✅ **주장 가능 (헤지 포함)**: (i) injection mechanism = additive pre-softmax bias on cross-/memory-**attention** logits, (ii) fully training-free (no TempNet, no evidence head, no prototypes, no gradient TTA), (iii) real adverse-condition benchmarks (DELIVER/MUSES/MCubeS) with a VFM, (iv) N-modality SAM2 memory setting.

## Method — key prior works (with equations)

### 1. UNO — ICRA 2020, arXiv:1911.05611 ⚠️ 가장 인용-의무적인 선행 [VERIFIED-PDF, 양쪽 skeptic 모두 원문 재확인]

- **Mechanics (verified)**: 4개 uncertainty measure 분석 — (1) MC-dropout predictive entropy, (2) MC-dropout mutual information, (3) **deterministic single-pass softmax entropy** (training-free!), (4) learned "TempNet" data-dependent temperature. (skeptic2 추출문은 TempNet 제외 "three metrics"로 집계 — 실질 동일.) Scaling:
  $$p_i = \mathrm{Softmax}([l_i^1,\dots,l_i^C] \cdot \delta_{\min})$$
  (output-logit **multiplicative** temperature scaling), 이후 noisy-or fusion:
  $$I(y{=}c) = 1 - \prod_i \big(1 - p_i(y{=}c \mid x_i, \theta_i)\big)$$
- **Chosen config = `min(Ave.Temp, Ave.En)`** — 채택 구성은 **learned TempNet**과 deterministic entropy의 결합이므로 UNO의 최종 시스템은 fully training-free가 아님 (skeptic1 확인: TempNet은 NLL로 학습).
- **Task**: RGB-D semantic segmentation, **AirSim simulation only** (real adverse benchmark 아님, skeptic1 확인), degradations fog/snow/frost/noise, 2 modalities, fully independent late-fusion networks.
- **RBMA 대비 방어선**: injection point (output-probability vs attention-logit, representation에 영향), full training-freeness, real benchmarks, N-modality VFM.

### 2. Evidential dense-seg fusion lineage — 모두 LEARNED evidence [CONFIRMED by both skeptics]

- **TMC/ETMC** (ICLR'21 / TPAMI'22, arXiv:2204.11423): per-view learned evidence $e_v \to$ Dirichlet $\alpha_v = e_v + 1$, $u_v = C/\sum\alpha$, Dempster-Shafer 결합. Classification only. [ABSTRACT-ONLY]
- **arXiv:2309.05919** (Inf. Fusion 2025): DST mass functions + **learned contextual-discounting reliability coefficients** $m_i^\beta$, Dempster ⊕; medical voxel seg — evidential dense-seg fusion은 **존재함**. [VERIFIED by skeptics via primary source]
- **MEFN** (arXiv:2406.18327): softplus evidence → Dirichlet + learned uncertainty calibrator + uncertainty perceptual loss; PET/CT. [VERIFIED via primary]
- **UTFNet** (GRSL 2023, IEEE 10273407): Dirichlet-parameterized **learned** evidence + DST fusion, RGB-T seg. skeptic1은 IEEE 원문으로 확인, skeptic2는 secondary source만 — injection locus(feature vs decision)는 여전히 paywall 미해결. [ABSTRACT-ONLY+]
- **USNet** (arXiv:2203.04537, ICRA'22): evidential MEC + uncertainty-aware fusion, RGB-D road seg. 문헌 인덱스로만 검증. [ABSTRACT-ONLY]

**결론 (양쪽 skeptic CONFIRMED)**: 발견된 모든 evidential dense-seg fusion은 learned evidence heads / learned discounting에서 reliability를 유도 — **raw softmax entropy of an auxiliary decode를 쓰는 사례 없음**, attention-logit 주입 없음.

### 3. TTA lineage — entropy는 loss/pseudo-label 레벨에만 [CONFIRMED by both skeptics]

- **Tent/EATA/SAR**: entropy = minimization objective / sample filter. Loss-level.
- **READ (ICLR 2024)** — **residual risk 해소됨**: skeptic1이 ICLR proceedings PDF 추출로 확인 — "self-adaptive attention" = source model에서 상속한 **Q,K,V projection의 W/B를 confidence-aware loss 하에 gradient TTA로 업데이트**하는 것. Additive logit 항 아님. 벤치마크는 audio-visual event classification / action recognition — dense seg 아님. "attention × reliability" 사분면의 최근접 개념적 이웃으로 인용 필수; 구분: READ = parameter adaptation via loss (classification), RBMA = closed-form entropy bias on logits, zero updates (dense seg).
- **MM-TTA (CVPR'22)**: cross-model consistency → pseudo-label 선택. **Latte (ECCV'24, arXiv:2403.06461)**: ST-voxel **raw prediction entropy**(training-free 신호!)로 high-entropy voxel **filtering** — loss cell 점유; A-신호의 plausibility 근거로 유용.
- **Night-TTA (arXiv:2307.04470, IEEE TAI'23)**: RGB-T dense seg에서 ensemble logits + TTA; **CAR weighting formula 미해결 (잔여 리스크)** — entropy-weighted라면 output-level near-miss로 승격됨.

### 4. 기타 근접 클래식/신규

- **Blum et al.** (IROS'18, arXiv:1807.11249): 독립 학습된 per-modality expert의 Bayes/Dirichlet **output fusion**, calibration-fit only. Training-free-ish output cell의 고전 베이스라인.
- **Seeing Through Fog** (CVPR'20): **sensor-measurement entropy** (decoder softmax 아님) → feature-exchange gating, **detection**. "training-free entropy → fusion steering"의 역사적 점유자 — signal source와 injection 차이를 명시하고 인용.
- **HyperDUM** (CVPR 2025, arXiv:2503.20011) [VERIFIED-HTML, 양쪽 skeptic CONFIRMED]: HDC projection $\varphi_m(z) = \Phi^{d\times C} \otimes z_m^{pooled}$ + patch bundling; uncertainty = **labeled data로 학습한 class prototypes**와의 hypervector 거리; **learnable weighting module $\Omega(z_m, u_m)$**이 각 feature extractor 뒤·fusion block 앞에서 per-modality **feature reweighting**. Not training-free (원문 표현: "traditional bundling technique, which requires labels for supervised learning of prototypes"). DELIVER 상 가장 가까운 uncertainty 기반 경쟁자 (skeptic 탐색에서도 더 가까운 것 없음).
- **QMF** (ICML'23): energy-score confidence → decision weights, classification. **UDML** (arXiv:2603.19681) [VERIFIED-HTML]: 기존 uncertainty estimator가 경미한 열화에 둔감 + "dual suppression" bias를 보임 → learned noise estimator로 feature multiply; classification. **우리 A-신호의 한계/ablation 축으로 직결**.
- **AECF** (arXiv:2505.15417): entropy가 gate regularizer/curriculum에 등장하나 gate 자체는 learned, classification.
- **2505.06635** (Reducing Unimodal Bias w/ Functional Entropy Reg.): multimodal semseg의 **loss-level** entropy 정규화; 우리 벤치마크 계열 — related work 인용. (gains +13.94/+3.25/+3.64의 dataset명 미추출 — 잔여 리스크.)
- 2026 triage: **ModalPatch** (2603.02481, 3D det, feature-level), **RSGMamba** (2604.12319, learned self-gate SSM — P30 router 근접), **SGMA** (2603.02505, attention→reliability 역방향).

## Quantitative results (verbatim rows)

| Work | Benchmark / metric | Number (verbatim) | Verification | Split |
|---|---|---|---|---|
| HyperDUM (2503.20011) Table 4, DELIVER semseg, CMNeXt backbone | baseline mean mIoU | **66.30** | [VERIFIED-HTML] (skeptic-reconfirmed) | [val] (CMNeXt-val protocol cluster, cf. [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]]) |
| HyperDUM, same table | HyperDUM mean mIoU | **67.59 (+1.29)**; Night 64.21, Foggy 66.85 | [VERIFIED-HTML] (skeptic-reconfirmed) | [val] |
| MemorySAM (2503.06700), DELIVER | mIoU | **65.38** | [ABSTRACT-ONLY] | [unknown — split 확정은 Track 3/8] |
| UNO (1911.05611), AirSim RGB-D under degradation | mIoU gain vs SOTA fusion baselines | **+28%** (abstract claim) | [VERIFIED-PDF for method; number ABSTRACT-ONLY] | [unknown, simulation] |
| Blum et al. (1807.11249) | IoU over best single modality | up to **+5%** | [ABSTRACT-ONLY] | [unknown] |
| Night-TTA (2307.04470) | mIoU boost claim | **+13.07%** | [ABSTRACT-ONLY] | [unknown] |
| Func-entropy reg (2505.06635), 3 multimodal semseg datasets | gains | **+13.94 / +3.25 / +3.64** | [ABSTRACT-ONLY, dataset names unverified] | [unknown] |

## Limitations (of the prior art, and of our negative claim)

**Prior art 한계 (우리 기회):**
- UNO: simulation-only, 2 modalities, 완전 독립 late-fusion 네트워크, 최적 구성이 learned TempNet에 의존, output-level only.
- Evidential lineage: evidence head + 특수 loss 학습 필요; reliability가 global per-modality/class (per-pixel 아님); DS machinery 무거움.
- HyperDUM: labeled prototypes + weighting module 학습 필요; feature-multiply class.
- TTA lineage: gradient updates 필요 (READ), 혹은 신호가 loss/pseudo-label에 갇힘 (Latte, MM-TTA).

**우리 negative claim의 한계 (반드시 논문에 반영):**
- Attention-logit cell의 unoccupied 판정은 **absence of evidence** — 양 skeptic 모두 최종 판정 "uncertain (unfalsified)". SAE(2603.16558)가 타 도메인에서 동일 메커니즘을 이미 사용 → "mechanism 자체가 최초"가 아니라 "**multimodal dense-seg fusion에의 적용이 최초 (to our knowledge)**"로 한정.
- Night-TTA CAR formula, UTFNet injection locus, emergentmind aggregator가 언급한 "prior bias matrices" 원 논문 — 3건 미추적 (아래 잔여 리스크).

## Improvement directions

1. **Injection-point ablation을 taxonomy의 증거로 전환**: 동일 $B_i$ 신호를 (i) UNO-style output-average weight, (ii) HyperDUM-style feature multiply, (iii) RBMA attention-logit bias로 주입해 (iii) 승리를 실증 — 표의 각 행이 ablation 행이 됨.
2. **UDML 비판 선제 대응**: raw softmax entropy는 경미한 corruption에 과신 — graded degradation 하 $B_i$ calibration curve (ECE / uncertainty-error correlation) 보고; $\lambda$가 hard suppression이 아닌 soft reweighting임을 over-suppression 방어 논리로 사용.
3. Spatially-varying $B_i$ (per-pixel/patch entropy) vs global per-modality — evidential lineage의 "global reliability" 한계를 넘는 확장 실험.
4. Post-softmax scaling vs pre-softmax additive bias 비교 ([[relatedworks/42_attention_logit_bias_novelty_defense]]의 ablation 표와 합치).

## Comparison to RBMA-P29-P30 (mechanism-class)

Signal(행) × Injection(열) 분류 — mechanism-class 어휘: feature-multiply | learned-gate | output-scale | loss-level | condition-token | **logit-additive-bias**.

| Work | Signal | Mechanism-class (injection) | Dense seg? | Training-free signal? |
|---|---|---|---|---|
| UNO (1911.05611) | raw softmax entropy 분석 + learned TempNet (min 결합) | output-scale (multiplicative temp + noisy-or) | YES (sim RGB-D) | partially (entropy 변형만) |
| Blum et al. (1807.11249) | softmax-vector statistics | output-scale (Bayes/Dirichlet) | YES | yes (calibration-fit) |
| Seeing Through Fog (CVPR'20) | sensor-measurement entropy | feature-multiply (exchange gating) | no (det) | yes |
| TMC/ETMC | learned Dirichlet evidence | output-scale (DS opinion) | no (clf) | no |
| 2309.05919 / MEFN / UTFNet / USNet | learned evidence + discounting | output-scale / feature-multiply | YES | no |
| HyperDUM (2503.20011) | labeled-prototype similarity | feature-multiply (learned Ω) | YES (DELIVER) | no |
| QMF / UDML | energy score / learned noise est. | output-scale / feature-multiply | no (clf) | partially / no |
| READ (ICLR'24) | confidence (entropy-like) | loss-level + attention **parameter adaptation** (gradient TTA of Q,K,V W/B) | no (AV clf) | no |
| MM-TTA / Latte | consistency / raw ST entropy | loss-level (pseudo-label select/filter) | YES (2D-3D) | yes (signal) / self-training |
| Night-TTA | 미해결 (pixel distribution agg.) | output-scale ensemble + loss-level TTA | YES (RGB-T) | TTA |
| AECF / 2505.06635 | gate entropy / functional entropy | learned-gate / loss-level | no / YES | no / train-time only |
| RSGMamba / SGMA / UMFNet | learned gate / attention-derived / learned Gaussian | learned-gate / (역방향) / **multiplicative** gating | YES | no |
| SAMURAI / SAM2Long / SAMed-2 | motion/quality scores | memory-frame **selection** (not logit bias) | no (VOS/med) | mostly yes |
| SAE (2603.16558) | entropy | **logit-additive-bias류 attention modulation** — 단 LVLM hallucination, seg 아님 | no | **yes** |
| **RBMA (ours)** | **raw per-modality decoder softmax entropy** | **logit-additive-bias (SAM2 memory cross-attention)** | **YES (DELIVER/MUSES/MCubeS)** | **yes** |
| P29 SDC | unsupervised condition prototype | condition-token → FiLM on Soft-MoE LoRA router | YES | yes (signal) |
| P30 router | learned router anchored by RBMA $B_i$ | learned-gate (training-free anchor) | YES | anchor yes |

**(raw-entropy signal) × (logit-additive-bias injection) × (multimodal dense seg) 조합의 점유자는 발견되지 않음** — 단, 위 Verdict의 epistemic 단서 적용. P30 관련: RSGMamba의 learned self-gate가 최근접 2026 learned-gate — training-free 신호로 anchor한 router는 여전히 구분됨.

## Application to ours (RBMA/P29/P30 적용방향)

1. **Claim 문구 확정**: novelty = *injection mechanism* (additive pre-softmax **memory-attention** bias) + *fully training-free signal* + *real adverse benchmarks with a VFM*. "first to use predictive entropy for multimodal fusion" 금지 (UNO/Latte/STF). "pre-softmax" 단독 표현 금지 (UNO Eq.6이 output-logit pre-softmax scaling). "to our knowledge" 헤지 필수 (universal negative).
2. **Must-cite set (A-신호 문단)**: UNO, Blum'18, Seeing-Through-Fog, TMC/ETMC, 2309.05919, UTFNet, HyperDUM, QMF/UDML, READ, MM-TTA/Latte, Tent/EATA/SAR, 2505.06635, **+ SAE(2603.16558, 메커니즘 최근접, 타 도메인)**.
3. **선제 ablation**: 동일 $B_i$의 3-way injection 비교 (output-average / feature-multiply / logit-bias) — 리뷰어의 "왜 attention logit인가"를 실증으로 전환.
4. **Robustness caveat (UDML발)**: graded corruption 하 $B_i$ 민감도·calibration 보고; $\lambda$ soft-reweighting 논리로 over-suppression 반박.
5. **P29**: 본 트랙 무관 신규 위협 없음. **P30**: RSGMamba를 learned-gate 최근접으로 인용, RBMA-anchored router의 구분 유지.
6. **제출 전 필수 추적 (잔여 리스크)**: ① Night-TTA CAR 수식 (IEEE TAI 원문) — entropy-weighted ensemble logits면 output-level near-miss 승격, ② UTFNet injection locus (paywall), ③ emergentmind가 paraphrase한 "prior bias matrices modulate attention logits" 원 논문 추적, ④ 2505.06635 dataset명. (READ 수식 리스크는 **해소** — skeptic이 ICLR proceedings PDF로 확인: gradient TTA of Q,K,V projections, additive 항 없음.)

## Related-work paragraph candidate (English)

> **Uncertainty-driven multimodal fusion.** Converting per-modality uncertainty into fusion behavior is well established, but prior work operates at three control points distinct from ours. At the *output* level, UNO scales each modality's decoder softmax by an uncertainty-derived temperature — including a single-pass predictive-entropy variant — before noisy-or fusion for RGB-D segmentation in simulation, and Blum et al. fuse independently trained per-modality decodes with calibrated Bayesian and Dirichlet rules. At the *feature* level, evidential methods learn per-modality evidence heads or discounting coefficients (TMC/ETMC, UTFNet, USNet, deep evidential medical fusion), and HyperDUM reweights pre-fusion features with a learnable module driven by distances to label-supervised hyperdimensional prototypes. At the *loss or pseudo-label* level, test-time adaptation minimizes or filters by prediction entropy (Tent, EATA, SAR; Latte's spatial-temporal entropy filtering; MM-TTA's consistency-based label selection), and READ adapts fusion-attention parameters via gradient updates under a confidence-aware objective for audio-visual classification. Entropy has also steered detection fusion from raw sensor measurements rather than decoder predictions (Bijelic et al.), and training-free entropy-guided attention-logit modulation has appeared outside dense prediction, for LVLM hallucination mitigation. To our knowledge, however, no existing method injects a training-free, label-free per-modality predictive-entropy signal as an *additive pre-softmax bias on cross- or memory-attention logits* for multimodal dense segmentation: prior signals are either learned (evidence heads, prototypes, TempNet) or consumed at the output, feature, or loss level, leaving the attention-logit control point — where modality memories compete for retrieval — unexplored. RBMA occupies exactly this point.

## Links

- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]] — 선행 uncertainty/evidential fusion ledger (2026-06-24)
- [[relatedworks/42_attention_logit_bias_novelty_defense]] — RBMA logit-bias 방어 공식화 + ablation 표
- [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] — DELIVER two-cluster 프로토콜 (66.30=val 확인)
- [[relatedworks/03_unimodal_bias_entropy_relatedwork]] — 2505.06635 loss-level entropy
- [[relatedworks/01_memorysam_relatedwork]] — base architecture
