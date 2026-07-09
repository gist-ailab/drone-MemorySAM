---
title: Uncertainty and Reliability Fusion Related Work for RBMA
aliases: [TMC]
tags: [related-work, uncertainty, reliability, evidential-fusion, multimodal-segmentation, rbma, key-paper]
created: 2026-06-24
source: OpenAlex API verification; arXiv API verification; [[sources/01_source_index_multimodal_segmentation]]; [[sources/02_openalex_top_venue_literature_database]]; [[relatedworks/02_dgfusion_relatedwork]]; [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
status: verified-draft
---

# Uncertainty and Reliability Fusion Related Work for RBMA

## Scope

This note collects reliability-, uncertainty-, and trust-aware multimodal fusion work for positioning [[RBMA]] as **reliability-biased SAM2 memory attention**. The key distinction is whether a method uses uncertainty only as a training loss or output calibration signal, as feature/decision scaling, as learned gating/modality selection, as evidential belief fusion, or as an **additive pre-softmax attention-logit bias**.

## Verified source ledger

| Method / paper | Verification source | Venue / year | Verified facts | RBMA positioning |
|---|---|---:|---|---|
| UTFNet — *Uncertainty-Guided Trustworthy Fusion Network for RGB-Thermal Semantic Segmentation* | DOI: [10.1109/LGRS.2023.3322452](https://doi.org/10.1109/LGRS.2023.3322452); OpenAlex abstract; [[sources/01_source_index_multimodal_segmentation]] | IEEE GRSL 2023 | RGB and thermal quality varies by sample; UTFNet estimates modality uncertainty with a UEEF module, models predicted probabilities with a Dirichlet distribution, and integrates evidence through Dempster–Shafer theory. | Strong evidence that uncertainty can guide trustworthy fusion; it is RGB-T feature/evidential fusion, not SAM2 memory-attention logit bias. |
| HyperDUM — *Hyperdimensional Uncertainty Quantification for Multimodal Uncertainty Fusion in Autonomous Vehicles Perception* | DOI: [10.1109/CVPR52734.2025.02078](https://doi.org/10.1109/CVPR52734.2025.02078); OpenAlex abstract | CVPR 2025 | Quantifies feature-level epistemic uncertainty for multimodal fusion using hyper-dimensional computing; estimates channel- and patch-wise uncertainty and adaptively weights sensor features; reports gains in 3D detection and semantic segmentation. | Very relevant because uncertainty is feature-level/spatial; still feature weighting, not memory-attention logit bias. |
| Deep evidential fusion with uncertainty quantification and reliability learning for multimodal medical image segmentation | DOI: [10.1016/j.inffus.2024.102648](https://doi.org/10.1016/j.inffus.2024.102648); OpenAlex metadata | Information Fusion 2024 | Title and venue verified; OpenAlex marks it as uncertainty/reliability fusion. Abstract was unavailable in OpenAlex in this run. | Use cautiously: strong evidential-fusion related work but domain is medical segmentation; verify PDF before final claims. |
| TMC — *Trusted Multi-View Classification With Dynamic Evidential Fusion* | DOI: [10.1109/TPAMI.2022.3171983](https://doi.org/10.1109/TPAMI.2022.3171983); Semantic Scholar / OpenAlex abstract | TPAMI 2022 | Uses variational Dirichlet class-probability distributions and Dempster–Shafer theory to dynamically integrate multi-view evidence; targets reliability and robustness for noisy/corrupted/OOD data. | Foundational evidential multi-view reliability baseline; classification, not dense segmentation or SAM2 memory. |
| CAFuser — *Condition-Aware Multimodal Fusion for Robust Semantic Perception of Driving Scenes* | arXiv:2410.10791; DOI [10.48550/arXiv.2410.10791](https://doi.org/10.48550/arxiv.2410.10791); [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]] | arXiv 2024 / RA-L 2025 per existing note | Uses RGB to classify environmental conditions and create a condition token that guides multimodal fusion; uses modality-specific feature adapters; reports MUSES and DeLiVER results in abstract. | Condition-aware gating/adaptation; not predictive uncertainty and not pre-softmax memory-logit bias. |
| DGFusion — *Depth-Guided Sensor Fusion for Robust Semantic Perception* | arXiv:2509.09828; [[relatedworks/02_dgfusion_relatedwork]] | RA-L 2026 per existing note | Uses auxiliary depth head, local depth tokens, and a global condition token to condition cross-modal fusion; targets spatially varying reliability. | Closest in spirit for local reliability, but depth-specific and not SAM2 memory attention. |
| Conflict-guided evidential fusion / semantic conflicts | Pattern Recognition paper *Resolving semantic conflicts in RGB-T semantic segmentation*, DOI: [10.1016/j.patcog.2025.111398](https://doi.org/10.1016/j.patcog.2025.111398); OpenAlex metadata | Pattern Recognition 2025 | Title/venue/year/citation count verified; OpenAlex abstract unavailable. The title confirms a conflict-resolution framing for RGB-T segmentation. | Cite only after PDF verification for exact module. It supports the claim that RGB-T conflict is an active problem. |

## Mechanism taxonomy

| Mechanism class | What is changed? | Representative sources | Strength | Limitation for RBMA novelty |
|---|---|---|---|---|
| **Evidential uncertainty fusion** | Branch outputs are converted to evidence / Dirichlet opinions and fused, often with Dempster–Shafer theory. | [[UTFNet]], [[TMC]], deep evidential fusion | Explicit uncertainty and reliability language; strong reviewer-recognizable theory. | Usually fuses decisions/features; does not alter attention competition over memory tokens. |
| **Feature-level uncertainty weighting** | Sensor features are scaled or weighted according to uncertainty. | [[HyperDUM]] | Spatial/channel uncertainty can localize unreliable regions. | Weighting occurs on features, not as additive attention logits inside SAM2. |
| **Condition-guided feature modulation** | A condition token or condition classifier modulates fusion. | [[CAFuser]], [[DGFusion]] | Good for adverse weather/night/domain shifts. | Condition may be global or proxy-based; not necessarily predictive uncertainty. |
| **Depth-guided local reliability** | Depth or LiDAR supervises local tokens that guide fusion. | [[DGFusion]] | Spatially varying reliability; strong driving-scene relevance. | Requires depth/LiDAR and is not modality-general. |
| **Semantic-conflict resolution** | Resolves inconsistent RGB/T predictions/features. | Pattern Recognition 2025 conflict paper | Directly motivates modality conflict. | Exact mechanism requires PDF verification; not yet established as attention-logit bias. |
| **Pre-softmax attention-logit bias** | Add a reliability prior to attention logits before softmax, changing token competition. | [[RBMA]] proposed | Directly targets SAM2 memory attention and preserves softmax normalization. | Needs empirical ablation versus feature gating/output scaling. |

## Established facts

- RGB-T and RGB-X segmentation papers explicitly report that **modality quality varies across samples and conditions**; UTFNet uses this as the motivation for uncertainty-guided trustworthy fusion.
- Evidential learning is an established reliability framework in multi-view learning: TMC uses Dirichlet evidence and Dempster–Shafer aggregation and is published in TPAMI.
- Current robust driving-scene fusion methods such as CAFuser and DGFusion adapt fusion using condition/depth cues, supporting the premise that fusion should be input-dependent.
- HyperDUM verifies that feature-level epistemic uncertainty can be used for multimodal fusion in autonomous-vehicle perception, including semantic segmentation settings.

## Open questions / verification gaps

- The exact mechanisms of **Conflict-guided evidential multimodal fusion**, **ReliFusion**, **READ**, **AG-Fusion**, and **EQUISeg** were not fully verified in this run. Search via arXiv/OpenAlex/Semantic Scholar either returned no exact match or only metadata without abstracts. Keep them in the candidate list but do not make detailed claims until PDFs are obtained.
- Semantic Scholar was rate-limited (HTTP 429) during part of the run; OpenAlex and arXiv were used as fallbacks.
- Deep evidential fusion in Information Fusion is verified at metadata level, but its full method details require the publisher PDF or preprint.

## Ready-to-use related-work paragraph candidates

### Paragraph A — reliability fusion background

Uncertainty-aware multimodal fusion has recently become a central theme in robust perception. UTFNet estimates RGB and thermal uncertainty with an evidential fusion module and uses Dempster–Shafer theory to guide trustworthy RGB-T segmentation, while TMC establishes a broader multi-view evidential-learning paradigm based on Dirichlet opinions and dynamic evidence aggregation. HyperDUM further moves uncertainty estimation into the feature-fusion stage by estimating channel- and patch-wise epistemic uncertainty for autonomous-vehicle perception. These works show that multimodal fusion should be reliability-aware, but they primarily modulate features or fuse evidential predictions. In contrast, RBMA injects reliability directly into the pre-softmax attention logits of SAM2-style memory attention.

### Paragraph B — driving-scene condition and local reliability

Driving-scene fusion methods also demonstrate the need for input-dependent sensor weighting. CAFuser uses an RGB-derived condition token and modality-specific adapters to guide multimodal fusion under changing environmental conditions, and DGFusion adds local depth tokens plus a global condition token to model spatially varying reliability. RBMA follows the same high-level motivation but replaces condition/depth-specific modulation with predictive reliability and applies it at the memory-attention decision point, where modality tokens compete for attention.

## Links

- [[relatedworks/41_unimodal_bias_and_modality_collapse]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/02_dgfusion_relatedwork]]
- [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
- [[relatedworks/03_unimodal_bias_entropy_relatedwork]]

---

## 2026-07-02 deep-research update (Track 2 — 원문 수준 검증 + adversarial verification)

Source: Track 2 deep verification (findings + two independent adversarial skeptic passes). 아래 내용은 위 본문(2026-06-24)의 ledger를 원문(PDF/HTML) 수준으로 확정/수정한다. 기존 내용은 보존; 충돌 시 이 섹션이 우선.

### A. 기존 ledger 항목의 확정/수정

1. **CAFuser (RA-L'25) — CA² vs CAA 메커니즘 확정** [VERIFIED-PDF, 두 skeptic 모두 원문 재확인]
   - CT 생성: 최상위 RGB feature → 2enc+2dec Transformer → CT; **verbo-visual contrastive loss로 supervised** (MUSES 메타데이터 템플릿 "A {weather} driving scene at {time}..." — 조건 라벨/텍스트 필요 → unsupervised 아님. P29 SDC 방어 포인트).
   - **CA²**: CT를 FC 통과 후 **cross-attention QUERY set에 concat** ("concatenate this adjusted CT with the 49 RGB tokens... forming an enhanced, condition-aware query"), attention 후 CT 토큰 제거. Ablation Table VI: query-concat **59.7 PQ** > K/V-concat 59.1 PQ. **Additive logit bias 아님 — condition-token(query-concat) class.**
   - **CAA**: CT → FC(4 outputs) → softmax(합=1) → **각 modality feature map에 스칼라 곱** — learned-gate/feature-multiply class.
   - 수치 [VERIFIED-PDF]: MUSES PQ — OneFormer* 55.2 / MUSES baseline 53.6 / CAA 59.4 / **CA² 59.7**; CA² per-condition PQ: Clear 61.4 / Fog 57.5 / Rain 59.6 / Snow 57.2 / Day 59.5 / Night 57.3. MUSES mIoU: CAA 78.5 / CA² 78.2 (GeminiFusion 75.3). DELIVER CLDE (Table III): CA² **67.8 [val] / 55.6 [test]**, CAA 68.6 [val] / 55.2 [test]. Backbone: OneFormer + Swin-T, 공유 백본 + modality adapters (CA² 모델 77.7M, Table IV).

2. **DGFusion (RA-L'26) — 현 최고 published condition-adaptive seg** [VERIFIED-PDF; skeptic들이 URVIS-2026 벤치마크 연구(arXiv 2604.16984, 최고 ~54.6 PQ)까지 교차확인 — 초과 수치 미발견]
   - 메커니즘: LiDAR를 **입력+depth GT 겸용**(multi-task); local depth tokens `t_d = Pool_mean(Conv(d))` + global condition token을 **query set에 concat** `F_q = [F_rgb, t_c, t_d]`; 표준 `Softmax(QK^T)V` (Eq. 6), attention 후 토큰 제거. Robust depth loss: log-L1 + τ=0.8 quantile outlier filter + edge/panoptic-edge smoothness; `L_depth = 0.9·L_logL1 + 0.05·L_es + 0.05·L_pes`. **Learned, logit-additive 아님.**
   - 수치 [VERIFIED-PDF]: MUSES PQ **61.03** vs CAFuser 59.70 (per-condition 이득: Snow +2.57, Night/Rain +1.63, Fog +1.34 — adverse에서 최대); MUSES mIoU **79.5** vs 78.2; DELIVER [test]: CLE 51.6 / CLDE **56.7** (vs CAFuser 55.6). Hyperparams: batch 8, 180k(MUSES)/200k(DELIVER) iters, AdamW+poly, Swin-T.

3. **HyperDUM (CVPR'25)** — 메커니즘·수치 확정, 상세는 신규 노트 [[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]]. 핵심: **learnable uncertainty-weighting layer를 fine-tune** (feature-multiply, training-free 아님); DELIVER Table 4 = **[val] 프로토콜** (CMNeXt 66.30 ↔ CAFuser Table III 66.3 val 일치), HyperDUM mean **67.59**.

4. **DELIVER two-cluster 논쟁 — 프로토콜로 해소 확정** [두 skeptic 독립 재확인]: CMNeXt(MiT-B2) CLDE **66.3 = val / 53.0 = test, 같은 모델** (CAFuser Table III). HyperDUM류 val-cluster vs CAFuser/DGFusion류 test-cluster. 우리 비교표는 반드시 [val]/[test] 태그 분리 (→ [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]]).

5. **UTFNet (GRSL'23)**: 여전히 [ABSTRACT-ONLY] (IEEE paywall). Dirichlet evidence head 기반 UEEF — learned evidential class로 유지; 정확한 가중식 인용 금지.

### B. 신규 진입자/near-miss — 신규 노트로 분리

- [[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]] — HyperDUM per-paper (DELIVER val 표 포함)
- [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]] — SAE: 유일한 training-free entropy→additive-logit 선행 (LVLM 도메인)
- [[relatedworks/46_attention_reweighting_detection_nearmisses]] — ModalPatch(post-softmax multiplicative), ReliFusion(output-scale, 재검증 캐비앳), SAM2Long(SAM2 memory에 multiplicative key-scaling, training-free, unimodal)
- [[relatedworks/47_reliability_fusion_2025_2026_new_entrants]] — RSGMamba, UP-Fuse, AW-MoE, EQUISeg, GeomPrompt, MULTIAQUA, READ, AECF, SGMA 스윕

### C. 갱신된 mechanism-class taxonomy (signal source × training-free × injection)

| Method | Task | Signal source | Training-free signal? | Injection location | Mechanism class |
|---|---|---|---|---|---|
| **RBMA (ours)** | multimodal seg | raw per-modality decoder softmax entropy | **YES** | **additive PRE-softmax logits, SAM2 memory cross-attn** | **logit-additive-bias** |
| CAFuser-CA² (RA-L'25) | seg | learned CT (text-supervised) | no | token concat into query set | condition-token |
| CAFuser-CAA | seg | learned CT | no | per-modality scalar × features | learned-gate / feature-multiply |
| DGFusion (RA-L'26) | seg | learned depth tokens (LiDAR-GT) + CT | no | token concat into query set | condition-token (spatial) |
| HyperDUM (CVPR'25) | seg + det | HDC prototype distance | no (prototypes+FT) | learned feature reweighting | feature-multiply |
| UTFNet (GRSL'23) | RGB-T seg | evidential (Dirichlet) head | no | evidence-guided weighting | loss/feature-level (evidential) |
| RSGMamba ('26 pre) | RGB-X seg | learned MLP gates | no | SSM C-matrix multiply | learned-gate |
| EQUISeg ('25 pre) | multimodal seg | self-guided mutual gating | no | feature-level | learned-gate |
| UP-Fuse ('26 pre) | 3D panoptic | learned divergence uncertainty | no | cross-modal feature modulation | feature-multiply |
| ModalPatch ('26 pre) | 3D det | learned variance (NLL) | no | W̃=W·[1−softmax(U)], POST-softmax | attn-multiplicative (post) |
| ReliFusion ('25 pre) | 3D det | learned confidence (CMCL) | no | confidence × attn OUTPUT (⚠ 재검증 요) | output-scale |
| SAM2Long ('24) | video seg (unimodal) | SAM2 occlusion score | **yes** | multiplicative key scaling, SAM2 memory attn | attn-multiplicative (key) |
| AW-MoE ('26 pre) | 3D det | supervised weather routing | no | MoE expert selection | MoE-routing |
| AECF ('25) | classification | entropy (learned gate) | no (gate learned) | gated fusion layer | learned-gate |
| SAE (2603.16558) | LVLM hallucination | attention-distribution entropy | **YES** | **additive PRE-softmax** (decoder→visual) | logit-additive-bias (non-fusion) |
| Missing-modality masks | various | availability (binary) | yes (trivial) | −∞ logit mask | hard logit mask |

### D. Adversarial verification 판정 요약 (주장별)

| 주장 | 판정 | 비고 |
|---|---|---|
| CAFuser CA²=query-concat, CAA=softmax scalar multiply | **confirmed** (skeptic1+2, 원문) | Table VI 59.7 vs 59.1 재현 |
| DGFusion 최고 published + learned query-concat | **confirmed** (skeptic1+2) | URVIS'26 교차확인 포함 |
| DELIVER two-cluster = val/test 프로토콜 | **confirmed** (skeptic1+2) | 66.3≈66.30 앵커 |
| "모든 reliability 경쟁자가 trained + feature/output/post-softmax 주입" | **uncertain** | HyperDUM·RSGMamba·ReliFusion(부분)만 원문 spot-check; UTFNet/ModalPatch/UP-Fuse/EQUISeg 미재검증 → 논문에서는 전칭("all") 대신 "the methods we examined"로 한정할 것 |
| "additive pre-softmax reliability bias 선행 없음" | **uncertain (unrefuted)** | → [[relatedworks/42_attention_logit_bias_novelty_defense]] 2026-07-02 update의 fenced claim 참조 |

### E. 갱신된 related-work paragraph candidate (English, Paragraph A/B 대체 후보)

Condition- and reliability-aware multimodal fusion has progressed from global condition tokens (CAFuser, RA-L'25 — a text-supervised condition token concatenated into cross-attention queries, or softmax-normalized per-modality feature weights) to spatially varying depth-guided conditioning (DGFusion, RA-L'26), and from learned uncertainty heads — evidential (UTFNet), hyperdimensional prototypes (HyperDUM, CVPR'25), variance regressors modulating post-softmax attention weights (ModalPatch), contrastively supervised confidences scaling attention outputs (ReliFusion), and self-gated state-space readouts (RSGMamba) — toward robustness under degradation. The methods we examined either require dedicated training of the reliability signal or inject it multiplicatively at the feature, output, or post-softmax-attention level; condition tokens act by enlarging the query set rather than by explicitly biasing attention scores. In contrast, we derive a training-free per-modality reliability from the predictive entropy of a lightweight per-modality decode and inject it as an additive pre-softmax bias on SAM2 memory cross-attention logits — a mechanism whose closest precedents are entropy-driven additive logit modulation for LVLM hallucination within a single image modality (SAE, 2026) and training-free multiplicative key scaling in SAM2 memory attention for unimodal long-video segmentation (SAM2Long).

### F. 남은 검증 gap

- UTFNet 정확한 가중식 (IEEE paywall) — 정량 인용 시 도서관 경유.
- ReliFusion CW-MCA 수식(Eqs. 13–14) — findings의 pdftotext 추출은 있으나 adversarial 재검증 실패 → PDF 재확인 후 인용.
- CAFuser DELIVER per-condition 분해 미추출 (MUSES per-condition만 확보; DELIVER 조건별은 HyperDUM Table 4 [val]로 대체 중).
- SAE 정량 수치, SAM2Long 수식 — 원문 표 미발췌.
- 2026 preprint 전원(RSGMamba/ModalPatch/UP-Fuse/AW-MoE) venue watch.
