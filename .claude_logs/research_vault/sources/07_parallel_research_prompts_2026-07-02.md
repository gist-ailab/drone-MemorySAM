---
title: Parallel Deep-Research Prompts — VFM + Condition-Adaptive Multimodal Fusion SOTA
tags: [research-prompts, deep-research, multimodal-segmentation, detection, vfm, rbma]
created: 2026-07-02
source: gap-diff between drone-MemorySAM .claude_logs (doc 10/12 open TODOs) and this vault
status: ready-to-run
---

# Parallel Deep-Research Prompts — 2026-07-02

목표: 멀티 센서 기반 멀티모달 semantic segmentation + object detection에서 (1) VFM(SAM2/SAM3 등)을 잘 활용하고, (2) 이미지 상황(비/눈/암영 등)에 adaptive하게 모달리티 피쳐를 fusion하는 방법론으로 SOTA 달성.

8개 트랙은 상호 독립 — 병렬 실행 가능. 각 트랙 프롬프트 앞에 아래 공통 블록을 붙여서 사용.

트랙 ↔ 기존 열린 TODO 매핑 (drone-MemorySAM `.claude_logs/12_novelty_and_related_work.md` §4):
- Track 2·3·4 → TODO 1b (DELIVER 프로토콜), 2 (A-신호 evidential 선행), 3 (DGFusion 정량), 4 (CAFuser 조건화 원문)
- Track 6 → TODO 5/6 (MoE-LoRA 조건 라우팅, FiLM 라우터)
- Track 7 → TODO 7/8 (class-token on memory features, reliability-anchored gate)
- Track 5 → detection 확장 (doc 12 §2.6)
- Track 8 → 스쿠프 경보 (MemorySAM 인용 논문 전수 체크)

---

## ⚠️ 실행 전 필터 — 볼트가 이미 답을 가진 항목 (2026-07-02 전수 인벤토리 결과)

리서치 에이전트가 중복 조사하지 않도록, 아래 항목은 조사 범위에서 빼거나 "확인만" 수준으로 낮출 것:

- **Track 2.1 (DGFusion 정량+메커니즘): ✅ 해소** — `relatedworks/02` + `09`: MUSES test PQ **61.03** / mIoU **79.5**, DELIVER test CLDE **56.7**·CLE 51.6 (vs CAFuser 55.6). 메커니즘(LiDAR=입력+depth-GT, robust log-depth loss, local depth tokens + global condition token) 검증 완료. 잔여: per-condition breakdown 표, 학습 하이퍼파라미터 세부.
- **Track 2.2 (CAFuser): 숫자만 ✅** — CA² test 55.6/val 67.8, CAA test 55.2/val 68.6; MUSES 78.2(CA²)/78.5(CAA). 단 **CA² vs CAA 메커니즘 구분(condition token 주입 위치)은 미해소** → 원문 확인 그대로 필요. (볼트 경고: CAFuser Table III는 pdftotext 열붕괴 가능 → 시각 재확인.)
- **Track 3 (two-cluster): ✅ 대부분 해소** — **66.30 = CMNeXt MiT-B2 · val · RGB-DEL** (StitchFusion Table 7 + CAFuser Table III "mIoU-val 66.3"), **53.0 = MiT-B2 · test · CLDE**, **59.18 = MiT-B0** (MemorySAM Table 1 — B2가 아님). 잔여: MemorySAM 65.38의 split 확정, val 2005/test 1897 count 원문 확인, MUSES/MCubeS 공식 dataset card(split/클래스) 추출, per-condition 표.
- **Track 1: 그대로 필요** — OmniSegmentor는 abstract-stub만(DELIVER 숫자 없음), MM-SAM-adapter(2509.10408)는 인덱스 라인만 존재.
- **Track 6: 그대로 필요** — MoE-Adapters4CL/Mod-Squad는 볼트에 부재. 볼트 보유 근접군: MoE-LoRA SAM(2412.04220, modality-routing), M⁴-SAM(2605.11760, modality dispatcher), AW-MoE(2603.16261, **supervised** weather routing) → "unsupervised image-derived condition latent → LoRA routing" 셀은 현재까지 빈 것으로 보임(확정 조사 필요).
- **Track 8 HIGH-watch 명단 추가**: RSGMamba(2604.12319, reliability **self-gated** Mamba — 학습형이지만 P30 router와 근접), EQUISeg(2509.24505), GeomPrompt(2604.11585), ModalPatch(2603.02481, **uncertainty-guided** cross-modality fusion), M⁴-SAM(2605.11760), OmniSegmentor(2509.15096). 전부 볼트에 abstract-stub만 있음 → 원문 정독 대상.
- 볼트 자체 경고: OpenAlex DB(3,010건)는 노이즈 많음 — 인용 전 per-paper 검증 필수; 2026 dated stub들과 JEPA 노트는 abstract-only.

---

## 공통 컨텍스트 블록 (모든 프롬프트 앞에 붙여서 사용)

```
[PROJECT CONTEXT — prepend to every track]
Our project: multimodal semantic segmentation (+ object detection extension) using
Vision Foundation Models (SAM2/SAM3). Core method = RBMA (Reliability-Biased Memory
Attention): training-free per-modality predictive entropy B_i = 1 − H(softmax(Dec_i(f_i)))/log C
injected as ADDITIVE PRE-SOFTMAX BIAS into SAM2 memory cross-attention logits
(Attention = softmax(QK^T/√d + λ·B)V). Base architecture = MemorySAM (arXiv:2503.06700,
modalities-as-frames on SAM2). Extensions: P29 SDC (unsupervised image-derived condition
prototype → FiLM modulation of Soft-MoE LoRA router), P30 (class-token decoder on fused
memory features + learned modality router anchored by RBMA reliability).
Benchmarks: DELIVER (RGB-D-E-L, 25cls), MUSES, MCubeS, MULTIAQUA (maritime RGB-T-L).
Goal: SOTA on condition-adaptive (rain/snow/night/fog) multimodal fusion.
Known direct competitors: DGFusion (2509.09828), CAFuser (2410.10791), MemorySAM,
OmniSegmentor (2509.15096), MM-SAM-adapter (2509.10408), HyperDUM (CVPR'25), UTFNet.

[VERIFICATION RULES]
- Every quantitative number MUST come from the original paper PDF/HTML table — quote
  verbatim with table number, split (val/test), modality config, backbone size.
- Tag each claim: [VERIFIED-PDF] / [ABSTRACT-ONLY] / [UNVERIFIED-BLOG].
- Always record: arXiv ID, venue+year, official code URL.
- Cover 2025-06 ~ 2026-07 arXiv especially (newest first).
- If a claim contradicts another source, report BOTH numbers and the protocol difference —
  do not silently pick one. (DELIVER has a known "two-cluster" baseline discrepancy:
  CMNeXt-B2 = 66.30 vs 59.18 vs 53.0 depending on protocol.)

[OUTPUT FORMAT]
Obsidian note per topic, with YAML frontmatter (title/tags/created/source/status:
verified-draft), sections: Problem setting / Novelty / Method (with equations) /
Quantitative results (verbatim table rows) / Limitations / Comparison to RBMA-P29-P30
(mechanism-class: feature-multiply | learned-gate | output-scale | loss-level |
condition-token | logit-additive-bias) / Related-work paragraph candidate (English).
```

---

## Track 1 — VFM 백본의 멀티모달 dense prediction 활용 지형 (2025–2026)

```
Survey how Vision Foundation Models (SAM/SAM2/SAM3, DINOv2/v3, Perception Encoder,
EVA-02, InternImage, Depth Anything) are adapted as MULTIMODAL backbones for dense
prediction (semantic seg + detection), 2024→2026-07.

Must answer:
1. Complete list of SAM/SAM2/SAM3-based multimodal segmentation methods after MemorySAM
   (2503.06700): MM-SAM-adapter (2509.10408), FusionSAM (2408.13980), OmniSegmentor
   (2509.15096), MemorySAM follow-ups, anything citing MemorySAM. For each: adaptation
   strategy (LoRA/adapter/prompt/full-FT), how non-RGB modalities enter the model
   (channel concat / separate encoder / frames / prompts), DELIVER-MUSES-MCubeS numbers.
2. Does ANY method exploit SAM2's memory attention for modality fusion besides MemorySAM?
   (direct threat check to our base claim)
3. DINOv2/v3-based multimodal seg competitors — do they beat SAM2-based on DELIVER?
4. What is the current #1 on DELIVER (any protocol)? List top-5 with protocol details.
5. Multi-scale limitation: SAM3 uses plain ViT (single-scale) — any work adding
   multi-scale/FPN to SAM3 or PE for dense seg? (explains our SAM3-RBMA ~24 plateau)
Deliverable: one note per major method + a "VFM-multimodal landscape" synthesis table.
```

## Track 2 — 조건-적응형(비/눈/야간) reliability fusion 직접 경쟁자 정밀 조사

```
Deep-verify the condition-adaptive / reliability-aware multimodal fusion competitors.
This is our paper's main related-work section and novelty defense.

Must answer (original-PDF level):
1. DGFusion (2509.09828, RA-L'26): EXACT mechanism — how depth-GT + robust depth loss
   produce "spatially-varying sensor reliability"; where depth tokens condition fusion;
   full DELIVER/MUSES mIoU tables vs CAFuser/MAGIC (verbatim, with split). Hyperparams.
2. CAFuser (2410.10791, RA-L'25): where exactly the CLIP/text condition token is
   injected — CA² (condition-aware cross-attention) vs CAA (addition); is it a
   pre-softmax logit modification or feature/output modulation? DELIVER test 55.6 and
   MUSES 78.2 confirmation + per-condition (night/rain/snow/fog) breakdown tables.
3. HyperDUM (CVPR'25) and UTFNet (GRSL'23): exact uncertainty→weight mapping
   (feature-multiply? output-scale?), DELIVER numbers for HyperDUM.
4. NEW 2025-2026 entrants we may have missed: search "condition-aware fusion",
   "reliability-aware multimodal segmentation", "adaptive modality weighting",
   "weather-adaptive fusion", "sensor degradation robust segmentation" on arXiv
   2025-06→2026-07 (e.g. RSGMamba, AW-MoE, GeomPrompt, MultiAqua-related, ModalPatch).
5. CRITICAL novelty check: does ANY published work inject reliability/uncertainty/
   condition as an ADDITIVE PRE-SOFTMAX ATTENTION-LOGIT BIAS in multimodal fusion?
   Search: "attention bias fusion", "logit bias attention multimodal", "attention
   mask reliability", ALiBi-style biases applied to cross-modal attention.
   Report any near-miss with its exact injection point.
Deliverable: per-competitor notes + updated mechanism-class taxonomy table
(signal source × injection location), flagging any occupant of the logit-bias cell.
```

## Track 3 — DELIVER/MUSES/MCubeS 벤치마크 프로토콜 & SOTA 표 확정

```
Resolve the DELIVER benchmark "two-cluster" protocol problem and build a
publication-ready SOTA table for DELIVER, MUSES, MCubeS (+ MULTIAQUA if public).

Known discrepancy: CMNeXt-B2 RGB-D-E-L is reported as 66.30 (CMNeXt original,
2303.01480), 59.18 (MemorySAM paper), 53.0 (DGFusion "mIoU-test"). Splits:
train 3983 / val 2005 / test 1897.

Must answer:
1. For EACH paper reporting on DELIVER (CMNeXt, MemorySAM, StitchFusion, GeminiFusion,
   CAFuser, DGFusion, MAGIC/MAGIC++, AnySeg, OmniSegmentor, MM-SAM-adapter, HyperDUM,
   U3M, Reducing-Unimodal-Bias 2505.06635, + any newer): which split (val/test),
   input resolution, modality config, backbone, and the exact mIoU. Verbatim table rows.
2. Determine WHY the clusters differ: val-vs-test? resolution? re-implementation?
   Check official GitHub eval scripts (CMNeXt DELIVER repo, MemorySAM repo) for the
   default eval split. Quote the code/config line if findable.
3. MemorySAM 65.38: is it val or test? (repo config / issue tracker / author response)
4. Same audit for MUSES (CAFuser 78.2 vs GeminiFusion 75.3) and MCubeS
   (MemorySAM 52.88 etc.).
5. Per-condition (cloud/fog/night/rain/snow + sensor-failure cases: motion blur,
   overexposure, LiDAR-jitter, event low-res) breakdown tables where available —
   we need these for the condition-adaptive story.
Deliverable: one benchmark-protocol note + machine-readable SOTA table (markdown),
each row tagged [val]/[test]/[unknown] and source table number.
```

## Track 4 — A-신호 novelty 확정: decoder predictive entropy의 dense-seg 선행 여부

```
Novelty kill-check for our "A signal": training-free, GT-free, per-modality DECODER
PREDICTIVE ENTROPY (H of softmax of a per-modality decode) used to weight/bias
multimodal DENSE segmentation fusion.

Must answer:
1. Evidential deep learning for dense seg fusion: TMC/ETMC lineage extended to
   segmentation? Search "evidential multimodal semantic segmentation", "Dirichlet
   segmentation fusion", "subjective logic segmentation", medical multimodal
   (MRI T1/T2/FLAIR) evidential fusion — do any use raw softmax entropy (not learned
   evidence heads)?
2. Test-time adaptation literature (entropy minimization: Tent, EATA, SAR, READ) —
   any that turn per-modality entropy into FUSION weights (not loss weights)?
3. Uncertainty-weighted fusion classics (Kendall & Gal aleatoric, temperature-scaled
   confidence fusion, Bayesian fusion in robotics/remote sensing) — nearest matches
   to "entropy of an auxiliary per-modality decode, no extra training".
4. For each near-miss: is the signal (a) learned head, (b) MC-dropout/ensemble,
   (c) raw softmax entropy? And is usage (a) loss, (b) feature weight, (c) output
   average, (d) attention logit?
Verdict format: occupied / partially-occupied / unoccupied for the exact cell
"raw per-modality decoder softmax-entropy → dense-seg fusion weight", with citations.
```

## Track 5 — 멀티모달 객체 검출: 조건-적응형 fusion + VFM (검출 확장 근거)

```
Map condition-adaptive multimodal fusion in OBJECT DETECTION (2D & 3D/BEV), and VFM
usage in multimodal detection — to justify extending our reliability mechanism to
detection heads.

Must answer:
1. Adverse-weather-adaptive detection SOTA 2024-2026: camera-LiDAR(-radar) 3D det
   with explicit reliability/condition handling beyond ReliFusion (2502.01856) —
   e.g. AW-MoE, ModalPatch, weather-conditioned BEV fusion, radar-fallback methods.
   For each: signal source, injection location (feature/output/query/logit?), dataset
   (nuScenes-adverse, K-Radar, DENSE/STF, aiMotive), numbers verbatim.
2. RGB-T detection under day/night: current SOTA (M3FD, FLIR, LLVIP, DroneVehicle),
   any illumination-adaptive gating — mechanism class for each.
3. VFM in multimodal detection: SAM/DINOv2/Grounding-DINO features fused with
   LiDAR/thermal for detection — how are VFM features injected?
4. Does ANY detection work bias attention logits by sensor reliability? (novelty
   check in the det domain — we believe precedent = 0)
5. Maritime/drone domain: MULTIAQUA, MACVi/MODS/USVInland leaderboards — current
   best multimodal detection/seg methods and their fusion mechanisms.
Deliverable: detection-side taxonomy table (same mechanism-class axes) + note on
"seg→det transferability of logit-bias reliability" with head/representation caveats.
```

## Track 6 — 조건 라우팅 MoE-LoRA 선행 (P29 SDC 방어)

```
Novelty check for P29 SDC: UNSUPERVISED image-derived condition latent (global feature
stats → prototype/cluster bank) used to ROUTE/MODULATE LoRA experts (FiLM on a
Soft-MoE gate) — no condition labels, no text/CLIP, no extra sensors.

Must answer:
1. MoE-LoRA routing prior art: MoE-Adapters4CL (NeurIPS'24), Mod-Squad (CVPR'23),
   VLMo/BEiT-3 MoME, LD-MoLE, DynMoLE, MoLE variants, AdaMoLE, X-LoRA — what drives
   each router (task-ID? learned token gate? text?), and has anyone used
   unsupervised visual condition clustering?
2. Condition/domain-conditioned experts in perception: weather-MoE segmentation/
   detection (AW-MoE etc.), domain-indicator routing, test-time domain clustering
   (SwAV/DeepCluster-style) feeding a router.
3. FiLM-modulated routers: any precedent of FiLM(condition latent) → MoE gate?
4. Expert/gate collapse: known fixes (load-balancing loss, z-loss, anchoring) —
   any work that anchors a learned gate with a training-free signal? (also serves
   P30 router defense)
Verdict: nearest 3 works + exact unoccupied combination statement for SDC.
```

## Track 7 — Class-token 디코더 on fused memory features (P30 방어)

```
Novelty check for P30's class-token decoder: learnable per-class query tokens
cross-attending to SAM2 MULTIMODAL MEMORY features (post-RBMA fused), decoded to
masks — vs MaskFormer/Mask2Former on a single-modality backbone.

Must answer:
1. Query/mask-token decoders on top of FUSED multimodal features: any Mask2Former/
   MaskDINO variant whose pixel decoder consumes multi-sensor fused features with
   explicit fusion module before queries? (CAFuser uses OneFormer — where does
   fusion happen relative to queries? exact answer needed)
2. Class/query tokens interacting with SAM/SAM2 features: SEEM, Semantic-SAM,
   OMG-Seg, SAM-based semantic heads — do any attach class queries to SAM2 memory
   or multimodal tokens?
3. Rare-class collapse remedies in multimodal seg (our P28 failure mode): class-
   balanced losses vs query-based decoders — evidence that query decoders help
   rare classes (ADE20K/DELIVER tail classes).
4. High-resolution query decoding on single-scale ViT (SAM3/PE): ViTDet-style
   simple FPN + query head precedents.
Verdict: nearest 3 works + unoccupied-combination statement for
"class-token decoding on reliability-biased multimodal memory features".
```

## Track 8 — 2026 최신 스윕 & RBMA 셀 재점검 (경보 트랙)

```
Fresh-sweep arXiv 2026-01 → 2026-07 (cs.CV) for anything that could scoop RBMA/P29/P30.

Queries (run all, newest-first):
- "multimodal segmentation" + (reliability | uncertainty | condition | adverse)
- "SAM2 memory attention" / "memory attention fusion" / "modality as frame"
- "attention logit bias" / "additive attention bias" / (ALiBi + multimodal)
- "training-free uncertainty fusion", "entropy-guided fusion"
- "LoRA expert routing" + (condition | weather | domain)
- DELIVER / MUSES / MCubeS leaderboard新 entries; MULTIAQUA citations
- SAM3 / Perception Encoder + segmentation adaptation
For each hit: 3-line triage (mechanism class, dataset+number, threat level to
RBMA/SDC/P30: HIGH/MED/LOW). HIGH threats get a full verified note.
Also: check MemorySAM's citing papers (Semantic Scholar/Google Scholar citations)
one by one — anyone modifying its memory attention?
Deliverable: threat-triage table + full notes for HIGH items only.
```

---

## 결과 배치 규칙

- Track 1 → `relatedworks/5x_vfm_multimodal_landscape*` (신규 5x 번호대)
- Track 2 → 기존 `40/42` 갱신 + 신규 경쟁자는 `4x` 추가
- Track 3 → `09_benchmark_tables_deliver_muses_mcubes.md` 갱신 (SOTA 표 확정판)
- Track 4 → `43_a_signal_entropy_priorart.md` (신규)
- Track 5 → `1x` 검출 번호대 갱신 + `15_condition_adaptive_detection.md` (신규)
- Track 6 → `50_moe_lora_condition_routing.md` (신규, P29 방어)
- Track 7 → `51_class_token_fused_memory_decoder.md` (신규, P30 방어)
- Track 8 → `sources/08_threat_watch_2026H2.md` (triage 표)
- 완료 후 drone-MemorySAM `.claude_logs/12_novelty_and_related_work.md` §4 TODO 체크오프 + 이 볼트 `00_relatedworks_index.md` 갱신 (양방향 동기화)

---

## 2026-07-02 완료 기록

8개 트랙 전부 완료 (adversarial verification 포함). drone-MemorySAM `.claude_logs/12_novelty_and_related_work.md` §4 TODO 체크오프는 해당 파일이 다른 머신(drone)에 있어 **미수행 — 아래 기록 기반으로 drone 쪽에서 별도 동기화 필요**. 이 볼트 `00_relatedworks_index.md` 갱신은 완료.

| Track | 상태 | 결과 위치 | 핵심 결론 (1줄) |
|---|---|---|---|
| 1 (VFM 지형) | ✅ | `relatedworks/52`–`57` (신규 6개) | SAM4D = 두 번째 SAM2-memory-attention 점유자(promptable) → "MemorySAM only"는 multimodal *semantic* seg로 한정; StitchFusion 68.18 = clean-val 선두(70.3 Swin-T 미검증), SAM3 multi-scale neck은 기발표(SAM3-UNet) |
| 2 (reliability fusion 경쟁자) | ✅ | `40`·`42` 갱신 + 신규 `44`(HyperDUM)·`45`(SAE)·`46_attention_reweighting_detection_nearmisses`·`47`(2025–26 신규 진입자) | logit-bias 셀 "unrefuted but uncertain"으로 하향; universal negative 금지 — "the methods we examined" 헤지 필수 |
| 3 (벤치마크 프로토콜) | ✅ | `09` 갱신(§U1–U9) + 신규 `46_benchmark_protocol_split_resolution` | two-cluster = split×backbone (66.30=B2-val / 53.0=B2-test / 59.18=B0-val); 목표선: val 70.34 / test 57.35 / MUSES 81.07 mIoU · 61.03 PQ. 잔여: CMNeXt Tab.2 / HyperDUM Tab.4 시각 PDF 재확인 |
| 4 (A-신호 선행) | ✅ | 신규 `43_a_signal_entropy_priorart` | coarse 셀 PARTIALLY-OCCUPIED (UNO ICRA'20 필수 인용); 정확 셀(training-free entropy → additive attention-logit bias, multimodal dense seg) UNOCCUPIED — "unfalsified, not proven" 헤지 |
| 5 (검출 확장) | ✅ | 신규 `15_condition_adaptive_detection` + `14` 갱신 | detection 도메인에서 해당 셀 미점유("to our knowledge"); 최소 실험 = MEFormer에 inference-only λ·B 주입 (nuScenes-C) |
| 6 (P29 MoE-LoRA 라우팅) | ✅ | 신규 `50_moe_lora_condition_routing` | P29 셀 UNOCCUPIED (최근접: MoCLE/AW-MoE/MoFME); 단 DAMP·MLE-SAM 근접 점유 → scoped claim wording만 허용 |
| 7 (P30 class-token 디코더) | ✅ | 신규 `51_class_token_fused_memory_decoder` | broad 셀 점유(CAFuser/DGFusion/BiXFormer/DF2RQ); 정확 셀 외부 미점유 — 단 stock SAM2 디코더와의 gray zone → 명시적 차별화 + ablation 의무 |
| 8 (위협 감시) | ✅ | `sources/08_threat_watch_2026H2` + 신규 `60`(PRIMED)·`58`(SAE)·`59`(BiXFormer)·`48`(M⁴-SAM), `42` 추가 갱신 | **REFUTED ×2**: logit-bias 셀 점유(PRIMED+SAE) → RBMA novelty = training-free entropy × SAM2-memory-attention × RGB-X seg로 축소; "first multimodal SAM3"는 SAMCM-SR에 의해 폐기. 제출 전 필독: PRIMED, SAE, ICRCV underwater |

번호 충돌 주의: `46_`이 2개 (Track 2 `46_attention_reweighting_detection_nearmisses` vs Track 3 `46_benchmark_protocol_split_resolution`), Track 8 HIGH 노트는 45–47 충돌 회피로 58/59/60/48로 재번호.

drone 쪽 §4 TODO 매핑 (체크오프용): TODO 1b→Track 3 완료, TODO 2→Track 4 완료, TODO 3→Track 2 완료(DGFusion), TODO 4→Track 2 완료(CAFuser CA²/CAA 메커니즘 확정), TODO 5/6→Track 6 완료, TODO 7/8→Track 7 완료, det 확장(doc 12 §2.6)→Track 5 완료, 스쿠프 경보→Track 8 완료(MemorySAM 13개 인용 논문 전수 확인, memory attention 변경 없음).
