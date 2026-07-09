---
title: Condition-Adaptive Multimodal Fusion in Object Detection + VFM Usage — Track 5 synthesis (det-extension justification for RBMA/P29/P30)
tags: [related-work, synthesis-note, multimodal-object-detection, condition-adaptive, reliability, adverse-weather, rgb-t, vfm, sam2, rbma, track5]
created: 2026-07-02
source: parallel deep-research Track 5 (sources/07_parallel_research_prompts_2026-07-02.md) + adversarial verification (skeptic1/skeptic2, 2026-07-02)
status: verified-draft
---

# Condition-Adaptive Multimodal Fusion in OBJECT DETECTION + VFM usage — Track 5 synthesis

범위: 2D & 3D/BEV 멀티모달 검출에서 명시적 reliability/condition 처리 (2024–2026-07), 멀티모달 검출에서의 VFM 주입 방식, maritime/drone 도메인, 그리고 "reliability → additive pre-softmax attention-logit bias" 셀의 detection-도메인 novelty check.

검증 태그: [VERIFIED-PDF] = arXiv HTML/abs 원문에서 수식·표 직접 확인, [ABSTRACT-ONLY] = 초록/랜딩 페이지만, [UNVERIFIED-BLOG] = 3자 요약. 핵심 주장은 2026-07-02 adversarial verification(skeptic1/skeptic2, 원문 재확인 + 반증 검색)을 거쳤고, 반박·불확실 판정은 아래에 그대로 반영했다.

Wikilinks: [[relatedworks/14_multimodal_detection_survey_note]], [[relatedworks/42_attention_logit_bias_novelty_defense]], [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]], [[relatedworks/10_bevfusion_relatedwork]], [[relatedworks/11_transfusion_relatedwork]], [[relatedworks/13_futr3d_relatedwork]]

---

## Problem setting

멀티모달 검출(camera-LiDAR-radar 3D/BEV, RGB-T 2D)에서 센서 신뢰도는 조건(비/눈/안개/야간, 센서 고장/드롭)에 따라 급변한다. 기존 검출 fusion(BEVFusion, TransFusion, CMT 계열)은 신뢰도를 암묵적으로만 다룬다. 질문 다섯 가지:

1. adverse-weather-adaptive 3D 검출에서 명시적 reliability/condition 처리는 어디까지 왔나 (ReliFusion 이후)?
2. RGB-T 검출의 illumination-adaptive gating은 어떤 mechanism-class인가?
3. VFM(SAM/SAM2/DINOv2)은 멀티모달 검출에 어떻게 주입되는가?
4. **검출 도메인에서 attention logits를 sensor reliability로 bias하는 선행이 존재하는가?** (RBMA det-확장의 novelty cell)
5. maritime/drone 도메인(MULTIAQUA/MaCVi)의 현황은?

## Novelty (검증 결과 요약 — adversarial verification 반영)

핵심 셀 **"sensor reliability/uncertainty/condition 신호 → ADDITIVE PRE-SOFTMAX attention-logit bias, in detection"** 은 2026-07 기준 **미점유로 판단되나, 아래 두 가지 단서를 반드시 달고 주장해야 한다**:

- **[확정된 사실]** 검출 디코더에서 additive pre-softmax bias 자체는 존재한다 — MEFormer (2407.19156, Eq.9)가 attention logits에 `M_{i,j} = α·||φ̂_{LC,i} − φ̂_{A,j}|| + β` (BEV box-center 거리, **geometric** 신호)를 더한다 [VERIFIED-PDF, skeptic1·2 모두 원문 재확인]. 신호가 reliability가 아니므로 우리 셀은 비어 있다.
- **[반증 검증에서 나온 counter-example — 표현 수위 조정 필요]** "검출 디코더의 pre-softmax additive bias는 MEFormer가 유일" 이라는 원래 문구는 **과도하게 강한 표현으로 반박됨**: **SMCA-DETR (arXiv:2101.07448, ICCV 2021)** 이 DETR 디코더 logits에 log-Gaussian spatial prior를 pre-softmax로 더한다 (skeptic2 확인). 단 이 역시 geometric·single-modality 신호이므로 **reliability-신호 셀 자체는 여전히 미점유**. 논문에서는 "additive logit bias는 검출 디코더에서 이미 수용된 machinery (SMCA-DETR, MEFormer)이나, 그 신호가 reliability인 사례는 없다"로 서술할 것.
- **[미해소 near-neighbor — 정직하게 명시]** **SeBFusion/BCAF** (MDPI Applied Sciences 2026-03, doi:10.3390/app16062943): LiDAR-camera adverse-weather 3D 검출 cross-attention에서 "estimates the confidence of each modality and adaptively reweights the bidirectional information flow" — **원문 403 차단으로 additive vs multiplicative 미확인** (skeptic1: uncertain). 문구상 learned multiplicative reweighting으로 추정되고, 어쨌든 **learned이므로 training-free 셀은 무관하게 생존**하지만, 엄밀한 universal negative("additive pre-softmax reliability bias는 검출에 없다")는 이 논문을 확인하기 전까지 "to our knowledge" 수위로만 주장해야 한다.

reliability **신호**를 쓰는 검출 선행(ReliFusion, ModalPatch)은 전부 **learned + multiplicative post-softmax**로 확인됨(아래 Method). 따라서 정확한 조합 "training-free entropy-derived reliability → additive pre-softmax logit bias"의 검출 선행 = **0건 (search-supported negative; 전수 증명 아님)**.

부속 novelty 셀들 (모두 skeptic 확인):
- **P29 SDC (unsupervised condition routing)**: 검출의 condition-routing은 전부 supervised — AW-MoE(2603.16261)는 Stage 2에서 GT weather label로 classifier 학습(K-Radar 7-class, ~99% 라우팅 정확도, FiLM도 prototype도 아님; label-free는 inference에서만), WCBR(2604.05405)은 "weather-supervised learning strategy" 명시. Unsupervised condition-latent routing은 검출에서 미발견 (최근접 label-free router WM-MoE 2303.13739은 weather-removal restoration, 검출 아님). **단, universal-absence는 반증 실패 기반이지 전수 증명이 아님을 명시.**
- **RBMA base (SAM2 memory attention 기반 fusion)**: 검출(bbox)에서 SAM2 memory attention을 fusion에 쓰는 선행 없음. 최근접은 salient object detection(=segmentation)인 M4-SAM(2605.11760, RGB-D 융합은 memory 이전 encoder에서 발생, memory attention은 순수 temporal)과 SAM-DAQ(2511.09870, RGB-D VSOD). VFM 사용은 encoder-feature fusion(RoboFusion, IJCAI 2024)과 distillation(2510.10287 DINOv2→BEV, 2605.10130 Thermal-Det) 수준 — 단 "limited to" 열거는 불완전(예: SAM-Guided Semantic Knowledge Fusion for VI det — 역시 encoder-feature-level이라 셀은 유지됨).

## Method (per-paper, 수식 포함)

### 1. ReliFusion (arXiv:2502.01856) — reliability 신호, multiplicative post-softmax [VERIFIED-PDF]
- venue: arXiv 2025-02 + Canadian AI conf. (비상위). 코드 미발견.
- Reliability module = CMCL(Cross-Modality Contrastive Learning): clean vs corrupted pair를 contrastive로 분리, confidence는 **learned sigmoid head**: `C_LiDAR = σ(W_L·z_L + b_L)`, `C_Cam = σ(W_C·z_C + b_C)` (Eq.12).
- CW-MCA: confidence가 **post-softmax attention output에 곱해짐** (Eq.13–15, skeptic1·2 원문 재확인):
  - `F_{L→C} = C_LiDAR · softmax(Q_C K_L^T/√d_k) V_L`
  - `F_{C→L} = C_Camera · softmax(Q_L K_C^T/√d_k) V_C`, `F_fused = F_{L→C} + F_{C→L}`
- mechanism-class: **learned-gate → output-scale (multiplicative, post-softmax)**. logits에는 아무것도 더하지 않음.

### 2. MEFormer (arXiv:2407.19156) — additive pre-softmax bias, geometric 신호 [VERIFIED-PDF]
- venue 미확정 (README/abs 무언급 — HTML 내 "In ICCV" 문자열은 참고문헌임. **인용 전 확인 필수**).
- MOAD(modality-agnostic decoding): 공유 디코더가 LiDAR-camera/LiDAR-only/camera-only 예측을 모두 생성. PME(Proximity-based Modality Ensemble): **attention logits에 additive bias** (Eq.9):
  - `M_{i,j} = α · ||φ̂_{LC,i} − φ̂_{A,j}|| + β` — LC-branch 박스중심과 modality-agnostic branch 박스중심의 BEV 거리.
  - 원문 verbatim: "we add attention bias M to the attention logit and apply the softmax function".
- mechanism-class: **logit-additive-bias (geometric proximity)** — 기구는 우리와 동일, 신호가 다름. RBMA det-확장의 최고의 "feasibility precedent + non-occupancy evidence" 쌍.

### 3. ModalPatch (arXiv:2603.02481) — learned variance, multiplicative post-softmax mask [VERIFIED-PDF]
- arXiv 2026-03, 코드 github.com/Castiel-Lee/MM3Det_MD. "first plug-and-play solution … arbitrary modality drop" (verbatim).
- history 기반 temporal compensation + uncertainty = **learned MLP variance head**: `σ²_M = MLP(F̂^{t0}_M)` (Eq.4), **NLL loss로 학습** (Eq.5). 2-stage 학습 필요 → **not training-free** (skeptic1·2 확인).
- 주입 (Eq.7): deformable attention weights에 **multiplicative post-softmax mask**: `W̃ = W · [1 − softmax(U^{t0}_pts)]`.
- mechanism-class: learned-gate → attention-weight multiply (post-softmax), 공간 dense. 대상은 modality **drop**(이진 부재)이지 graded degradation이 아님.

### 4. AW-MoE (arXiv:2603.16261) — supervised weather routing [VERIFIED via skeptic; 본문 일부 확인]
- IWR: 이미지 특징으로 weather **classification** — Algorithm 1 Stage 2에서 **GT weather label로 배치마다 classifier 학습** (K-Radar 7-class, Table I ~99% cls acc; skeptic1·2 원문 확인). top-K Weather-Specific Experts 라우팅 + UDMA 증강. mechanism-class: **supervised condition-token → learned-gate (expert routing)**.

### 5. Weather-Conditioned Branch Routing (arXiv:2604.05405) [ABSTRACT-ONLY]
- LiDAR / 4D-radar / condition-gated fusion 3분기 + visual·semantic prompt에서 뽑은 condition token으로 lightweight router가 sample별 soft weight 예측. 붕괴 방지 = "weather-supervised learning strategy with auxiliary classification and diversity regularization" (abstract verbatim). mechanism-class: condition-token → branch-level soft aggregation, **weather supervision 필수**. P30 router-collapse 논의에도 관련 (그들: supervision+diversity reg / 우리: training-free RBMA reliability anchoring).

### 6. L4DR (arXiv:2408.03677, AAAI 2025 Oral) [ABSTRACT-ONLY]
- LiDAR-4D radar: MME + Foreground-Aware Denoising(최초 early-fusion), IM2 병렬 백본 + Multi-Scale Gated Fusion(MSGF). mechanism-class: implicit learned-gate (feature-multiply, multi-scale). reliability 신호 비노출.

### 7. SAMFusion (arXiv HTML 2508.16408; project page = ECCV 2024 표기 — 인용 전 venue 확인) [VERIFIED-PDF]
- RGB + gated NIR + LiDAR + radar, 공유 BEV에서 Cross-Modal Adaptive Blending: **거리-의존 learned Gaussian weighting**이 range별 LiDAR-vs-radar 기여를 조절, gated-camera feature는 additive 결합. mechanism-class: learned-gate + hand-designed range prior (feature level). 명시적 reliability estimate 없음.

### 8. RGB-T 검출 (day/night)
- **IAF R-CNN** (1803.05347, PR 2019) / **IATDNN+IAMSS** (1802.09972, Inf. Fusion 2019) / **CCIFNet** (2024): illumination 추정치가 color/thermal 스트림의 **최종 detection confidence를 gating** — mechanism-class: condition-signal → **output-scale (score-level)** [ABSTRACT-ONLY / UNVERIFIED-BLOG].
- **Modality-Decoupled RGB-T via Query Fusion** (2601.08458) [VERIFIED-PDF]: DETR식 dual-branch, 디코더 layer마다 **top-k confidence-selected query** 교차 주입: Eq.(6) `Q_fused_rgb = [Q_rgb, Ψ_RGB(Q_tir)]`. illumination/reliability gating 없음 — query-level selection/concat.
- **TFDet** (2305.16580): target-aware feature enhancement, illumination gating 아님 [ABSTRACT-ONLY].
- 패턴: RGB-T 검출 적응성 = (a) score-level illumination gate (2018-19 계보), (b) confidence 기반 query/feature selection (2026). **attention logits를 illumination/reliability로 bias하는 사례 없음.**

### 9. VFM in multimodal detection
- **RoboFusion** (2401.03907, **IJCAI 2024** proc.141 — venue verified): SAM-AD(robust image encoder) + AD-FPN + wavelet denoising + self-attention re-weighting. 주입 class: **VFM-as-robust-encoder + feature fusion**. SAM2 memory 미사용 (skeptic2 원문 확인). KITTI-C/nuScenes-C 수치는 3자 요약 기반 — 인용 전 PDF 확인.
- **Bridging Perspectives** (2510.10287): DINOv2 → BEV **distillation-time only** (skeptic2 확인: SAM2 없음).
- **Thermal-Det** (2605.10130): frozen RGB teacher cross-modal distillation, open-vocab thermal det (skeptic2 확인: SAM2 없음). "CVPR 2026" 표기는 시점상 의심 — 확인 전 인용 금지.
- **SAM-Guided Semantic Knowledge Fusion (VI det)**: skeptic1 반증 검색에서 발견된 추가 사례 — 역시 SAM-guided **encoder-feature-level** fusion, memory attention 아님.
- **결론 (skeptic1·2 확인)**: bbox 검출에서 SAM2 memory attention을 fusion 기구로 쓰는 선행 없음. 최근접 MemorySAM(2503.06700)·M4-SAM은 segmentation 도메인.

### 10. Maritime/drone
- **MULTIAQUA** (2512.17450): 동기화·캘리브레이션된 RGB+thermal+LiDAR maritime dataset, day/night subset [ABSTRACT-ONLY]. **MaCVi @ CVPR 2026** multimodal semantic seg challenge: MULTIAQUA low-light subset, 4-class, metric M = (val mIoU + **nighttime test mIoU**)/2, 3개 모달 전부 소비 필수 [VERIFIED page] — RBMA의 자연스러운 실전 진입점 (night = condition).
- M2SODAI(NeurIPS'23 D&B, RGB+hyperspectral det), MODS(T-ITS, RGB-stereo) [ABSTRACT-ONLY]. **maritime 멀티모달 검출에서 condition-adaptive fusion 선행 미발견** — 도메인 셀도 열려 있음.

## Quantitative results (verbatim rows)

| Paper | Dataset/split | Metric | Value | Tag |
|---|---|---|---|---|
| ReliFusion | nuScenes [test], Table 1 | mAP/NDS | **70.6 / 73.2** (CMT 70.4/73.0; BEVFusion 69.2/71.8; TransFusion 68.9/71.7) | [VERIFIED-PDF] [test] |
| MEFormer | nuScenes [val] | NDS/mAP | **73.9 / 71.5** (w/o PME 73.7/71.3; FPS 3.1 A6000) | [VERIFIED-PDF] [val] |
| MEFormer (robustness, Table 2) | nuScenes [val], sensor missing | NDS/mAP | LiDAR-only 69.5/63.6; camera-only 48.0/42.5 | [VERIFIED-PDF] [val] |
| MEFormer (Table 3) | nuScenes [val], 4-beam LiDAR | NDS/mAP | 63.4 / 55.9 | [VERIFIED-PDF] [val] |
| ModalPatch (Table I) | nuScenes [val], 50% independent modality drop | mAP/NDS | UniBEV 35.49/50.65 → **46.32/56.91**; BEVFusion 27.74/50.37 → 31.43/50.60; CMT 27.21/49.92 → **44.21/56.89**; MEFormer 27.88/50.64 → **44.11/57.36** | [VERIFIED-PDF] [val] |
| SAMFusion (Tables 1,3) | SeeingThroughFog, Pedestrian 50–80 m, 3D AP | AP | Day **40.16** (DeepInteraction 28.55); Night **27.14** (20.53); Fog **34.31** (+17.2 over next); Snow **41.45** (+15.62) | [VERIFIED-PDF] [unknown split] |
| AW-MoE | "real-world dataset" (K-Radar로 추정) | adverse-weather perf. | "~15% improvement over SOTA" | [ABSTRACT-ONLY] [unknown] |
| WCBR (2604.05405) | K-Radar | — | "state-of-the-art" (수치 없음) | [ABSTRACT-ONLY] [unknown] |
| L4DR | VoD + simulated fog | 3D mAP | "up to **+20.0%** over LiDAR-only" | [ABSTRACT-ONLY] [unknown] |
| Query-Fusion RGB-T (Table 1) | FLIR / M3FD | mAP, mAP50 | FLIR **43.8**/83.1; M3FD **55.9**/90.4 | [VERIFIED-PDF] [unknown split] |
| TFDet | KAIST, LLVIP / FLIR, M3FD | vs prev best | +0.65%, +4.1% / +2.2%, +1.9% | [ABSTRACT-ONLY] [unknown] |
| RoboFusion-L | KITTI-C snow sev.1→5 | AP | 86.69→83.67 (LoGoNet 55.07→45.02) | [UNVERIFIED-BLOG — IJCAI PDF 확인 필요] [unknown] |

## Detection-side mechanism-class taxonomy (signal × injection; adversarial-verification 반영판)

| Signal ↓ / Injection → | output/score-scale | feature-multiply / gate | branch-/expert-routing | query selection | attn-weight multiply (post-softmax) | **attn-logit additive (pre-softmax)** |
|---|---|---|---|---|---|---|
| illumination estimate (learned) | IAF R-CNN, IATDNN, CCIFNet | — | — | — | — | — |
| supervised weather class | — | AW-MoE (WSE feat.) | AW-MoE, WCBR 2604.05405 | — | — | — |
| learned confidence (contrastive) | — | — | — | — | ReliFusion (output-multiply) | — |
| learned variance (NLL) | — | — | — | — | ModalPatch | — |
| learned confidence (미확인 주입점) | — | SeBFusion/BCAF (403-blocked, **watch**) | — | — | ? | ? (추정상 multiplicative) |
| learned degradation divergence | — | UP-Fuse 2602.19349 (panoptic, 주입점 불명, **watch**) | — | — | ? | — |
| proposal confidence | — | — | — | Query-Fusion RGB-T | — | — |
| geometric proximity / spatial prior | — | — | — | — | — | **MEFormer** (box-center dist.), **SMCA-DETR** (log-Gaussian, single-modality) |
| range prior (Gaussian) | — | SAMFusion | — | — | — | — |
| implicit (learned gate) | — | L4DR MSGF | — | — | — | — |
| **training-free predictive entropy (ours)** | — | — | — | — | — | **RBMA — 검출에서 미점유 (search-supported)** |

## Limitations

- **Universal negative의 한계**: 미점유 판정은 반증 검색 실패에 기반하며 전수 증명이 아님. 특히 SeBFusion/BCAF(app16062943) 전문 미확인 → 논문에는 "to our knowledge" 수위 + SMCA-DETR·MEFormer를 mechanism-precedent로 선인용하는 방어적 서술 필요.
- venue 미확정 다수: MEFormer(ICCV 표기 금지), SAMFusion(ECCV 2024?), Thermal-Det(CVPR 2026 표기 의심), AW-MoE/WCBR(preprint).
- 수치 공백: AW-MoE/WCBR K-Radar 표 미추출, RoboFusion corruption 표는 3자 요약, UP-Fuse 주입점 미해소 (MED threat — full text 필요).
- ReliFusion의 CMT 대비 이득은 +0.2 mAP로 근소 — "reliability 신호가 검출에 유효" 근거로는 ModalPatch(+10~17 mAP under drop)와 SAMFusion(fog +17.2 AP)이 더 강함.

## Improvement directions (det-확장 설계 제안)

1. **B_i 신호원**: 검출엔 per-modality dense class decode가 없음. 후보 — (a) BEV center/class heatmap의 softmax entropy, (b) **MOAD-style modality-agnostic decoding의 per-modality query classification logits (가장 깔끔 — MEFormer가 공짜로 제공)**, (c) 경량 seg-style auxiliary head.
2. **공간 해상도 상향**: keys가 사는 공간(BEV cell)별 entropy → spatially varying bias — seg-side의 global per-modality bias보다 오히려 풍부한 신호.
3. **Hungarian matching 상호작용**: λ·B가 초기 학습에서 matching을 불안정화할 수 있음 → **inference-only training-free 주입을 1차 실험으로** (seg 결과와 대칭).
4. **Sparse 모달(radar) 보정**: 본질적 고엔트로피 decode → /log C 정규화 + per-sensor temperature calibration.
5. **벤치마크**: nuScenes-C/K-Radar/STF에서 CMT·MEFormer에 plug-in; maritime은 MaCVi'26 MULTIAQUA(현재 seg 트랙)로 condition-adaptive 스토리 실증.

## Comparison to RBMA-P29-P30 (mechanism-class)

| Ours | 최근접 det 선행 | 그들의 class | 우리 class | 차별점 (3축) |
|---|---|---|---|---|
| **RBMA** `softmax(QK^T/√d + λ·B)V` | ReliFusion / ModalPatch (신호 겹침), MEFormer/SMCA-DETR (기구 겹침) | learned-gate → multiplicative post-softmax / geometric logit-additive-bias | **training-free entropy → logit-additive-bias (pre-softmax)** | (i) additive pre-softmax vs multiplicative post-softmax, (ii) training-free entropy vs learned head(contrastive/NLL), (iii) graded per-condition reliability vs drop-보상·geometry |
| **P29 SDC** | AW-MoE, WCBR | supervised weather cls → expert/branch routing (learned-gate) | **unsupervised condition prototype → FiLM on Soft-MoE LoRA router** | label-free 학습 (그들은 inference만 label-free); FiLM·prototype 기반은 검출에 부재 |
| **P30** | WCBR (diversity reg로 collapse 방지) | supervision + regularization | **training-free reliability anchoring of learned router** | anchor 신호가 GT-free·training-free |
| **RBMA base (SAM2 memory fusion)** | RoboFusion (encoder), 2510.10287/Thermal-Det (distill), M4-SAM (seg-saliency) | VFM-encoder-feature / distillation | **modalities-as-frames + memory attention fusion** | 검출에서 SAM2 memory attention fusion 선행 0건 |

## Application to ours (RBMA/P29/P30 적용방향)

1. **det-확장 정당화 논리 (paper §det-extension)**: "기구는 이미 수용됨(SMCA-DETR, MEFormer의 pre-softmax additive bias) + 신호는 이미 유효함(ReliFusion·ModalPatch의 reliability 이득) + 조합은 미점유" — 이 3단 논법을 그대로 사용. MEFormer의 MOAD가 per-modality 예측을 공짜로 제공하므로 **MEFormer + RBMA-bias가 최소변경 실험 설계**.
2. **1차 실험**: 학습 없이 inference-time에만 λ·B 주입 (CMT 또는 MEFormer 체크포인트, nuScenes-C corruption suite) — seg-side training-free 결과와 대칭 구도.
3. **위협 관리**: ModalPatch(2026-03)가 최근접 위협 — 차별 3축(additive/pre-softmax, training-free, graded)을 실험표에 명시적 대조군으로. SeBFusion/BCAF·UP-Fuse는 camera-ready 전 full-text 확인 (watch-list 등재).
4. **P29/P30**: AW-MoE·WCBR를 "supervised routing의 한계" 인용 앵커로; 우리 router-collapse 방어는 diversity-reg가 아닌 training-free anchoring이라는 대비.
5. **도메인 스토리**: MaCVi'26 (MULTIAQUA night) 참가로 condition-adaptive 주장의 외부 검증 확보; maritime det 셀은 후속 논문 여지.

## Related-work paragraph candidate (English)

> Reliability- and condition-aware fusion has recently entered multimodal detection, but through mechanisms distinct from ours. ReliFusion learns per-modality confidence scores via contrastive pretraining and multiplies them onto the post-softmax cross-attention output (F = C·softmax(QK^T/√d)V), while ModalPatch trains an MLP variance head with an NLL loss and applies it as a multiplicative post-softmax mask on deformable-attention weights (W̃ = W·[1−softmax(U)]); both therefore learn the reliability signal and inject it multiplicatively after the softmax. Condition-routed detectors such as AW-MoE and weather-conditioned branch routing rely on supervised weather classification to drive expert or branch selection, and classic RGB-T detectors (IAF R-CNN, IATDNN) gate final detection scores by an estimated illumination measure. Additive pre-softmax attention biases do exist in detection decoders — SMCA-DETR adds log-Gaussian spatial priors and MEFormer adds a box-center proximity bias M = α·dist + β to the attention logits — but in all published cases the biasing signal is geometric, not sensor reliability. To our knowledge, no detector injects a training-free, entropy-derived per-modality reliability as an additive pre-softmax attention-logit bias, which is precisely the mechanism RBMA transfers from segmentation.

## References (arXiv IDs)

- ReliFusion 2502.01856 · MEFormer 2407.19156 · ModalPatch 2603.02481 · AW-MoE 2603.16261 · WCBR 2604.05405 · L4DR 2408.03677 (AAAI'25 Oral) · SAMFusion 2508.16408 · UP-Fuse 2602.19349 · SMCA-DETR 2101.07448 (ICCV'21) · SeBFusion/BCAF doi:10.3390/app16062943
- RGB-T: IAF R-CNN 1803.05347 · IATDNN 1802.09972 · TFDet 2305.16580 · Query-Fusion 2601.08458
- VFM: RoboFusion 2401.03907 (IJCAI'24) · Bridging Perspectives 2510.10287 · Thermal-Det 2605.10130 · M4-SAM 2605.11760 · SAM-DAQ 2511.09870
- Maritime: MULTIAQUA 2512.17450 · M2SODAI (NeurIPS'23 D&B) · MaCVi macvi.org
