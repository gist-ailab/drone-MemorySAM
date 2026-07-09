---
title: Reliability/Condition-Adaptive Fusion — 2025–2026 New Entrants Sweep (RSGMamba, UP-Fuse, AW-MoE, GeomPrompt, EQUISeg, MULTIAQUA, READ, AECF, SGMA)
tags: [related-work, sweep, reliability, condition-aware, multimodal-segmentation, rbma, p29, p30, threat-triage]
created: 2026-07-02
source: Track 2 deep-research 2026-07-02 (arXiv sweep 2025-06→2026-07) + adversarial verification additions (skeptic1/skeptic2)
status: verified-draft
---

# 2025–2026 신규 진입자 스윕 — 신뢰도/조건 적응형 fusion

Track 2 item 4의 결과. 개별 major 경쟁자(HyperDUM=44, SAE=45, ModalPatch/ReliFusion/SAM2Long=46)는 별도 노트; 여기는 나머지 신규 진입자 + adversarial 검증에서 추가 발견된 near-miss들. 통합 taxonomy는 [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]] 2026-07-02 update 참조.

## MED threats (기재 순 = 위협도 순)

### RSGMamba (arXiv 2604.12319, v2 2026-04-15, preprint) [VERIFIED-PDF]
- **Reliability-aware Self-Gated Mamba**: uncertainty gate `g_u = σ(G_u(f))` (LayerNorm+MLP, **learned** — entropy 아님; skeptic2가 원문에서 재확인) + consistency gate `g_c = σ(G_c([f^rgb, f^x, |f^rgb−f^x|]))`; **SSM effective C(state-readout) matrix에 곱셈** 적용: `C_eff^rgb = g_u^rgb·(1−g_c)·C^rgb`, `C_eff^x = g_u^x·g_c·C^x`.
- 수치 [VERIFIED-PDF]: RSGMamba-B (SegMAN-B, 48.6M) NYUDv2 **58.8** / SUN-RGBD **54.0** mIoU (vs DFormerV2-L 58.4/53.3); MFNet **61.1**; PST900 **88.9**. [unknown split → 관행상 표준 test지만 원문 표기 재확인 요] **DELIVER/MUSES 없음, adverse-weather 평가 없음.** Code URL 없음.
- 위협: "reliability-aware" 네이밍이 겹치나 mechanism = learned gates on SSM readout (Mamba 계열). Training-free 신호도, attention-logit도 아님.

### UP-Fuse (arXiv 2602.19349, 2026-02-22) [ABSTRACT-ONLY]
- Uncertainty-guided LiDAR-camera fusion, 3D **panoptic** seg (2D range view); uncertainty maps는 "diverse visual degradations 하 representational divergence"로 **학습**; **feature-multiply** 방식 cross-modal modulation. Panoptic nuScenes / SemanticKITTI / Panoptic Waymo. Abstract에 수치 없음, code 없음.
- 위협 MED (uncertainty-guided *panoptic seg*), mechanism class는 별개.

### AW-MoE (arXiv 2603.16261, 2026-03-17) [ABSTRACT-ONLY]
- All-Weather MoE, 멀티모달 3D det: Image-guided Weather-aware Routing (**supervised** weather discrimination) → top-K Weather-Specific Experts; LiDAR+4D radar dual-modal augmentation. "~15% adverse-weather improvement". Code: github.com/windlinsherlock/AW-MoE (미공개).
- **P29 SDC 함의:** routing이 supervised weather classification → "unsupervised condition-prototype → FiLM-router" 셀은 여전히 비어 있는 것으로 관찰됨 (단정 불가 — Track 6에서 확정).

### EQUISeg (arXiv 2509.24505, 2025-09-29) [ABSTRACT-ONLY]
- "Balanced Modality Contributions": 4-stage Cross-modal Transformer Block + Self-guided Module (상호 유도 gating). Abstract에 uncertainty/reliability 신호 명명 없음, 수치/데이터셋 불명. 위협 LOW-MED — **Track 8에서 전문 정독 필요** (gate 세부 미확인).

## Adversarial 검증에서 추가된 near-miss (skeptic 발견 — 인용 후보)

### READ (ICLR 2024, OpenReview TPZRq4FALB; github XLearning-SCU) [skeptic1 발견, 미정독]
- Test-time-**learned** attention-layer modulation으로 'reliability bias' 대응 — audio-visual **classification**. Test-time adaptation 계열이므로 Track 4(A-신호)와도 교차. 정독 후 42번 노트 near-miss 표에 편입 여부 결정.

### AECF (arXiv 2505.15417) [skeptic1/2 확인: LEARNED]
- Entropy-gated fusion layer — 단 **learned** gate이고 classification. "entropy 기반이지만 training-free가 아닌" 대표 사례로 인용 가치.

### SGMA (arXiv 2603.02505) [ABSTRACT-ONLY]
- Remote sensing incomplete-modality: attention-derived reliability를 **sampling/training schedule**에 사용 (logit bias 아님, loss/schedule-level).

### ICCV'25 functional-entropy regularization (arXiv 2505.06635)
- Parameter-free이지만 **training-time loss** — inference-time reliability 주입 아님. (기존 41번 노트의 Reducing Unimodal Bias 라인과 동일 계열.)

## LOW threats / 비경쟁 (triage만)

| 항목 | 한 줄 판정 |
|---|---|
| GeomPrompt (2604.11585, CVPR'26 URVIS WS) [ABSTRACT-ONLY] | RGB→geometric prompt 합성으로 depth 결손 보상 (SUN RGB-D +6.1 mIoU DFormer 기준, 7.8ms) — **modality compensation, reliability weighting 아님**. LOW |
| MULTIAQUA (2512.17450) [ABSTRACT-ONLY] | 해양 멀티모달 데이터셋(RGB/thermal/IR/LiDAR) + daytime-only 학습→야간 강건 전략. Mechanism-level reliability fusion 주장 없음 — **우리 4번째 벤치마크 후보** (MaCVi @ CVPR 2026 challenge). |
| MdaIF (2511.12525) / CAWM-Mamba (2603.02560) / AWM-Fuse | adverse-weather image fusion/restoration (픽셀 출력) — seg fusion 아님 |
| SeBFusion (MDPI Appl. Sci. 16(6):2943) [UNVERIFIED-BLOG] | 3D det confidence-aware bidirectional fusion, non-tier venue |
| UGG-ReID (2507.04638) | uncertainty-guided graph, ReID 도메인 |
| Frequency-Guided RGB-T seg (2605.26273), UAV RGBT benchmark (2604.26893) | 제목/스니펫에 reliability 주입 주장 없음 — Track 8 항목 |

## Limitations (이 스윕 자체의)

- ABSTRACT-ONLY 항목 다수 (UP-Fuse, AW-MoE, EQUISeg, GeomPrompt, MULTIAQUA, SGMA, READ) — 정량 인용 전 원문 필수.
- "모든 경쟁자가 learned 신호"라는 전칭 주장은 adversarial 검증에서 **uncertain** 판정 (UTFNet/ModalPatch 외 다수 미재검증) — 논문에서는 "the competitors we examined"로 한정할 것.

## Improvement directions

- EQUISeg/UP-Fuse 전문 정독 (Track 8), READ 정독 (Track 4 교차), MULTIAQUA baseline 표 확보 (Track 3).
- 2026 preprint들(RSGMamba, ModalPatch, UP-Fuse, AW-MoE) venue watch — 채택 시 비교표 갱신.

## Comparison to RBMA-P29-P30 (mechanism-class)

- 신규 진입자 전원이 learned-gate / feature-multiply / MoE-routing / loss-level — **logit-additive-bias 셀 점유자 없음** (단, 전칭 부정은 확증 불가; 42번 노트 verdict 참조).
- P29 방어: AW-MoE(supervised weather routing)가 최근접 — unsupervised 셀은 관찰상 빈칸.
- P30 방어: RSGMamba의 learned self-gate가 "router anchored by training-free signal"과 가장 가까우나 anchoring 개념 없음.

## Application to ours (RBMA/P29/P30 적용방향)

- RBMA related work의 "recent entrants" 각주/문단에 RSGMamba·UP-Fuse·EQUISeg를 한 문장씩 배치 (naming-collision 방지: "reliability-aware"가 이미 쓰인 이름임을 인지하고 우리 약어/명명 차별화).
- MULTIAQUA를 4번째 벤치마크로 확보하면 MaCVi challenge 가시성 + 해양 도메인 일반화 주장 가능.
- AECF/READ는 "entropy/reliability를 쓰되 학습이 필요한" 대조군으로 intro 한 문장 인용.

## Related-work paragraph candidate (English)

A wave of 2025–2026 work attaches reliability or condition signals to multimodal fusion through learned machinery: RSGMamba gates state-space readout matrices with MLP-derived uncertainty and consistency gates; UP-Fuse learns degradation-divergence uncertainty maps to modulate LiDAR-camera features for panoptic segmentation; AW-MoE routes weather-specific experts via supervised image-based weather discrimination; and entropy-gated fusion layers (AECF) and test-time attention modulation against reliability bias (READ) address classification settings. Across these entrants, the reliability signal is trained and injected at the feature, gate, routing, or schedule level; we found no method that derives a training-free reliability from per-modality predictive entropy or that biases cross-modal attention logits additively.

## Links

- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]]
- [[relatedworks/46_attention_reweighting_detection_nearmisses]]
