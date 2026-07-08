---
title: Attention-Reweighting Near-Misses — ModalPatch, ReliFusion, SAM2Long (multiplicative / post-softmax / output-scale)
tags: [related-work, novelty-defense, attention, reliability, 3d-detection, sam2-memory, rbma, near-miss]
created: 2026-07-02
source: Track 2 deep-research 2026-07-02; arXiv:2603.02481 [VERIFIED-PDF]; arXiv:2502.01856 [VERIFIED-PDF via pdftotext — skeptic could not independently re-verify mechanism, see caveat]; arXiv:2410.16268 [skeptic1 addition]
status: verified-draft
---

# Reliability × Attention 근접 사례 3건 (모두 additive pre-softmax가 아님)

RBMA novelty cell 방어에서 "reliability가 attention에 들어간다"는 반례 후보 3건. 세 건 모두 주입 방식이 **multiplicative(post-softmax) / output-scale / multiplicative key-scaling**으로 RBMA(additive pre-softmax)와 다르고, 도메인도 다름.

---

## 1. ModalPatch (arXiv 2603.02481, 2026-03-03; U. Alberta/U. Tokyo) [VERIFIED-PDF]

### Problem setting
Modality drop(센서 탈락) 하의 멀티모달 3D object detection. Plug-and-play 모듈. nuScenes, drop rate {10, 30, 50}%.

### Novelty / Method
- Uncertainty = **learned variance MLP** `σ²_M = MLP(F̂_M)`, NLL loss로 학습.
- 주입: deformable-transformer attention weight를 **POST-softmax에서 multiplicative 재가중**: `W̃ = W · [1 − softmax(U^{t0}_pts)]`.
- Mechanism class: **attn-multiplicative (post-softmax)**, learned.

### Quantitative results [VERIFIED-PDF] [unknown split → nuScenes val 관행이나 원문 재확인 요]
- 50% drop에서: UniBEV +10.83 mAP, CMT +17.00, BEVFusion +3.69.
- Code: https://github.com/Castiel-Lee/MM3Det_MD

### RBMA와의 차이 (4중)
(i) multiplicative, (ii) post-softmax, (iii) learned variance (training-free 아님), (iv) detection/modality-drop (dense seg 아님).

---

## 2. ReliFusion (arXiv 2502.01856) — 검증 캐비앳 포함

### Problem setting
Camera-LiDAR 3D detection에서 센서 손상/오정렬 하의 신뢰도 기반 fusion.

### Method — ⚠ 검증 상태 주의
- Findings 단계 기록(pdftotext): CW-MCA Eqs. 13–14 `F_{L→C} = C_LiDAR · softmax(Q_C K_L^T/√d_k) V_L` — 학습된 confidence 스칼라(CMCL contrastive + L_conf, λ=[1.0, 0.1, 0.2, 0.05])가 **attention OUTPUT에 곱해짐(output-scale, post-softmax)**. [VERIFIED-PDF via pdftotext]
- **⚠ Adversarial 재검증(skeptic2)에서는 abstract만으로 이 "post-softmax multiplicative" 특성을 재확인하지 못함** — abstract는 confidence가 learned인지, attention 어느 지점에 들어가는지 명시하지 않음. **정량 인용 전 PDF 수식 재확인 필수.** 확실한 것: trained Reliability module + confidence-weighted mutual cross-attention (CW-MCA)이라는 구성 자체 (skeptic1 확인).
- Venue: arXiv-only preprint (2026-07-02 기준 채택처 미확인).

### Quantitative results
- nuScenes test mAP 70.6 / NDS 73.2 (Table 1) [VERIFIED-PDF via pdftotext, 재검증 안 됨] [test].

---

## 3. SAM2Long (arXiv 2410.16268) — SAM2-memory 최근접 이웃 (skeptic1 발견, 추가 인용 필수)

### Problem setting
장시간 비디오 object segmentation에서 SAM2 memory의 error accumulation. **Training-free.**

### Method
- Occlusion/confidence score 기반 reliability를 **SAM2 memory cross-attention에 주입 — 단, MULTIPLICATIVE key scaling (`w·M`)**, additive logit bias 아님.
- 단일 모달(비디오 프레임), 신호는 SAM2 자체 occlusion score.
- Mechanism class: **attn-multiplicative (key-scale), training-free, unimodal**.

### RBMA와의 관계
- **"SAM2 memory attention에 reliability를 넣은 선행"으로 리뷰어가 가장 먼저 던질 반례** — 반드시 선제 인용.
- 차이 3중: (i) multiplicative key scaling vs additive logit bias, (ii) 신호 = SAM2 occlusion score(시간축 신뢰도) vs per-modality predictive entropy, (iii) unimodal video vs multimodal sensor fusion (modalities-as-frames).
- 세부 수치/수식은 원문 재확인 후 인용 ([ABSTRACT-ONLY] 수준으로만 검증됨).

---

## Limitations (공통)

세 방법 모두: learned 신호(SAM2Long 제외) + multiplicative/output-scale 주입. Softmax 이후 곱셈은 attention 분포 자체의 재정규화 없이 mass를 깎는 방식이라, 토큰 간 경쟁(competition)을 바꾸는 additive pre-softmax와 수학적으로 다름 (42번 노트의 defense 논리).

## Improvement directions

- ModalPatch의 drop-rate 프로토콜(10/30/50%)은 RBMA의 modality-corruption 실험 설계에 차용 가치 있음.
- SAM2Long의 training-free memory-reliability 아이디어를 멀티모달로 확장한 것이 사실상 RBMA의 포지션 — "SAM2Long이 시간축에서 한 것을 우리는 모달리티축에서, additive로" 라는 서사 가능.

## Comparison to RBMA-P29-P30 (mechanism-class)

| Method | Signal | Training-free | Injection | Class |
|---|---|---|---|---|
| ModalPatch | learned variance (NLL) | no | W̃ = W·[1−softmax(U)], post-softmax | attn-multiplicative (post) |
| ReliFusion | learned confidence (CMCL) | no | confidence × attn output (⚠ 재확인 요) | output-scale |
| SAM2Long | SAM2 occlusion score | **yes** | multiplicative key scaling in SAM2 memory attn | attn-multiplicative (key) |
| **RBMA** | predictive entropy | **yes** | **additive pre-softmax logit bias** | **logit-additive-bias** |

## Application to ours (RBMA/P29/P30 적용방향)

- 세 편 모두 related work의 "reliability meets attention" 문단에 배치하고, 주입점 차이를 표로 명시 — 특히 SAM2Long은 SAM2-memory 계열이므로 MemorySAM 문단 바로 옆에서 처리.
- Ablation "post-softmax multiplicative vs pre-softmax additive"의 실제 문헌 근거가 ModalPatch/SAM2Long — ablation 정당화 인용으로 사용.
- ReliFusion 수치 인용 시 반드시 PDF 수식(Eqs. 13–14) 재확인 후 사용 (adversarial 재검증 미완).

## Related-work paragraph candidate (English)

Reliability signals have recently been coupled to attention itself, but always multiplicatively or after normalization: ModalPatch regresses per-modality variance with an NLL loss and down-weights deformable-attention weights post-softmax (W̃ = W·[1−softmax(U)]) for 3D detection under modality drop; ReliFusion scales cross-attention outputs by a contrastively supervised confidence; and SAM2Long, the nearest SAM2-memory neighbor, injects training-free occlusion-based reliability into SAM2's memory cross-attention via multiplicative key scaling for long-video segmentation within a single modality. None of these adds a reliability term to the pre-softmax logits, where token competition is decided, and none targets multimodal dense segmentation.

## Links

- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]]
- [[relatedworks/01_memorysam_relatedwork]]
