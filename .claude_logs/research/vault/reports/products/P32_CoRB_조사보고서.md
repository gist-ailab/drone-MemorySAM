---
title: "P32-B CoRB — RBMA 신뢰도 신호 재설계 조사보고서"
tags: [P32, RBMA, corroboration, reliability, multimodal-segmentation, memorysam]
created: 2026-07-05
status: Phase0-PASSED / P32-B 학습중(B200)
---

# P32-B (CoRB) 조사보고서 — 왜 P28은 안 됐고, 무엇을 바꿔 이번엔 작동하는가

> 요약: MemorySAM 계열의 신뢰도 기반 라우팅(RBMA=P28)이 SOTA를 못 뚫은 **진짜 원인은 게이트 설계가 아니라 신뢰도 신호 자체의 붕괴**였다. P32-B는 그 신호를 "자기확신(self-entropy)"에서 "**상호검증(cross-modal corroboration)**"으로 교체한다. 무학습 진단(Phase 0)에서 LiDAR 신뢰도 AUROC가 **0.22 → 0.81**로 반전되어 가설이 실측 입증됐고, 현재 `LoRA_Sam_P32`가 B200에서 학습 중이다.

---

## 0. 한눈에

| 항목 | 내용 |
|------|------|
| **베이스** | **P28 = RBMA** (Reliability-Biased Memory Attention) |
| **P28의 병목** | 신뢰도 = per-modal **self-entropy** → decoder 용량과 얽혀(confounded) event/LiDAR가 항상 저신뢰. AUROC **event .30 / LiDAR .22** (우연 0.5 이하 = anti-calibrated) |
| **P32-B의 변경** | 신호를 self-entropy → **corroboration**(다른 모달과의 합의도) + **unique-info veto**(혼자 옳은 센서 보호)로 교체. training-free 유지, RBMA logit-bias 배관 그대로, 학습 파라미터 λ만 |
| **작동 증거(무학습)** | corr_veto가 **어떤 모달도 anti-calibrated로 남기지 않음**. LiDAR .22→.81, event .30→.54. P31 workhorse(depth) .90도 veto가 보호 |
| **현재 상태** | `LoRA_Sam_P32(LoRA_Sam_P31)` 구현 완료, B200 DDP 학습중(~22h/200ep). Gate #2 = Test mIoU vs P31 54.75 / P28 55.27 |

---

## 1. 베이스가 왜 P28인가 — RBMA란 무엇인가

### 1.1 계보: MemorySAM → RBMA
우리 모델의 토대는 **MemorySAM**이다. SAM2의 시간축 **memory attention**을 시간이 아니라 **모달리티 축**으로 전용한다 — 각 센서(RGB / depth / event / LiDAR)를 한 장면의 "프레임"으로 인코딩한 뒤, memory attention의 cross-attention으로 서로를 참조·융합한다.

**P28 = RBMA(Reliability-Biased Memory Attention)** 는 이 memory attention에 **신뢰도 편향(reliability bias)** 을 더한 것이다. 핵심 노벨티 한 줄:

> 각 모달리티의 **training-free 신뢰도**를 SAM memory cross-attention의 **pre-softmax logit에 additive bias**로 주입한다.

```
Attention = softmax( QKᵀ/√d  +  λ · B ) V
B_i(x)    = 1 − H(softmax(D_iᵢ(f_i)))(x) / log C      # 모달 i 단독 디코드의 예측 불확실성 (GT-free)
λ         = 학습 스칼라 (self.lambda_bias)
```

- **신호(A축)**: per-modality decoder `D_i`의 **예측 엔트로피** = training-free predictive uncertainty. 학습형 quality/evidential head 없음, GT 없음, 추가 loss 없음.
- **기구(B축, 헤드라인 노벨티)**: 그 신호를 attention **logit에 가산**. 신뢰 낮은 모달의 memory 토큰이 attention 경쟁에서 눌리되 **Value는 보존**(feature를 0으로 죽이지 않음 → 정보 병목 없음).

### 1.2 코드상 위치 (직접 확인)
- 신호 계산: `sam_lora_image_encoder_seg.py` → `LoRA_Sam_P28._compute_bias_source` (line 8028)
  - `p = softmax(aux_logits_i)` → `ent = −Σp·log p / logC` → `rel = 1 − ent` → 모달 간 zero-mean centering.
  - `torch.no_grad()` 로 감싸 신호는 무학습, **오직 λ만 학습**.
- 주입점: `RoPEAttention._p27_attn_bias` 를 통해 SDPA `attn_mask` 로 pre-softmax 가산 (P27에서 확립한 배관 계승).

### 1.3 선행연구 대비 위치 (왜 신규성이 있나)
신뢰도를 쓰는 선행연구는 전부 **feature-multiply / attention-output-scale / loss-level** 이다 (UTFNet·HyperDUM=학습 evidential feature 가중, ReliFusion=출력 스케일, DGFusion=depth-GT 토큰 conditioning, CAFuser=CLIP-text). **attention logit에 additive bias**를 준 전례는 0건. → RBMA = (SAM foundation) × (reliability-aware fusion)의 교차점, 기구는 logit-additive bias로 유일.

---

## 2. P28은 왜 SOTA를 못 뚫었나 — 신호의 붕괴 (핵심 진단)

RBMA의 노벨티(무학습 신호 + logit bias)는 건재하다. 문제는 **신호 A의 의미**에 있었다.

### 2.1 self-entropy는 "정보량"이 아니라 "decoder 용량"을 잰다
`1 − H(softmax(D_i(f_i)))/logC` 는 "모달 i의 decoder가 얼마나 확신하는가"이다. 그런데 event·LiDAR의 per-modal decoder는 애초에 약해서 **어디서나 고엔트로피** → 신호가 "이 위치에서 이 센서가 믿을 만한가"가 아니라 "이 센서 decoder가 약하다"를 인코딩한다. self-confidence가 정답 여부와 얽혀(confounded) 있는 것.

### 2.2 정량 증거 (doc 16 §7, DELIVER)
correctness-AUROC (신뢰도로 per-modal 정답/오답을 얼마나 잘 가르나, 0.5=우연):

| 모달 | self-entropy AUROC | 판정 |
|------|:---:|------|
| img (RGB) | 0.773 | 정상 |
| depth | 0.621 | 정상 |
| **event** | **0.296** | **우연 이하 (anti-calibrated)** |
| **LiDAR** | **0.215** | **우연 이하 (심각)** |

- 융합 가중은 사실상 uniform [.27,.28,.23,.23]인데 실제 기여(drop-Δ)는 [8.4, 23.5, **0.02, 0.01**] → **event/LiDAR가 사실상 미사용**(빼도 성능 불변). 모델이 RGB+Depth 2-모달로 퇴화.
- 즉 **신호가 우연 이하이니 그 위에 어떤 라우팅(게이트·MoE·FiLM)을 얹어도 geometry 모달을 영구히 죽인다.** P10~P31의 라우팅 재설계가 P9를 못 넘은 근본 이유.

**결론**: 라우팅이 안 된 게 아니라, **라우팅이 참조하는 신호가 틀렸다.** → 게이트를 고치기 전에 신호를 고쳐야 한다.

---

## 3. 무엇을 바꿨나 — P32-B (CoRB): Corroboration-Biased Memory Attention

### 3.1 아이디어 한 줄
"이 모달이 **스스로** 얼마나 자신 있나(self-entropy)" 대신 → "이 모달의 주장이 **다른 모달들의 합의와 얼마나 상호검증(corroborate)되나**"를 신뢰도로 쓴다.

```
p_i          = softmax(D_i(f_i))                          # per-modal posterior (기존 그대로)
p̄_{−i}(x)    = mean_{j≠i} p_j(x)                          # leave-one-out 합의
corr_i(x)    = 1 − D_B(p_i, p̄_{−i})   또는  Bhattacharyya 계수 Σ_c √(p_i·p̄_{−i})
```

**왜 이게 근본 수리인가**: self-entropy는 decoder 용량과 얽혀 있지만, **합의도는 용량과 분리된다.** 약한 decoder라도 "그 위치에서 다수 모달과 같은 클래스를 가리키는가"는 정보 유무를 반영한다. → anti-calibrated 신호가 정보성 신호로 반전.

### 3.2 필수 안전장치 — Unique-Info Veto
합의 기반의 알려진 위험: 야간에 thermal/LiDAR만 물체를 보는 곳은 "다수와 불일치"로 오히려 벌점받는다(=혼자 옳은 workhorse 처벌). 이를 막는 **veto**:

```
g_i         = clamp( selfent_i − max_{j≠i} selfent_j , 0, 1 )    # 모달 i가 나머지보다 "더" 확신하는 정도
corr_veto_i = g_i · selfent_i + (1 − g_i) · corr_i
```
- 혼자만 confident한 센서(다수가 못 보는 곳에서 홀로 확신) → self-confidence를 유지(벌하지 않음).
- 나머지는 corroboration을 따름. **threshold-free, training-free.**

### 3.3 P32-B의 정체성: 교체 ⟂ 보정 (합성)
- P31의 calibration loss(per-modal temperature)와 **직교**한다. calibration은 보정 후 self-confidence를 살리고, corroboration은 raw 모달에서 강하다.
- corr_veto는 **둘을 모두 살리는 blend** → 순수 대체보다 우세.
- **config-gated** (`CORROBORATION.ENABLE`, OFF → P31 byte-identical). RBMA logit-bias 배관·주입점 그대로, 학습 파라미터는 λ만.

---

## 4. 작동 증거 — Phase 0 무학습 진단 (GATE PASSED)

기존 체크포인트로 eval 1회만 돌려 신호를 사전 검증(도구 `tools/eval_reliability_auroc.py`). **학습 없이** self-entropy vs corroboration의 correctness-AUROC를 비교.

### 4.1 P28 (무보정 baseline) — 극적 반전

| 모달 | self-entropy | corr(best) | Δ |
|------|:---:|:---:|:---:|
| img | 0.773 | 0.704 | −0.069 |
| depth | 0.621 | 0.701 | +0.080 |
| **event** | **0.296** | **0.543** | **+0.247 (0.5 돌파)** |
| **LiDAR** | **0.215** | **0.808** | **+0.593 (극적 수리)** |

- self-entropy = [.773,.621,.296,.215] 로 doc 16 §7 값 정확 재현 → **도구 검증됨**.
- GATE(event/LiDAR corroboration > 0.5): **PASS**. 무학습으로 anti-calibrated geometry 모달 반전.

### 4.2 신호형 확정 — corr_veto (worst-modality 기준 최고)

두 모델 모두에서 **가장 약한 모달의 AUROC를 최대화**하는 변형을 선택:

| 신호 | P28 worst-modal AUROC | P31 worst-modal AUROC |
|------|:---:|:---:|
| self-entropy | 0.215 (LiDAR) | 0.322 (img) |
| corr 순수 | 0.543 | **0.283 (depth 붕괴!)** |
| **corr_veto** | **0.543** | **0.603** |

- 순수 corroboration은 P31의 workhorse depth를 0.90→0.28로 죽인다 → **veto가 0.71로 회복**.
- **corr_veto만이 어떤 모달도 anti-calibrated로 남기지 않는다** → 신호형으로 확정.

### 4.3 이것이 왜 결정적인가
Phase 0은 **학습 전에** "신호 수리"를 무학습으로 증명했다. 즉 P32-B가 실패하더라도 그것은 신호 문제가 아니라 fusion/decoder 문제로 국소화된다 — 근본 병목(R3)은 데이터로 제거됨.

---

## 5. 구현 & 현재 상태

- **모델**: `LoRA_Sam_P32(LoRA_Sam_P31)` — `_compute_bias_source` override, bias 소스 = **corr_veto**.
  - P31 `consistency_bias`(line 8421-8424)가 이미 Bhattacharyya 합의를 2차 항으로 계산 → corroboration을 **1차 신호로 승격** + soft veto gate.
  - 전 기능 config-gated (OFF → P28~P31 byte-identical). 검증: py_compile, 모델 corr_veto == 도구 (오차 4.7e-6), GPU smoke PASS.
- **config**: `b200-deliver_rgbdel_P32_physaug.yaml` = 순수 ablation (P28 base: AMF uniform, SDC/CTD/router OFF) + corroboration ON.
- **학습**: B200 tmux `jemo:p32corrb`, GPU 2-5 DDP, ~22h/200ep. 로그 `logs/b200-deliver_rgbdel_P32_physaug/P32_20260705_220251.log`.
- **판정 게이트 #2**: Test mIoU vs P31 54.75 / P28 55.27. 공식 목표 = DELIVER val ≥66.51 / test ≥56.71.

---

## 6. 로드맵에서의 위치

P32는 라우팅 실패 4원인(R1 게이트입력 조건부재 / R2 상수가 loss 지름길 / R3 신호붕괴 / R4 soft가중은 select 불가)을 각각 직격하는 **5개 컨셉 A~E**로 구성. 우선순위:

1. **Phase 0 진단** ✅ PASSED (본 보고서 §4)
2. **P32-B (CoRB)** ← 지금 여기, B200 학습중. R3(신호붕괴) 직격 = 헤드라인
3. **P32-C (PruneMem)** — memory token hard-pruning + modality dropout. R4 직격 (반드시 B 이후)
4. **P32-A (PhysCond) / D (ProtoTable)** — 조건 라우팅, MULTIAQUA/MUSES 무대
5. **P32-E (CCR)** — condition-contrastive 정칙화, 상시 부착

> DELIVER 공식목표는 P32-B/C의 dead-modality 회복분 + P31 레버(backbone unfreeze) **병행**이 전제. 라우팅 단독으로는 day→test class-transfer 갭(Mode B)이 닫히지 않음.

---

## 관련 노트
- 로드맵 원본: `.claude_logs/23_seg_arch_proposals_P32.md`
- Phase 0 상세: `.claude_logs/24_p32_phase0_results.md`
- P28/P29 실패분석: `.claude_logs/16_failure_analysis_P28_P29.md`
- RBMA 노벨티: `.claude_logs/12_novelty_and_related_work.md`
- [[00_MOC_26_MultimodalSeg]] · [[PROJECT_TRACKING_26_MultimodalSeg]] · [[relatedworks/02_dgfusion_relatedwork]] · [[relatedworks/41_unimodal_bias_and_modality_collapse]]
