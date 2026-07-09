---
title: "P32-B CoRB — 신뢰도 신호 재설계 리포트 (그림판)"
tags: [P32, RBMA, corroboration, reliability, multimodal-segmentation, memorysam, report, figures]
created: 2026-07-06
status: Phase0-PASSED / P32-B 학습중(B200)
supersedes_text: "[[P32_CoRB_조사보고서]] (텍스트 상세판)"
figures_source: "/mnt/HDD2/src/logs/P32_reliability_figs_20260706/ (재생성 스크립트 포함)"
data_source: ".claude_logs/24_p32_phase0_results.md, 16_failure_analysis_P28_P29.md"
---

# P32-B (CoRB) — 왜 P28은 안 됐고, 무엇을 바꿔 이번엔 작동하는가 (그림판)

> **한 줄**: MemorySAM 계열 신뢰도 라우팅(RBMA=P28)이 SOTA를 못 뚫은 진짜 원인은 게이트가 아니라 **신뢰도 신호 자체의 붕괴**였다. P32-B는 신호를 "자기확신(self-entropy)"에서 "**상호검증(cross-modal corroboration)**"으로 바꾼다. 무학습 진단(Phase 0)에서 LiDAR 신뢰도 AUROC가 **0.22 → 0.81**로 반전되어 가설이 실측 입증됐다.
>
> 아래 모든 그림의 수치는 **학습 없이(training-free)** 기존 체크포인트(P28 test-ep178, P31 test-ep182)에 신호만 교체해 측정한 실측값이다. 그림 원본·재생성 스크립트는 `figures_source` 참조.

---

## 0. 한 장 요약 (storyboard)

![P28→P32 storyboard](../assets/P32_CoRB/fig0_storyboard.png)

네 칸이 곧 이 리포트의 논리 전개다: **문제 → 수리 → 안전장치 → 신호 선택**. 각 칸을 아래에서 하나씩 푼다.

| 항목             | 내용                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **베이스**        | **P28 = RBMA** (Reliability-Biased Memory Attention)                                                                                        |
| **P28의 병목**    | 신뢰도 = per-modal **self-entropy** → decoder 용량과 얽혀 event/LiDAR가 항상 저신뢰. AUROC **event .30 / LiDAR .22** (우연 0.5 이하 = anti-calibrated)        |
| **P32-B 변경**   | 신호를 self-entropy → **corroboration**(다른 모달과의 합의도) + **unique-info veto**(혼자 옳은 센서 보호). training-free 유지, RBMA logit-bias 배관 그대로, 학습 파라미터 λ만 |
| **작동 증거(무학습)** | corr_veto가 **어떤 모달도 anti-calibrated로 남기지 않음**. LiDAR .22→.81, event .30→.54, P31 workhorse(depth) .90도 veto가 보호                             |
| **현재 상태**      | `LoRA_Sam_P32(LoRA_Sam_P31)` 구현·검증 완료, B200 DDP 학습중. Gate #2 = Test mIoU vs P31 54.75 / P28 55.27                                           |
|                |                                                                                                                                             |

---

## 1. 베이스: P28 = RBMA란 무엇인가

우리 모델의 토대는 **MemorySAM** — SAM2의 시간축 memory attention을 **모달리티 축**으로 전용해, 각 센서(RGB/depth/event/LiDAR)를 한 장면의 "프레임"으로 인코딩한 뒤 cross-attention으로 융합한다. **P28 = RBMA**는 여기에 신뢰도 편향을 더한다:

> 각 모달리티의 **training-free 신뢰도**를 SAM memory cross-attention의 **pre-softmax logit에 additive bias**로 주입.

```
Attention = softmax( QKᵀ/√d  +  λ · B ) V
B_i(x)    = 1 − H(softmax(D_i(f_i)))(x) / log C      # 모달 i 단독 디코드의 예측 불확실성 (GT-free)
λ         = 학습 스칼라 (self.lambda_bias)            # 유일한 학습 파라미터
```

- **신호 축(A)**: per-modal decoder의 예측 엔트로피 = training-free predictive uncertainty. 학습형 evidential head 없음, GT 없음, 추가 loss 없음.
- **기구 축(B, 헤드라인 노벨티)**: 그 신호를 attention **logit에 가산**. 신뢰 낮은 모달의 memory 토큰은 눌리되 **Value는 보존**(feature 0-kill 없음 → 정보 병목 없음).
- 코드: `LoRA_Sam_P28._compute_bias_source` (`sam_lora_image_encoder_seg.py:8028`), 주입점 `RoPEAttention._p27_attn_bias`(SDPA `attn_mask`). 신호 계산은 `torch.no_grad()`.
- **선행연구 대비**: 신뢰도 활용 선행은 전부 feature-multiply / output-scale / loss-level (UTFNet·HyperDUM·ReliFusion·DGFusion·CAFuser). **attention logit에 additive bias**는 전례 0건.

---

## 2. 문제: self-entropy 신뢰도가 약한 모달에서 붕괴한다

![P28 self-entropy anti-calibrated](../assets/P32_CoRB/fig1_p28_selfentropy_anticalibrated.png)

세로축은 **correctness AUROC** — "이 신뢰도 값이 실제로 맞는 픽셀을 예측하나?"를 잰다(0.5=우연). `1 − H(softmax(D_i))`는 사실 "모달 i의 **decoder가 얼마나 확신하나**"이지 "모달 i에 **정보가 있나**"가 아니다. event/LiDAR의 per-modal decoder는 애초에 약해 어디서나 고엔트로피 → 신호가 "이 센서 decoder가 약하다"를 인코딩한다.

결과: **event 0.30 / LiDAR 0.22로 우연(0.5) 아래 = anti-calibrated**(확신할 때 오히려 더 틀림). 신호가 우연 이하이니 그 위에 어떤 라우팅(게이트·MoE·FiLM)을 얹어도 geometry 모달을 영구히 죽인다.

### 진단 ↔ 증상 연결

![Mode C dead modality](../assets/P32_CoRB/fig5_dead_modality_symptom.png)

(a) anti-calibrated 신뢰도가 (b) 실제 성능에서 **drop-modality ΔmIoU ≈ 0**으로 이어진다 — event/LiDAR를 빼도 성능이 안 변함 = 사실상 미사용. 융합이 RGB+Depth 2-모달로 퇴화(**Mode C**). **결론: 라우팅이 안 된 게 아니라, 라우팅이 참조하는 신호가 틀렸다.**

---

## 3. 수리: self-entropy → cross-modal corroboration

![Corroboration repair](../assets/P32_CoRB/fig2_corroboration_repair.png)

핵심 교체: "이 모달이 **스스로** 확신하나" → "이 모달의 주장이 **다른 모달들의 합의와 얼마나 상호검증되나**".

```
p_i        = softmax(D_i(f_i))                    # per-modal posterior (기존 그대로)
p̄_{−i}(x)  = mean_{j≠i} p_j(x)                    # leave-one-out 합의
corr_i(x)  = Σ_c √( p_i · p̄_{−i} )                # Bhattacharyya 계수 ∈[0,1]
```

**왜 근본 수리인가**: self-entropy는 decoder 용량과 얽혀 있지만, **합의도는 용량과 분리된다.** 약한 decoder라도 "그 위치에서 다수 모달과 같은 클래스를 가리키나"는 정보 유무를 반영한다. 그림처럼 event **+0.25**, LiDAR **+0.59**로 둘 다 우연 위로 반전 — **재학습 없이 신호 정의만 바꿔서** 얻은 결과다.

---

## 4. 안전장치: unique-info veto (왜 순수 corroboration은 위험한가)

![Veto protects workhorse](../assets/P32_CoRB/fig3_veto_protects_workhorse.png)

합의 기반의 알려진 위험: 야간에 thermal/LiDAR만 물체를 보는 곳은 "다수와 불일치"로 오히려 벌점받는다 = **혼자 옳은 workhorse 처벌**. 실제로 순수 corroboration은 P31의 workhorse인 depth를 **0.90 → 0.28로 죽인다**(그림 가운데). 이를 막는 threshold-free veto:

```
g_i         = clamp( selfent_i − max_{j≠i} selfent_j , 0, 1 )   # i가 나머지보다 "더" 확신하는 정도
corr_veto_i = g_i · selfent_i + (1 − g_i) · corr_i
```

- 혼자만 confident한 센서 → self-confidence 유지(벌하지 않음). 나머지는 corroboration을 따름.
- veto가 depth를 **0.28 → 0.71로 회복**시킨다(그림 오른쪽). **training-free, 학습 파라미터는 여전히 λ만.**

---

## 5. 신호형 확정: corr_veto (worst-modality 기준)

![Signal form selection](../assets/P32_CoRB/fig4_signal_form_selection.png)

5개 신호형을 "**worst-modality AUROC**"(가장 약한 모달리티도 얼마나 살아있나)로 비교했다 — 강모달 평균이 아니라 **최악 모달을 최대화**하는 게 목표(어떤 센서도 죽이지 않아야 하므로).

| 신호 | P28 worst | P31 worst |
|------|:---:|:---:|
| self-entropy | 0.215 (LiDAR) | 0.322 (img) |
| corr 순수 (Bhattacharyya) | 0.543 | **0.283 (depth 붕괴!)** |
| corr 순수 (JSD) | 0.509 | 0.278 |
| **corr_veto** | **0.543** | **0.603** |
| corr_max | 0.543 | 0.498 (img 계승 붕괴) |

**corr_veto만이 두 체크포인트 모두에서 어떤 모달도 우연 아래로 남기지 않는다** → 신호형으로 확정. corr_max는 평균은 비슷하나 self-entropy의 깨진 신호(P31 img 0.498)를 그대로 물려받아 탈락.

---

## 6. 구현 & 현재 상태

- **모델**: `LoRA_Sam_P32(LoRA_Sam_P31)` — `_compute_bias_source` override, bias 소스 = **corr_veto**. P31 `consistency_bias`(Bhattacharyya 합의를 2차 항으로 계산)를 **1차 신호로 승격** + soft veto gate.
- **config-gated**: 전 기능 OFF → P28~P31 byte-identical. 검증: py_compile, 모델 corr_veto == 도구(오차 4.7e-6), GPU smoke PASS.
- **학습**: B200 tmux `jemo:p32corrb`, DDP, ~22h/200ep. config `b200-deliver_rgbdel_P32_physaug.yaml`(순수 ablation: P28 base + corroboration ON).
- **판정 게이트 #2**: Test mIoU vs P31 54.75 / P28 55.27. 공식 목표 = DELIVER val ≥66.51 / test ≥56.71.

> ⚠️ 위 그림들이 증명하는 것은 **신뢰도 신호 품질(AUROC)의 수리**이고, **최종 세그멘테이션 mIoU 개선**은 학습중인 P32 결과로 확정된다. 리포트 인용 시 "신호 진단=검증됨 / 성능 반영=학습 대기"로 구분할 것.

---

## 7. 로드맵에서의 위치

P32 = 라우팅 실패 4원인(R1 조건부재 / R2 상수 지름길 / R3 신호붕괴 / R4 soft가중은 select 불가)을 직격하는 5개 컨셉 A~E. P32-B(CoRB)는 **R3(신호붕괴) 직격 = 헤드라인**. 다음은 P32-C(PruneMem, R4). DELIVER 공식목표는 dead-modality 회복분 + P31 레버(backbone unfreeze) 병행이 전제(라우팅 단독으로 day→test 갭 Mode B는 안 닫힘).

---

## 8. 노벨티 방어 (요약)

- **RBMA-mechanism = NEAR-MISS.** additive pre-softmax bias 자체는 PRIMED(2605.07154, 학습형)·SAE(2603.16558, attention-entropy)·SAM2Long(2410.16268, multiplicative)가 인접 셀 점유 → **메커니즘 단독 "first" 금지, 4축 conjunction만 주장.**
- **CoRB = NOVEL as conjunction.** 어떤 검증된 단일 연구도 4-pillar(training-free × posterior-Bhattacharyya × N≥3 leave-one-out consensus × additive pre-softmax into SAM2 memory) + unique-info veto를 결합하지 않음. RSGMamba(learned·feature-diff·pairwise·multiplicative)와 MAGIC++(feature-cosine·hard ranking·keep-fragile)를 **4축 모두에서** 이김.
- **가장 깨끗한 discriminator = posterior-space Bhattacharyya** (모두가 feature-space 합의를 쓰는데 우리만 클래스 posterior 공간). 이 한 문장이 RSGMamba·MAGIC++를 동시에 분리.
- **MUST-CITE:** RSGMamba(2604.12319), MAGIC++(2412.16876), PRIMED, SAE, SAM2Long. **미해결 debt:** SCRNet 유료 PDF 미확인.
- 상세: [[49_corb_novelty_defense]] · [[P32_CoRB_novelty_risk_register]]

## 관련 노트
- 텍스트 상세판: [[P32_CoRB_조사보고서]]
- 폴더 인덱스/로그: [[00_P32_CoRB_index]]
- 로드맵 원본: `.claude_logs/23_seg_arch_proposals_P32.md`
- Phase 0 상세: `.claude_logs/24_p32_phase0_results.md`
- P28/P29 실패분석: `.claude_logs/16_failure_analysis_P28_P29.md`
- RBMA 노벨티: `.claude_logs/12_novelty_and_related_work.md`
- [[00_MOC_26_MultimodalSeg]] · [[PROJECT_TRACKING_26_MultimodalSeg]] · [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]] · [[relatedworks/41_unimodal_bias_and_modality_collapse]]
