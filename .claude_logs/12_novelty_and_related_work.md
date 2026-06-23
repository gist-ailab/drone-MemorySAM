# 노벨티 & 관련 연구 (Canonical) — RBMA 포지셔닝

> 이 문서가 **공식(canonical) 요약**이다. `10_related_work.md`는 deep-research 원시 로그(시계열),
> `02_model_arch.md`는 모델 상세. 새 에이전트는 **이 문서를 먼저** 읽고 필요 시 10/02로 내려간다.
> 최종 업데이트: 2026-06-23

---

## 0. 우리 모델 한눈에 (Our models at-a-glance)

| 버전 | 핵심 | 데이터셋 best | 상태 |
|------|------|------|------|
| P8 | ConfidenceHeadV2 + sigmoid UAMM | MULTIAQUA M=78.45 | 완료 |
| **P9** | CrossModalFusionHead + max-norm UAMM | MULTIAQUA **M=81.98** | 장기 최선(MULTIAQUA) |
| P10/P11 | +ModalAuxHead/oracle KL / +MI loss | 79.27 / 77.09 | 취소(test 하락) |
| P12 | Input-conditioned Soft-MoE LoRA | - | 설계 |
| P13 | Energy-score fusion + expert-collapse fix | - | 부분성공 |
| P20~P23 | SharedGate / DeBA-FP / DeBA-BB(OOM) | - | ablation |
| P24/P25/P26 | SpatialQualityGating(SQG) 계열 (scalar→spatial, KL teacher, modality-cond MoE) | - | SQG 붕괴 진단 |
| P27 | **Additive attention bias** (RBMA 전구체, SQG logit을 attn에 가산) | - | 구현 |
| **P28** | **RBMA on SAM2** (Hiera-B+) | **DELIVER val~55** | 메인 후보 |
| **P28 Hiera-L** | RBMA on SAM2-**Hiera-Large** | (학습 대기) | **67 본命 경로** |
| **SAM3-RBMA** | RBMA를 SAM3(plain ViT)로 이식 + SAM mask-decoder repurpose | DELIVER val **~24 (plateau)** | **ablation**(single-scale 한계 입증) |

> 목표: DELIVER mIoU **≥67** (SOTA급, CMNeXt ~66.3). 경로 = **P28 Hiera-L**. SAM3는 ~24 천장(구조 한계) → ablation.

---

## 1. RBMA란 (Reliability-Biased Memory Attention)

**한 줄**: 각 모달리티의 **training-free 신뢰도**(예측 엔트로피)를 **SAM memory cross-attention의 pre-softmax logit에 additive bias**로 주입.

```
Attention = softmax( QKᵀ/√d  +  λ·B ) V
B_i = 1 − H(softmax(Decoderᵢ(fᵢ))) / log C      # 모달 i 단독 디코드의 예측 불확실성, GT-free
λ   = 학습 스칼라
```
- **신호(A축)**: per-modality 디코더의 **training-free predictive uncertainty**. 학습형 quality/evidential head 없음, GT 없음, 추가 loss 없음.
- **기구(B축, 헤드라인)**: 그 신호를 attention **logit**에 **가산**. degraded 모달의 memory 토큰이 attention 경쟁에서 눌리되 **Value 보존**(feature 0-killing 아님 → 정보 병목 없음).
- **토대**: MemorySAM의 modality-as-frame memory attention을 재활용 → RBMA는 그 memory attention에 reliability bias를 더하는 것.

---

## 2. 관련 연구 비교표 (Related work × RBMA 차별축)

축은 **(신호: 어떻게 신뢰도를 얻나) × (위치: 어디에 적용하나)**.

| 방법 | venue | 신뢰도/조건 신호 | 적용 위치 | RBMA와의 차이(구조) |
|------|------|------|------|------|
| **MemorySAM** (arXiv 2503.06700, 2025) | - | 없음(등가/학습 융합) | memory attention 융합 | **우리의 토대.** RBMA가 그 memory attn에 **reliability bias 추가** = 직접 확장 |
| **DGFusion** (arXiv **2509.09828**, RA-L 2026) | RA-L | **depth**(입력+GT, robust depth loss) → spatially-varying sensor reliability | depth token이 **cross-modal fusion을 condition** | **최근접 경쟁자.** 신호=depth-supervised vs **entropy training-free** / 위치=depth-token conditioning vs **logit additive bias** / depth 필수 vs **모달 불문** |
| CAFuser (RA-L 2025) | RA-L | **CLIP/text** condition token | cross-attention² | text/CLIP 의존 vs 우리 self-derived, 무텍스트 |
| StitchFusion (arXiv 2408.01343) | - | 없음 | multi-adapter **조기 융합** | 신뢰도 가중 없음 |
| U3M (arXiv 2405.15365) | - | unbiased multiscale | feature 융합 | 신뢰도 기반 아님 |
| Zheng et al. (arXiv 2505.06635) | ICCV'25 | functional **entropy** | **loss/최적화**(regularization) | 같은 entropy지만 attention이 아니라 loss 레벨 |
| UTFNet (GRSL'23, RGB-T) | GRSL | **학습 evidential/Dirichlet+DST** head | feature 가중 | 학습 evidential head+loss vs **training-free** |
| HyperDUM (CVPR'25, DeLiVER) | CVPR | **학습 hyperdimensional** uncertainty | feature 가중 | 학습 prototype vs training-free; feature가중 vs **logit bias** |
| TMC/ETMC (ICLR'21/TPAMI) | ICLR | evidential SL | (분류) | dense-seg 아님, 분류 |
| **ReliFusion** (arXiv 2502.01856, 3D det) | - | **학습** reliability 모듈 | cross-attn **출력**에 곱 | 출력 스케일 vs **softmax 내부 logit 가산** |
| **READ** (ICLR'24) | ICLR | confidence | **loss** 가중(TTA) | loss vs attention-logit |
| **DAFusion** | ⚠️ **ref 미확인** | ? | ? | **사용자 확인 필요** — 아래 TODO |

---

## 3. 노벨티 판정 (deep-research verdict, 리뷰 방어용)

- **헤드라인 = 기구(B)**: "reliability를 **SAM memory-attention pre-softmax logit에 additive bias**로". feature-multiply / output-scale / loss-level 일색인 선행연구에 **logit-additive bias 전례 0건**. + MemorySAM 핵심 메커니즘 개조 서사.
- **신호(A) 단독 노벨티는 약함**: "per-modality 불확실성으로 dense-seg 융합 가중"은 **UTFNet/HyperDUM이 점유**. → A는 **"학습형 evidential/HD head 없이 training-free"**라는 점만 보조 차별점. **전면에 내세우지 말 것.**
- **반드시 명시 구분(must-write)**:
  1. **ReliFusion/READ vs RBMA** = "출력 재가중/loss" ↔ "**softmax 내부 logit additive bias**". 한 문장으로 못 박기.
  2. **DGFusion vs RBMA** = "depth-supervised, depth-token conditioning" ↔ "**training-free entropy, logit bias, 모달 불문**". DGFusion도 "spatially-varying sensor reliability"라 가장 헷갈림 → 가장 날카롭게.
  3. **Value 보존**(정보 병목 없음): logit bias는 Value를 안 죽임 → feature-zeroing/binary-gating(UAMM 상수수렴, SQG 붕괴) 대비 장점.
- **실험 전제**: 노벨티만으론 부족. **헤드라인 mIoU = P28(SAM2-Hiera-L)**, SAM3-RBMA(~24)는 "백본 무관성 + single-scale 한계" ablation. = (기구 노벨티) × (경쟁 mIoU) × (백본 ablation) 3종 세트.

---

## 4. 열린 lit-check TODO (반드시 채울 것)

1. **DAFusion 정확한 ref 확정** — 웹 검색에서 멀티모달 **세그멘테이션** 논문으로 특정 안 됨(영상 융합 MTKDFusion의 teacher로만 등장). 사용자에게 arXiv/제목/연도 요청 → 확정 후 표 2행 채우기.
2. **A 신호(decoder predictive uncertainty)가 evidential/TMC dense-seg에서 이미 쓰였는지** — 별도 lit-check (deep-research 미확인 1건).
3. DGFusion 정량(DELIVER/MUSES mIoU vs CAFuser/MAGIC) + 전체 하이퍼파라미터 — 비교표용 수치 미확보(원문 abstract만 확인).

---

## 5. 근거 / 더 읽기
- RBMA 신규성 deep-research 원문·판정: `10_related_work.md` §"신규성 조사 A vs B"(L329~), §"A 신호 신규성 확정"(L367~).
- SAM3 이식 분석: `10_related_work.md` §"SAM3 이식성"(L392~), `11_sam3_rbma_plan.md`.
- 모델 상세(P8~P28 forward/한계): `02_model_arch.md`.
- 실험 수치: `03_experiment_log.md`.
- 출처(웹): DGFusion [arXiv:2509.09828](https://arxiv.org/abs/2509.09828), 코드 `github.com/timbroed/DGFusion`. StitchFusion [arXiv:2408.01343](https://arxiv.org/pdf/2408.01343). U3M [arXiv:2405.15365](https://arxiv.org/pdf/2405.15365).
