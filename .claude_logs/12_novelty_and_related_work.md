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
| **CAFuser** (arXiv 2410.10791, RA-L 2025) | RA-L | **CLIP/text condition token**(verbo-visual contrastive, 환경조건 인지) | Condition-Aware **Cross-Attention(CA²)** / Addition(CAA), multi-level | text/CLIP·조건 supervision 의존 vs **self-derived training-free**; CA²는 condition으로 attention 변조하나 **softmax logit additive bias는 아님**. **DELIVER test 55.6** |
| StitchFusion (arXiv 2408.01343) | - | 없음 | multi-adapter **조기 융합** | 신뢰도 가중 없음 |
| U3M (arXiv 2405.15365) | - | unbiased multiscale | feature 융합 | 신뢰도 기반 아님 |
| Zheng et al. (arXiv 2505.06635) | ICCV'25 | functional **entropy** | **loss/최적화**(regularization) | 같은 entropy지만 attention이 아니라 loss 레벨 |
| UTFNet (GRSL'23, RGB-T) | GRSL | **학습 evidential/Dirichlet+DST** head | feature 가중 | 학습 evidential head+loss vs **training-free** |
| HyperDUM (CVPR'25, DeLiVER) | CVPR | **학습 hyperdimensional** uncertainty | feature 가중 | 학습 prototype vs training-free; feature가중 vs **logit bias** |
| TMC/ETMC (ICLR'21/TPAMI) | ICLR | evidential SL | (분류) | dense-seg 아님, 분류 |
| **ReliFusion** (arXiv 2502.01856, 3D det) | - | **학습** reliability 모듈 | cross-attn **출력**에 곱 | 출력 스케일 vs **softmax 내부 logit 가산** |
| **READ** (ICLR'24) | ICLR | confidence | **loss** 가중(TTA) | loss vs attention-logit |

---

## 2.5 최신 멀티모달 시맨틱 세그 지형 (DELIVER 중심, 2023~2026)

### ⚠️ DELIVER 숫자 "두 cluster" 문제 (2026-06-23 원문 대조 확정 — 매우 중요)

동일 조건(CMNeXt-**B2**, RGB-D-E-L)인데 **논문마다 baseline 숫자가 셋으로 갈림** → 단순 val↔test 토글이 아니라 **재구현/재평가 프로토콜 자체가 다름** (DELIVER 문헌 고질병):

| 출처 | CMNeXt(B2) | 그 논문 method | split |
|------|-----------|---------------|-------|
| CMNeXt 원논문 (2303.01480) | **66.30** | — | 3983/2005/1897, split 명시 X |
| MemorySAM (2503.06700) | **59.18** | **MemorySAM 65.38** | 명시 안 함 (高 cluster) |
| DGFusion (2509.09828) | **53.0** | **DGFusion 56.7** | 명시적 **"mIoU-test"** |

- **Cluster A (高 ~59–66)**: CMNeXt 원논문 66.30, MemorySAM(CMNeXt 59.18 / MemorySAM **65.38**). ← **우리 구조적 base(MemorySAM)가 여기**.
- **Cluster B (低 ~53–57, test 통일)**: CMNeXt 53.0 < StitchFusion 53.4 < GeminiFusion 54.5 < CAFuser-CAA 55.2 < CAFuser 55.6 < **DGFusion 56.7**. ← **우리 직접 경쟁자(DGFusion=reliability-aware, CAFuser)가 여기**. MUSES: CAFuser 78.2 > GeminiFusion 75.3.

> **두 cluster 숫자를 한 표에 섞으면 안 됨** (MemorySAM 65.38 ≠ DGFusion 56.7, 비교 불가).
> **"67 목표" = Cluster A(66.30) 기준**. 하지만 RBMA 직접 경쟁군(B)의 SOTA는 **56.7**.
> **우리 평가 split 확정 (코드 확인)**: `train_sam3_rbma.py:46,50` 트레이너는 **val(2005장)과 test(1897장)를 둘 다** 평가. 그동안 본 **"val~55" = DELIVER val split**. → DGFusion 56.7/CAFuser 55.6은 **test**라서 우리 val과 직접 비교 불가. **Cluster B 비교용 숫자는 로그의 P28 *test* mIoU**(이미 계산됨)를 써야 함.
> **포지셔닝별 봐야 할 숫자**: (a) DGFusion/CAFuser(reliability-aware 직접 경쟁, Cluster B) 대비 → P28 **test** mIoU. (b) MemorySAM(구조적 base, Cluster A, 65.38, split 미표기→val 추정) 대비 → P28 **val** mIoU.
> **남은 TODO**: (1) P28 test mIoU 실측치 doc 12/03에 기록. (2) **유일한 안전책 = CMNeXt(가능하면 MemorySAM)를 우리 단일 프로토콜(같은 split·해상도)로 직접 재평가**해 같은 표에 넣기. 남의 표 숫자 그대로 인용 시 리뷰어가 cluster 불일치 지적. (3) MemorySAM split(val/test) 원저자 코드/표에서 확정.

**Taxonomy (관련연구 작성용):**
1. **Arbitrary/any-modal** (가변·결측 모달): CMNeXt(SQ-Hub, CVPR'23, arXiv 2303.01480, DELIVER 벤치마크 제안), MAGIC(modality-agnostic), AnySeg(2411.17141, uni/cross-modal distillation), **OmniSegmentor**(2509.15096, pretrain-finetune+ImageNeXt, 새 SOTA 주장).
2. **Condition/Reliability-aware fusion**: CAFuser(CLIP condition), **DGFusion**(depth reliability, RA-L'26), HyperDUM(CVPR'25, hyperdimensional uncertainty), UTFNet(evidential). ← **RBMA가 속하는 칸**.
3. **Foundation-model(SAM/SAM2) 기반**: **MemorySAM**(2503.06700, SAM2 memory attn = 우리 토대), MM-SAM-adapter(2509.10408, DELIVER SOTA 주장), FusionSAM(2408.13980).
4. **Cross-modal attention/transformer fusion**: GeminiFusion(intra+inter-modal attn), CMX(RGB-X), FTransUNet.
5. **Adapter/early-fusion 효율**: StitchFusion(2408.01343, 대형 pretrained를 encoder로), dual-prompt.

→ **RBMA 위치 = (3) SAM foundation + (2) reliability-aware의 교차점**, 단 메커니즘이 **logit-additive bias**라 (2)의 누구(CAFuser=condition-cross-attn, DGFusion=depth-token, HyperDUM/UTFNet=feature가중)와도 다름.

## 2.6 멀티모달 객체 검출 (fusion 백본 공유 — "head만 다름")

사용자 지적대로 멀티모달 **검출**도 결국 *cross-modal fusion + head*. 주요 camera-LiDAR 3D det:

| 방법 | venue | 융합 방식 | reliability |
|------|------|------|------|
| BEVFusion | NeurIPS'22/ICRA'23 | 공유 **BEV** 공간에서 융합 | 없음(등가) |
| TransFusion | CVPR'22 | LiDAR query가 image에 cross-attend | 없음 |
| DeepInteraction | NeurIPS'22 | 모달 표현 분리 유지 + interaction | 없음 |
| CMT | ICCV'23 | position-guided query, 명시 융합 모듈 X | 없음 |
| FUTR3D | CVPR'23 | query 기반 통합 센서 융합 | 없음 |
| **ReliFusion** | arXiv 2502.01856 | cross-attn | **학습 reliability → 출력 스케일** |

→ 검출도 대부분 **reliability 미처리(등가 융합)**, 다뤄도 출력/feature 스케일(ReliFusion). **attention logit additive bias는 검출에도 전례 0.** 단 framing 주의: 검출은 **BEV 투영/3D query head + geometry**가 추가 → "**fusion 백본은 공유, head·표현공간(2D dense vs BEV/3D)이 다름**". RBMA의 reliability-bias 메커니즘은 두 분야(seg/det) 공통으로 비점유 영역.

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

1. ~~DAFusion~~ → **CAFuser 오기였음**(2410.10791). 표 정정 완료. (이름 혼동 해소.)
1b. **DELIVER 프로토콜 확정** — CMNeXt 66.30(원논문) vs 53.0(CAFuser 표) 불일치. 우리 P28 val~55가 어느 쪽과 비교 가능한지(같은 test split/모달 구성인지) 확인 → "67" 목표의 정확한 기준선 설정. OmniSegmentor/MM-SAM-adapter의 DELIVER 정량 수치도 미확보(원문 PDF 확인 필요).
2. **A 신호(decoder predictive uncertainty)가 evidential/TMC dense-seg에서 이미 쓰였는지** — 별도 lit-check (deep-research 미확인 1건).
3. DGFusion 정량(DELIVER/MUSES mIoU vs CAFuser/MAGIC) + 전체 하이퍼파라미터 — 비교표용 수치 미확보(원문 abstract만 확인).

---

## 5. 근거 / 더 읽기
- RBMA 신규성 deep-research 원문·판정: `10_related_work.md` §"신규성 조사 A vs B"(L329~), §"A 신호 신규성 확정"(L367~).
- SAM3 이식 분석: `10_related_work.md` §"SAM3 이식성"(L392~), `11_sam3_rbma_plan.md`.
- 모델 상세(P8~P28 forward/한계): `02_model_arch.md`.
- 실험 수치: `03_experiment_log.md`.
- 출처(웹): DGFusion [arXiv:2509.09828](https://arxiv.org/abs/2509.09828), 코드 `github.com/timbroed/DGFusion`. StitchFusion [arXiv:2408.01343](https://arxiv.org/pdf/2408.01343). U3M [arXiv:2405.15365](https://arxiv.org/pdf/2405.15365).
