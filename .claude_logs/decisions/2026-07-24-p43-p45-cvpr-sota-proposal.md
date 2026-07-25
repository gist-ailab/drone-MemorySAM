---
created: 2026-07-24
scope: CVPR 타깃 차기 아키텍처 제안 3안 (P43 PanopticDual / P44 BMR / P45 FogStyle) — 멀티에이전트 딥리서치 6축(모달불균형·상호증류·fog·panoptic·condition-adaptive·SOTA지형) 교차 종합
gates: 각 안에 사전등록 (본문 §게이트). 헤드라인 후보 = P43(PQ SOTA 축)
supersedes: 없음 (P42와 상보 — P42는 P44의 crude 선행 실험으로 재해석)
---

# P43~P45 — CVPR SOTA 제안 3안 (딥리서치 6축 교차, 2026-07-24)

> 딥리서치 에이전트 6기 병렬 수행: ① 모달불균형(OGM-GE 계열) ② cross-modal distillation ③ fog 강건성 ④ panoptic/mask-cls ⑤ condition-adaptive ⑥ SOTA 지형+노벨티 갭. 아래 수치·arXiv ID는 전부 에이전트가 원문/리더보드에서 검증한 것.

## 0. 🔴 전략 판정 — 어느 축이 진짜 SOTA인가 (Codabench 실측, 2026-07-24 검증)

**MUSES Semantic mIoU 리더보드 (Codabench 14005, live)**:

| # | 방법 | arXiv | 모달 | test mIoU | 비고 |
|---|---|---|---|---|---|
| 1 | GtA | **없음(미발표·익명)** | **카메라 단독** | **82.39** | 논문·코드 없는 유령 엔트리 |
| 2 | MM-SAM-Adapter | 2509.10408 | RGB+L | 81.07 | **frozen-SAM = 우리와 같은 레시피** |
| 3 | MM-SAM-Adapter | 2509.10408 | RGB+E | 79.92 | |
| 4 | DGFusion | 2509.09828 | C+L+R+E | 79.49 | |
| 5 | CAFuser | 2410.10791 | C+L+R+E | 78.19 | |
| — | **우리 P38-m2f** | | C+L+E | **79.025** | 4위권 |

**MUSES Panoptic PQ 리더보드 (Codabench 13987, live)**: DGFusion **61.03**(1위) > CAFuser 59.70 > CAFuser-CAA 59.38 > OneFormer(cam) 55.21 > Mask2Former baseline 53.60. **GtA는 PQ 보드에 없음. frozen-VFM 참가자 0.**

**판정**:
1. **mIoU는 융합 방법에게 죽은 SOTA 축** — 1위가 미발표 카메라단독(82.39), 2위가 frozen-SAM 2모달(81.07). 융합으로 aggregate mIoU를 이길 깨끗한 자리가 없다.
2. **PQ가 유일한 현실적 SOTA 축** — 1위 61.03은 사정권이고, 우리는 현재 PQ 산출 자체가 불가(per-pixel head = 구조적 배제). 경쟁 논문(MUSES/CAFuser/DGFusion)의 주지표가 PQ다.
3. **fog는 논문 서사 축** — "융합이 카메라단독을 이기는 조건은 fog뿐"(MM-SAM-Adapter fog 74.12 vs cam-only ~72.6). 우리 fog −13pt(경쟁자 −5~−7)는 수리 가능한 실패지 물리 천장이 아님.

→ **제안 우선순위: P43(PQ, SOTA 헤드라인) > P44(fog/불균형, 서사+robustness) > P45(fog 보조, 토글)**.

## 1. 진단 ↔ 문헌 대응표 (제안 공통 근거)

| 우리 실측 | 문헌 근거 (검증된 arXiv) | 함의 |
|---|---|---|
| drop-modality dMIoU(lidar,event)≈0 = **비RGB 미사용**; img 제거 시 fused rank ↑(+5.3) = **img 과지배** | EQUISeg 2509.24505 — **같은 데이터셋(DELIVER+MUSES)에서 같은 현상 보고**(event/lidar cos-sim 8.25/11.35로 붕괴 → prototype-KL로 29.82/32.24 회복); MoBaNet/MCRM 2603.17705(frozen-VFM+PEFT 균형 마스킹); MMPareto 2405.17730(aux·주손실 gradient Pareto 통합, **유일하게 seg(NYUv2) 검증**) | 불균형 수리는 loss/gradient 레벨로 (P44) — 단 EQUISeg가 최근접 선행이라 인용·차별화 필수 |
| lidar는 저정보가 아니라 미사용 (AnySeg lidar-only 32.13) | AnySeg 2411.17141, MM-SAM-Adapter fog 74.12 | fallback 만들면 fog 회복 여지 실재 |
| fog −13pt, 경쟁자 −5~−7 | FIFO 2204.01587(fog=style, 스타일 불변 학습), DGFusion fog +1.34 PQ(단 depth-GT=금지) | fog 헤드룸 ≈ +6~7pt(fog 국소) ≈ 전체 +1.5~2 |
| PQ 산출 불가(per-pixel head) + P38 zero-init 잔차 no-op + P30 query 대체 붕괴 | **PMT 2603.25398 — frozen DINOv3 + M2F 헤드의 검증 레시피**: naive 접붙임(EoMT식) PQ 6.8 붕괴, ViT-Adapter/multi-scale은 56.4, **multi-depth lateral +2.2 PQ**; Mask2Former 2112.01527(PointRend 손실, multi-scale round-robin) | dual-head 공동학습이 정답 형태 — 우리 P30/P38 실패가 문헌(PMT)의 실패 모드와 정확히 일치 |
| OGM/AGM/PMR 계열 주의 | MMPareto 재현 표에서 OGM 69.19/AGM 70.06/PMR 66.32 **< uniform baseline 71.10** | gradient 변조 단독은 위험 — Pareto 통합(A안)이 안전 |
| 조건적응 gate는 우리 계보에서 3세대 사망 | READ(ICLR'24)/RSGMamba 2604.12319 = inference-gate 계열; CAFuser/DGFusion는 **학습 시 외부신호**(CLIP text/depth-GT) 필요 | **train-time-only + 양단(train·inference) label-free 레인이 미점유** (P44 포지셔닝) |

## 2. 제안 1 — **P43 PanopticDual** (헤드라인, SOTA 축) 🥇

### 구조 (키1 준수 — zero-init 잔차 금지, 대체 금지)

- **공유 트렁크(불변)**: DINOv3-L frozen + per-modal LoRA → fusion → 256ch → SimpleFPN {1/4,1/8,1/16,1/32}. **+ PMT식 multi-depth lateral**: frozen DINOv3 중간 블록 ≥3곳에서 SimpleFPN으로 lateral tap (PMT 실증 +2.2 PQ, frozen 인코더의 소물체 보험).
- **Head A — per-pixel FPN(유지, 주 손실)**: 기존 그대로. mIoU 소유, thin-class 앵커.
- **Head B — mask-classification(신규, 주 손실)**: M2F식 쿼리 디코더(N queries, masked cross-attn {1/32,1/16,1/8} round-robin, class logits + mask embed·1/4 feature dot). **자체 Hungarian 손실(CE+BCE+Dice, PointRend 포인트 샘플링)** — P38처럼 β-잔차로 얹는 게 아니라 **독립 주 손실**. `panoptic_inference` → **PQ 산출**.
- **공동학습**: `L = L_pixel(CE+deep-sup) + λ·L_mask(Hungarian)`, λ warmup 0.1→1.0 (수 ep). 두 헤드가 공유 SimpleFPN을 두고 경쟁 — dense 손실이 1/4 feature를 thin-class에 날카롭게 유지(P30 붕괴 방지 기제).
- **추론**: semantic = Head A(mIoU), panoptic = Head B(PQ). 두 지표 모두 보고.
- 전 항목 토글: `P43.M2F_HEAD`(Head B off = P39.1 등가) / `P43.LATERAL`(lateral off) / λ 스케줄 config.

### 노벨티 (정직)

- **주장 금지**: 헤드 자체(PMT/Mask2Former 선점), "OneFormer를 멀티모달에"(CAFuser/DGFusion 선점).
- **주장 가능(미점유 셀)**: ① **frozen-VFM × per-modal LoRA × multimodal panoptic** — MUSES PQ 경쟁자 전원이 Swin-T finetune, PMT는 단일모달. ② **dual-head 공동학습 기제 + 실패 특성화**: query-only(P30 붕괴) vs zero-init 잔차(P38 no-op) vs 공동학습(ours)의 3-way ablation = 논문급 기제 기여. ③ PQ 지표 접근 자체가 구체 산출물.
- 리뷰 방어: "Mask2Former를 붙였을 뿐" 공격에는 ①+② **수치로만** 생존 — PQ가 CAFuser(59.7)급 미달이면 기여 붕괴. 헤드 신규성은 절대 전면에 세우지 말 것.

### 헤드룸 (캘리브레이션)

- frozen DINOv3-L+M2F 헤드 ≈ 56.1~56.4 PQ(COCO, PMT). 우리 semantic 품질은 이미 SOTA권(79.025 vs DGFusion 79.49) → **현실 착륙 지대 58~61 PQ**. CAFuser(59.70) 경쟁 = 기본 목표, **DGFusion(61.03) 돌파 = SOTA 업사이드**.
- ⚠️ MUSES val→test 전이 낙차 ~3.2pt 일관([[muses-dataset-setup]]) — val-PQ로 SOTA 주장 금지, test 제출로만.

### 게이트 (사전등록, falsifiable)

| 시점 | 게이트 | falsify |
|---|---|---|
| ep30 | ① val PQ 상승 추세 & **PQ_thing > 0**(P30 붕괴 시그니처 = thing≈0) ② Head A thin-class IoU가 dense-only 대비 **−1pt 이내**(>2pt 하락 = Head B가 feature 강탈) ③ 쿼리 비었지 않음(mean mask activation, P38 no-op 시그니처 검출) | 하나라도 실패 = λ/해상도 조정 1회 후 재실패 시 kill |
| 완주 | MUSES **val mIoU ≥ 82.22 유지**(Head A 무손상) & **test PQ ≥ 59.7**(CAFuser) — SOTA 업사이드 61.03 | PQ<53.6(baseline)이면 접목 실패 확정 |
| DELIVER | P36 fair(val 67.74/test 55.62) + thin-class(Wall≥13/Water≥9.5/RailTrack≥62) | 단일 아키텍처 제약 |
| 공정성 | physaug 금지·TTA 금지·val-best·radar 미포함(ISSUE-025) | |

## 3. 제안 2 — **P44 BMR (Balanced Modality Rebalance)** (fog/robustness 축, P42 후계) 🥈

P42(무조건 img 마스킹 FRAC)는 이 계열의 crude 1호기다. P42 결과와 무관하게 아래가 원리적 완성형 — P42 게이트 통과 시 "위에 얹기", 미달 시 "원리적 재시도".

### 변경 목록 (전 항목 토글, 키1 준수 — 전부 주손실/gradient 레벨, 모듈 추가 0)

| # | 변경 | 근거 | 형태 |
|---|---|---|---|
| **B-1 (주)** | **MMPareto gradient 통합** — 기존 per-modal aux CE(deep-sup, 이미 존재)와 주 CE의 gradient를 naive 합산 대신 Pareto 방향(전 목표와 내적≥0)+크기 복원으로 통합. per-modal LoRA gradient에 적용 | MMPareto 2405.17730(**seg 검증 유일**; OGM/AGM/PMR은 재현서 baseline 미달) | optimizer 레벨, 모듈 0, 추론 불변 |
| **B-2** | **peer 상호증류** — per-modal aux logit 간 대칭 KL(`Σ_{i≠j} KL(p_i‖p_j)`, teacher 없음, λ warmup ep10~) + **관계형 대응**(branch 간 cos-sim map KL — feature copy 아님, CKA 붕괴 방지) | DML(CVPR'18)·EQUISeg 2509.24505·AnySeg CMD 형 | aux 손실(주손실 합산) |
| **B-3** | **MCRM식 조건부 국소 마스킹** — P42의 전역 img-drop을 **영역 단위**(RGB 국소 corrupt, 해당 영역 lidar 유지)로 승격 + 생존 모달 aux에 hard-pixel 가중(P42 M-3와 동일 방향). **마스킹 영역은 랜덤 사각형이 아니라 실제 커버리지 패턴(FOV 경계·sparse 투영·무반환 지대)을 모사해 샘플링**(§7-b) | MCRM 2603.17705(frozen-VFM+PEFT 실증) | 학습 전용, 추론 full-modality |

### 노벨티 (정직 — EQUISeg가 최근접 위협)

- **주장 금지**: "모달 불균형을 driving seg에서 처음"(EQUISeg가 같은 데이터셋에서 선점), "frozen-VFM 균형 마스킹"(MoBaNet), gradient 변조/Pareto 자체(OGM·MMPareto·Pareto-LoRA 2606.17296).
- **주장 가능(미점유)**: ① **gradient-consensus 재균형을 frozen-VFM의 per-modal LoRA에** — EQUISeg는 full SegFormer 학습, MoBaNet은 원격탐사. ② **drop-modality dMIoU≈0→양수의 인과 실증**(EQUISeg는 유사도 회복만 보고, 인과 drop-test 없음). ③ **train-time-only·양단 label-free·no-inference-gate** 레인 — CAFuser/DGFusion은 학습 시 외부신호(CLIP text/depth-GT) 필요, READ/RSGMamba는 inference gate(우리 계보에서 3세대 사망 — 반증 이력이 오히려 포지셔닝 근거가 됨).
- 서사: "graceful degradation without an inference-time gate" — aggregate SOTA 주장 금지.

### 헤드룸: fog +2~5pt(국소), 전체 +1~2pt. 82.39 돌파 불가 명시.

### 게이트 (사전등록)

- ep30: ① **dMIoU(lidar) ≈0 → >1**(직접 지표 — P42와 동일 측정) ② lidar-aux gradient와 주 gradient 내적 양수 전환(MMPareto 자체 진단) ③ CKA(img,lidar) ≥0.5 유지·하락 없음. 미달 시 kill.
- 완주: MUSES val ≥82.22 & **fog ≥68**(62.67 대비 +5, 경쟁자 갭 −7 수준) · DELIVER P36 fair + thin-class.
- falsify: dMIoU↑인데 fog 불변 = "lidar 정보 task 무용" = 정직한 종결(P42 게이트와 동일 논리).

## 4. 제안 3 — **P45 FogStyle** (보조 토글, P44에 합성 가능) 🥉

- **F-1**: fog-invariant fused-feature 일관성(FIFO 2204.01587 이식) — fused feature의 style 통계(Gram)에 fog-pass filter, clear↔adverse style 거리 최소화. 날씨 라벨 불요(배치 내 style 대비). 학습 전용 손실.
- ⚠️ 입력 증강으로 구현하면 physaug 공정성 라인 침범 — **feature-space 손실로만**.
- 단독 제안이 아니라 P44 위 토글(`P45.FOGSTYLE`). 게이트: fog mIoU 추가 +1 이상, clear 무퇴행.

## 5. 실행 계획 (분석 선행 → 스모크 → 슬롯)

1. **학습 0 선행 (즉시 가능)**: ① P38 ckpt로 **PQ 하한 실측** — 기존 m2f_head `panoptic_inference` 경로로 val PQ 산출(P43 착륙 지대 캘리브레이션, 분석 세션 위임 가능) ② P42 진행분에서 dMIoU(lidar) 궤적 확인(P44 B-3 전제).
2. **구현 순서**: P43(헤드라인·독립 축, hpca100 슬롯) → P44(P42 판정 후, yeon/jarvis) → P45(P44 위 토글).
3. P43·P44는 **독립 변수라 병렬 가능**(서버 슬롯 허용 시). 최종 조합 런 = P43+P44 승자 레시피.
4. 구현 시 코드검수 파이프라인 의무([[code-review-pipeline]]: fresh-eyes 7종 + 스모크 grad/등가 assert + 로더 실측 + ep30 토글 즉검).
5. CVPR 논문 패키지(3안 종합 시): 헤드라인 = **first frozen-VFM multimodal panoptic(PQ 58~61)** + fog 서사("융합이 카메라단독을 이기는 유일 조건") + P41 negative-finding 방법론(Phase-0 판별→사전등록 게이트) + P30/P38/P43 3-way 실패 특성화 ablation.

## 6. 제약 준수 확인

- 반증 경로 회피: attn-bias ✗ / gate·calib·veto 추론 ✗ / zero-init 잔차 ✗(P43 Head B=독립 Hungarian 주손실, P44=gradient·aux 레벨) / fusion-rank ✗(P41 반증 — P44는 rank가 아니라 **사용률**(dMIoU) 타깃, 지표부터 다름) / 외부신호 ✗(전 안 label-free) / conv head 즉시 대체 ✗(P43 dual-head 유지).
- 단일 모델: P43/P44/P45 모두 DELIVER·MUSES 동일 아키텍처.
## 7. 토론 반영 (2026-07-25, user 비판 3건 — 서사·설계 조정)

**(a) per-class 라우팅 서사 축소.** user 지적: "clear에서 RGB 절대 우위인데 클래스별 모달 선호가 성립하려면 비RGB가 이기는 클래스가 있어야" — 타당. 우리 실측이 방어하는 명제는 "클래스별 모달 선호"가 아니라 **"전역 조건 게이트의 클래스별 오배분 방지"**다(night RoadLine: img competence .798/depth .001인데 전역 게이트가 depth .432 배분; gate/calib은 night thin-class에서 끄면 +35.9/+26.0 = 전역 신호 유해 실증). RGB 열화는 클래스별 불균등하므로 조건 단위 단일 결정은 반드시 일부 클래스를 죽인다 — 이것이 per-class의 존재 이유이고, 논문에서도 이 형태로만 주장한다. router 이득이 "클래스-모달 특화" 때문인지 "thin-class 독립 자유도" 때문인지는 미분리 → 학습0 검증 ①로 판별.

**(b) coverage-aware 요구사항 신설 (partial FOV/coverage).** user 지적: MULTIAQUA처럼 비RGB가 RGB FOV를 전부 커버 못 하는 데이터셋에서, 클래스 조건 라우팅이 비가시 영역에 그 모달을 배분할 위험 — 타당하며 일반화된다(MUSES lidar sparse 투영·무반환도 동일 구조 = partial coverage는 예외가 아니라 기본 상태). 취약점의 실체는 "zero-fill 입력에서도 백본이 그럴듯한 feature를 생성해 router가 무효 데이터임을 모른다"(P40 C-1 lidar 유효성 가드와 동일 문제). **처방 = 결정론적 presence 마스킹**: 입력 기하에서 유도되는 validity mask(thermal FOV 경계=캘리브레이션, lidar=투영 리턴 존재)로 router/gate softmax를 유효 모달 위에서 재정규화. 학습 파라미터 0. **반증된 'quality 추정 게이트'와 구분** — 죽은 것은 학습/자기추정 품질 재가중이고, 데이터 부재는 추정이 아님. B-3 마스킹도 커버리지 패턴 모사로 샘플링(학습↔추론 분포 정렬). MULTIAQUA 확장 시 이것이 선행 조건.

**(c) GtA 무시 결정 (user).** 융합 연구로서 비교군 = 융합 방법. 그 경우 **mIoU 융합 1위는 DGFusion(79.49)이 아니라 MM-SAM-Adapter 81.07(RGB+L, frozen-SAM)** — 같은 frozen-VFM 레인의 무조건화 평융합이라 리뷰의 실제 비교 대상. 우리 79.025에서 +2pt라 P44 fog 회복만으로는 빠듯 → PQ 축(P43) 병행 구조는 GtA 제외 후에도 유효(PQ 보드에 MM-SAM-Adapter 부재).

**학습0 검증 2건 — ✅ 완료·판정 (2026-07-25, [analysis/2026-07-25-router-coverage-verification.md](../experiments/analysis/2026-07-25-router-coverage-verification.md))**:
1. **V1**: "클래스별 모달 특화" 강한 해석 **기각**(clear 비RGB 인과 기여 미미) · "전역 오배분 방지+조건 적응" 약한 해석 **실증**(비RGB argmax 클래스 clear 4→fog 13). 🔴 **가중↔인과 괴리 발견**(fog 13클래스 비RGB argmax인데 drop-lidar +0.3) = P44 학습시-강제 전제의 독립 재확인.
2. **V2**: 🔴 **§7-b 실패 확정 — lidar 가중이 커버리지 밖에서 오히려 높음**(안 0.10~0.18 vs 밖 0.25~0.37 ≈ uniform 퇴화) → **V-1 presence 재정규화를 P44 본학습 config 기본 on으로 전환**(eval 토글로 ablation 분리).

- 근거 arXiv (에이전트 검증): 2603.25398(PMT) · 2112.01527 · 2107.06278 · 2211.06220 · 2203.16527 · 2308.03747 · 2405.17730(MMPareto) · 2203.15332 · 2308.07686 · 2211.07089 · 2509.24505(EQUISeg) · 2603.17705(MoBaNet/MCRM) · 2411.17141(AnySeg) · 2508.03060(CHARM) · 2505.12861 · 2204.01587(FIFO) · 2103.02370 · 2212.09068 · 2312.04265(Rein) · 2509.10408(MM-SAM-Adapter) · 2509.09828(DGFusion) · 2410.10791(CAFuser) · 2401.12761(MUSES) · 2508.16408(SAMFusion) · 2410.03010 · 2307.14126 · 2606.16639 · 2604.12319 · 2405.09321 · 2311.10707.
