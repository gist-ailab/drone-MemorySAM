---
created: 2026-09-17
author: fable (background 세션 "MMSAM | 생각정리")
status: 🟡 탐색 완료 — 후보 등재, 사용자 선택 대기 (정책 충돌 항목 2건은 사용자 결정 필요)
depends: decisions/2026-09-17-p53-detail-branch-proposal.md · decisions/2026-09-07-daily-cycle-experiment-cards.md §0·§0-2 · status/current.md(모달리티 정합 비교표)
---

# 성능 개선 전략 탐색 — MoE-LoRA · 융합 대안 · 학습 전략 (2026-09-17)

> **user 질문(2026-09-17)**: "성능 개선 전략을 탐색해 달라. MoE LoRA를 쓰고 있나? MoE는 구조와 학습 전략(초반부터가 아니라 중반부터 학습하는 식)이 있다던데, 다른 융합 방법까지 열어 두고 고민해야 한다."
>
> 방법: 세 축(① MoE-LoRA 구조·학습 일정, ② 백본 안/후기 융합 대안 + 경쟁 SOTA 해부, ③ 어댑터 학습 일정·손실·도메인 전이·frozen VFM 레시피)을 딥리서치 에이전트 3개로 병렬 조사(arXiv ID 인용 의무, 수치는 원문 확인분만). 판정은 이 세션이 우리 실측(카드 §5, registry)과 대조해 내렸다. 예상 이득은 전부 【추측】이며, 판정 규약은 카드 §0(시드 분산 ±0.9, 3시드 평균, legal 재채점)을 따른다.

## 0. 한 줄 답

1. **MoE-LoRA는 쓰지 않는다.** 현재는 센서별 LoRA(Q/V r8)를 입력 모달에 따라 고정 스위치한다(`encoder.py` `MultiModalLoRAQKV`; 라우터·전문가 혼합 없음).
2. **문헌은 MoE-LoRA 도입을 지지하지 않는다.** 우리 고정 스위치는 문헌의 "오라클 태그 라우팅"이고, 학습된 이산 라우팅이 이를 이긴 사례가 없다(SMEAR 2306.03745: 오라클 61.4 vs top-k 60.0 vs 소프트 병합 62.0; THOR 2110.04260: 학습 라우팅 = 무작위 라우팅). 토큰 단위 라우팅이 최악이고(MoCLE 2312.12379: 토큰 58.1 < 클러스터 61.8) 작은 데이터에서는 sparse가 과적합한다(ST-MoE 2202.08906; MoE sweet spot 2411.18322 "작은 데이터셋에선 이득 거의 없음, OOD 저하").
3. **"중반부터 라우팅" 일정은 근거가 없다.** dense→sparse 전환(sparse upcycling 2212.05055·Drop-Upcycling 2502.19261·NVIDIA 2410.07524)은 원 학습의 10~60% 추가 예산 + LR 리셋 + 부분 재초기화가 있어야 분화하고, 짧은 파인튜닝 예산에서는 dense와 구분되지 않는다. 파인튜닝에서는 라우팅 분화 동역학 자체가 관찰되지 않는다(2604.04230). 라우터 warm-up·온도 스케줄·z-loss는 안정성 도구이지 정확도 변수가 아니다(Ling 2503.05139, NVIDIA: z-loss 효과 없음, MoLE 2404.13628: 온도로 균형 잡으면 악화).
4. **융합 구조를 바꿔서 얻을 이득은 시드 분산 이하다.** 백본 안 토큰 교환(StitchFusion 2408.01343)은 DELIVER test +0.4, gated 합산↔attention 융합은 ±0.3(CAFuser 2410.10791). 우리 융합부에는 이미 2층 cross-modal attention이 있다.
5. **경쟁 SOTA가 적은 모달로 이기는 이유는 융합이 아니라 백본 적응·세부 특징·해상도다.** MM SAM-adapter(2509.10408)는 SAM ViT-L 풀 파인튜닝 + ConvNeXt-S 측면 가지 deformable cross-attn 주입이고, 같은 논문 ablation에서 "frozen+LoRA" 53.97 vs adapter 57.14(+3.2), frozen→FT +1.8. DELIVER test는 RGB-easy 지배(easy 57.62 ≈ 전체 57.35)라 보조 모달 기여가 작고, depth 한 모달이 +5.1(DGFusion CLE 51.6 vs CLDE 56.7), event·LiDAR는 거의 0(MLE-SAM RD 63.57 > RDE 62.69).
6. **시드 분산을 넘길 가능성이 있는 후보는 손실·평가 가중치·세부 경로 쪽이다**(아래 §2). 비용 0인 것 둘(Lovász 보조손실, EMA 가중치 평가)이 있고 둘 다 우리 레포에서 시도된 적이 없다.

## 1. 축별 조사 요약

### 1-1. MoE-LoRA 구조·일정 (에이전트 A)

| 구조 | 근거 | 우리에의 함의 |
|---|---|---|
| A 공유 + B 다중(HydraLoRA 2404.19245) | 따로 학습한 LoRA들의 A는 수렴, B만 갈라짐; r8 B 3개가 r32 단일을 이김(MMLU 47.22 vs 46.59, LLM) | 우리 4모달 LoRA의 A 코사인 유사도를 재면(비용 0) A 공유 가능 여부가 즉시 나온다 |
| 공유 전문가 + 특화(DeepSeekMoE 2401.06066, MoCLE universal +1.5, MLoRE 2403.17749 generic path) | 공유 경로가 있을 때만 전문가 분리가 이득 | 우리 gated-MLP 트렁크가 이미 공유 경로 역할 |
| 라우터 없는 확률적 전문가 + 병합(MoSA 2312.02923, AdaMix 2205.12410, THOR) | FGVC LoRA 88.44 → MoSL 89.92(+1.48), 추론 비용 0; B(up)만 희소가 최선 | 정규화 효과. dense 세그 미확인 |
| 토큰 top-k MoE-LoRA(MixLoRA 2404.15159 등) | LLM 이득; r32에서 수렴 난조 | 비추천(SMEAR·THOR·MoCLE·ST-MoE) |
| 조건 단위 융합 가중(CAFuser CAA) | 학습된 고정 가중 59.1 = 평균 59.1 → 조건부 59.4(+0.3), 조건 대조 손실 +0.4 PQ | 우리 고정 γ_m을 조건부로 바꾸는 실험이 직접 계승. 이득이 노이즈 경계 |
| 세그 직접 선행: MLE-SAM 2412.04220 | 제목과 달리 백본 LoRA는 모달별 고정(우리와 동일), "MoE"는 융합 가중; 라우팅 단독 58.35 < 균등 61.87 | 우리 설계의 정당화 |
| 진단 도구: SpawnLoRA 2609.03150 | 라우팅으로 분리돼도 어댑터 내부 그래디언트 충돌; rank 확장은 충돌을 키움(E2 무이득과 일치) | 조건별 그래디언트 코사인 ≤0인 조건쌍이 있어야 조건 전문가에 근거 |

### 1-2. 융합 대안·경쟁 SOTA (에이전트 B)

| 항목 | 근거 | 판정 |
|---|---|---|
| MM SAM-adapter 해부 | ViT-L 풀 FT(layer-decay 0.9), ViT-Adapter injector/extractor ×4, ConvNeXt-S ×2(모달별), Road-Fusion, 1024 crop, 100ep; 2모달만 지원 | 이득 원천 = 백본 FT(+1.8) > 측면 adapter vs LoRA(+3.2) > 융합 연산(Add→Road +1.05) |
| DGFusion 해부 | CAFuser 위에 학습 시 depth 보조 헤드(+0.64 PQ, 추론 불변) + 윈도별 depth 토큰(+0.69 PQ); DELIVER는 GT depth를 입력 겸 감독 | 기하 토큰을 쿼리에 붙여 attention이 쓰게 함(가중 예측 아님) |
| GtA(MUSES 82.39) | 논문·구조 미확인. 공개 벤치마크 사이트 2025-12-31 폐쇄 → 익명 리더보드 항목 가능성 | 카메라 단독 82는 대형 VFM + M2F + 고해상도/TTA로 설명 가능(GOOSE 2606.18582: DINOv3-L+ViT-Adapter+M2F +13.1) |
| 백본 안 융합 | TokenFusion 불안정(전부 교환이 최선), StitchFusion test +0.4, GeminiFusion "마지막 4층만 융합해도 전층 이득의 80%", 통제 실험(MoBaNet CPIA +1.4, DPLNet MPG +0.9, CoLA +1.1)은 타 도메인 | 상한 +1~1.5, DELIVER test 재현 없음 |
| 후기 융합 대안 | CA² vs CAA ±0.3; FiLM 게이트(scale·shift·gate + validity, URVIS 2604.16984) +1.1 PQ(RGB-only 대비); 융합 특징 증류 −0.5~−0.9(AnySeg) | attention 교체 단독 비추천 |
| frozen VFM 축 | 토큰 밀도 2배가 어떤 업샘플러보다 큼(ViT-Up 2606.14024: Cityscapes LP +6.0); ViT-Adapter(2205.08534) add +1.4 vs extractor +2.1; frozen에서 injector 금지(DINOv3도 제거); SPAR 2604.02252: 마지막 2블록 MLP > QKV | 세부 가지는 "합산"이면 SPM-add 수준, extractor면 +2 |

### 1-3. 학습 일정·손실·도메인 전이 (에이전트 C)

| 항목 | 근거 | 판정 |
|---|---|---|
| 단계적 일정(헤드 먼저→LoRA, progressive unfreeze, LP-FT) | 세그+LoRA 실증 없음; Rein 2312.04265에서 Full-FT 61.7 < LoRA 62.7 | 비추천 |
| LoRA+/rsLoRA/DoRA, layer-wise decay | r8에서 무효, dense 미확인, frozen엔 대상 없음 | 비추천 |
| **Lovász-Softmax**(1705.08790, ABL 2102.02696) | Cityscapes Class IoU +4.8(weighted-CE fine-tune), CE 79.5→+Lovász 80.2(+0.7); **pole 66.1→68.3, TL 72.2→75.4, rider +3.9**; 긴 스케줄에 유리; 비용 0 | **1순위**. 우리 약점 클래스와 정확히 겹침. OHEM 병용 상호작용 미확인 |
| 클래스 가중 CE / Focal / Recall loss(2106.14917) | 소수 클래스 IoU 55.23→29.58 / 51.63 / 53.36 | 비추천 |
| **EMA 가중치 평가**(GOOSE 2505.11769, SWA 2012.12645) | +1.12(소물체 최대) / +1.0 AP; 분산 감소 | **2순위**. 비용 0. val→test 효과는 추측 |
| 스타일 증강, consistency, 경계 보조, 주파수 입력(2605.27962) | DINOv2급에서 0 또는 저하; 열화 증강 p=1.0은 −1.9 | 비추천(모달 드롭아웃 역효과와 같은 패턴) |
| 다중 층 특징(DINOv2 lin1→lin4 +4.0) | — | **이미 적용됨(TAPS = E1)** |
| 마스크 분류 헤드로 교체(DINOv3 선형→M2F +7.1) | 비용 4.1×, FPN 대비 순이득 미확인 | 우리는 M2F-lite를 병렬 헤드로 이미 씀. 주 헤드 교체는 P43 계보, 보류 |
| copy-paste / 소물체 삽입 | 경량망 소물체 +2%; 4모달 픽셀 정렬이라 구현 가능 | Lovász 다음 후보 |

## 2. 후보 등재 (단일 변수, 사전 등록 게이트)

기준선 = E1 확정 레시피(TAPS on, C3 mfeat λ0.1), 스크린 40ep 시드 821 + 같은 시드 기준선 대조, legal 재채점(1024·BS1·native GT). 게이트는 카드 §0·§0-2를 그대로 쓴다: **G1** 24클래스(RailTrack 제외) Δ ≥ +0.5, **G2**(얇은 객체 표적 실험만) Pole·Pedestrian·Static·TrafficLight 평균 Δ ≥ +2.0, **G3** 조건별 −0.5 미만 없음, **G4** 우리 최고 단일 런 56.99·DGFusion 56.71 거리 병기. 통과 시 200ep 3페어.

| 순위 | 카드 | 무엇을 바꾸나 | 예상【추측】 | 근거 | 비용 | 정책 |
|---|---|---|---|---|---|---|
| **1** | **E18 Lovász** | OHEM CE 유지 + 픽셀 헤드 로짓에 Lovász-Softmax 보조손실(배치 내 존재 클래스만, 가중 0.5), M2F 손실 무변경 | +0.5~+2 (얇은 객체 집중) | §1-3 | 0 | 충돌 없음 |
| **2** | **E19 EMA** | 학습 가중치 EMA(α 0.9995)를 val 루프에서 평가·저장, 헤드라인 규칙에 "EMA val-best"를 병기 규칙으로 추가 | +0.5~+1, 시드 분산 감소 | §1-3 | 0 | 헤드라인 선택 규칙 개정 필요(카드 §0) |
| **3** | **E17 세부 가지**(기동 대기) → 통과 시 **E17x extractor** | E17 = 합산 주입(제안서 P53). 통과하면 결합을 ViT-Adapter extractor(피라미드=Q, 4탭 토큰=K/V, deformable cross-attn ×4, injector 없음)로 교체 | E17 +0.5~1.5, E17x 추가 +1~2 | §1-2 ViT-Adapter add +1.4 vs extractor +2.1; MM SAM adapter vs LoRA +3.2 | E17x 약 24M, deformable attn 구현 | 충돌 없음 |
| **D0** | **진단 2종(학습 0)** | (i) 4모달 LoRA A 행렬 층별 코사인 (ii) 각 모달 LoRA의 조건별(DELIVER 조건 5×케이스) 미니배치 그래디언트 코사인 | — | HydraLoRA 관찰 II, SpawnLoRA | 0 | — |
| 4 | **E20 공유 A + 모달 B + 공유 LoRA** (D0-i ≥ 0.9일 때만) | ΔW_m = B_m A + B_s A, 라우터 없음. 파라미터 3.1M→약 2.4M | 0~+0.5 | HydraLoRA, DeepSeekMoE, MoCLE universal, MLoRE | 0 | 충돌 없음 |
| 5 | **E21 확률적 분할-B LoRA + 병합** | 각 모달 LoRA의 B를 2~3조각으로 나눠 배치마다 1조각 활성, 두 forward 간 KL 일관성, 추론 시 병합(라우터·추론 비용 0) | +0.3~+1 | MoSA/AdaMix/THOR | 학습 forward 2× | 충돌 없음 |
| 6 | **E22 라벨 없는 조건 게이트** | 고정 tanh(γ_m) 대신 RGB 최상위 특징 풀링에서 조건 토큰 → 소프트맥스 모달 가중치. **조건 라벨 손실은 쓰지 않는다**(정책) | +0.3~+0.6 | CAFuser CAA, URVIS FiLM | 수천 파라미터 | 라벨 손실 없는 변형만 허용 |

### 🔴 user 결정(2026-09-17 추가): DGFusion식 depth 보조 감독 **허용, 단 +α 필수**

> user: "dgfusion 방식 써도 되는데 거기의 한계를 우리의 무언가로 해결해서 +alpha가 있어야 해."

- **E23 = DGFusion 기제 이식(기반)**: 학습 시 depth 보조 헤드(LiDAR 투영/GT depth 감독, robust L1 + τ 분위 필터 + edge smoothness — DGFusion 손실 설계 그대로) + 우리 융합부 2층 cross-modal attention에 윈도별 로컬 depth 토큰을 K/V로 추가. 추론 구조는 depth 토큰 생성만 추가. 게이트: 24클래스 Δ vs E1 ≥ +0.5(재현 근거 +1.0~1.3), 조건별 G3, SOTA 거리 G4. **E23 단독 결과는 SOTA 주장에 쓰지 않는다**(재현일 뿐).
- **+α는 기준선 실패 분석(analysis/2026-09-17-baseline-failure-analysis-plan.md D4/D6)에서 도출**한다. D4에서 검증할 DGFusion 한계 가설(사전 등록, 추측 금지):
  - H1 윈도 평균 풀링 depth 토큰은 얇은 구조를 지움 → DGFusion−CAFuser 클래스별 차가 Pole·TrafficLight·Pedestrian에서 ≈0인가.
  - H2 depth 감독 신호(LiDAR 투영)가 야간·비·lidarjitter에서 희소·노이즈 → depth 헤드 오차가 커지는 조건에서 seg 이득이 사라지는가(이미지별 depth 오차 vs ΔmIoU 상관).
  - H3 유도가 RGB 쿼리 한정 → 비-RGB 모달 특징은 기하 정렬이 안 됨(모달 제거 추론에서 RGB 외 기여 ≈0인가).
  - H4 depth 토큰은 attention만 바꾸고 해상도는 못 바꿈 → 원거리 얇은 객체(면적 구간 최소)에서 이득 없음.
- 가설별 +α 후보(D4 결과로 택1, 단일 변수): H1/H4 → **기하 유도 세부 경로**(예측 depth·depth 경계로 E17 세부 가지의 게이트/샘플링을 구동: "geometry-aware detail"); H2 → **depth 불확실성 인지 유도**(depth 헤드의 학습된 불확실성으로 depth 토큰 기여를 조절 — 엔트로피 바이어스 계열과 다름을 명시); H3 → **모달 대칭 기하 토큰**(모든 센서 스트림에 depth 정렬 토큰, "전 모달 융합" 렌즈와 정합); 공통 → **교차 모달 기하 일관성**(LiDAR 유도 depth를 각 모달 스트림의 공통 타깃으로 두는 내부 신호 감독).
- 시드 규약 불변: 단일 런 최고와 시드 평균 병기, 판정 3시드 평균.

### 사용자 결정이 필요한 후보(정책·비용 충돌) — depth 항목은 위로 이동

| 후보 | 예상 | 충돌 |
|---|---|---|
| **토큰 밀도 상향(1536 또는 세밀 stride 슬라이딩)** | 문헌 +1.5~3 | **우리 실측이 반대**: 1024 학습 3런 test 54.85/54.55 vs 768 54.39(registry, "유해" 판정). 세부 가지가 실패했을 때만 재고 |
| **마스크 분류 주 헤드(M2F 픽셀 디코더)** | +2~4 추정 | 비용 4×, P43 계보 재개 |
| **마지막 2~4블록 MLP LoRA / 부분 해동** | SPAR: MLP > QKV | E2(전 선형층 r32) 무이득, "frozen 백본" 원칙 충돌 → 후순위 |

## 3. 비추천(문헌·실측 근거)

- 백본 토큰 단위 MoE-LoRA(MixLoRA/MoELoRA/LoRAMoE), Soft-MoE 삽입, sparse upcycling, "단일 LoRA→복제·분화" 2단계, 라우터 warm-up·온도 스케줄·z-loss를 성능 변수로, 강한 로드밸런스(희소 모달 데이터 불균형에서 역효과), 전문가 ≥8개, 층별 전문가 수 차등(LLM 근거뿐), MUSES에서 조건 전문가 먼저(조건별 25~34장).
- TokenFusion식 교환, StitchFusion식 전층 stitch(test +0.4), gated 합산→attention 단독 교체, 신뢰도·불확실성 가중(우리 4세대 실패와 정합), 특징 업샘플러(FeatUp/ViT-Up/RaysUp, +0.6), 융합 특징 증류, DFormerv2식 훈련 없는 depth-attention prior(RBMA 실패 재현 위험), Event·LiDAR 추가 기여 기대.
- 클래스 가중 CE·Focal·Recall loss, ABL/InverseForm 단독, consistency·경계 보조·주파수 입력·복원 전처리, 스타일 증강(DINOv2급에서 0), LoRA+/rsLoRA/DoRA/LR 분리, LP-FT·헤드 워밍업·progressive unfreeze, layer-wise decay, 시드 앙상블 증류(N×), 1024 crop(이미 무효).

## 4. 실행 순서 제안

1. **지금(코드 0~반나절)**: D0 진단 2종은 학습 없이 기존 체크포인트로 돌린다(E20·E22 근거 확보). E18(Lovász)·E19(EMA)는 각각 손실 한 줄·트레이너 EMA 한 블록이라 labcode 위임 후 검수, 빈 슬롯 순서대로 40ep 스크린(같은 시드 기준선 902·903 재사용).
2. **E17 결과 후**: 통과 → E17x extractor 설계서 별도 작성. 미달 → 세부 가지 폐기, 토큰 밀도 상향은 여전히 후순위(실측 반대).
3. **사용자 결정 후**: depth 보조 감독은 허용되면 E23으로 등재(DGFusion 손실 설계 그대로 이식: robust L1 + τ 분위 필터 + edge smoothness, +1.05 PQ 차이).
4. 판정은 전부 3시드 평균. EMA(E19)가 분산을 줄이므로 E19 → E18 순으로 확정 런을 돌리면 이후 카드 판정 비용이 준다.

## 5. 출처(arXiv)

MoE-LoRA: 2404.19245, 2404.15159, 2402.08562, 2404.13628, 2308.00951, 2312.02923, 2205.12410, 2110.04260, 2306.03745, 2403.17749, 2401.06066, 2312.12379, 2604.02338, 2609.03150, 2401.17868, 2605.11760, 2212.05055, 2502.19261, 2410.07524, 2510.01185, 2406.04801, 2202.08906, 2503.05139, 2412.04220, 2410.10791, 2605.03555, 2411.18322, 2604.04230.
융합·SOTA: 2509.10408, 2509.09828, 2410.10791, 2604.16984, 2508.10104, 2606.18582, 2205.08534, 2406.01210, 2204.08721, 2408.01343, 2604.02948, 2404.04256, 2504.04701, 2407.11344, 2509.15096, 2412.04220, 2312.00360, 2401.16923, 2603.17705, 2604.03314, 2411.17141, 2509.24505, 2403.10516, 2606.14024, 2606.22749, 2604.02252, 2601.05927.
학습 전략: 2312.04265, 2407.18568, 2202.10054, 2402.12354, 2312.03732, 2402.09353, 2406.10973, 1705.08790, 2102.02696, 2104.02745, 2106.14917, 2605.27962, 2012.07177, 2505.11769, 2012.12645, 2102.08604, 2505.12745, 2204.02548, 2412.12050, 2504.07691, 2304.07193, 2404.12172, 2510.12764.

미확인(명시): GtA 논문·구조, MFESeg ablation 수치, MLE-SAM MUSES 분할, 경쟁 SOTA 3편 전부 시드·분산 미보고(단일 런), FPN 대비 M2F 주 헤드 순이득, Lovász×OHEM 상호작용, EMA의 val→test 전이.
