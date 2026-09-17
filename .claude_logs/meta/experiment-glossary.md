---
created: 2026-09-17
author: fable(생각정리 세션) — opus 서브에이전트가 카드 문서·제안서·plan·current·arch 원문에서 추출, 생각정리 세션이 검수
status: 참조 문서. 🔴 규칙은 "약어를 쓸 때마다 같은 문장·같은 행에 ①무엇을 보는 실험 ②왜 ③구조 ④결과 해석을 붙인다"이다(user 2026-09-17). 이 문서는 그 설명을 채울 때 참조하는 정본이지, 답변에서 설명을 생략할 근거가 아니다.
---

# 실험·게이트·구조 약어 용어집 (일일 카드 프로그램, 2026-09-17 기준)

출처 약칭: **카드** = decisions/2026-09-07-daily-cycle-experiment-cards.md · **P53제안** = decisions/2026-09-17-p53-detail-branch-proposal.md · **전략탐색** = decisions/2026-09-17-strategy-exploration-moe-lora-fusion-training.md · **plan** = experiments/plan.md · **current** = status/current.md · **arch** = models/arch-evolution.md · **인수인계** = meta/monitoring-session-handover.md

## 1. 카드(실험) 계열 — DELIVER 본판

| 약어 | 풀어 쓴 이름 | ① 무엇을 보나 | ② 왜 | ③ 어떻게(바꾼 변수·config·길이·시드) | ④ 결과가 말해 주는 것 | 출처 |
|---|---|---|---|---|---|---|
| B0 | 기준선 스크린 | 모든 카드의 공통 대조군 | 카드 간 공통 분모 | 변경 없음. DELIVER 4센서, P46 C3-only λ0.1, PhysAug off, DGFUSION_AUG on, 768², 40ep·EVAL 5, 시드 20260821 | legal val 64.97 / test 53.78, RailTrack 31.98, 24클래스 54.69. 통과선 test 54.78·폐기선 54.28 | 카드 §0·§5 |
| B0s2 | 기준선 시드2 | 기준선의 시드 요동 | 페어 판정용 같은 시드 분모 | SEED만 20260902 | test 55.07(+1.29), RailTrack 62.47(30.49 차), 24클래스 54.76(0.07 차) → 24클래스 부기준의 근거 | 카드 §5-6 |
| E0 | 특징 정보 프로브(구조 사다리 S0) | 클래스·센서 정보가 원 특징에 있는데 어댑터·헤드가 버리나 | 구조 카드 전에 정보 잔존 확인 | 선형 헤드만 학습, raw/adapted/fused 세 지점 | raw 27.8 < adapted 35.6 < fused 45.1 → "어댑터가 버린다" 기각. 단 depth 중간층에 Water 47.6·RailTrack 23.9 잔존(fused 16.6·0.0) → E1·E3 유지, E5·E6 하향 | 카드 §1·§5 |
| E1 | 중간층 4탭 읽기(S1) | 중간 블록 6/12/18/24 출력을 함께 읽으면 센서별 클래스 근거가 사나 | E0의 depth 중간층 잔존 | `MODEL.TAPS: [6,12,18,24]`, MODE per_modal, 약 4.3M, non-zero init | 스크린 Δ24 +0.57/+0.78(가장 안정), 확정 3시드 24클래스 55.49, legal val 67.89. MUSES +0.52(통과선 미달), MCubeS +0.10(종결). 헤드라인 후보 아님 | 카드 §1·§5-1·§5-29 |
| E2 | 전 선형층 LoRA(S2) | Q/V만으로 용량이 부족한가 | 용량 병목 가설 | `LORA_TARGETS: [qkv, proj, fc1, fc2]`, r32·α64, 56.6M | test +0.72(회색), 시드2 −1.37, RailTrack 62→4.36 → 폐기 | 카드 §5-1·§5-7·§5-9 |
| E3 | 센서별 클래스 prototype | 클래스 정체성을 센서마다 독립 지지하면 혼동이 줄까(user 가설) | DELIVER는 형상은 맞고 클래스를 헷갈린다 | `P46.C3_PROTO.SRC: permodal` | test +1.40이나 24클래스 −0.14, 시드2 −1.74 → 단독 폐기, E13 구성요소로 생존 | 카드 §5-2·§5-11 |
| E3b | 센서 간 prototype 일치 항 | 센서별 prototype을 서로 당기면 더 좋아지나 | E3 해석 검증 | `AGREE_LAMBDA 0.1` | Δ +0.32, 24클래스 −0.71, RailTrack 70.37→56.99 → 폐기("센서별 prototype은 서로 달라야 한다") | 카드 §5 |
| E4 / E4b / E4c | 혼동 쌍 margin(auto k5 / 명시 5쌍 / RailTrack 3쌍·margin 0.25) | 클래스 쌍별 margin으로 혼동을 분리할 수 있나 | 클래스 혼동 직격 | `CONFUSION_PAIRS`, MARGIN 0.5→0.25 | E4 −0.03 폐기, E4b +1.36→24클래스 +0.08 보류(Adam ComplexFloat 사망, ep35 기준), E4c +1.54→+0.13, 얇은 객체 −0.53 → 계열 종결 | 카드 §5-1·§5-5·§5-8·§5-12 |
| E9 | 검증셋 사전 로짓 보정 | 혼동이 클래스 사전의 도메인 이동 탓인가 | 학습 0 카드 | 추론 시 logit − τ·log p_val, τ 0/0.5/1.0 | 53.57/53.45/51.89 단조 하락 → 폐기(재현율↔정밀도 교환뿐) | 카드 §5 |
| E12 | E1+E2 결합 | 특징 축과 용량 축이 더해지나(예측 +1.8) | 서로 다른 경로로 RailTrack을 푸는 듯 | TAPS + 전 선형층 r32 | +0.61, RailTrack 23.13 < B0 → 가산 기각, E1이 E2에 흡수 → "결합 축은 용량 규모가 비슷해야" | 카드 §5-9 |
| E13 | E1+E3 결합 | 특징 축과 손실 축이 포화인가 가산인가 | E3 이득의 RailTrack 편중 | TAPS on + C3_PROTO permodal(λ0.1, 워밍업 5ep), 확정 EVAL 5 | 스크린 3시드 Δ24 평균 +0.91, 확정 3시드 24클래스 55.33(σ 0.24), val을 잃고 test를 번다. MUSES는 E1보다 우위(3페어 통과), MCubeS 3시드 음수, DELIVER 24클래스·조건별 E1 ≥ E13 | 카드 §5-4·§5-18·§5-22·§5-33 |
| E14 | 얕은 탭 어블레이션 | 얇은 객체 병목이 탭 깊이 문제인가 | E13이 TrafficLight·Static을 벌었다 | TAPS [4,8,12,24] | 24클래스 +0.54 통과, 얇은 객체 46.25 < E13 49.71(−3.46) 미달, TrafficLight −15.98 → 탭 깊이 방향 종결 | 카드 §5-18 |
| E15 | 탭 투영 mean 어블레이션 | E13을 1/4 파라미터로 경량화 가능한가 | 투영 16개가 필요한지 | `TAPS.MODE: mean`(투영 4개, 1.1M; `shared`는 코드에 없음) | 24클래스 54.40 = E13 −1.58, B0보다 낮음 → 경량화 불가(단 원인 분리 안 됨) | 카드 §5-15·§5-19 |
| E17 | 고해상도 세부 가지(DETAIL_BRANCH) = E1 + 세부 경로 | 얇은 객체 손실이 stride-16 격자 밖 정보 부재 탓인가 | 얇은 객체 4클래스 합 −1.85 vs DGFusion | conv 스템 stride-4/8 → 1×1 투영 → FPN에 tanh(g)·detail(g init 0.1), 292,130 파라미터 | 진행 중(hpca100, 09-17 08:23 기동, gate 0.10→0.40/0.22 상승) | P53제안 |
| E17x | E17의 ViT-Adapter extractor 판 | 합산 대신 deformable cross-attn 결합 | add +1.4 vs extractor +2.1 | 피라미드=Q, 4탭=K/V, ×4, 약 24M | 설계만(E17 통과 시) | 전략탐색 §2 |
| E18 | Lovász-Softmax 보조손실 | IoU 직접 최적화가 얇은 클래스를 살리나 | pole +2.2·TrafficLight +3.2 문헌 | OHEM CE 유지 + Lovász 0.5 | 후보(미시도) | 전략탐색 §2 |
| E19 | 가중치 EMA 평가 | EMA가 시드 분산을 줄이고 소물체를 올리나 | GOOSE +1.12 | EMA 0.9995 val 평가·저장 | 후보(헤드라인 규칙 개정 선행) | 전략탐색 §2 |
| E20 / E21 / E22 | 공유 A LoRA / 분할-B 확률 LoRA / 라벨 없는 조건 게이트 | LoRA 구조·정규화·조건 가중 | HydraLoRA / MoSA / CAFuser CAA | 각각 단일 변수 | 후보(E20은 D0 결과 조건부) | 전략탐색 §2 |
| E23 | DGFusion식 depth 보조 감독 이식 + α | depth 유도 기제를 옮기면 그만큼 벌리나, 그 한계를 우리 기제로 넘나 | user 결정(+α 필수) | depth 보조 헤드 + 융합 cross-attn에 depth 토큰; +α는 D4 가설 H1~H4로 택1 | 설계 대기. E23 단독은 SOTA 주장 불가 | 전략탐색 §2 |
| E-LoRA A/B/C | 센서별 r16 / 완전공유 r16 / 공유 r8+센서별 잔차 r8 | 센서별 LoRA 용량이 필요한가 | P50 +0.74가 미정렬 증거 | 시드 821, 200ep, 6.29M/1.57M/3.93M | 1페어: 24클래스 +0.02/+0.36/+0.41, "C ≥ A−0.3" 통과. 25클래스 순위는 RailTrack이 만든다. 3시드 판정 09-20 | 카드 §5-19·§5-23 |
| D0 | LoRA 진단 2종(학습 0) | A 행렬 수렴·조건별 그래디언트 충돌 | E20·E22 근거 | 기존 ckpt로 코사인 측정 | 미실행 | 전략탐색 §2 |
| E5 / E6 / E8 / E10 | deformable 픽셀 디코더 / 블록 간 교환 어댑터 / 클래스 표적 copy-paste / 상위 절반 블록 부분 FT | — | — | — | 전부 미실행(E5·E6은 E0로 하향) | 카드 §1 |

## 2. 데이터셋 접미와 런 종류

| 약어 | 뜻 | 출처 |
|---|---|---|
| `M` 접미(E1M·E13M·E7) | MUSES 이식판(3센서, PhysAug off). 판정은 공식 native 1080×1920 val 250장(`tools/eval_muses_official.py`)으로만. E7 = MUSES PhysAug-off 기준선 | 카드 §1 |
| `Mc` 접미(E1Mc·E13Mc·B0Mc) | MCubeS 이식판(4모달, 200ep). 게이트 = 3시드 평균 Δ ≥ +0.5, final-epoch 기준(로더가 test를 val로 읽어 val-best = test-best) | 카드 §4·§5-21·§5-30 |
| screen40 | 40ep 스크린. 통과 = Δtest ≥ +1.0 + 악조건 −0.5 미만 없음, 부기준 24클래스 Δ ≥ +0.5, 조기 kill ep20 −1.5 | 카드 §0 |
| confirm200 | 200ep 확정 3페어(같은 시드 기준선/카드 쌍). 게이트 = 3페어 mean Δtest ≥ +1.0 and 24클래스 ≥ +0.5. 1페어는 "예비"로만 | 카드 §0-1 |
| 시드 20260821/902/903 | DELIVER 표준 3시드. 시드4 값은 문서 미명시 | 카드 §0 |
| 시드 3407 | 코드 기본 시드. MUSES E7·E1M·E13M 초판은 SEED 키가 없어 전부 3407. 파일명 `seed2`는 P39.1 레시피 이름 | 카드 §5-17 |
| N6 5시드 | DELIVER C3-only 기준선 5런(base·815·816·821·822) | 카드 §5-24 |

## 3. 게이트·판정 규칙

| 약어 | 뜻 | 근거·결과 | 출처 |
|---|---|---|---|
| G1 | 24클래스 게이트: (25×mean − RailTrack)/24 ≥ 분모 + 0.5, 원자료에서 매번 재계산, 분모 이름 병기(스크린 B0 54.69 / 확정 seed821 54.68) | RailTrack이 26.96~72.03 요동·val↔test 역방향. E3·E4b·E4c를 뒤집고 E2s2를 살림. E13 3시드 24클래스 σ 0.24(25클래스 1.11). DELIVER 전용 | 카드 §0·§0-1·§5-6·§5-8 |
| G2 | 얇은 객체 게이트: Pole·Pedestrian·Static·TrafficLight 평균 Δ vs E1 ≥ +2.0 | E17·E18 표적. E14는 −3.46 미달 | P53제안 §4 |
| G3 | 악조건 게이트: 조건별 24클래스 Δ −0.5 미만 없음(DELIVER 5조건, 379~380장) | E1·E13 3시드 통과, night 최대 이득, sun 최소 | 카드 §5-27·§5-30 |
| G4 | SOTA 거리 게이트: 56.99(우리 최고 단일 런)·DGFusion 56.71 대비 거리 병기. 미달이면 통과라도 "기준선 대비 이득 카드" | 2026-09-17 신설(user 지적) | 카드 §0-2 |
| §0-2 모달리티 정합 | 1차 = 같은 모달 집합(DGFusion 56.71·79.5, StitchFusion 55.9), 2차 = 적은 모달(MM SAM-adapter 57.35·81.07, GtA 82.39). 단일 런 최고와 시드 평균 병기 | DELIVER +0.28 · MUSES +0.29 · MCubeS +2.17 | 카드 §0-2 |
| 중간 epoch 비교 금지 | 판정은 40ep 완주(val-best)에서만 | B0 vs B0s2 ep5/10/15 부호 두 번 반전 | 카드 §5-3·§5-10 |

## 4. 구조·손실 약어

| 약어 | 뜻 | 출처 |
|---|---|---|
| C1 RCS / C2 MCC / C3 PROTO | P46 세 구성요소: 희소 클래스 샘플링 / 마스킹 문맥 일관성(EMA teacher) / 클래스 prototype 대조 손실(EMA bank, 학습 전용, λ0.1). 현행 최선 레시피는 C3만("C3-only") | arch §P46 |
| TAPS / MODE per_modal·mean | 다중 블록 탭(E1 키). 도입 커밋 7d83c11(이전 체크아웃은 키 무시 → total_trainable 대조 필수) | 카드 §1·§5-25 |
| gated_mlp 트렁크 | P39.1 R-1: fused += tanh(γ_m)·MLP_m(f_m), γ init 0.1(0이면 학습 불가) | arch §P39.1 |
| VICReg | P39.1 R-2: per-modal 토큰 var/cov 정규화(LiDAR ×1.0) | arch §P39.1 |
| router(P36) | per-class reliability-anchored router, 헤드 잔차. 실제 증분 +0.10 test / +0.76(5조건) — router_off Δ+38~42는 co-adaptation | arch §P36 |
| arbiter | P39 V5: dense·query 경로 per-class 중재 Λ | arch §P39 |
| M2F-lite | Mask2Former-lite 쿼리 헤드(P38). β zero-init로 추론 no-op였던 사례; 현재 병렬 보조 헤드 | arch §P38 |
| DETAIL_BRANCH | E17의 config 키(격자 밖 정보 축) | P53제안 §3 |

## 5. 세대 이름

P36 router 세대(구 내부최고 67.74/55.62) · P38 M2F-lite · P39.1 rank 수리(현행 트렁크) · P43 PanopticDual(E1 탭의 유래) · P44 BMR(gradient 균형, 반증) · P46 CTR(C3 prototype, 헤드라인 56.99 본 런) · P47 UniBal · P52 RxDINO(DELIVER −0.92, 종결) · P53 세부 가지(E17)

## 6. 평가 규약

| 약어 | 뜻 | 출처 |
|---|---|---|
| legal 재채점 | val.py 1024·native GT·BS1 + 하네스 가드 8파일 SHA256 | 카드 §0 |
| val-best top1 규칙 | 학습기 val-best 체크포인트 고정(2026-09-16) | current |
| test-best 인용 금지 | 철회 사고 2회(57.60, 57.05) | current |
| N6 재선택 | 저장 ckpt를 legal val로 재선택 — 다른 규칙, 척도 열로 구분 | 카드 §5-24 |
| 53.83 / 54.39 | 같은 5런의 다른 선택 규칙 값(top1 / N6). 규칙 이름 없이 "5시드 평균" 금지 | 카드 §5-24 |
| 56.99 | P46 C3-only 본 런 ep70. 무시드 런, 시드 고정 재런이 재현 못 함. ckpt 정본 = NAS ckpts/P46_c3only_base_ep70_test5699_20260730 (ISSUE-035) | current·인수인계 |
| 전이율 | Δtest/Δval — 반드시 (Δval, Δtest) 쌍으로 | 카드 §5-14 |
| MUSES 조건별 규칙 | 조건당 25~34장 → 3시드 반복 시에만 실재 | 카드 §5-31b |
| EMM / RMM / NM | 결측 모달 벤치마크 2503.18445: 15조합 zero-fill(정규화 후)·Bernoulli 기대값 / 픽셀 드롭 r / salt-and-pepper(Gaussian은 릴리스 주석 처리). 도구 tools/missing_modality_eval.py. 수치 계열 2종 혼용 금지 | related-work-raw |

## 7. 문서 간 정의가 다른 약어(주의)

1. G1이 세 뜻: 카드 §0(분모 기준선) / P53제안·전략탐색(분모 E1 확정 시드1 55.26) / P52 게이트(DELIVER ≥ 54.95−0.3, 별개 체계). G2·G3도 P52 쪽은 이름만 같다.
2. E7은 카드 §1 1일차 표에 섞여 있으나 MUSES 기준선이다.
3. "5시드 평균" 53.83 vs 54.39는 선택 규칙 차이.
4. 분모 54.69(스크린 B0) vs 54.68(확정 seed821) — 이름 병기 필수.
5. E13M 악조건 게이트의 비교 대상이 문면에 없었음 → E7 기준으로 확정(2026-09-12), 조건별 손실 해석은 §5-31b로 무효화.
6. MCubeS 척도 혼용(final vs val-best)으로 E13Mc Δ가 −0.62 → +0.40 → −0.63 변동(정본 마지막).
7. MUSES 우리 최고는 79.788(79.025는 7월 기록).
8. `seed2`는 레시피 이름(실제 시드 3407).
9. bengio 점유·jarvis GPU0 예약 상태가 plan 내 모순(확인 대상).
