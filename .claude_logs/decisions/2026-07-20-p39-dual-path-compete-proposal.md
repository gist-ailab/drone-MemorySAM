---
created: 2026-07-20
scope: P39 모델 제안 (설계안, 구현 전) — 실패-키 문서(2026-07-20) 전 키 반영
constraint: 단일 아키텍처로 DELIVER·MUSES 모두 커버 (user 지정) — 데이터셋 적응은 학습된 모듈로만
gates: DELIVER = P36 fair val 67.74/test 55.62 + thin-class(Wall/Water/RailTrack) IoU · MUSES = P38 val 82.22 (신규 내부 최고)
---

# P39 — Dual-Path Compete (DPC) 제안

## 0. 설계 원칙 (실패-키 → 규칙 변환)

| 키 | 규칙 |
|---|---|
| 키1 zero-init 잔차 사장 (4연속) | **소극 잔차 결선 전면 금지.** 모든 신규 경로는 (a) 주 손실을 직접 받거나 (b) 기존 경로와 경쟁(path dropout)한다 |
| 키2 router 유일 실적 + co-adaptation | router를 **직접 감독**으로 "의존"에서 "기여"로 전환하고, deep-sup의 의존 해소는 유지 |
| 키3 FUSED rank 7/256 병목 | 로짓 근처 모듈 추가 금지. **융합 트렁크의 rank를 직접 넓히고**, query 경로는 병목을 **우회** |
| 키4 문제 위치 상이 (클래스축 vs 도메인축) | 해법을 클래스 단위로 학습되는 **per-class 중재**로 — 데이터셋별 상이한 병목에 같은 기제가 다르게 적응 |
| 키5 event 기여 = 데이터셋 속성 | 모달 하드 제거 없음. query가 **모달 토큰을 직접 attend**해 데이터셋별로 스스로 배분 |
| 반증 C1~C6 | attn-bias·gate류 신규 없음 · conv head 즉시 대체 없음(P30) · 무감독 게이트 없음 |

## 1. 구조 (P38 대비 변경 5개, 전부 토글 가능)

베이스 유지: frozen DINOv3-L + per-modal LoRA(r8) + cross-modal fusion + SimpleFPN + FPNSegHead + per-class router. 이하 변경:

### V1 — Trunk Rank Expansion (키3, 융합 트렁크)
`fused' = fused + Σ_m P_m(f_m)` — 모달별 선형 투영 P_m(1024→1024, small-random init, **zero-init 아님**)을 트렁크에 **가산 합류**. 게이트 뒤에서 소실된 모달 부분공간을 주 경로에 복원 → rank 상한을 Σ modal rank로 확장. 주 경로 소속이므로 첫 스텝부터 CE gradient를 받음(키1 충족). +4.2M params.

### V2 — Modal-token Query Attention (키3 우회 + 키5)
m2f query의 cross-attn 소스를 [기존] fused map (N tokens) → [P39] **per-modal 토큰 합집합 (M·N tokens + modality embedding)**. query가 융합 병목(rank 7)을 거치지 않고 인코더 피쳐(rank 20~36)를 직접 봄. 모달 배분은 attention이 학습 — MUSES는 event/lidar를, DELIVER는 event를 무시하는 식으로 **데이터셋 적응이 파라미터가 아니라 학습으로** 일어남 (단일 모델 제약 충족). mask dot-product는 기존대로 FPN stride-4 feat.

### V3 — Anchored + Free Queries (키2 thin-class + P30/P37b 교훈 합성)
query 100개 = **앵커 K개(클래스 고정 할당, Hungarian 없음 — P37b 방식, 단 이번엔 마스크 손실 직접 감독) + 자유 (100−K)개(Hungarian — 인스턴스/PQ 담당)**. 앵커 query는 thin/희소 클래스가 매칭에서 굶는 문제(P38 요인: Hungarian 기아)를 구조적으로 제거 — Wall/Bridge/Other도 매 스텝 손실을 받는 전용 query 보유. PQ 추론은 자유 query가 thing을, 앵커 query가 stuff를 담당.

### V4 — Balanced Point Sampling (P38 요인2 직접 수정)
mask BCE/dice 포인트 샘플링을 uniform 12,544 → **GT 영역별 최소 쿼터(클래스당 ≥256pt) + 잔여 uniform**. thin 마스크(RoadLine·Wall·RailTrack 띠)가 포인트 예산에서 소멸하는 문제 제거. 학습 손실만 변경, 추론 불변.

### V5 — Compete-and-Arbitrate 결합 (키1 핵심 적용, β 잔차 폐기)
dense 경로(conv head + router 잔차)와 query 경로(anchored+free semantic 조립)의 결합을 zero-init β 대신:
- **학습: path dropout 경쟁** — 확률 p_d=0.25 dense-only CE / p_q=0.25 query-only CE / 나머지 결합 CE. 양 경로 모두 주 손실을 단독으로 감당해야 하므로 어느 쪽도 무임승차(no-op 고착)가 불가능(키1-b).
- **추론: per-class 학습 중재** `final_k = dense_k + softplus(Λ_k)·query_k` — Λ는 K-dim 학습 파라미터(init 0 → softplus≈0.69, 죽은 시작 아님). query-only 턴에서 query 경로가 주 손실을 받으므로 Λ의 gradient가 실질임. 데이터셋·클래스별로 "어느 head를 믿을지"가 학습됨 — DELIVER thin-stuff는 dense+router로, MUSES 대형 동적 객체는 query로 기우는 것이 기대 동작이자 **검증 대상 예측**.
- **router 직접 감독**: `CE(up(routed_logits), gt)`(w=0.4)를 추가 — router가 결정경로 의존이 아니라 자립 기여로 학습(키2). deep-sup은 유지(의존 해소 실적).

## 2. 손실 스택

`L = CE_compete(final|dense|query) + w_r·CE(routed) + mask-cls(anchored 고정매칭 + free Hungarian, V4 샘플링, 2/5/5, deep-sup) + aux_ce + cal (기존 유지)`

## 3. 사전 등록 판정 게이트 & 예측

| 벤치 | 게이트 | P39 예측 (falsifiable) |
|---|---|---|
| DELIVER | P36 fair val 67.74/test 55.62 **+ Wall≥13/Water≥9.5/RailTrack≥62 (P36 수준 복원)** | V3+V5로 thin-class 회복 유지 + V1/V2로 상승 여지 |
| MUSES | **val 82.22 (P38)** 이상 | V2가 event/lidar 정보를 직접 쓰므로 ≥ 유지 예상 |
| 모듈 | 학습 직후 `module_ablation.py` 토글 즉검 (완주 후 발견 금지, 체크리스트 E) | p39_arbiter_off / p39_trunkexp_off / p39_anchored_off / router_off 각각 |Δ|>0.5 & agreement<0.99 (no-op 조기 탈락 기준) |

## 4. 리스크와 방어

- **P30 재발(query-only 붕괴)**: p_q=0.25 한정 + V3 앵커 + V4 쿼터가 원인(소물체 기아)을 직접 제거. dense 경로는 추론에 항상 존재.
- **V1 rank 확장이 게이트 서사 훼손?** — 게이트/신뢰도는 유지(무해 확인됨), V1은 게이트 뒤 보강이라 스토리 충돌 없음.
- **변경 5개 = 다변수**: 전 항목 토글 구현 의무화로 ablation 표에서 분해 (deadline상 1-변수 5세대는 불가 — 키 문서가 지목한 결합 문제는 단독 변경으로 풀리지 않음).
- physaug: 공정선 유지(헤드라인 off, ablation 행만) — 키6은 레시피 변수로 별도 관리.

## 5. 실행 계획 (제안)

1. 구현: V1~V5 + 토글 5종 + config 2벌(deliver/muses — ROOT·모달만 상이, 아키 동일) — 1일
2. 스모크(yeon 빈 GPU 2ep) → hpca100/jarvis 첫 빈 슬롯 투입 (DELIVER 우선, MUSES는 jarvis P38 완주 후 이어달리기)
3. ep30 조기판정: module_ablation 토글 즉검 + val 궤적 vs P36/P38 동 epoch — no-op 검출 시 조기 중단(2026-07-16 EPOCHS 사고 규칙)
