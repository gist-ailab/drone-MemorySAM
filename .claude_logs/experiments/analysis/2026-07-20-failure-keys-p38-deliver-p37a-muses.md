# 종합 실패-키: P38-DELIVER × P37a-MUSES 표준분석 (차기 구조 설계 인계 문서, 2026-07-20)

> **용도**: 새 모델 구조를 설계·구현할 에이전트 인계용. 단일 설계안이 아니라 **"어디가 문제이고 어디서 실패했는가"의 키 목록**.
> **근거 로그** (NAS 루트 `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/`):
> - DELIVER: `P38_eval_20260719/` (P38 test-best ep62, 5-cond 전 스테이지) + 판정 [2026-07-19-p38-m2f-standard-analysis.md](2026-07-19-p38-m2f-standard-analysis.md)
> - MUSES: `P37a_muses_std_20260719/` (P37a ep110, 6-cond 전 스테이지) + `P37a_muses_20260718/` (CEFR 라우팅 프로브) + 판정 [2026-07-18-p37a-muses-cefr-output-analysis.md](2026-07-18-p37a-muses-cefr-output-analysis.md)
> - 계보 배경: [2026-07-12-p29-p34-standard-analysis.md](2026-07-12-p29-p34-standard-analysis.md), [../../decisions/2026-07-16-p36-novelty-critical-review.md](../../decisions/2026-07-16-p36-novelty-critical-review.md)

## A. 두 데이터셋에서 교차 검증된 실패 키 (구조적 문제)

### 키 1 — "zero-init 잔차/게이트 결선"이 모듈을 사장시킨다 (최상위 키)
- 증상: 제안 모듈이 **전부 추론 no-op**으로 수렴하는 패턴이 결선 방식과 정확히 겹침.
  - m2f β 잔차: 학습 후 β=0.133까지만 열림 → off Δ **+0.04~+0.12** (DELIVER)
  - CEFR σ(a) blend: 0.018→0.121~0.17까지만 열림 → off Δ **+0.04~+0.17** (MUSES), per-class 라우팅 **0/19 미분화**(전역 event-틸트로 퇴화)
  - P37b mask_proj: 게이트 전용 소비 + 무손실 → 영구 random (버그로 확정)
- 실패 판정: collapse-safe를 위해 zero-init 소극 결선을 쓰면, 기존 경로(gate-fused/FPN)가 이미 손실을 다 내리므로 **새 경로는 gradient 몫을 못 받고 "약간 열리다 만" 상태로 고착**된다. 4번 연속 같은 사망 방식.
- 설계가 건드릴 지점: 새 모듈은 (a) 주 손실을 직접 받거나(deep-supervision — P38에서 유일하게 학습 동역학을 바꾼 장치), (b) 기존 경로를 대체·경쟁시키거나(예: 경로 dropout, β 강제 스케줄), (c) 아예 넣지 말 것. "잔차로 살짝 얹기"는 반증 완료.

### 키 2 — router 잔차 = 유일한 지배 모듈이나, 기여가 아닌 co-adaptation 의존 + 단일 실패점
- 증상: p36_router_off Δ = MUSES **+22.6~+35.6**(agreement 0.56~0.72), DELIVER P36 +38~42. 표현 자체가 router 경로 위에 얹혀 있어 끄면 붕괴.
- 반전: P38(m2f deep-sup 존재)에서는 **+1.6~2.4로 급감** — deep-supervision이 router 우회 표현을 만들 수 있음을 실증. 단 P38은 성능(게이트 미달)을 못 얻어 절반의 성공.
- 실적: thin/rare-class 회복은 계보 전체에서 **router 잔차만** 해냄 (P36: Wall 6.0→13.3, Water 5.3→9.5, RailTrack 56.1→62.5; P31 동일 계열).
- 설계가 건드릴 지점: "router의 thin-class 회복력"과 "deep-sup의 의존성 해소"를 **동시에** 갖는 결합이 미해결 과제. P38은 m2f가 router를 희석해 thin-class를 되잃음(Wall 13.3→≈7).

### 키 3 — fusion이 정보 병목: per-modal rank 20~36 → FUSED rank 6.8~8.0/256
- 두 데이터셋 공통: 인코더 피쳐는 건강(dead-ch≈0, rank 20~36, DINOv3 계보 유지)한데 **융합 직후 effective rank가 7 안팎으로 붕괴**. (DELIVER FUSED 7.1~7.7, MUSES 6.8~8.0.)
- 제안 모듈들이 전부 fusion 이후·로짓 근처에서 작동하며 no-op이 된 것과 정합 — **병목 위에서 뭘 더해도 안 변함**.
- 설계가 건드릴 지점: fusion 자체(256ch 압축·게이트 가중합 구조)가 표현을 좁힘. 로짓 근처 모듈 추가보다 **융합 단계의 용량/구조**가 개입 지점.

### 키 4 — 성능 문제의 "위치"가 데이터셋마다 다르다 (단일 해법 없음)
| | DELIVER | MUSES |
|---|---|---|
| per-domain spread | **2.58** (도메인 균일) | **14.88** (도메인시프트 실재) |
| 최약 조건 | night 52.1 (경미) | **fog 62.7** (clear 74.5, night 77.6 — night는 약점 아님!) |
| 병목 클래스 | **thin-class 사망**: Other≤5, Bridge≤1, Wall≈7, Water≈11 (도메인 불변) | 희소 동적 클래스의 조건별 결손: rider/traffic light 0@fog, truck/bus 0@rain 등 (⚠️ 일부는 소표본 클래스-부재 아티팩트 가능 — 판독 시 GT 존재 확인 필요) |
| 필요한 해법 방향 | 클래스 축 (thin-class 표현/손실) | 도메인 축 (fog 강건화) + 희소클래스 |

## B. 데이터셋 조건부 키 (모달리티)

### 키 5 — 모달 기여는 데이터셋 속성이다: "event 무용"은 DELIVER 한정
- DELIVER (P38): event adapter **dead** (Δacc −0.003~+0.020, cloud/fog 음수) — P32·P34·det ablation 포함 4세대 일관.
- MUSES (P37a): **event 강력** (Δacc **+0.24~+0.29** 전 조건), lidar는 필수(+0.31~+0.42, off 시 acc 0.45~0.54로 붕괴), img조차 +0.03~+0.10.
- 함의: (a) DELIVER 4-modal에서 event 제거/교체는 정당, (b) MUSES 3-modal은 전부 유효 — 멀티센서 융합 스토리는 MUSES에서 세울 것, (c) P37a CEFR의 전역 event-틸트(0.385)는 미분화의 산물이지만 방향 자체는 실제 정보량과 정합.

### 키 6 — 학습 레시피 키: physaug가 P34 우위의 실체
- P34→P35 하락(−1.12)의 전부가 physaug 제거(Static −13.8, Pole −5). P36+physaug bounded run이 Day-Val 계보 최고(68.76)로 실증. 새 모델 학습 시 physaug 포함이 아키텍처보다 큰 단일 변수일 수 있음 — ablation 시 반드시 분리할 것.

## C. 반증된 경로 (재시도 금지)

1. **RBMA/CoRB attn-bias** 계열 — 4세대·2백본에서 효과 0 또는 유의한 순손해(P32 p=4.5e-22).
2. **reliability gate/calib/veto** — 3세대 no-op (이번 두 분석에서 각각 재재현: |Δ|≤0.5).
3. **CEFR class-expected routing** — per-class 미분화(0/19), 전역 재가중으로 퇴화.
4. **무감독 threshold mask 게이트** (P37b) — 영구 random.
5. **m2f semantic 잔차 헤드** (P38) — 추론 no-op + thin-class 퇴행 + 게이트 미달.
6. **query-decoder로 conv 헤드 즉시 대체** (P30) — 소물체 붕괴.
7. **fusion rank/η² 개입** (P41 FCR, 2026-07-23) — fused between-class 분산비 η²를 supervised aux로 최대화(0.35→0.94, 2.7×)해도 **mIoU 무이득**(≈P38). 학습0 Phase-0 판별 + 사전등록 게이트로 확정한 **airtight falsification**: 키3의 "fusion rank 붕괴"는 증상이지 성능 병목 아님. decode가 이미 클래스정보 추출(P38 decode η² 0.63)해 fusion 사전정렬은 중복. 상세 = [../../decisions/2026-07-22-p41-fusion-spectral-discrimination-proposal.md]. **→ MUSES 병목은 fusion이 아니라 fog(clear 75.85/fog 62.67/night 78.05, −13pt).**

## D. 실증된 경로 (유지·조합 대상)

1. **DINOv3-L frozen + per-modality LoRA** — P34 도약의 실체 (SAM2 대비 +2.6, rank·CKA 건강). 백본은 확정, 건드리지 말 것.
2. **per-class router 잔차** — thin-class 회복 유일 실적.
3. **physaug** — 최대 단일 레시피 변수.
4. **deep-supervision** — router 단일 의존을 해소한 유일 장치 (P38에서 구조 효과만 실증, 성능 미달).
5. **MUSES night 강건성** — night 77.6 > fog 62.7: 야간이 아니라 fog·희소클래스가 남은 문제.

## E. 구현 인계 체크리스트

- 새 모듈은 반드시 **사전 등록된 판정 게이트**와 함께: (현 기준) DELIVER = P36 fair val 67.74/test 55.62 + thin-class(Wall/Water/RailTrack) IoU, MUSES = P37a val 81.16. val-best ckpt 규칙([[seg-report-sota-gap]]) 준수.
- 모듈 순기여는 `tools/module_ablation.py` 토글로 학습 직후 즉시 검증 (no-op 조기 검출 — 이번처럼 완주 후 발견하지 말 것). 신규 모듈엔 토글을 함께 구현할 것.
- 표준분석은 `tools/seg_analysis_pipeline.py` 하나로 DELIVER(test/5-cond)·MUSES(val/6-cond, `--split val --conditions clear,fog,rain,snow,day,night`) 모두 지원 (develop ≥21d112e).
- MUSES per_domain 표의 clear/snow/day 열 누락은 analyze 스크립트의 DELIVER 조건명 고정 탓 — 원시 로그 `per_domain/best__{clear,snow,day}.log`에 수치 있음 (clear 74.49/snow 77.29/day 80.91).
