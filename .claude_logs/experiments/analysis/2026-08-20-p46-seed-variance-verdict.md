---
created: 2026-08-20
type: P46 C3-only 진짜-시드 분산 판정 (DELIVER 근SOTA 주장 반증)
---

# P46 C3-only 진짜-시드 분산 — "사정권" 주장 붕괴 (2026-08-20)

> 실측 = 모니터링 세션(jarvis seed815/816 @768 완주 → yeon val.py legal 재평가). 교란 검증(@768 확정) + 판정 = discussion 세션(fable). 정본 측정기 = val.py native-GT @1024.

## 1. 수치 (legal, @768학습/@1024평가 — base와 동일 프로토콜)

| 런 | val | test |
|---|---|---|
| base(본run, ep70) | 69.44 | **56.99** |
| seed815(ep150, SEED 20260815) | 66.88 | 53.25 |
| seed816(ep90, SEED 20260816) | 67.62 | 53.08 |
| **mean ± std (n=3, sample)** | **67.98 ± 1.32** | **54.44 ± 2.21** |

- @768 학습 **로그 config 덤프로 확정**(`IMAGE_SIZE: [768,768]`) — @1024 오염(원장 P46@1024 유해 −2pt) 아님. config develop 커밋(c4d5166).
- **base가 val·test 둘 다 3점 중 최댓값.** 선별(pool에서 고름)은 아니나(이 recipe 첫 학습이 base) 결과적으로 분포 상단.

## 2. 판정 — 3주간의 근SOTA 서사 반증

1. **"사정권" 주장 붕괴.** current.md는 "SOTA 격차 −0.36 < 단일런 편차 0.59 → 사정권"이었으나, 그 0.59는 **가짜-시드**(`fix_seeds(3407)` 하드코딩, `C3.SEED`는 C1 off 시 inert) = GPU 비결정성만 잰 값. **참 시드 std = test 2.21**(3.7배). mean test 54.44는 MM-SA(57.35) **−2.91**, DGFusion(56.71) −2.27 — 근SOTA 아님.
2. **단일런 56.99를 헤드라인으로 못 쓴다.** 3점 중 최댓값이고 mean이 −2.5. "DGFusion no-tradeoff +0.28 상회"도 mean 기준 −2.27로 소멸.
3. **분산 성격**: val·test **양의상관**(base 둘 다 최고, 시드 둘 다 낮음) = test-도메인 노이즈가 아니라 **런 품질 분산**(어떤 시드는 전반적으로 더 나은 해로 수렴). val std 1.32도 유의 → 파인튠 수렴 자체가 seed에 민감.

## 3. 미결 — 분산이 줄일 수 있는 것인가 (후속 판단 재료)

- **seed-init vs ckpt-선택**: 트레이너 val-best(768-리사이즈-GT 낙관지표)가 legal의 나쁜 대리 → val-best 선택이 legal 관점에선 준-랜덤일 수 있다. **legal-val 기준 재선택**(각 시드의 저장 ckpt를 val.py legal val로 재선택)으로 저-시드가 base에 근접하면 분산은 선택 아티팩트(축소 가능), 아니면 근본. **미검 — 값싼 후속(eval만).**
- **경쟁자 비대칭**: SOTA 논문들은 대개 **단일런 보고**(variance 미공개, 종종 best-ish). 우리 mean-of-3 vs 그들 단일런 = 우리가 엄밀성으로 불리하게 비교됨. 공정 비교면 우리 best(56.99) vs 그들 단일, 또는 양쪽 mean(그들 미공개). **정직한 보고 = mean±std 주 + best 병기 + 비교 비대칭 명시.**

## 4. 함의 (서사·다음)

- **DELIVER 헤드라인 = 근SOTA 주장 철회.** 남는 것: ① mean±std 엄밀 보고(대다수 경쟁자보다 rigorous) ② 분석 기여(캠페인 음성결과·모달 잉여/상보 분해·오라클)는 시드-강건(수치 헤드라인 아닌 기제) ③ 분산 축소 시도(legal-val 재선택 / 다중 시드 / EMA·last-k 평균).
- 🔴 **MUSES 평행 리스크**: MUSES test 79.788도 **단일 제출**(seed2만 Codabench). 이것도 유리한 draw일 수 있다 — MUSES 시드 test 분산 **미측정**(val은 81.8~82.62 spread 확인). 근SOTA 아닌 "융합계보 1위" 주장이라 덜 취약하나, 헤드라인 쓰기 전 재확인 필요 항목으로 등재.
- **user 전략 판단 필요**(내가 대신 못 정함): 근SOTA 포기하고 "엄밀한 음성-결과+분석" 논문으로 갈지, 분산 축소(재선택/다중시드)에 자원 넣어 헤드라인 회복 시도할지.

## 5. 원장/기록 반영

- ledger **H18 신설**: "P46 C3-only legal test는 seed에 강건(단일런 대표 가능)" → ✗ 반증(σ=2.21, base=max).
- current.md DELIVER 행 + 시드 서술 갱신(완료). memory `seg-report-sota-gap` 갱신 필요.
- 종합 §의 "DELIVER 56.99(SOTA −0.36)" 문구 전부 → "54.44±2.21(−2.9)"로 정정 대상.

관련: [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md) H18 · [seg-report-sota-gap 메모리] · configs `jarvis-deliver_rgbdel_P46_ctr_c3only_lam01_seed2026081{5,6}.yaml`(develop c4d5166) · 원시 yeon `logs/eval_seed{815,816}/`
