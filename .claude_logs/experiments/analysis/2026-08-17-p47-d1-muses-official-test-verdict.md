---
created: 2026-08-17
type: MUSES 공식 test 판정 (P47-MUB D-1)
---

# P47-D1 MUSES 공식 test 판정 — 폐기 (2026-08-17)

> Codabench 제출 = user (zip `muses_P47MUB_D1_4modal_ep172_submission.zip`, ep172 val-best = legal ckpt). 판정 = discussion 세션(fable).

## 1. 수치

| 런 | 공식 test | 비교 |
|---|---|---|
| **P47-D1 4모달** (ep172, val 82.58) | **78.790** | — |
| P39.1-rank **4모달** seed2 (val 82.35) | 79.571 | D-1 **−0.781** |
| P39.1-rank **3모달** (val 82.62) | **79.788** (계보 최고) | D-1 −0.998 |
| P38-m2f | 79.025 | D-1 −0.235 |

조건별: clear 78.669 / fog 77.705 / rain 77.575 / snow 77.496 / day 79.551 / night 74.666.
교차: clear_day 79.93 · **fog_night 63.68(최약)** · snow_day 69.47 < **snow_night 74.19 — 주야 역전 3회째 재현**(P34·P38에 이어). 주야 갭 4.89.

## 2. 게이트 대조 (사전 등록 → 결과)

- Primary "4모달 val ≥82.62(3모달 역전)" → **미달**(82.58, −0.04). 공식 test가 확증: 역전은커녕 4모달 base 대비도 후퇴.
- **val→test 낙차 3.79** — 계보 일관치(3모달 2.83 · 4모달 base 2.78)보다 **~1.0pt 초과**. D-1 밀도화(projected_to_rgb_dgf, 4.99× 밀도)는 **val 분포에 과적합**하고 test 일반화를 해친다.

## 3. 판정

1. **P47-D1 폐기.** "LiDAR 투영 밀도화 = 정보축 이득" 가설은 val에서만 성립(+0.23), 공식 test 에서 −0.78 역행. val 이득의 test 미전이 **3번째 사례**(P46-C3 MUSES 이식 −0.765, P49.1과 정합) — MUSES에서 val 단독 이득은 제출 근거가 못 된다는 규칙 강화.
2. **4모달 공식 기록은 79.571(P39.1 seed2) 유지.** 4모달>3모달 역전은 여전히 미달성 — 남은 등록 레버 = **P47-2 UniBal**(구현완료·A100 대기, modality-laziness 처방).
3. 부수 확정: snow 주야 역전 3회 재현(주간 눈 노면의 GT/도메인 특이 가능성 — 분석 절 재료), fog_night 63.68이 전 조건 최약(radar 포함에도 — H11 정합).

## 4. 반영

- registry: D-1 행 상태 → 폐기 판정. memory `muses-dataset-setup` 갱신(제출 4건째).
- Codabench 슬롯 소모 1 — 이후 4모달 제출은 val 82.62 초과 + 낙차 보정 추정 79.8↑ 전망일 때만.

관련: [../../decisions/2026-08-03-p47-mub-muses-proposal.md](../../decisions/2026-08-03-p47-mub-muses-proposal.md) §3 D-1 · [2026-08-16-p49-1-muses-official-verdict.md](2026-08-16-p49-1-muses-official-verdict.md)
