# P46-3(C3-only) λ 스윕 — DELIVER 판정 (2026-08-03 작성 / 🔴 **2026-08-04 전면 정정**)

> 🔴 **정정 고지.** 원본 제목은 "DELIVER test SOTA 돌파 확정(λ0.2 완주 57.05)", 파일명은 `...-deliver-sota.md`였다. **그 주장을 철회한다** — 57.05는 **test-best 체크포인트** 값으로, 우리 규약([[seg-report-sota-gap]])과 이 벤치의 두 관행(CMNeXt=val-best / CAFuser·DGFusion=final-iter) **어느 쪽으로도 쓸 수 없다**. 아래는 두 합법 프로토콜로 재계산한 결과다.

## 1. 배경

[2026-07-30-p46-c3only-vs-c1c3-attribution.md](2026-07-30-p46-c3only-vs-c1c3-attribution.md)에서 C-3(prototype consistency) 단독이 핵심 기제로 확정된 뒤, C3 prototype loss 가중치 λ(config `LAMBDA`)의 민감도를 jarvis/yeon에 스윕 배치했다. 학습·평가 모두 **@768 동일 프로토콜**(해상도 mismatch 없음).

## 2. 정정된 결과 — 두 합법 프로토콜 (C1_RCS off · C2_MCC off · C3_PROTO on·cross_view off, P39.1 base 동결)

| λ | 상태 | val-best@ep | **test @ val-best** | **final-iter(ep200) test** | final-iter val |
|---|---|---|---|---|---|
| — (base P39.1-rank) | 완주 | 67.60@106 | 54.34 | 53.95 | 65.88 |
| **0.05** | 완주 | **68.57@62** | **55.62** | **55.69** | 65.81 |
| 0.1 | 완주 | 67.79@70 | *미감사* | *미감사* | — |
| 0.15 | 진행 ep112/200 | 67.02@92 | 54.63 | — | — |
| 0.2 | 완주 | 67.47@118 | 54.60 | **55.69** | 66.78 |
| 0.2-seed2 | 진행 ep106/200 | 67.74@62 | 55.55 | — | — |
| 0.3 | 완주 | 67.83@170 | 54.52 | 55.04 | 67.67 |

**종반 안정성** — ep180~200 test 11점의 폭: λ0.05 55.15~55.74 · λ0.2 55.47~55.73 · λ0.3 54.91~55.56 · base 53.86~54.11. **모두 ±0.3 내**라 final-iter가 스파이크에 휘둘리지 않는다(= 이 수치는 믿을 만하다).

## 3. 판정

1. 🔴 **DELIVER test SOTA(DGFusion 56.71) 돌파 없음.** 최고 legal test ≈ **55.7 (−1.0)**. 두 프로토콜이 일치한다(55.62 / 55.69). 우리 재현 DGFusion(56.73)을 기준으로 잡아도 결론 불변.
2. ✅ **P46-3의 효과는 실재하고 견고하다.** base 대비 test **+1.35**(val-best) / **+1.74**(final-iter), val **+0.97**(val-best). 서로 다른 epoch를 고르는 두 규칙이 같은 방향·비슷한 크기를 낸다 = 우연 아님.
3. **λ 최적은 0.05–0.2 평탄, 0.3에서 악화.** final-iter에서 λ0.05 = λ0.2 = **55.69 동률**, λ0.3만 −0.65.
4. **val은 λ와 함께 오르고 test는 아니다**(final-iter val: 65.81→66.78→67.67 단조 증가 vs test 55.69→55.69→55.04). val-test 궤적 괴리가 이 계열의 특징이며, 이것이 test-best 선택이 유난히 위험했던 이유이기도 하다.

## 4. 🔴 방법론 교훈 — test-best는 헤드라인만이 아니라 *하이퍼파라미터 선택*을 오염시킨다

| 기준 | λ 순위 | 도출됐을 결론 |
|---|---|---|
| test-best (**부정**) | λ0.2(57.05) > λ0.3(56.80) > λ0.05(56.78) | "λ0.2가 정점, 민감도 낮아 견고" |
| val-best (합법) | λ0.05(55.62) > λ0.2(54.60) > λ0.3(54.52) | "λ가 작을수록 좋다" |
| final-iter (합법) | λ0.05 = λ0.2(55.69) > λ0.3(55.04) | "0.05–0.2 평탄, 0.3 악화" |

**순위가 규칙마다 뒤집힌다.** test-best를 믿었다면 잘못된 λ로 후속 실험을 전부 돌렸을 것이다. 학습 로그의 `Best:` 필드는 val과 test가 **독립적으로** 갱신되므로 두 `Best:`를 나란히 인용하면 자동으로 test-peeking이 된다 — 이번 오보의 구조적 원인이다.

**작업 규칙**: 스윕 표에서 **순위를 매기기 전에** 런마다 val-best epoch N을 찾고 `[Test] epoch:N ` 줄의 `mIoU:`를 읽어라. `Best:` 괄호값을 test 열에 넣지 마라.

## 5. Ckpt

- `outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_ctr_c3only_lam005/*/epoch62_68.57_top1_checkpoint.pth` (val-best, **보고용 정본**)
- `outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_ctr_c3only_lam02/*/epoch118_67.47_top1_checkpoint.pth` (val-best)
- ⚠️ `test_epoch*` 계열 ckpt는 **보고·제출에 쓰지 마라** (test-best = test peeking).

## 6. 🔴 미해결 / 다음

1. **RailTrack val<test 역전** — [2026-07-30 attribution 문서](2026-07-30-p46-c3only-vs-c1c3-attribution.md) §3의 역전이 완주분에서도 원인 미상(미확인).
2. **λ0.1 미감사** — 이 표에서 유일하게 test@val-best / final-iter를 안 뽑았다. 완주 로그가 있으므로 확인 가능.
3. **λ0.2-seed2 / λ0.15 완주 후 표 갱신**(재현성).
4. **λ < 0.05 탐색**: base(λ=0)가 54.1이므로 하한은 존재하나, 평탄구간 하단이 어디인지 미확인. GPU 여유 시.
5. **P34 계보와 직접 비교 금지(보류 유지)** — P34(val 68.19/test 56.62)는 ISSUE-026(ColorAugSSD) 픽스 **이전** 런이고 P39.1 계열이 픽스 후 첫 클린 런이다. 두 계보의 test 우열을 단정하지 말 것.

관련: [[seg-report-sota-gap]] · `2026-07-30-p46-c3only-vs-c1c3-attribution.md` · `decisions/2026-08-03-p47-mub-muses-proposal.md`
