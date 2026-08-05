# P46-CTR C3-only — fair-eval 최종 판정 (val-best only, 2026-08-06)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 결과표 (val-best ckpt만 — test-best 사용 안 함)

| Run | ckpt(val@ep) | 해상도 | val mIoU | test mIoU | RailTrack test |
|---|---|---|---|---|---|
| 본(original) | 67.79@ep70 | @768 | 67.80 | 55.91 | 72.03 |
| 본(original) | 67.79@ep70 | @1024 | **69.44** | **56.99** | 67.69 |
| seed2 | 67.18@ep62 | @768 | 67.20 | 55.12 | 55.73 |
| seed2 | 67.18@ep62 | @1024 | 68.90 | 56.40 | — |

## 판정 (코디네이터)

**legal 최고 = 본(original) val-best@ep70, @1024 평가 → val 69.44 / test 56.99.**

- **현행 DELIVER SOTA(MM SAM-adapter, val 69.60 / test 57.35) 대비 val −0.16 / test −0.36 = 미돌파.**
- 단 **이전 test-SOTA 기준이던 DGFusion(val 66.51 / test 56.71)은 val·test 동시 상회(+2.93 / +0.28) = no-tradeoff 우위.**
- **RailTrack 67.69~72.03**(base 4.02, DGFusion 64.47 상회) — class-transfer 복구가 헤드라인 수치를 실제로 견인한 것으로 판정.

## 프로토콜 명시

1. **val-best ckpt만 사용**(test-best 미사용) — seg-report-sota-gap 규약 준수.
2. **학습 @768 / 평가 @1024** — 해상도 선택은 **val 기준으로 정당화**(val@1024 69.44 > val@768 67.80; test 수치를 보고 고른 게 아니다). 경쟁군(DGFusion 등)의 native-res 평가 관행과 정합.
3. **seed2는 test 56.40으로 −0.59** — 단일 런 편차가 존재함을 보여주는 재현성 단서(본 run 대비 낮지만 방향은 동일 — 양쪽 모두 @1024가 @768보다 우수, RailTrack 모두 게이트 압도적 통과).

## 미결

- 논문 기재 시 **train/eval 해상도 mismatch(@768 학습 → @1024 평가)를 명시**해야 한다.
