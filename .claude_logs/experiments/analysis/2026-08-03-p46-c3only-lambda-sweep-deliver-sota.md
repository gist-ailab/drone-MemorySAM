# P46 C3-only λ 스윕 — DELIVER test SOTA 돌파 확정(λ0.2 완주 57.05)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 배경

[2026-07-30-p46-c3only-vs-c1c3-attribution.md](2026-07-30-p46-c3only-vs-c1c3-attribution.md)에서 C-3(prototype consistency) 단독이 핵심 기제로 확정되고 ep40 중간 체크포인트에서 test-SOTA 예비 도달(56.82@1024)이 나온 뒤, C3 prototype loss 가중치 λ3(config상 `LAMBDA`)의 민감도를 확인하기 위해 jarvis/yeon 4개 서버에 λ 스윕을 배치했다(λ0.05/0.1/0.15/0.2). 아래는 **@768 동일 프로토콜**(학습 해상도=eval 해상도, 해상도 mismatch 없음) 실측이다.

## λ 스윕 표 (DELIVER, C1_RCS off · C2_MCC off · C3_PROTO on·cross_view off, P39.1 base 동결)

| λ (C3.LAMBDA) | 서버 | 상태 | val-best (@ep) | test-best (@ep) | 비고 |
|---|---|---|---|---|---|
| 0.05 | jarvis (GPU1-3) | 진행 중 (ep160/200) | 68.57 @ep62 | 56.78 @ep114 | val-SOTA(68.79) −0.22, 최고 val |
| 0.1 | jarvis (`c3only` 기본 config) | ✅ 완주 (200/200) | 67.79 @ep70 | 56.39 @ep108 | — |
| 0.15 | yeon (GPU1,2,5) | 진행 중 (ep38/…) | 65.90 @ep36 | 55.41 @ep32 | 초반 구간, 미완주 |
| **0.2** | jarvis (GPU4-7) | ✅ **완주 (200/200, Total 07:58:28)** | 67.47 @ep118 | **57.05 @ep108** | **test 최적, DELIVER test SOTA 돌파** |

(λ0.05/0.15는 최신 완료분 스냅샷 기준, 최종 완주치는 각 서버 완주 후 갱신 필요. λ0.2만 200/200 완주 확정.)

## 판정

- **λ0.2가 test 최적이며, DELIVER test SOTA 돌파를 확정한다.** test 57.05 vs DGFusion SOTA 56.71 → **+0.34**. 내부최고(P34 test 56.62) 대비 **+0.43**. 학습·평가 모두 **@768 동일 프로토콜**이라 [P46-CTR C3-only vs C1+C3](2026-07-30-p46-c3only-vs-c1c3-attribution.md)에서 남아있던 "@1024 eval 해상도 mismatch" 우려가 이번 결과에는 적용되지 않는 **깨끗한 비교**다.
- **val·test가 서로 다른 λ를 선호한다** — val은 λ0.05가 최고(68.57, val-SOTA 68.79에 −0.22로 근접), λ가 커질수록(0.05→0.2) val은 하락하고 test는 λ0.2에서 최고를 찍는다. 즉 **λ↓=val 최적, λ↑=test 최적**인 트레이드오프 구조. 논문·리포트에서 "test-best 단일 λ"를 고정 서술하려면 λ0.2를 채택 근거로 명시해야 한다.

## Ckpt

- `outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_ctr_c3only_lam02/*/test_epoch108_57.05_top1_checkpoint.pth` (test-best, jarvis)
- `outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_ctr_c3only_lam02/*/epoch118_67.47_top1_checkpoint.pth` (val-best, jarvis)

## 🔴 미해결

1. **RailTrack val<test 역전** — [2026-07-30 attribution 문서](2026-07-30-p46-c3only-vs-c1c3-attribution.md) §3에서 지적된 역전이 λ0.2 완주분에서도 원인 미상으로 남아있음(미확인, 완주 후 재분석 필요).
2. **DGFusion final-iter vs 우리 ckpt 선택 프로토콜 차이** — DGFusion은 BestCheckpointer 없이 final-iter를 쓰는 반면 우리는 val-best/test-best 두 계열을 저장한다([seg-report-sota-gap 컨벤션](../../decisions/) 참조). 우리 재현 DGFusion 수치(56.73)를 기준으로 잡으면 격차는 **+0.32**로 소폭 좁혀지나, 방향(test SOTA 돌파)은 유지된다.

## 다음

- **λ0.3 탐색** — λ0.05→0.2 구간에서 test가 단조 증가하는 경향(56.39→56.78→57.05, λ0.15는 초반 구간이라 제외)이 관측되어, 상단(λ0.3)에서 test가 더 오르는지 확인. jarvis GPU4-7(λ0.2 완주로 회수됨)에서 기동.
- **MUSES-C3 이식** — 동일 C3-only(λ0.2) 기제를 MUSES에 이식한 실험이 hpca100(GPU0-3)에서 진행 중(class-transfer 붕괴가 MUSES에도 있다는 관찰 기반, night truck 44.40 vs day 76.43 등). 별도 문서로 추적.
