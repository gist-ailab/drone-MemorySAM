---
created: 2026-09-04
author: 이 세션 (worktree p30-det)
status: 🟢 캘리브레이션 완료 — P52 config 작성 시 반영
---

# P52 UniBal-adaptive 캘리브레이션 (P47-2 UniBal 고정런 데이터 기반)

> 선행조건: [2026-08-31-p52-rxdino-adaptive-amendment.md](../../decisions/2026-08-31-p52-rxdino-adaptive-amendment.md) §4-2 "UniBal 고정런 완주·판정" — 이 문서가 그 판정이다.

## 0. 결론

- **P47-2 UniBal 고정런(yeon, `yeon-muses_rgbelr_P47_2_unibal_4modal_recovery.yaml`) 2026-09-03 17:16 완주 확인.** epoch 300/300, best val mIoU **82.06@ep164**, 공식(harness-guard) 재평가 **81.72** → **G2 = 81.42**(이미 확정, 재확인만).
- **UniBal-adaptive 기본 하이퍼파라미터 중 `CAP`가 실측 laziness gap 범위 대비 크게 느슨했다** (기본 2.0, 실측 최대 gap ≈0.64) — 이 상태로는 가장 게으른 모달(radar)조차 λ_u_max의 약 12~15%밖에 못 받는다(아래 §2). **`CAP: 0.7`로 낮출 것을 권고.** `LAMBDA_U_MAX`(0.4)·`EMA_M`(0.99)·`WARMUP_EP`(0)는 변경 불필요 — 근거는 아래.

## 1. 데이터: P47-2 고정런 실측 (per-modal uni-modal CE, N=126 로그 포인트, ep0~300 전 구간)

로그 소스: `yeon:/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/outputs/ReliaDINO/yeon_muses_rgbelr_P47_2_unibal_4modal/MUSES_ReliaDINO-ViTL16_iler/train.log`, `[P47-2] uni_aux:... per-modal ce:...` 라인.

고정런은 λ_u=0.4를 **4모달 균일 적용**(uni_aux = 0.4 × mean(per-modal CE) — 로그값과 일치 확인)했다. per-modal CE 자체(λ 적용 전, UniBalAdaptive.observe가 받는 것과 동일한 신호):

| 모달 | CE mean | CE range | acc mean | gap(L/mean−1) mean | gap range |
|---|---|---|---|---|---|
| img (RGB) | 0.103 | 0.097–0.112 | 96.3% | **−0.583** | [−0.607, −0.551] |
| lidar | 0.227 | 0.212–0.252 | 91.9% | **−0.082** | [−0.099, −0.063] |
| event | 0.267 | 0.247–0.291 | 90.7% | **+0.082** | [+0.060, +0.108] |
| radar | 0.391 | 0.359–0.440 | 86.1% | **+0.583** | [+0.539, +0.636] |

**핵심 관찰**: 이 순서(img≪lidar<event≪radar)와 gap 크기는 **ep0 근처부터 ep300까지 거의 불변**이다(std 0.008~0.021, 순서 역전 0회). 즉 "laziness"는 학습 동역학이 아니라 **센서 물리(RGB 조밀·radar 희소/노이즈)에 뿌리박은 구조적 성질**이다 — 온라인 컨트롤러가 노이즈를 잘못 추적할 위험은 낮고, 반대로 EMA/warmup을 공격적으로 걸 필요도 없다는 뜻이다.

## 2. 문제: 기본 CAP=2.0은 실측 gap 범위(최대 0.636)에 비해 지나치게 느슨하다

`λ_u,m = LAMBDA_U_MAX · clamp(gap_m, 0, CAP) / CAP` (p52.py `UniBalAdaptive.lambdas`). CAP가 분모이므로 실측 gap이 CAP에 크게 못 미치면 λ_u,m은 LAMBDA_U_MAX 근처에도 못 간다.

실제 코드(`UniBalAdaptive`)로 검증(§1 평균값 주입):

| CAP | img | lidar | event | radar(mean gap) | radar(최악 gap=0.636) |
|---|---|---|---|---|---|
| 2.0(기본) | 0 | 0 | 0.0162 | **0.1166** | 0.1534 |
| 0.7(권고) | 0 | 0 | 0.0463 | **0.3331** | **0.4000**(saturate) |
| 0.6 | 0 | 0 | 0.0540 | 0.3887 | 0.4000(saturate) |

기본값(CAP=2.0)으로는 radar가 "P47-2가 균일 λ_u=0.4로 실증한 개선"의 **29%밖에** 못 받는다 — adaptive가 고정런보다 약하게 작동해 G2를 못 맞출 위험. CAP=0.6은 radar를 거의 항상 포화(변별력 손실)시킨다.

## 3. 권고: `CAP: 0.7`

- radar가 평균 동작점에서 λ_u≈0.33(고정런 실증치 0.4의 83%), gap이 조금만 더 벌어지면(≥0.7) 0.4로 포화 — **고정런에서 검증된 강도에 근접하되 완전 포화는 피해 그래디언트 신호를 유지**한다.
- event는 λ_u≈0.046 — 미세한 넛지. lidar/img=0 — **"게으른 모달만 강화"** 계약 그대로 재현.
- G4 정성 게이트("MUSES=radar λ_u↑") 예측과 정확히 일치하는 패턴 확인됨(§2 표).
- `LAMBDA_U_MAX=0.4`는 그대로 둔다 — 고정런이 이미 이 값으로 82.06을 실증했으므로 상한을 새로 추정할 근거가 없다(바꿀 이유 없음).
- `EMA_M=0.99`, `WARMUP_EP=0`도 그대로 — §1에서 본 대로 gap이 ep0 근처부터 이미 정착돼 있어 warmup을 늘릴 근거가 없고, momentum을 조일 근거(고빈도 노이즈)도 관측되지 않았다.

## 4. 한계 — DELIVER·MCubeS엔 실측 근거 없음

이 캘리브레이션은 **MUSES(4모달: img/lidar/event/radar)만의 실측**이다. UniBal-adaptive는 3벤치 완전 동일 config로 공유되는 파라미터이므로, DELIVER·MCubeS의 모달 조합에서 laziness gap이 이 범위(최대 ~0.64)를 크게 벗어나면 CAP=0.7이 최적이 아닐 수 있다. 대응하는 UniBal 고정런이 없어 사전 검증 불가 — **P52 본런 착수 후 [UB-ADPT] 로그(λ_u 궤적)로 벤치별 사후 확인 필요**(G4 판정의 일부이기도 함). 필요 시 벤치별로 CAP만 조정하는 것은 "3벤치 동일 config" 원칙 위반이므로, 만약 다른 벤치에서 명백히 부적합하면 컨트롤러 자체(정규화 방식)를 재검토해야 한다 — 이번 개정에서는 조정하지 않는다.

## 5. P52 config 반영값 (discussion 세션이 config 3벌 작성 시 사용)

```yaml
MODEL:
  UNIBAL_ADAPTIVE:
    ENABLE: true
    LAMBDA_U_MAX: 0.4
    CAP: 0.7
    EMA_M: 0.99
    WARMUP_EP: 0
    WARMUP_SMALL: 0.05
```

검증 스모크(합성 gap 주입, 실제 `p52.UniBalAdaptive` 클래스로 실행 — 인위 데이터 아님, §1 관측치 그대로 재생): 코드 그대로 통과. 별도 pytest 스모크는 config 작성 시 §4 계약("합성 붕괴/게으름 주입 시 λ 반응·eval 불변")과 함께 커밋.

관련: [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md) H20/H22 · [status/current.md](../../status/current.md) §P52
