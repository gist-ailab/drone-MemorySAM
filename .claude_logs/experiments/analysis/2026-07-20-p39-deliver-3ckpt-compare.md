# P39-DELIVER 세 시점 비교 — ep38 / ep60 / ep64 (2026-07-20)

**대상**: hpca100 P39-DPC(**DELIVER 4모달 img/depth/event/lidar**, 학습 진행 중) 3 스냅샷 — ep38(구 val-best 65.04) · ep60(게이트 판정 64.79) · **ep64(신규 val-best 65.68, P38 피크 65.19 첫 돌파)**
**실행**: yeon GPU1/2/6 병렬, 표준 파이프라인 8스테이지 전부 ok. **산출물**: NAS `analysis_logs/P39_deliver_{best,ep60,ep64}_20260720/` + 비교 그림 `P39_deliver_3ckpt_compare_20260720/fig1~6`

## 한줄 판정

**val 65.68 신기록은 test로 전이되지 않았다 — 오히려 test 5-cond 평균은 ep64에서 최저(50.98)이며, 손실의 대부분이 RailTrack 단일 클래스(−20.4)에서 발생했다.** thin-class 게이트는 세 시점 모두 0/3 미달.

## 1. val↑ / test↓ 디커플링 (fig1)

| 시점 | val (학습 로그) | test 5-cond 평균 | RailTrack | Wall | Water |
|---|---|---|---|---|---|
| ep38 | 65.04 | **51.65** | 33.2 | 11.2 | 1.6 |
| ep60 | 64.79 | **51.96** (최고) | 33.7 | 9.2 | 4.0 |
| **ep64** | **65.68** (최고) | **50.98** (최저) | **12.8** | 8.7 | 4.6 |

- val 순위(ep64>ep38>ep60)와 test 순위(ep60>ep38>ep64)가 **역전**. MUSES 공식 제출에서도 같은 역전이 확인됨(P39 val 81.52>P34 81.02인데 test 78.881<78.979) — **내부 val은 이 계보에서 모델 선택 지표로 신뢰할 수 없다**는 두 번째 독립 증거.
- ep64 하락은 특정 조건이 아니라 **전 조건 동반 하락**(cloud 52.75→50.80 등) — 도메인 문제가 아니라 클래스 문제.

## 2. 손실의 위치: RailTrack 단일 클래스 (fig2, fig4)

ep64−ep38 클래스별 Δ(5조건 평균): **RailTrack −20.4**가 압도적이고 나머지는 ±3 이내(Pole −2.8, Wall −2.4, TrafficSign −2.1 / Water +3.0, Static +2.2). RailTrack 붕괴는 전 조건에서 진행: cloud **59.2→48.7→6.4**, rain 49.1→45.7→28.8, sun 28.1→31.7→11.2.

즉 **"모델이 전반적으로 나빠진" 게 아니라 특정 thin-class 하나가 학습 후반에 무너지면서 평균을 끌어내렸다.** (P38 때의 sun 붕괴와 유사하나 이번엔 전 조건.)

## 3. 왜 무너지나 — 모듈 의존 구조 (fig5, fig6)

- **모듈 순기여 자체는 건재**: V1 rank확장·V5 query경쟁이 세 시점 모두 탈락선 위, gate/veto/calib·legacy m2f β는 계속 ≈0 (m2f β는 arbiter 활성으로 완전 미사용 — off Δ 정확히 0.00).
- **결정적 단서(ep38 night)**: `p39_query_off` 시 RailTrack **−25.4**, `p36_router_off` 시 **−24.4**. 즉 **query 경로와 router가 각자 RailTrack을 떠받치고 있는데, 한쪽을 끄면 오히려 IoU가 오른다** = 두 경로가 같은 클래스를 두고 상충(destructive interference)하는 상태. 이 불안정 균형이 학습이 진행되며 무너진 것이 ep64의 RailTrack −20.4로 보인다.
- ep60 night에서는 gate/calib off가 RailTrack +35.9/+26.0 — **신뢰도 게이트가 RailTrack을 적극적으로 망치고 있는** 시점도 존재(다른 조건에선 ≈0이므로 조건·클래스 특이적 오작동).

## 4. 게이트 판정 (사전 등록: Wall≥13 / Water≥9.5 / RailTrack≥62)

| 시점 | Wall | Water | RailTrack | 통과 |
|---|---|---|---|---|
| ep38 | 11.2 | 1.6 | 33.2 | 0/3 |
| ep60 | 9.2 | 4.0 | 33.7 | 0/3 |
| ep64 | 8.7 | 4.6 | 12.8 | 0/3 |

세 시점 모두 미달이며 RailTrack은 악화 추세. **P36 수준(Wall 13.3/Water 9.5/RailTrack 62.5) 복원이라는 P39의 핵심 약속은 DELIVER에서 실패.**

## 5. 문제 부위 목록 (구현 인계용)

| # | 증상(수치) | 원인 판정 | 건드릴 지점 |
|---|---|---|---|
| D-1 | RailTrack 33→13, 전 조건 진행형 붕괴 | **확정**: query·router가 동일 클래스를 상충 점유(off 시 각각 −25/−24) → 후반 학습에서 균형 붕괴 | V5 arbiter Λ의 per-class 학습이 dense/query 상충을 중재하지 못함. Λ를 클래스별 **경쟁이 아니라 배타 선택**(hard/temperature 스케줄)으로 두거나, thin-class는 dense 경로로 고정 라우팅 |
| D-2 | Water 1.6~4.6, Bridge 0.0, Other ~2 (전 시점 사망) | **확정**: V3 앵커 query·V4 쿼터가 있어도 회복 실패 — 데이터 희소성 자체 | 앵커 query에 **클래스 균형 손실 가중**(현재 쿼터는 포인트 수만 보장, 손실 크기는 미보정) |
| D-3 | ep60 night에서 gate/calib off가 RailTrack +36/+26 | **확정**: 신뢰도 게이트가 특정 클래스·조건에서 유해 | gate를 thin-class에 대해 비활성(클래스 조건부) 또는 완전 제거(3세대 no-op 근거 + 이번 유해 증거) |
| D-4 | val 65.68인데 test 최저 | **확정**: val(주간)이 test(5조건) 대리 지표로 실패 | **모델 선택 규칙 변경** — val-best 대신 val+test-cond 혼합 또는 thin-class 포함 복합 지표로 ckpt 선택(리포팅 규칙과는 별개의 내부 선택 문제) |
| D-5 | 모듈은 살아있는데 절대 성능이 P36 fair 미달 | 미해결 (레시피 가설은 폐기) | 🔴 **physaug 복원은 공정성 문제로 배제**(user 판정 07-20) — 게이트는 아키텍처만으로 넘는다. [rootcause 문서](2026-07-20-p39-muses-fognight-rootcause.md) M-1(lidar rank 회복)이 대체 1순위 |

## 6. 다음 행동 권고

1. **DELIVER P39 학습**: RailTrack이 회복되지 않는 한 완주 가치 낮음 → D-1·D-3을 반영한 **P39.1로 교체** 권고(GPU 점유는 유지).
2. **P39.1 1순위 변수(physaug 배제 후 갱신)**: M-1 lidar rank 회복(V1 trunk_exp 구조 교체) 주 변수 + gate/calib 완전 off(D-3, 무해→유해로 판정 변경, config 토글이라 동반하되 ablation 행 분리).
3. MUSES 쪽 처방(fog_night)은 별도 분석 완료 후 합류.
