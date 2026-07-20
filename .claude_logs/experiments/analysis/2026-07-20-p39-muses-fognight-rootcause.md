# P39-MUSES fog_night 붕괴 원인 규명 + P39.1 스펙 (2026-07-20)

**대조**: **MUSES 3모달(img/lidar/event)** — P39-DPC ep146(공식 test 78.881) vs P38-m2f ep156(79.025), val, **조합 셀**(fog_night/clear_night/snow_night/rain_night/fog_day — 조합 CASE 지원 커밋 dee524f로 처음 가능).
**산출물**: NAS `analysis_logs/P39_muses_fognight_20260720/`(+`figs/fig1~4`).

## 한줄 판정

**fog_night −12.05의 원인은 V2 attention 노이즈가 아니라 lidar 표현 자체의 붕괴다** — P39의 lidar effective rank가 **전 조건에서 4.6~4.8**(P38은 22.9~24.7)로 무너져 있고, 그 결과 **카메라가 죽는 유일한 셀(fog_night)에서 lidar가 대체 역할을 못 한다**(모달 제거 손실 6.33 → **1.24**). clear_night처럼 카메라가 살아있는 셀에서는 무증상이라 P39가 오히려 이긴다(+3.69).

## 증거

### E1 — lidar 표현 붕괴 (fig1, 전 조건 일관)

| 모델 | img rank | **lidar rank** | event rank | FUSED |
|---|---|---|---|---|
| P38 ep156 (fog_night) | 21.9 | **24.7** | 20.3 | 8.5 |
| **P39 ep146 (fog_night)** | 21.5 | **4.8** | 25.7 | 7.8 |
| P39 (clear/snow/rain/fog_day) | 21~31 | **4.6~4.8** | 26~30 | 7.8~9.5 |

img/event는 정상(오히려 event는 P38보다 높음) — **lidar 브랜치만** 선택적으로 무너졌다. cross-modal CKA도 img~lidar 0.72(P38) → **0.38**(P39), lidar~event 0.88 → 0.45: lidar가 다른 모달과 공유하는 구조를 잃었다.

### E2 — 그 결과: 안개+야간에서 대체 모달 부재 (fig2)

모달 제거 시 mIoU 손실(diag drop-modality):

| 셀 | 모델 | img | **lidar** | event |
|---|---|---|---|---|
| fog_night | P38 | 7.38 | **6.33** | 0.51 |
| fog_night | **P39** | 8.21 | **1.24** | −1.48 |
| clear_night | P38 | 15.03 | 1.82 | 0.71 |
| clear_night | P39 | 24.29 | 1.34 | 0.47 |

- P38은 fog_night에서 lidar 의존도가 급등(1.82 → 6.33)한다 = **정상적인 센서 대체**. P39는 그대로 1.2 수준 = 대체 실패.
- P39는 clear_night에서 img 의존이 24.3으로 극단적(P38 15.0) — **카메라 편중 모델**이 됐고, 카메라가 죽는 셀에서 그대로 무너진다.
- 신뢰도 AUROC(img)도 fog_night만 0.70(P39) vs 0.79(P38) — 카메라가 나쁘다는 것조차 잘 못 알아본다.

### E3 — 모듈 토글(fig3): V2/V5가 범인이 아니다

fog_night: `p39_query_off` **+0.20**(query는 미미하게 기여), `p39_trunkexp_off` +1.72, `p36_router_off` +1.73, **`p34_gate_off` −0.89 · `p34_calib_off` −0.32(=끄는 게 이득, 유해)**.
→ 초기 가설("V2가 노이즈 증폭")은 **반증**. query를 꺼도 회복되지 않는다. 대신 **gate/calib이 fog_night에서 유해**하다는 DELIVER 판정(D-3)이 MUSES에서도 재현.

### E4 — 셀별 공식 test (fig4)

clear_night +3.69, snow_night +1.93로 **야간 개선은 실재**(주야 격차 5.14→3.73). fog_night만 −12.05로 전체를 −0.144로 되돌림.

## 원인 가설 (다음 실험에서 확정할 것)

lidar rank 붕괴는 P39 고유이므로 V1/V2/V5 중 하나가 원인이다. 가장 유력한 것은 **V1 trunk expansion**: `fused + Σ P_m(f_m)`의 선형 1×1 투영이 lidar에게 **저rank 지름길**을 제공해, LoRA adapter가 풍부한 표현을 학습할 유인을 잃었을 가능성(투영이 rank를 병목시키는 형태로 수렴). V2(query가 modal 토큰 직접 attend)도 img/event 토큰에 attention이 쏠리면 lidar 경로의 gradient가 얕아질 수 있다. **판별 실험**: P39.1에서 (a) trunk_exp off, (b) trunk_exp를 비선형/고rank로 교체, (c) lidar 브랜치에 rank 보존 정규화 — 셋 중 어느 것이 lidar rank를 24 수준으로 되돌리는지.

## P39.1 스펙 (physaug 제외 — user 공정성 판정 07-20)

**동결**: DINOv3-L frozen + per-modal LoRA + SimpleFPN/FPNSegHead + router 직접감독 + deep-sup + V3/V4. **PHYSAUG off 유지**(헤드라인은 아키텍처만으로).

| # | 변경 | 근거 | 검증 게이트 |
|---|---|---|---|
| **M-1 (1순위)** | **lidar rank 회복** — V1 trunk_exp를 (a) off 또는 (b) 2-layer 비선형+LayerNorm으로 교체. 동시에 per-modal feature에 rank 보존 항(예: 배치 공분산 log-det 정규화, 가중 0.01) 옵션 토글 | E1/E2: lidar rank 4.8, fog_night 기여 1.24 | ep30에 `feature_stats`로 **lidar rank ≥ 15** & fog_night drop-lidar **≥ 4.0** |
| **M-2** | **gate/calib 완전 off** (config) | E3 + DELIVER D-3: 3세대 no-op이었고 이번엔 thin-class·fog_night에서 유해 | 토글 즉검에서 |Δ|<0.2 유지(무해) |
| **M-3** | **V5 Λ 배타화** — per-class Λ에 온도 스케줄(초반 soft → 후반 hard-argmax)로 dense/query 상충 제거 | DELIVER D-1: query_off −25.4 & router_off −24.4(RailTrack 상충) | RailTrack 후반 붕괴 없음(ep 후반 −5 이내) |
| **M-4** | **앵커 query 클래스 균형 손실 가중** (V4는 포인트 수만 보장) | DELIVER D-2: Water/Bridge/Other 사망 | Water ≥ 9.5 |
| **M-5** | **ckpt 선택 규칙 변경** — val-best 대신 (val + test-cond 평균) 또는 thin-class 포함 복합 지표 | D-4/MUSES: val↔test 순위 역전 2건 | 선택 ckpt가 test에서도 상위 |

**사전 등록 게이트**: DELIVER = P36 fair(val 67.74/test 55.62) + thin-class(Wall≥13/Water≥9.5/RailTrack≥62) · MUSES = **P38 val 82.22 / 공식 test 79.025**, **fog_night 셀 ≥ 74**(P38 수준 복원 = 전체 +1.2pt 효과).
**중단 조건**: ep30 즉검에서 M-1 게이트(lidar rank ≥15) 미달 시 다른 후보(V2 or V5 원인)로 전환하고 그 런은 중단.
**1-변수 원칙**: M-1을 주 변수로, M-2는 config 토글이라 동반 적용하되 ablation 행으로 분리 보고.
