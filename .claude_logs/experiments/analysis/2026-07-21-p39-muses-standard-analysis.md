# P39-DPC MUSES 3모달 표준분석 (공백 보완) — ep146 (2026-07-21)

**대상**: P39-DPC **MUSES 3모달(img/lidar/event)** `epoch146_81.52_top1`(jarvis, 공식 test 78.881) — 기존 분석은 토글 즉검(ep66)과 fog_night 대조뿐이어서 **표준 파이프라인 전체(D1~D5+viz)**로 공백을 메움.
**실행**: yeon GPU0(8-GPU det 학습과 동거, OOM 감시 하에 통과), 8스테이지 전부 ok. **산출물**: NAS `analysis_logs/P39_muses_std_20260721/`(21MB).

## 한줄 판정

**"lidar adapter가 죽었다"는 가설은 반증됐다** — adapter는 lidar에 **가장 강하게** 작동한다(Δacc +0.47~+0.64, dW 최대). 진짜 문제는 **표현이 저rank(≈4.7)로 압축된 채 융합이 그것을 악조건에서 활용하지 못하는 2단 실패**다.

## ① adapter 적응도 (D3/D3B) — 가설 수정의 핵심

| 모달 | adapter on/off **Δacc** | acc(on)→acc(off) | dW norm | feat_cos |
|---|---|---|---|---|
| img | +0.018 ~ +0.093 | 0.92 → 0.84~0.90 | 12.6 | 0.53~0.58 |
| **lidar** | **+0.47 ~ +0.64 (최대)** | **0.82~0.87 → 0.19~0.34** | **16.5 (최대)** | **0.115 (거의 직교)** |
| event | +0.21 ~ +0.23 | 0.79~0.87 → 0.57~0.66 | 11.5 | 0.43~0.46 |

- lidar adapter를 끄면 그 모달 정확도가 **0.87 → 0.23**(night)으로 붕괴 = adapter가 lidar를 **전적으로 담당**. dead adaptation 아님.
- feat_cos 0.115 = adapter가 lidar 피쳐를 거의 직교 방향으로 재작성. 그 결과가 **rank 4.6~4.8의 압축 코드**([fog_night 문서](2026-07-20-p39-muses-fognight-rootcause.md) E1) — 즉 **"압축을 만든 주체가 adapter 자신"**이다.
- ⚠️ `adapter_health.json`의 per-modality 키는 DELIVER 이름(`depth`)으로 표시됨 — MUSES에선 index1 = **lidar**(16.48)로 읽어야 함(도구 라벨링 아티팩트).

## ② 피쳐 통계 (D2N) / ④ 클래스×조건 (D1)

- per-modal rank: img 21~31 · **lidar 4.6~4.8** · event 26~30, FUSED 7.8~9.5 — 전 조건 동일 패턴.
- D1 mIoU: **day 81.62 · clear 75.06 · snow 76.78 · night 76.87 · rain 71.93 · fog 62.36**, spread **14.51**.
  - fog가 최약(62.36)이며 night(76.87)는 약점이 아님 — P37a(fog 62.73/night 77.61)와 동일 구조. **P39는 야간을 개선했지만 fog 병목은 그대로**.
  - 조건-소멸 클래스: fog에서 traffic light/rider/train/motorcycle/bicycle = 0.0, rain에서 truck 0.0·bus 45.5. (일부는 소표본 클래스 부재 아티팩트 가능 — 공식 test 셀 수치와 교차 확인 필요.)

## ③ 모듈 A/B (D5, 6조건)

| toggle | day | night | 판정 |
|---|---|---|---|
| **p39_trunkexp_off** (V1) | +0.61 | **+2.50** | 최대 기여, 야간에 특히 큼 |
| **p39_query_off** (V5) | **+1.00** | **−0.28** | 주간 기여 / **야간 역효과** — DELIVER night·rain 음수와 동일 패턴 재현 |
| p36_router_off | +0.73 | +1.05 | 정상 기여(의존 해소 유지) |
| p34_gate_off / veto / calib | −0.05~+0.23 | ≈0 | no-op (fog_night에선 유해 −0.89) |
| p38_m2f_off | 0.00 | 0.00 | legacy β 완전 미사용 |

## 종합: P39.1 M-1의 표적 재정의

기존 M-1은 "lidar rank 회복(adapter를 되살려라)"였으나, adapter는 이미 최대로 일하고 있다. 정확한 처방은 **두 갈래**다:

1. **표현 압축 완화** — adapter가 lidar를 5차원으로 접지 않도록: LoRA rank 상향(현 r8) 또는 per-modal rank 보존 정규화(배치 공분산 log-det). V1 trunk_exp의 선형 투영이 이 압축을 유도했는지는 trunk_exp off 실험으로 판별.
2. **악조건 활용 강제** — lidar 정보(자체 acc 0.87)를 fog_night에서 융합이 쓰게: 학습 중 **신뢰도 조건부 모달 드롭아웃**(카메라 열화 상황을 만들어 lidar로 풀게)으로 대체 능력을 표현에 새김. (P33의 무조건적 드롭아웃은 무효였으나 **조건부**라는 점이 차별점.)

**게이트(유지)**: MUSES 공식 test ≥79.025(P38) & **fog_night 셀 ≥74**, DELIVER P36 fair + thin-class. **V5 query 경로는 야간/악조건에서 음수**이므로 M-3(Λ 배타화)과 함께 조건부 비활성도 후보.
