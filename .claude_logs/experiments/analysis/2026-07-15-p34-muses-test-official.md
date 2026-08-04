---
created: 2026-07-15
scope: P34-ReliaDINO MUSES 공식 test 서버 상세 결과 (Codabench #14005, submission 850776)
source: https://www.codabench.org/competitions/14005/detailed_results/850776/
---

# P34-ReliaDINO — MUSES 공식 Test 상세 결과 (2026-07-15 제출)

> **mIoU (Overall) = 78.979** — 프로젝트 **최고 신뢰 수치** (test GT는 서버 전용 → 훔쳐보기 구조적 불가, **val-best ep276 단일 제출**).
> 비교선: **CAFuser 78.5를 모달리티 1개 덜 쓰고 상회(+0.48)**, DGFusion 대비 −0.52. SOTA "주장"은 불가(백본 파라미터 ~10×, ~~DGFusion 공표 수치 3종 불일치 79.5/79.49/79.72~~ **← 오독, 2026-08-04 정정**: 불일치가 아니라 **79.72 = val · 79.49 = test(리더보드) · 79.5 = 논문 Table II의 test 반올림**이다(원문 2509.09828v3 Table II + README 직접 확인). 🔴 이 오독 때문에 그동안 MUSES **val 델타를 산출하지 않았다** — 이후 보고는 val SOTA 79.72 기준 델타를 병기할 것. 단 ①우리 val은 내부 letterbox 1024² 지표라 공식 원해상도 재평가 필요(P34 전례 −0.16/+0.01) ②그들은 final-iter, 우리는 val-best라 **프로토콜을 맞추려면 우리도 final-iter val**을 써야 한다) — 논문 서사는 "모달 효율 + 야간 강건성"으로.

## 1. 조건 요약

| 구분 | mIoU | | 구분 | mIoU |
|---|---|---|---|---|
| **Overall** | **78.979** | | Daytime | 79.818 |
| Clear | 79.042 | | **Nighttime** | **76.373** |
| Fog | 71.615 | | Rain | 78.070 |
| Snow | 77.397 | | | |

**Day→Night 하락이 −3.45에 불과** — DELIVER에서 확인된 4모달 균형 보정(AUROC [.85,.78,.87,.70])이 실제 야간 강건성으로 발현. 최약 조건 = **Fog(71.6)**.

## 2. 서브카테고리 (Weather × Time)

| | Day | Night |
|---|---|---|
| Clear | 80.255 | 74.820 |
| Fog | 70.758 | **59.993** ⚠️ |
| Rain | 78.113 | 73.488 |
| Snow | 69.216 | 75.995 (Day보다 ↑) |

⚠️ **Fog Night 59.99 = 전체 최약 셀**. 단 per-class를 보면 IoU 0.000(traffic light/person/truck/train)과 100.000(bus/motorcycle)이 공존 — **표본 극소 셀**이라 수치 자체보다 "fog+night 복합 열화" 경향만 취할 것. Snow Day(69.2) < Snow Night(76.0) 역전도 표본 구성 영향.

## 3. Per-Class IoU × 조건 (19 클래스)

| Class | All | Clear | Fog | Rain | Snow | Day | Night |
|---|---|---|---|---|---|---|---|
| road | 97.13 | 98.38 | 96.48 | 96.30 | 96.95 | 97.20 | 97.02 |
| sidewalk | 86.76 | 92.99 | 77.64 | 87.36 | 83.34 | 85.32 | 88.86 |
| building | 92.82 | 94.26 | 89.64 | 93.67 | 91.14 | 93.19 | 92.28 |
| wall | 80.01 | 79.82 | 62.25 | 83.32 | 82.01 | 78.01 | 82.14 |
| fence | 65.25 | 55.22 | 75.55 | 67.88 | 62.77 | 66.63 | 62.63 |
| pole | 62.06 | 61.41 | 64.16 | 64.06 | 57.81 | 62.72 | 60.90 |
| traffic light | 69.02 | 68.17 | 82.14 | 66.70 | 67.50 | 72.91 | 62.23 |
| traffic sign | 72.30 | 73.10 | 67.36 | 76.43 | 68.78 | 69.29 | 76.35 |
| vegetation | 89.51 | 89.00 | 89.18 | 90.62 | 89.52 | 91.15 | 85.16 |
| terrain | 79.47 | 81.21 | 85.26 | 76.39 | 64.92 | 79.19 | 80.12 |
| sky | 96.78 | 97.54 | 97.09 | 95.65 | 96.23 | 97.32 | 93.86 |
| person | 69.56 | 68.69 | 69.16 | 70.02 | 68.96 | 72.63 | 65.01 |
| rider | 59.99 | 60.12 | 39.10 | 63.54 | 45.52 | 60.56 | 59.49 |
| car | 93.55 | 92.36 | 90.23 | 93.96 | 94.94 | 92.56 | 94.56 |
| truck | 76.52 | 71.29 | 86.86 | 62.56 | 85.28 | 78.00 | 60.09 |
| bus | 94.60 | 93.26 | 95.48 | 92.84 | 97.21 | 95.94 | 85.04 |
| train | 92.82 | 91.73 | **0.00** | 93.02 | 95.25 | 94.86 | 92.10 |
| motorcycle | **54.41** | 62.47 | 16.37 | 42.56 | 60.50 | 53.01 | 54.93 |
| bicycle | 68.05 | 70.79 | 76.74 | 66.45 | 61.91 | 76.06 | 58.33 |

## 4. 판독 (강점/약점)

**강점**
- 대형 구조물·정적 클래스 상향 안정: road/sky/building/car/bus/train 90+ 
- **야간에서 오히려 오르는 클래스**: sidewalk(+3.5), wall(+4.1), traffic sign(+7.1), car(+2.0) — 비RGB 모달(LiDAR/event)이 야간 기여하는 정황
- wall 80.0 — DELIVER에서 dead였던 클래스가 MUSES에선 정상 (DELIVER Wall 문제 = 데이터셋 고유 재확인)

**약점 (개선 타깃)**
- **motorcycle 54.4 (전체 최하)**, rider 60.0 — 소형 동적 객체
- Fog 조건의 rider(39.1)·motorcycle(16.4)·train(0.0 — fog에 train 표본 자체가 희소할 가능성)
- 야간 하락 큰 클래스: truck(−17.9), bus(−10.9), bicycle(−17.7), traffic light(−10.7)

## 5. 제출 메타

- 대회: Codabench #14005 (MUSES benchmark), submission **850776**, 계정 maengjemo
- 모델: P34-ReliaDINO (DINOv3 ViT-L/16 frozen + per-modal LoRA + reliability-gated fusion), MUSES 3모달(RGB+event+lidar — CAFuser 4모달 대비 1개 적음)
- ckpt: MUSES val-best ep276 (config `b200_muses_rgbel_P34_reliadino`, B200)
- 원본 페이지 조회는 로그인 필요. 이 문서가 전 수치의 사본 (표는 detailed_results.html 그대로).
