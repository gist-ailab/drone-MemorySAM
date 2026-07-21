# MUSES 공식 test 결과 — P38-m2f 3모달 ep156 (2026-07-20)

**제출**: Codabench comp 14005 · zip `muses_P38_m2f_3modal_ep156_submission.zip`
**모델**: P38 MaskQueryLite(M2F) 3모달(img/lidar/event), ckpt `epoch156_82.22_top1_checkpoint.pth` (내부 val 82.22)
**결과**: **Overall mIoU 79.025** — 프로젝트 MUSES 공식 test 신기록

## 요약 판정

| 제출 | 내부 val | 공식 test |
|---|---|---|
| P34 3모달 ep276 | 81.02 | 78.979 |
| P34 4모달(+radar) ep182 | 80.76 | 78.256 (radar −0.72) |
| **P38-m2f 3모달 ep156** | **82.22** | **79.025** (+0.046) |

- SOTA(GtA, camera-only) **82.39** 대비 **−3.365** → MUSES "승리" 주장 불가.
- 🔴 **val 개선의 test 전이율이 낮다**: P34→P38 val **+1.20** → test **+0.046**. val→test 낙차 ~3.2pt로 두 제출 일관.
- 주 축은 **주야 격차 5.14**(day 80.253 / night 75.118). 날씨는 평평(spread 0.8).
- 이상 지점: **snow_day 70.584 < snow_night 74.867** (통념과 반대).
- 야간 붕괴 클래스: **truck 76.43(day) → 44.40(night)**, bus 96.12 → 80.85.

---

## 원본 출력 (verbatim)

```
Overall mIoU: 79.025

Evaluating: Full Test Set (all) — 750 images
mIoU: 79.025
road 97.18 | sidewalk 87.51 | building 92.97 | wall 80.43 | fence 66.61
pole 61.46 | traffic light 71.08 | traffic sign 73.93 | vegetation 89.07 | terrain 79.31
sky 96.24 | person 70.88 | rider 57.68 | car 93.73 | truck 73.89
bus 94.24 | train 93.47 | motorcycle 55.07 | bicycle 66.72

Evaluating: Clear Weather (clear) — 225 images
mIoU: 78.218
road 98.31 | sidewalk 93.06 | building 94.39 | wall 78.07 | fence 56.95
pole 62.10 | traffic light 70.39 | traffic sign 75.29 | vegetation 88.76 | terrain 79.39
sky 97.47 | person 69.23 | rider 56.69 | car 92.27 | truck 67.45
bus 91.73 | train 89.77 | motorcycle 54.22 | bicycle 70.62

Evaluating: Fog Weather (fog) — 175 images
mIoU: 77.524
road 96.43 | sidewalk 77.79 | building 90.31 | wall 65.60 | fence 77.43
pole 61.86 | traffic light 82.83 | traffic sign 68.82 | vegetation 88.05 | terrain 84.60
sky 95.74 | person 71.73 | rider 38.05 | car 92.04 | truck 88.11
bus 95.70 | train 100.00 | motorcycle 21.95 | bicycle 75.92

Evaluating: Rain Weather (rain) — 175 images
mIoU: 78.096
road 96.41 | sidewalk 87.92 | building 93.74 | wall 84.36 | fence 68.36
pole 62.76 | traffic light 69.09 | traffic sign 77.60 | vegetation 90.13 | terrain 75.99
sky 95.09 | person 71.09 | rider 60.43 | car 94.18 | truck 58.48
bus 92.97 | train 93.63 | motorcycle 45.84 | bicycle 65.74

Evaluating: Snow Weather (snow) — 175 images
mIoU: 78.329
road 97.20 | sidewalk 85.27 | building 91.18 | wall 82.53 | fence 64.38
pole 57.94 | traffic light 69.50 | traffic sign 70.51 | vegetation 89.45 | terrain 68.06
sky 95.96 | person 71.69 | rider 55.60 | car 95.12 | truck 77.37
bus 96.82 | train 97.09 | motorcycle 65.59 | bicycle 56.98

Evaluating: Daytime (day) — 450 images
mIoU: 80.253
road 97.14 | sidewalk 85.98 | building 93.37 | wall 79.45 | fence 69.11
pole 63.65 | traffic light 74.46 | traffic sign 70.67 | vegetation 90.94 | terrain 79.16
sky 97.23 | person 73.50 | rider 57.77 | car 92.68 | truck 76.43
bus 96.12 | train 95.16 | motorcycle 56.81 | bicycle 75.18

Evaluating: Nighttime (night) — 300 images
mIoU: 75.118
road 97.23 | sidewalk 89.72 | building 92.41 | wall 81.44 | fence 61.75
pole 57.87 | traffic light 65.19 | traffic sign 78.26 | vegetation 84.21 | terrain 79.66
sky 90.77 | person 67.06 | rider 57.61 | car 94.81 | truck 44.40
bus 80.85 | train 92.87 | motorcycle 54.41 | bicycle 56.74

조건 조합 셀 (mIoU only):
Clear Day (clear_day, 150)    : 80.222
Clear Night (clear_night, 75) : 71.877
Fog Day (fog_day, 100)        : 76.747
Fog Night (fog_night, 75)     : 74.728
Rain Day (rain_day, 100)      : 78.512
Rain Night (rain_night, 75)   : 73.510
Snow Day (snow_day, 100)      : 70.584
Snow Night (snow_night, 75)   : 74.867
```

---

## 분석용 파생 관찰 (원본 아님, 해석)

**주야 격차 (day − night)**: 전체 +5.14. 클래스별로 보면 야간 손실이 편중됨 —
truck **−32.03**(76.43→44.40) · bus −15.27 · bicycle −18.44 · vegetation −6.73 · sky −6.46 · traffic light −9.27.
반대로 야간이 더 나은 것: traffic sign **+7.59**(70.67→78.26, 반사재 추정) · car +2.13 · sidewalk +3.74 · wall +1.99 · terrain +0.50.

**날씨 축은 거의 평평**(clear 78.218 / fog 77.524 / rain 78.096 / snow 78.329, spread 0.805) — 날씨 강건성은 확보된 상태이고 **병목은 조도(야간)**.

**이상 셀 snow_day 70.584**: 8개 조합 중 최저이며 snow_night(74.867)보다 4.28 낮음. 다른 날씨는 전부 day > night인데 snow만 역전 → **주간 설상 고반사/과노출** 가설. 별도 조사 가치 있음.

**fog의 극단 분산**: train 100.00 / truck 88.11 인데 motorcycle 21.95 / rider 38.05 — 안개에서 대형·강한 형태는 유지되나 소형·얇은 객체가 붕괴.

**전역 약클래스**: motorcycle 55.07 · rider 57.68 · pole 61.46 · fence 66.61 · bicycle 66.72 — 전부 얇거나 작은 구조물.
