# P46-CTR C1+C3 ep40 — RailTrack 게이트 통과(재타깃 가설 확증, overall SOTA 미달)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 결과 요약

| 지표 | Base (P39.1) | C1+C3 ep40 @768 | C1+C3 ep40 @1024 | DGFusion (SOTA) | 게이트 | 통과 여부 |
|---|---|---|---|---|---|---|
| **RailTrack test** | **4.02** | **59.10** (+55.1) | **60.14** (+56.1) | 64.47 | ≥40 | 🟢 **압도적 통과 (양 해상도 모두)** |
| Overall test mIoU | 52.47 | 54.92 (+2.45) | **56.12** (+3.65) | 56.71 | ≥56.62 | 🟡 @768 미달 / **@1024 -0.50 근접(거의 통과)** |

- ckpt: `jarvis_deliver_rgbdel_P46_ctr_c1c3/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.36_top1_checkpoint.pth` (val 67.36@ep40, 중간 체크포인트 — 학습은 ep200까지 계속 진행 중).
- eval: `tools/eval_reliadino_ckpt.py --cfg configs/jarvis-deliver_rgbdel_P46_ctr_c1c3.yaml --split both --gpu 0/1`, lecun(idle GPU, jarvis 학습 무간섭), env `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. 해상도 두 가지: @768(학습 프로토콜과 동일) + @1024(EVAL.IMAGE_SIZE만 override, 나머지 config 동일). 체크포인트 로드: missing=0, unexpected=0 (아키텍처 완전 일치, 두 해상도 공통).
- val split 재현(@768): mIoU 67.37 (학습 로그의 67.36과 사실상 일치 — eval 스크립트가 학습-시점 프로토콜을 정확히 재현함을 확인). val(@1024): 69.16.

## 25클래스 test IoU (C1+C3 ep40) — @768 vs @1024

| Class | @768 | @1024 |
|---|---|---|
| Building | 88.69 | 89.31 |
| Fence | 45.37 | 47.66 |
| Other | 2.26 | 3.25 |
| Pedestrian | 73.37 | 77.41 |
| Pole | 45.76 | 48.12 |
| RoadLine | 80.77 | 83.16 |
| Road | 96.57 | 96.78 |
| SideWalk | 72.50 | 73.29 |
| Vegetation | 82.09 | 83.39 |
| Cars | 92.35 | 93.20 |
| Wall | 10.84 | 12.57 |
| TrafficSign | 50.49 | 52.68 |
| Sky | 97.77 | 98.03 |
| Ground | 7.14 | 6.85 |
| Bridge | 0.02 | 0.03 |
| **RailTrack** | **59.10** | **60.14** |
| GroundRail | 76.63 | 79.47 |
| TrafficLight | 32.17 | 32.92 |
| Static | 25.57 | 27.15 |
| Dynamic | 8.97 | 10.01 |
| Water | 10.96 | 10.56 |
| Terrain | 68.03 | 70.12 |
| TwoWheeler | 63.00 | 64.95 |
| Bus | 93.45 | 94.34 |
| Truck | 89.22 | 87.53 |

(@768: overall test mIoU = 54.92, mAcc = 65.93, mF1 = 64.04, n=1897 · @1024: overall test mIoU = 56.12, mAcc = 66.67, mF1 = 65.07, n=1897)

## 판정 (코디네이터)

- **Primary falsifiable 게이트(RailTrack test 4→≥40) 통과 = class-transfer 진단→처방→검증 구조가 실증됨(논문 core claim).** C-1 RCS(희소클래스 샘플링) + C-3 PROTO(prototype consistency) 조합이, 사전 진단한 RailTrack under-learning(class-transfer 붕괴)을 실제로 복구시켰다는 falsifiable 예측이 맞아떨어졌다. **@768/@1024 양쪽 해상도에서 일관되게 통과**(59.10 / 60.14), 해상도 아티팩트가 아님을 재확인.
- Wall/Water/Bridge는 게이트에서 제외된 클래스(§9 재타깃 근거: DGFusion도 test IoU 0~4로 동반 붕괴 확인, 복구 불가·SOTA 자체가 못 넘는 영역) — 이번 결과에서도 여전히 저조(10~13대/10대/0점대)한 것은 예상된 범위이며 실패로 보지 않는다.
- Overall test: @768(54.92)은 secondary gate(56.62)·DGFusion(56.71) 미달이었으나, **@1024(56.12)는 −0.50/−0.59로 거의 근접** — 해상도만으로 다수 클래스(Pedestrian +4.04, GroundRail +2.84, Terrain +2.09 등 사소 클래스 전반 개선)가 함께 오르며 gap이 크게 좁혀짐. **RailTrack 회복 자체는 overall 개선의 일부일 뿐**(다른 붕괴 클래스가 여전히 천장) — 하지만 해상도 변경만으로 gate 근접까지 온 것은 추가 여지가 있음을 시사(해석은 코디네이터 몫).

## 이상 관찰(해석 보류)

- **RailTrack val < test 역전(양 해상도 공통)**: @768 val 18.53 < test 59.10, @1024 val 19.37 < test 60.14. val이 test보다 낮은 역전이 해상도와 무관하게 일관 재현됨 — 분포차(달빛/조명/RailTrack 등장 빈도 등) 추정이나 원인 미확인, 해석 보류.

## 미완 / 다음 단계

- ep40은 **중간 체크포인트**다. 학습은 ep200까지 계속 진행 중이며, 완주 후 val-best ckpt로 재판정 예정.
- **C3-only ablation**(C1 RCS 없이 C3 PROTO 단독, jarvis GPU4-7)이 ep40에 도달하면 동일 per-class test eval을 수행해 C1과 C3 각각의 RailTrack 기여를 분해할 예정(2026-07-30 10:34 기준 ep38 진행 중, 도달 임박 — 코디네이터 지시 대기).
