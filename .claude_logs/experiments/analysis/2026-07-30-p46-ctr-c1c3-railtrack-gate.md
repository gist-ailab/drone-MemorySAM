# P46-CTR C1+C3 ep40 — RailTrack 게이트 통과(재타깃 가설 확증, overall SOTA 미달)

> 판정: 코디네이터(사용자). 이 문서는 그 판정과 근거 수치를 기록한다.

## 결과 요약

| 지표 | Base (P39.1) | C1+C3 ep40 | DGFusion (SOTA) | 게이트 | 통과 여부 |
|---|---|---|---|---|---|
| **RailTrack test** | **4.02** | **59.10** (+55.1) | 64.47 | ≥40 | 🟢 **압도적 통과** |
| Overall test mIoU | 52.47 | 54.92 (+2.45) | 56.71 | ≥56.62 | 🔴 미달 |

- ckpt: `jarvis_deliver_rgbdel_P46_ctr_c1c3/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.36_top1_checkpoint.pth` (val 67.36@ep40, 중간 체크포인트 — 학습은 ep200까지 계속 진행 중).
- eval: `tools/eval_reliadino_ckpt.py --cfg configs/jarvis-deliver_rgbdel_P46_ctr_c1c3.yaml --split both --gpu 0`, lecun(idle GPU, jarvis 학습 무간섭), env `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, 해상도 test@768(학습 프로토콜과 동일). 체크포인트 로드: missing=0, unexpected=0 (아키텍처 완전 일치).
- val split 재현: mIoU 67.37 (학습 로그의 67.36과 사실상 일치 — eval 스크립트가 학습-시점 프로토콜을 정확히 재현함을 확인).

## 25클래스 test@768 IoU (C1+C3 ep40)

| Class | IoU |
|---|---|
| Building | 88.69 |
| Fence | 45.37 |
| Other | 2.26 |
| Pedestrian | 73.37 |
| Pole | 45.76 |
| RoadLine | 80.77 |
| Road | 96.57 |
| SideWalk | 72.50 |
| Vegetation | 82.09 |
| Cars | 92.35 |
| Wall | 10.84 |
| TrafficSign | 50.49 |
| Sky | 97.77 |
| Ground | 7.14 |
| Bridge | 0.02 |
| **RailTrack** | **59.10** |
| GroundRail | 76.63 |
| TrafficLight | 32.17 |
| Static | 25.57 |
| Dynamic | 8.97 |
| Water | 10.96 |
| Terrain | 68.03 |
| TwoWheeler | 63.00 |
| Bus | 93.45 |
| Truck | 89.22 |

(overall test mIoU = 54.92, mAcc = 65.93, mF1 = 64.04, n=1897)

## 판정 (코디네이터)

- **Primary falsifiable 게이트(RailTrack test 4→≥40) 통과 = class-transfer 진단→처방→검증 구조가 실증됨(논문 core claim).** C-1 RCS(희소클래스 샘플링) + C-3 PROTO(prototype consistency) 조합이, 사전 진단한 RailTrack under-learning(class-transfer 붕괴)을 실제로 복구시켰다는 falsifiable 예측이 맞아떨어졌다.
- Wall/Water/Bridge는 게이트에서 제외된 클래스(§9 재타깃 근거: DGFusion도 test IoU 0~4로 동반 붕괴 확인, 복구 불가·SOTA 자체가 못 넘는 영역) — 이번 결과에서도 여전히 저조(10.84/10.96/0.02)한 것은 예상된 범위이며 실패로 보지 않는다.
- Overall test SOTA 미달(54.92 < 56.62/56.71) = **RailTrack 회복이 overall mIoU 개선으로 온전히 직결되지 않음** — 다른 붕괴 클래스(Wall/Water/Bridge/Other/Ground/Dynamic 등 한 자리~10대 IoU)가 여전히 전체 평균의 천장 역할을 하고 있다.

## 이상 관찰(해석 보류)

- **RailTrack val 18.53 < test 59.10** — val이 test보다 낮은 역전. 통상 val(주간 위주)이 test(야간 포함)보다 높은 경향과 반대. 분포차(달빛/조명/RailTrack 등장 빈도 등) 추정이나 원인 미확인, 해석 보류.

## 미완 / 다음 단계

- ep40은 **중간 체크포인트**다. 학습은 ep200까지 계속 진행 중이며, 완주 후 val-best ckpt로 재판정 예정.
- @1024 해상도 결과 추가 예정(현재 lecun GPU1에서 병행 실행 중).
- **C3-only ablation**(C1 RCS 없이 C3 PROTO 단독, jarvis GPU4-7)이 ep40 부근에 도달하면 동일 per-class test eval을 수행해 C1과 C3 각각의 RailTrack 기여를 분해할 예정(현재 ep25 진행 중, 코디네이터 지시 대기).
