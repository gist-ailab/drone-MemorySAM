# MUSES 기준선 원인 분석 — DGFusion 공식 가중치 val 재현 + 우리 모델과 클래스·조건 대조 (2026-10-01)

lab-plan DRN-261001-01(user 결정 2026-09-26). 측정·기록만 한다. 판정은 생각정리 세션.

## 1. 무엇을 했나

- DGFusion 저장소 README 의 공식 MUSES 가중치(`dgfusion_swin_tiny_bs8_180k_muses_clre.pth`, md5 7749b63b…, Google Drive)를 받아 **MUSES val 250장**을 README 공식 명령 그대로 추론했다(bengio GPU7, 4모달 radar 포함, `MODEL.TEST.DEPTH_ON False`).
- 예측(`sem_seg_predictions.json`, 클래스별 RLE)을 우리 공식 재채점(`tools/eval_muses_official.py`)과 같은 축(native 1080×1920, 19클래스, 조건 = 날씨/주야)으로 다시 채점했다(`score_dgf_muses.py`).
- 비교 대상 = 우리 MUSES PhysAug-off 3시드(824·825·826, val-best 공식 val 82.46·82.19·82.26). 같은 val 250장.

## 2. 재현 성립 (성공기준 ①)

| 항목 | README 게시값 | 재현 | 차 |
|---|---|---|---|
| mIoU-val | 79.72 | **79.7183** | −0.002 |
| PQ-val | 58.88 | **58.882** | +0.002 |

우리 채점기로 다시 계산한 전체 mIoU 도 79.7183 으로 DGFusion 자체 평가기와 일치한다(채점 축 동일 확인). 재학습 경로는 필요 없다.

## 3. 결과 (val 250장, 우리 3시드 평균 − DGFusion)

- 전체: DGFusion 79.72 / 우리 **82.30±0.14** / Δ **+2.58**(val 에서는 우리가 앞선다).
- 🔴 val 의 +2.58 이 test 에서는 −0.21(79.29 대 79.5)이 된다. 이 대조는 val 만 설명하며 val→test 격차의 원인은 아니다.

클래스별(3시드 모두 같은 부호인 것만 묶음):
- **우리 열세 3/3**: pole −6.20 · person −5.37 · traffic sign −4.34 · traffic light −3.16 (DELIVER 에서 진 얇은·작은 4클래스와 같은 종류).
- **우리 우위 3/3**: truck +18.27 · motorcycle +12.83 · wall +10.05 · bicycle +9.60 · fence +7.79 · rider +3.23 · terrain +1.95 · bus +1.44 · building +0.93 · vegetation +0.70 · sidewalk +0.61.
- 차이 없음(|Δ|<0.1): road · sky.

조건별 mIoU(표본 25~50장이라 방향만 읽는다):

| 조건 | n | DGFusion | 우리 | Δ | 시드 부호 |
|---|---|---|---|---|---|
| clear/day | 50 | 78.08 | 78.36 | +0.28 | +−+ |
| clear/night | 25 | 72.84 | 69.87 | **−2.98** | −−− |
| fog/day | 33 | 83.94 | 87.87 | +3.93 | +++ |
| fog/night | 25 | 74.16 | 75.43 | +1.27 | +−+ |
| rain/day | 34 | 66.34 | 69.62 | +3.29 | +++ |
| rain/night | 25 | 73.10 | 70.10 | **−3.01** | −−− |
| snow/day | 33 | 81.60 | 80.36 | −1.24 | −−− |
| snow/night | 25 | 75.11 | 74.52 | −0.59 | −−− |

- 3시드 모두 같은 부호로 뒤지는 조건: clear/night −2.98 · rain/night −3.01 · snow/day −1.24 · snow/night −0.59(눈은 작다). 안개 낮·비 낮은 3시드 모두 앞선다.
- 클래스×조건 셀에서 3시드 같은 부호이고 |평균 Δ|≥5 인 셀: 열세 23칸·우위 26칸(전체 목록은 NAS `compare.txt`). 열세 셀의 대부분이 pole·person·traffic sign·traffic light·fence 이고, 야간·눈 조건에 몰리는 경향이 있다. 일부 셀(clear/night bus 0.00, rain/night truck 0.00, rain/day train 0.00)은 그 조건의 해당 클래스 화소가 매우 적어 잡음이 지배하므로 해석하지 않는다.

## 4. 사전 가설 대비 (사실만)

- 가설("차이는 야간·눈과 원거리 소형 클래스에 집중, radar 를 쓰는 DGFusion 이 그 구간에서 앞선다"): 클래스 축(얇은·작은 4클래스 3/3 열세)과 야간 축(clear/night·rain/night 열세)은 방향이 맞는다. 눈은 열세 폭이 작고(−0.6~−1.2) 안개·비 낮에는 우리가 앞선다.
- 반증 조건(Δ 가 전 클래스에 균일): **성립하지 않는다** — 열세는 4클래스에 몰리고 나머지 다수는 우위다.
- **거리 구간 축은 이번에 측정하지 못했다.** MUSES 에는 DELIVER 같은 밀집 depth 가 없고 LiDAR 투영이 희소하다(LiDAR 점이 있는 화소에 한정해야 계산 가능). 계획서 성공기준의 "거리 구간" 부분은 미완이다.

## 5. 한계

- val 250장 단일 split, DGFusion 은 공개 가중치 1개(시드 분산 없음), 우리는 3시드.
- 우리 우위 클래스(truck·motorcycle·wall·bicycle·fence)는 val 에서 화소가 적은 희귀 클래스라 분산이 크다(시드 Δ 폭도 크다). 이것이 val→test 전이율이 낮은 이유인지는 이 분석으로는 모른다(미검증).
- 4모달(radar 포함) 대 3모달 차이가 열세 원인인지는 분리하지 못했다(DGFusion 에서 radar 를 빼는 측정이 없다).

## 6. 원본

NAS `analysis_logs/muses_baseline_dgfusion_official_20261001/`: `official_val/`(run.log·config·panoptic 예측·sem_seg_predictions.json) · `official_val_scored/`(report.json·혼동행렬 npy) · `compare.txt`·`compare.py`·`score_dgf_muses.py`·`run_muses_official.sh` · `md5_검증.txt`(266파일 bengio 원본과 일치). 가중치 원본: `ckpts/dgfusion_official_20261001/`(MUSES clre·DELIVER clde·cle 3개, md5 3홉 일치).

## 7. val→test 전이 손실은 어느 클래스·조건에서 나나 (우리 모델만, DGFusion 없이)

판정 세션 질문(2026-10-01): 진짜 원인은 val→test 전이 손실이므로, 우리 자신의 val 19클래스와 test 19클래스를 같은 표에 놓는다.
- val = 우리 PhysAug-off 3시드 평균(공식 재채점 report.json). test = Codabench 원문 시드 20260825(ep168, **PhysAug-on**) Full 750장 19클래스(`analysis_logs/muses_test/2026-09-18-P39_1-seed20260825-78.786.md`).
- ⚠️ 두 모델이 다르다(PhysAug-off 대 on). 같은 시드 val 전체 mIoU 차이는 +0.72(81.47 대 82.19)라 아래 낙차(최대 −23)에 비해 작지만, 같은 체크포인트의 val 클래스별 표는 아니다(미확보). 시드2(79.788)는 약클래스 4개만 기록돼 있어 교차 확인용으로만 쓴다.

전체 mIoU: val 82.30 → test 78.79, **−3.52**. (DGFusion 은 공식 README 기준 val 79.72 → test 79.49, −0.23.)

클래스별 (test − val), 낙차 큰 순:

| 클래스 | val | test | Δ |
|---|---|---|---|
| motorcycle | 79.42 | 55.99 | **−23.43** |
| truck | 90.28 | 73.12 | **−17.16** |
| fence | 75.94 | 62.34 | **−13.60** |
| traffic light | 77.10 | 70.63 | −6.47 |
| train | 97.75 | 93.94 | −3.81 |
| bicycle | 71.90 | 68.61 | −3.29 |
| terrain | 80.24 | 77.92 | −2.32 |
| vegetation · road · person · wall · sidewalk · pole | — | — | −1.05 ~ −0.23 |
| sky · building · car · traffic sign | — | — | +0.16 ~ +0.40 |
| bus | 93.48 | 95.05 | +1.57 |
| rider | 55.94 | 59.89 | +3.95 |

- **낙차의 대부분은 motorcycle·truck·fence 세 클래스**(합계 −54 포인트, 19클래스 평균으로 −2.8)에서 난다. 이 세 클래스는 §3 에서 val 에서 DGFusion 을 가장 크게 앞섰던 클래스(truck +18.27·motorcycle +12.83·fence +7.79)와 같다 → **val 의 우위 일부는 val 250장에서 이 희귀 클래스가 유리하게 나온 값으로 보이며 test 에서 사라진다**(가설, 클래스별 val 화소 수는 미확인).
- **얇은·작은 4클래스(pole·person·traffic sign·traffic light)는 전이가 안정적**이다: 평균 낙차 −1.74(pole −0.23·person −0.67·traffic sign +0.40·traffic light −6.47), 나머지 15클래스 평균 −3.99. 즉 val 에서 DGFusion 에 뒤졌던 이 4클래스가 test 에서 더 벌어질 근거는 이 표에 없고, test 에서도 비슷한 폭(약 −3~−6)으로 뒤질 가능성이 높다(DGFusion test 클래스별 IoU 가 없어 직접 확인은 못 함).
- 시드2 test 약클래스(motorcycle 58.07·rider 59.47·pole 62.07·fence 65.70)도 시드 20260825(55.99·59.89·62.59·62.34)와 같은 자리에서 낮아, motorcycle·fence 낙차는 시드 하나의 일이 아니다(2 test 시드).

조건별 mIoU (val 8조합 3시드 평균 대 test 8조합):

| 조건 | val | test | Δ |
|---|---|---|---|
| clear/day | 78.36 | 77.05 | −1.32 |
| clear/night | 69.87 | 75.48 | +5.61 |
| fog/day | 87.87 | 78.14 | **−9.73** |
| fog/night | 75.43 | 69.22 | −6.21 |
| rain/day | 69.62 | 78.85 | **+9.23** |
| rain/night | 70.10 | 73.90 | +3.81 |
| snow/day | 80.36 | 69.97 | **−10.40** |
| snow/night | 74.52 | 72.73 | −1.79 |

- val 조건별 값은 조합당 25~50장이라 조건 사이 순위가 test 와 자주 뒤바뀐다(fog/day 가장 높음 → test 에서 크게 하락, rain/day 가장 낮음 → test 에서 크게 상승, snow/day 는 test 최약). 조건별 val 열세·우위는 판정 근거가 못 된다(§0-2). 눈 낮 조합이 test 에서 69.97 로 약하다는 것은 기존 "snow_day<snow_night 역전 반복" 기록과 일치한다.
- test 는 Codabench 가 날씨·주야 축별 19클래스까지만 주므로(8조합별 클래스 표 없음) 클래스×조건 대조는 하지 않았다.

정리: val→test −3.52 의 주 원인은 얇은 클래스가 아니라 **희귀 대형·소형 클래스(motorcycle·truck·fence)의 val 과대평가**이고, 얇은 4클래스 열세는 val 과 test 에서 비슷하게 유지될 것으로 보인다(DGFusion test 클래스별 없이 추정).
