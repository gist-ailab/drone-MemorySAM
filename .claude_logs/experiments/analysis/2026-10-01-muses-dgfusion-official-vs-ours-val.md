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

## 8. H-M1 "val-best 선택 편향" 검증 — 기각

가설(판정 세션): 250장 val 에서 희귀 클래스 IoU 가 에폭마다 크게 요동해 val-best 선택이 운 좋은 에폭을 고르고(DGFusion 은 final-iter 라 선택이 없다), 그 우위가 test 에서 사라진다. 사전 반증 조건 = motorcycle·truck·fence 의 에폭 간 표준편차가 2 미만이면 기각.

자료: 시드 824·825(yeon)·826(jarvis) `train.log` 의 `[Val]` 줄(2에폭 간격, 학습기 내부 채점 = 레터박스 1024 축이며 공식 native 채점과 클래스 값이 약간 다르다). 824·825 는 09-22 리부트 뒤 재개분이라 로그가 에폭 76~78 부터다. 원본 NAS `analysis_logs/muses_physaugoff_trainlogs_20261001/`(md5 서버 원본과 일치).

| 시드 (val-best 에폭) | 마지막 40에폭 전체 mIoU 평균±sd | val-best | 선택 이득(val-best − 구간 평균) | final(ep300) val | best − final |
|---|---|---|---|---|---|
| 824 (ep202) | 82.45±0.11 | 82.71 | +0.26 | 82.37 | +0.34 |
| 825 (ep258) | 82.15±0.06 | 82.40 | +0.25 | 82.21 | +0.19 |
| 826 (ep172) | 82.06±0.06 | 82.43 | +0.37 | 82.06 | +0.37 |

세 클래스의 에폭 간 표준편차(마지막 40에폭 / 에폭 120 이후 전체 / val-best ±20에폭):
- 시드 824: motorcycle 0.30 / 1.63 / 0.73 · truck 0.83 / 1.03 / 1.05 · fence 0.28 / 0.80 / 0.89
- 시드 825: motorcycle 0.28 / 1.30 / 0.91 · truck 0.14 / 0.45 / 0.23 · fence 0.22 / 0.73 / 0.43
- 시드 826: motorcycle 0.62 / 1.02 / 0.85 · truck 0.20 / 0.40 / 0.54 · fence 0.41 / 0.88 / 1.07

- **어느 창에서도 표준편차가 2 미만(최대 1.63)이라 사전 반증 조건에 따라 H-M1 은 기각**이다.
- 선택 이득(val-best 에폭의 클래스 IoU − 그 창의 궤적 평균, 에폭 120 이후): motorcycle −0.43~+1.14 · truck +0.19~+0.56 · fence +0.53~+2.57. 세 클래스를 합쳐도 시드당 +1~+4 포인트로, test 낙차 합계 −54 포인트(§7)의 10% 미만이다. 전체 mIoU 의 선택 이득도 +0.2~+0.5(에폭 120 이후 평균 대비)로 val→test −3.52 에 비해 작다.
- val-best 와 final 의 val 차이는 +0.19~+0.37 뿐이라 DGFusion 처럼 final 을 골라도 val 우위는 거의 그대로다.
- **final ckpt 의 Codabench test 결과는 없다**: `MUSES_TEST_RESULTS_INDEX.md` 의 모든 제출이 val-best 한 개다(주석: "전 제출이 val-best ckpt 단일 선택"). 그래서 val-best 대 final 의 test 차이는 측정 불가다(필요하면 final ckpt 로 test 예측을 만들어 제출해야 하며 user 제출 사안).

해석: 에폭 간 요동(sd 0.14~1.6)은 희귀 클래스 낙차(−13~−23)를 설명하기에 한참 작다. 에폭 선택이 아니라, **val 250장 대 test 750장의 표본 차이 또는 여러 레시피를 val 로 고른 과정의 과적합**(레시피 수준 선택, 가설 H-M2)이 남는 후보다. 이 분석으로는 둘을 가르지 못한다. 덧붙여 DGFusion 도 같은 val·test 를 쓰는데 val→test 가 −0.23 뿐이라, 표본 차이만으로는 우리의 −3.52 가 설명되지 않는다(DGFusion 의 희귀 클래스 test IoU 없이는 단정 불가).


## 9. H-M3 1차 — 희귀 5클래스의 조건별 val 표본과 우위의 출처 (val 덤프만, GPU 불필요)

질문(판정 세션): ①우리 val 우위(truck +18·motorcycle +13·fence +8)가 어느 조건에서 나오는가, ②test 낙차(fog/day −9.7·snow/day −10.4)와 맞물려 악천후에서 이 클래스들의 val 화소가 몇 장면에 불과한가.

방법: 우리 3시드 조건별 클래스 IoU(`per_condition_official`)와 DGFusion 공식 가중치 조건별 IoU(§3 채점)를 나란히 놓고, 조건별 해당 클래스 GT 화소 수·해당 클래스가 있는 장면 수를 NAS MUSES `gt_semantic/val` 의 `_gt_labelTrainIds.png` 250장에서 직접 센다. ⚠ = GT 화소 500천 미만(전체 val 1장 화소 207만의 약 0.24장 분량) 또는 장면 5개 미만.

val 전체 표본(250장, 약 5.2억 화소): motorcycle 475천 화소(0.09%)·45장면 · truck 1,040천·32장면 · fence 9,582천·168장면 · rider 85천·45장면 · bus 199천·10장면.

조건별 val IoU: 우리 3시드 평균 / DGFusion / Δ(우리−DGF) | GT 화소 수(천) · 해당 클래스가 있는 장면 수 / 조건 장면 수.  ⚠ = GT 화소 < 500천(≈0.24장 분량) 이거나 장면 수 < 5 (표본 빈약)

### motorcycle
| 조건 | 장수 | 우리 | DGF | Δ | GT 화소(천) | 있는 장면 | 표시 |
|---|---|---|---|---|---|---|---|
| clear/day | 50 | 65.7 | 73.3 | -7.5 | 64 (14%) | 14/50 | ⚠ |
| clear/night | 25 | 90.9 | 62.5 | +28.4 | 85 (18%) | 8/25 | ⚠ |
| fog/day | 33 | - | - | - | 0 | 0 | GT 없음 |
| fog/night | 25 | - | - | - | 0 | 0 | GT 없음 |
| rain/day | 34 | 83.1 | 66.8 | +16.2 | 235 (49%) | 8/34 | ⚠ |
| rain/night | 25 | 21.6 | 24.3 | -2.7 | 5 (1%) | 5/25 | ⚠ |
| snow/day | 33 | 72.5 | 76.7 | -4.2 | 42 (9%) | 6/33 | ⚠ |
| snow/night | 25 | 73.5 | 59.9 | +13.6 | 44 (9%) | 4/25 | ⚠ |

### truck
| 조건 | 장수 | 우리 | DGF | Δ | GT 화소(천) | 있는 장면 | 표시 |
|---|---|---|---|---|---|---|---|
| clear/day | 50 | 92.6 | 79.4 | +13.2 | 586 (56%) | 12/50 |  |
| clear/night | 25 | 95.8 | 97.8 | -2.0 | 120 (12%) | 1/25 | ⚠ |
| fog/day | 33 | 85.9 | 90.6 | -4.7 | 4 (0%) | 3/33 | ⚠ |
| fog/night | 25 | - | - | - | 0 | 0 | GT 없음 |
| rain/day | 34 | 0.0 | 0.1 | -0.1 | 34 (3%) | 4/34 | ⚠ |
| rain/night | 25 | 0.0 | 26.5 | -26.5 | 1 (0%) | 3/25 | ⚠ |
| snow/day | 33 | 94.7 | 94.3 | +0.4 | 287 (28%) | 7/33 | ⚠ |
| snow/night | 25 | 77.6 | 83.3 | -5.7 | 8 (1%) | 2/25 | ⚠ |

### fence
| 조건 | 장수 | 우리 | DGF | Δ | GT 화소(천) | 있는 장면 | 표시 |
|---|---|---|---|---|---|---|---|
| clear/day | 50 | 76.2 | 65.6 | +10.7 | 2675 (28%) | 41/50 |  |
| clear/night | 25 | 74.8 | 70.8 | +4.0 | 847 (9%) | 15/25 |  |
| fog/day | 33 | 92.6 | 75.2 | +17.4 | 1314 (14%) | 17/33 |  |
| fog/night | 25 | 62.8 | 48.1 | +14.8 | 557 (6%) | 9/25 |  |
| rain/day | 34 | 80.2 | 65.5 | +14.7 | 1855 (19%) | 23/34 |  |
| rain/night | 25 | 69.5 | 76.8 | -7.3 | 1286 (13%) | 20/25 |  |
| snow/day | 33 | 69.8 | 67.8 | +2.0 | 840 (9%) | 28/33 |  |
| snow/night | 25 | 52.9 | 67.0 | -14.1 | 208 (2%) | 15/25 | ⚠ |

### rider
| 조건 | 장수 | 우리 | DGF | Δ | GT 화소(천) | 있는 장면 | 표시 |
|---|---|---|---|---|---|---|---|
| clear/day | 50 | 70.9 | 64.0 | +6.9 | 60 (71%) | 14/50 | ⚠ |
| clear/night | 25 | 22.4 | 15.3 | +7.1 | 7 (9%) | 9/25 | ⚠ |
| fog/day | 33 | - | - | - | 0 | 0 | GT 없음 |
| fog/night | 25 | - | - | - | 0 | 0 | GT 없음 |
| rain/day | 34 | 33.1 | 39.5 | -6.4 | 9 (10%) | 8/34 | ⚠ |
| rain/night | 25 | 42.6 | 40.8 | +1.7 | 8 (9%) | 12/25 | ⚠ |
| snow/day | 33 | - | - | - | 0 | 0 | GT 없음 |
| snow/night | 25 | 47.9 | 50.2 | -2.3 | 1 (1%) | 2/25 | ⚠ |

### bus
| 조건 | 장수 | 우리 | DGF | Δ | GT 화소(천) | 있는 장면 | 표시 |
|---|---|---|---|---|---|---|---|
| clear/day | 50 | - | - | - | 0 | 0 | GT 없음 |
| clear/night | 25 | 0.0 | 99.8 | -99.8 | 1 (0%) | 1/25 | ⚠ |
| fog/day | 33 | 97.7 | 97.8 | -0.1 | 174 (87%) | 4/33 | ⚠ |
| fog/night | 25 | - | - | - | 0 | 0 | GT 없음 |
| rain/day | 34 | 65.6 | 39.7 | +25.9 | 4 (2%) | 3/34 | ⚠ |
| rain/night | 25 | - | - | - | 0 | 0 | GT 없음 |
| snow/day | 33 | - | - | - | 0 | 0 | GT 없음 |
| snow/night | 25 | 69.5 | 67.4 | +2.1 | 20 (10%) | 2/25 | ⚠ |

### test 쪽 같은 클래스의 날씨·주야 축 IoU (시드 20260825, Codabench 원문)

| 클래스 | Full | Clear | Fog | Rain | Snow | Day | Night |
|---|---|---|---|---|---|---|---|
| motorcycle | 56.0 | 56.8 | 42.1 | 47.3 | 64.5 | 47.7 | 59.2 |
| truck | 73.1 | 65.8 | 88.0 | 62.9 | 70.5 | 77.9 | **26.6** |
| fence | 62.3 | **44.3** | 74.6 | 73.1 | 61.4 | 63.1 | 60.8 |
| rider | 59.9 | 61.1 | 44.7 | 60.3 | 53.2 | 59.9 | 59.9 |
| bus | 95.0 | 94.1 | 96.3 | 93.1 | 96.9 | 96.0 | 88.4 |

(test 는 날씨 4축·주야 2축까지만 클래스별 값을 주고 8조합별 클래스 표는 없다.)

### 읽을 수 있는 것 (사실만)

① **val 우위의 출처**
- **truck(+18.3)**: 우위는 clear/day 한 조합에서 나온다(GT 화소의 56%, 12장면, 92.6 대 79.4 = +13.2). snow/day(28%, 7장면)는 동률(+0.4)이고 나머지 조합은 화소가 4천~120천·1~4장면이다. 비(rain/day 34천·rain/night 1천 화소)에서는 두 모델 모두 IoU 0 근처다. 악천후에서 이긴다는 증거는 없다.
- **motorcycle(+12.8)**: 전체 475천 화소, 모든 셀이 ⚠이고 부호가 섞인다: rain/day +16.2(화소 49%, 8장면)·clear/night +28.4(18%, 8장면)·snow/night +13.6 / clear/day −7.5(14장면)·snow/day −4.2·rain/night −2.7. 이기는 두 셀(8장면씩)이 우위 전체를 만든다.
- **fence(+7.8)**: 표본이 충분한 클래스(조합당 9~41장면, 화소 208천~2.7백만)이고 우위가 넓다: clear/day +10.7 · clear/night +4.0 · fog/day +17.4 · fog/night +14.8 · rain/day +14.7 · snow/day +2.0. 지는 조합은 rain/night −7.3·snow/night −14.1(⚠, 208천)뿐이다. 즉 fence 의 val 우위는 희소 표본 효과가 아니다.
- rider·bus 는 화소가 극히 적어(85천·199천) 우위·열세를 말할 수 없다(bus 는 10장면).

② **악천후 val 표본과 test 낙차의 맞물림**
- 악천후(fog·rain·snow) val 에서 이 클래스들이 보이는 장면은 매우 적다: truck 은 fog/day 3장면(4천 화소)·rain 7장면(35천)·snow 9장면(295천), motorcycle 은 fog 0장면·rain 13장면(240천)·snow 10장면(86천), rider 는 fog 0장면, bus 는 fog/day 4장면·rain/day 3장면.
- test 에서 크게 무너진 축은 그 공백과 겹친다: **truck Night 26.6**(val 의 야간 truck 은 clear/night 1장면 120천·rain/night 3장면 1천·snow/night 2장면 8천 = 6장면 약 129천 화소라 val 로는 야간 truck 약점이 사실상 관측되지 않는다) · **motorcycle Fog 42.1**(val fog 에 motorcycle GT 가 0) · motorcycle Day 47.7(val 주간 세 셀 65.7·83.1·72.5).
- 🔴 **fence 는 다르다**: test Clear 44.3(다른 날씨 61~75)로 크게 떨어진다. 그런데 val 에서 fence 는 clear 조합(clear/day 41장면·clear/night 15장면)이 표본이 넉넉하고 IoU 75 안팎이다. 즉 fence 낙차(−13.6)는 val 표본 부족이 아니라 **test clear 장면이 val 과 다른 분포**(또는 test 에서만 나타나는 유형)라는 쪽에 가깝다.
- fog/day −9.7·snow/day −10.4 (조건 mIoU 낙차)를 이 5클래스로 설명할 수는 없다: fog/day 에서 이 클래스들은 fence(17장면)를 빼면 화소가 거의 없고, test 는 8조합별 클래스 표가 없어 그 조합의 어느 클래스가 떨어졌는지는 확인하지 못한다.

정리: 희귀 클래스 낙차는 (가) **val 이 못 본 조건**(야간 truck·안개 motorcycle·rider)과 (나) **분포가 다른 test clear 의 fence** 의 두 갈래로 나뉜다. truck·motorcycle 의 val 우위는 화소가 몇 장면(truck clear/day 12장면, motorcycle 두 셀 8장면씩)에 매인 값이라 일반화를 보장하지 못한다. 결정적 검증은 DGFusion test 클래스별 IoU(user 결정 대기)다.

## 10. DGFusion 공식 가중치의 Codabench test 결과 대조 — 결정적 검증 (2026-10-02)

user 가 세션이 만든 제출 zip 으로 Codabench 에 제출한 결과 원문(전체 mIoU **79.494**, 논문 79.5 와 일치)을 받았다. 원문은 NAS `analysis_logs/muses_test_dgfusion_official_20261001/codabench_dgf_raw.txt`(md5 f3c57939…)에 보존했다. ⚠️ user 표기 제출 파일명은 `…_test_2.zip` 이라 세션이 만든 `…_20261001.zip`(md5 290c4a10…)과 같은 파일인지 md5 로는 확인하지 못했다(전체 mIoU 가 논문값과 맞아 같은 예측으로 판단).

비교: 우리 = 시드 20260825(PhysAug-on, ep168) Codabench 원문(78.786). DGFusion = 위 원문. 같은 test 750장, 같은 채점. (우리 test 는 이 시드 한 개만 19클래스 원문이 있다.)

### 10-1. test 에서 우리 − DGFusion

- 전체 mIoU 78.786 대 79.494 = **−0.708**.
- 클래스(Δ 오름차순): traffic light −7.72 · pole −6.95 · rider −5.34 · motorcycle −5.29 · person −4.44 · traffic sign −3.48 · sidewalk −1.15 · wall −0.90 · road −0.33 · car −0.32 · sky −0.08 · vegetation 0.00 · fence +0.27 · building +0.32 · terrain +0.90 · bicycle +3.60 · train +3.86 · truck +6.26 · bus +7.35.
- 묶음별 Δ 합계(전체 −13.4): **얇은·작은 4클래스(pole·person·traffic sign·traffic light) −22.6**(평균 −5.65) · 희귀 3클래스(motorcycle·truck·fence) +1.2(평균 +0.41) · 나머지 12클래스 +7.9(평균 +0.66). 즉 **test 에서 우리가 뒤지는 이유는 얇은·작은 4클래스**이고, val 에서 크게 앞섰던 희귀 3클래스는 test 에서 동률이다.
- 날씨·주야 축 mIoU Δ: Clear −1.91 · Fog +5.27 · Rain −1.20 · Snow +0.27 · Day −0.52 · Night −0.97. 8조합 Δ: clear/day **−4.29** · clear/night +5.40 · fog/day +5.23 · fog/night +8.41 · rain/day −0.05 · rain/night **−4.89** · snow/day +0.58 · snow/night **−4.50**.
- 축별 클래스 Δ 중 |Δ|≥10: 우리 열세 = Night truck −17.9 · Rain motorcycle −15.9 · Clear fence −12.2 · Snow truck −11.4 / 우위 = Fog motorcycle +38.0(DGF 4.09) · Night bus +21.4 · Fog bicycle +20.8 · Snow train +17.3 · Fog rider +17.1 · Fog truck +15.3 · Snow bus +13.5 · Fog wall +13.1 · Snow bicycle +12.9 · Fog fence +10.8 · Rain truck +10.4.

### 10-2. 각자의 val→test 낙차 (test − val; 우리 val = PhysAug-off 3시드 평균, DGF val = §3)

| 클래스 | DGF val→test | 낙차 | 우리 val→test | 낙차 | 우리−DGF 낙차 |
|---|---|---|---|---|---|
| motorcycle | 66.59→61.28 | −5.31 | 79.42→55.99 | **−23.43** | **−18.12** |
| truck | 72.01→66.86 | −5.15 | 90.28→73.12 | **−17.16** | −12.01 |
| wall | 66.93→77.41 | +10.48 | 76.98→76.51 | −0.47 | −10.95 |
| rider | 52.71→65.23 | +12.52 | 55.94→59.89 | +3.95 | −8.57 |
| fence | 68.15→62.07 | −6.08 | 75.94→62.34 | **−13.60** | −7.52 |
| bicycle | 62.30→65.01 | +2.71 | 71.90→68.61 | −3.29 | −6.00 |
| traffic light | 80.26→78.35 | −1.91 | 77.10→70.63 | −6.47 | −4.56 |
| (나머지 12클래스 | | 평균 +1.3 | | 평균 −0.5 | |)

- 전체: DGFusion 79.72→79.49(−0.22) / 우리 82.30→78.79(−3.52).
- 묶음 평균 낙차: **희귀 3클래스 DGF −5.51 대 우리 −18.06** · 얇은 4클래스 DGF −0.86 대 우리 −1.74 · 나머지 12클래스 DGF +1.31 대 우리 −0.47.

### 10-3. 판독 (사실만)

1. **MUSES test 격차(−0.71)의 몸통은 얇은·작은 4클래스(−22.6)이고, val 에서 크게 앞섰던 희귀 3클래스는 test 에서 동률이다.** val 과 test 에서 열세 클래스 목록이 같다(val: pole·person·traffic sign·traffic light 3/3 시드 열세) — 얇은 4클래스 열세는 전이 손실이 아니라 **val 과 test 모두에 있는 실재하는 열세**다. 평균 Δ 는 val −4.77 대 test −5.65(§3, §7 의 "val 열세가 test 에서 같은 폭으로 유지될 것" 예측이 맞았다).
2. **희귀 3클래스의 낙차는 부분적으로 test 가 어려운 분포라는 효과다**: DGFusion 도 같은 클래스에서 −5.1~−6.1 떨어진다(−5.51 평균). 그러나 우리 낙차 −18.06 은 그 3배라, 약 −5.5 는 split 효과, 남는 약 −12.5 는 우리 모델/레시피 고유(val 이 유리하게 읽힌 몫)로 본다. wall·rider 에서도 DGFusion 은 test 가 val 보다 +10~+12 오르는데 우리는 −0.5·+3.9 로 오르지 못한다.
3. H-M1(val-best 선택 편향) 기각(§8)과 합치면 **남는 후보는 H-M2 레시피 수준의 val 과적합**(우리는 여러 레시피를 val 로 고르며 올라왔고 DGFusion 은 단일 레시피·final-iter)이다. 이 자료는 H-M2 를 가르지 못한다(레시피별 test 가 필요).
4. 조건별: 우리는 Fog 에서 +5.27 앞서고(특히 Fog motorcycle DGF 4.09) clear/day −4.29·rain/night −4.89·snow/night −4.50 에서 뒤진다. 낮 맑음 −4.29 는 val clear/day(+0.28)와 다른 방향이다. 8조합은 표본 75~150장이라 방향만 읽는다.
5. 본문 §9 의 가설 "fence 낙차는 test clear 분포 차이"는 부분 확인된다: DGFusion fence 도 test Clear 56.46(전체 62.07 보다 낮음)이며 우리 Clear 44.26 은 그보다 −12.2 낮다. 즉 test clear 는 fence 에 어려운 장면이고 우리는 그 안에서 더 못한다.

### 10-4. 한계

- 우리 test 는 시드 20260825 한 개(PhysAug-on)뿐이다. val 쪽은 PhysAug-off 3시드 평균과 섞였다(§7 주의 그대로). 시드2(79.788)는 약클래스 4개만 기록돼 있어 교차 확인에만 쓸 수 있다.
- DGFusion test 는 공개 가중치 1개의 Codabench 값이다(시드 분산 없음).
- 조건별 클래스 Δ 는 표본이 작은 셀이 섞여 있어 개별 칸을 근거로 인용하지 않는다.
