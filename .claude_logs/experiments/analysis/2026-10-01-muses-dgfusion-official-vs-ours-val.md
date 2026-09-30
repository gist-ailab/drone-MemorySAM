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
