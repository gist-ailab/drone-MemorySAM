판정은 이 파일이 아니라 `lab-plan verdict DRN-261016-03 --analysis .claude_logs/tasks/DRN-261016-03/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-10-16. 새 학습·새 추론 없음. 기존 산출물만 재집계했다.
상태: **Q1·Q3 완료, Q2 는 덤프 대기**(manager 요청 inbox `20261016-171805-request-bc24a84a`).

## 입력(원본)
| 무엇 | 경로 | 비고 |
| --- | --- | --- |
| MUSES 1024 3시드 | NAS `analysis_logs/muphys_{824,825,826}_official_val_2026092*/report.json`·`hist_full.npy` | PhysAug-off 3시드(824 epoch202 · 825 epoch258 · 826 epoch172), 공식 native 1080×1920 val 250장 |
| MUSES 1536·2048 3시드 | NAS `analysis_logs/muses_res_val_20261016/<시드>_<해상도>/` | 824 1024 재측정 82.4631 = 09-25 82.4633(일치) |
| DELIVER 우리 | NAS `analysis_logs/thin_anatomy_test_20261011_res/run_all/{precision_recall,thickness_patch,boundary_distance}.csv` | Q2 레시피(열화 학습 + E1 교사 증류) 시드821, **test 1897장, 단일 시드**, legal v2 |
| 재집계 스크립트·결과 CSV | `.claude_logs/experiments/analysis/DRN-261016-03/` = NAS `analysis_logs/res_gain_decomp_20261016/`(10파일 md5 일치) | `muses_decomp.py`, `deliver_decomp.py` |

클래스 묶음 정의: DELIVER thin4 = Pole·Pedestrian·TrafficLight·Static(도구 정의). MUSES 얇은 7 = pole·traffic light·traffic sign·person·rider·bicycle·motorcycle(이 분석에서 정함), 나머지 12 = 넓은 영역.
기여도 = 클래스 ΔIoU ÷ 클래스 수(mIoU 가 단순 평균이므로 합이 ΔmIoU 와 정확히 같다).

## 사전 기준 대비 사실

### Q1-a MUSES: 1024→1536 이득의 출처(3시드 평균)
- 전체 ΔmIoU **+0.562**(82.30→82.86). 얇은 7클래스 기여 **+0.625(111%)**, 넓은 12클래스 −0.064.
- 상위: pole +3.06 · traffic light +2.66 · person +2.52 · traffic sign +2.05 · motorcycle +0.83(IoU 점). 이 4클래스(pole·TL·person·sign)만으로 +0.542 = **이득의 96%**.
- 혼동행렬(3시드 합 ÷3): 얇은 클래스 이득은 **FP 감소**에서 나온다. pole FP 0.75M→0.61M px(TP 2.60→2.63M), person FP 0.14→0.11M, traffic light FP 0.07→0.05M. 즉 더 찾는 것보다 윤곽 밖으로 번지는 예측이 줄었다.
- 조건별(낮): day 4조건 평균 +1.17, 이 중 얇은 7 기여 +1.04(89%).
- **성공 기준(이득 70% 이상을 설명하는 구간 특정) 대비: 클래스 수준은 충족**(얇은 4클래스 96%). 두께 구간 분해는 MUSES 예측 PNG 가 없어 **미실행**(MUSES 용 thin_error_anatomy 판은 새 코드 필요 → build 분리 대상).

### Q1-b DELIVER: 1024→1536 이득의 출처(Q2 시드821 test, 단일 시드)
- 전체 ΔmIoU **+1.477**(57.73→59.20). thin4 기여 **+0.448(30%)**, 나머지 21클래스 +1.029.
- 상위 10: Pedestrian +4.51 · Pole +4.50 · GroundRail +3.55 · TrafficSign +3.20 · TwoWheeler +3.02 · Fence +2.51 · RoadLine +2.34 · Wall +2.18 · Static +1.93 · Vegetation +1.76. 상위 8클래스 누적 70%. 하락은 RailTrack −0.92 하나.
- 클래스 묶음으로는 thin4 가 30% 뿐이지만, 상위 8 가운데 GroundRail·TrafficSign·TwoWheeler·Fence·RoadLine 도 가늘거나 작은 구조다. 그래서 클래스보다 **성분 두께**가 더 맞는 축이다.
- 두께 구간(모델 입력 1024 기준 16px 패치 단위, GT 성분 수 기준). "잘 그린 성분"(찾음 + 성분 IoU≥0.5) 증가분의 구간별 몫:

| 묶음 | <0.25 패치 | 0.25~0.5 | 0.5~1 | 1~2 | ≥2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| thin4(+9,957 성분) | 57% | 38% | 5% | 0% | 0% |
| 나머지 21(+32,289 성분) | 53% | 35% | 11% | 1% | 0% |

  - 두께 1 패치 이상은 찾음률·"찾았지만 못 그림" 비율 변화가 모두 0.015 이하다.
  - thin4 0.25~0.5 패치: 찾음률 .580→.680(DGF 재학습판 1024 .671), 못 그림 .445→.259(DGF .259, 같아짐).
- 경계 거리(25클래스 합, GT 픽셀 기준): 오류 픽셀 총량 104.70M→96.11M(−8.59M).
  - FP 감소는 **정답 경계 3px 이내**에서 −9.20M(−1px 이내 −4.42M, 1~3px −4.79M)이다. 16px 넘게 떨어진 FP 는 오히려 +1.53M 늘었다.
  - FN 감소 −8.60M 는 3~32px 구간 −5.09M · 3px 이내 −1.68M · 32px 초과와 예측 없음 −1.83M 이다.
- 조건별 ΔmIoU(1536): cloud +1.03 · fog +2.04 · night +0.93 · rain +2.28 · sun +0.84. thin4 기여는 모든 조건에서 +0.40~+0.51 로 비슷하다.
- **성공 기준 대비: 두께 구간으로는 충족**(잘 그린 성분 증가의 95~99% 가 1 패치 미만). 단 성분 수 기준이며 픽셀 가중이 아니다. 픽셀 기준으로는 FP 감소 전량이 경계 3px 이내다.

### Q2 DGFusion 대비: 미실행(덤프 대기)
- DGFusion 공식 가중치의 val 1024·1536 예측 PNG 가 어디에도 없다. native_axis_eval 은 summary.json 만 남긴다(lecun `native_axis_baseline_20261016/` 확인, PNG 0개).
- 지금 있는 DGFusion 덤프는 재학습판(dgffinal) test@1024 하나뿐이다. 같은 모델의 두 해상도 쌍이 없으므로 H-b 를 검증할 수 없다.
- 참고 사실(1024 단면, 덤프 있음): 1024 에서 DGF 재학습판의 thin4 0.25~0.5 패치 못 그림 비율 .259 는 우리 1024 .445 보다 이미 낮고, 우리 1536 .259 와 같다. H-b 의 전제("DGFusion 은 1024 에서 이미 얇은 성분을 그린다")와는 부합한다. 그러나 DGFusion 자신의 1536 증가 구간은 미측정이다.

### Q3 MUSES 2048 야간 하락(3시드 평균, 조건당 25장)
- 야간 4조건 평균 Δ(2048−1024) **−3.92**. 조건별 차이가 크다: clear/night +0.89 · fog/night **−7.01** · rain/night −4.71 · snow/night −4.86. 즉 "야간" 이 아니라 **열화+야간 3조건** 하락이다.
- 1536 에서도 이미 같은 3조건이 떨어진다: fog/night −2.29 · rain/night −0.85 · snow/night −1.65(clear/night +3.49). 야간 평균 1536 Δ = −0.32 이고, 이 중 얇은 7 −0.51 · 넓은 12 +0.18 이다.
- 야간 2048 하락 −3.92 의 분해: 얇은 7 **−1.92** · 넓은 12 **−2.00**(반반). 상위 3클래스 rider −0.66 · sidewalk −0.59 · wall −0.57 의 합은 **하락의 46%**.
- **성공 기준(상위 3클래스가 하락의 절반 이상) 대비: 야간 전체로는 미충족(46%)**. 조건 단위로는 충족: fog/night 75%(sidewalk·wall·fence = 넓은 영역), rain/night 64%(rider·motorcycle·pole = 얇음), snow/night 73%(rider·bicycle·truck).
- 시드 산포: fog/night 2048 Δ = −4.92/−4.71/**−11.39**(시드826 이 −11.39, 그중 wall −41.10). snow/night rider 2048 Δ = −39.55/−36.22/+0.46.
- 표본 크기: rider 는 250장 전체 TP 가 시드당 약 0.06M px 뿐이라 조건 25장 안에서는 극소 화소 클래스다. rider·motorcycle·bicycle 의 야간 Δ 는 한두 장면에 좌우될 수 있다(장면 단위 미확인).
- 전체 혼동행렬(조건 미분리)에서 2048 증가 오류: sidewalk→road FN 1.01→1.42M px · fence→vegetation FN 0.56→0.76M · wall FN 0.75→0.85M. 넓은 영역 클래스 경계가 인접 넓은 클래스 쪽으로 밀린다. 조건별 혼동행렬은 report 에 없어 야간만 분리하지 못했다.

## 원인 해석 (검증 여부 표시)
- [사실 기반 강한 지지] H-a(이득은 패치 이하 성분에서 나온다): DELIVER 두께 구간(1 패치 미만 95~99%)과 MUSES 클래스(얇은 4클래스 96%)가 같은 방향이다. 기제는 **경계 3px 이내 FP 감소**(DELIVER 픽셀 기준, MUSES 얇은 클래스 FP 감소)이므로, "더 찾는다" 보다 "윤곽을 더 정확히 놓는다" 가 주성분이다. 단 DELIVER 는 단일 시드·test, MUSES 는 두께 미측정.
- [미검증] DELIVER 에서 thin4 밖 클래스(GroundRail·RoadLine·Fence·Wall)가 이득의 70% 를 차지하는 것은, 이 클래스들의 이득도 같은 1 패치 미만 성분·경계 정밀화에서 온다는 해석과 맞다. 클래스별 두께 구간 표는 도구가 thin4/other 두 묶음으로만 내서 직접 확인하지 못했다.
- [미검증] H-b: Q2 덤프 전까지 판단 불가. 1024 단면 사실은 전제와 부합한다.
- [부분 반증] H-c(2048 야간 하락 = 넓은 영역 맥락 붕괴): 야간 하락의 절반(−2.00)은 넓은 영역, 절반(−1.92)은 얇은 클래스다. "하락이 얇은 클래스에 몰리면 기각" 조건에 정확히 해당하지 않고, 반반이다.
  - fog/night 는 H-c 대로 넓은 영역(sidewalk·wall·fence)이 무너진다.
  - rain/night·snow/night 는 소수 화소 얇은 클래스(rider·motorcycle·bicycle)가 주도하며 시드 산포가 크다.
- [가설, 미검증] 하락이 clear/night 가 아니라 열화+야간에서만 나오고 1536 에서 이미 시작된다. 그래서 단순 스케일 불일치보다 "고해상 입력에서 저신호(어둠+안개/비/눈) 영역의 국소 노이즈가 토큰마다 더 크게 보이는 것" 이 원인일 수 있다. 확인하려면 조건별 혼동행렬 또는 장면별 per-image mIoU 가 필요하다.

## 판정 제안
판정은 method 세션이 한다. 사실만 요약하면 다음과 같다:
- Q1: 사전 기준(이득 70% 이상을 설명하는 구간) 충족. DELIVER 는 두께 1 패치 미만, MUSES 는 얇은 4클래스.
- Q2: 미측정.
- Q3: 야간 전체 기준으로는 상위 3클래스 46% 라서 미충족. 조건별로는 충족하지만 조건마다 다른 클래스가 원인이다.

## 다음 결정
- Q2: manager 덤프(DGFusion 공식 val@1024/1536, E1 시드1 val@1024/1536) 수신 후 thin_error_anatomy 를 CPU 로 실행해 이 파일에 덧붙인다(대기 중, inbox `20261016-171805-request-bc24a84a`).
- Q3 원인을 더 좁히려면 MUSES 조건별 혼동행렬 또는 장면별 mIoU 저장이 필요하다. eval_muses_official.py 는 조건별 per-class IoU 만 남기고 hist 는 남기지 않으므로 코드 확장(build 항목)이 필요하다. method 가 필요성을 정한다.
- MUSES 두께 분해(Q1 MUSES 판)도 같은 이유로 build 항목이 필요하다.
