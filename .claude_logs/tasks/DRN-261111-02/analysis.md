판정은 이 파일이 아니라 `lab-plan verdict DRN-261111-02 --analysis .claude_logs/tasks/DRN-261111-02/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-12 01:3x (부분 전표). 요청 inbox `20261111-134639-request-d5fd124d`(method). 학습 없음. 판정·해석 문장 없음(요청대로 축별 값·Δ·부호 일치만).

## 대상·공통 설정
- A2 s821 `epoch40_68.93`(md5 f05b54d4…) · A2 s902 `epoch40_70.09`(0aab4cae…) · L6(증류 제거, KD_W 0) s821 `epoch40_69.39`(95d97cc3…) · L6 s902 `epoch30_69.24`(fa8239ba…). L6 의 MODEL 블록은 A2 와 같아 A2 평가 config 로 평가(yaml 로드 대조, 차이 키 0).
- 채점 = legal v2(native GT nearest-exact), 도구 `tools/mm_eval_v2.py`(seed 0, 전수). ④⑤ = 새 도구 `tools/pred_consistency_ece.py`(열화 = missing_modality_eval 의 zero_fill·bench_noise 그대로, batch 8 고정).
- 같은 체크포인트의 clean test(참고, Δ 해석의 기준선): A2 s821 58.61 · s902 57.09 / L6 s821 58.77 · s902 56.31 → **clean Δ(A2−L6) s821 −0.16 · s902 +0.78**.

## ② NM 벤치 정의 test (완료)
| | Low | Mid | High |
| --- | ---: | ---: | ---: |
| A2 s821 | 50.40 | 46.47 | 38.36 |
| L6 s821 | 51.28 | 47.60 | 40.21 |
| Δ s821 | −0.88 | −1.13 | −1.85 |
| A2 s902 | 50.81 | 47.37 | 39.67 |
| L6 s902 | 50.31 | 46.99 | 39.23 |
| Δ s902 | +0.50 | +0.38 | +0.44 |
| **Δ 2시드 평균** | **−0.19** | **−0.38** | **−0.71** |
| 부호 일치 | 아니오 | 아니오 | 아니오 |
- A2 = drone-demo 5090 batch 8, L6 = jarvis 4090 batch 8(벤치 S&P 는 배치 텐서 max/min 을 써서 배치 크기에 의존 → 두 모델 같은 batch 8). 원본: drone-demo `analysis_logs/kd_axes_20261111/A2_s*_test_nmbench`, jarvis `/SSDb/jemo_maeng/analysis_logs/kd_axes_20261111/L6_s*_test_nm_all`.

## ④ 예측 일치도 test (완료) — clean 예측과 열화 예측이 같은 유효 화소 비율(%)
| | EMM 14조합 평균 | NM Mid |
| --- | ---: | ---: |
| A2 s821 / L6 s821 / Δ | 92.99 / 93.27 / **−0.28** | 93.83 / 93.97 / **−0.14** |
| A2 s902 / L6 s902 / Δ | 93.13 / 91.22 / **+1.91** | 94.06 / 92.63 / **+1.43** |
| **Δ 2시드 평균** | **+0.82** | **+0.65** |
| 부호 일치 | 아니오 | 아니오 |

## ⑤ ECE(15 bin, %) test (완료) — 낮을수록 보정이 좋음
| | clean | NM Mid |
| --- | ---: | ---: |
| A2 s821 / L6 s821 / Δ(A2−L6) | 2.89 / 2.62 / +0.27 | 4.30 / 2.68 / +1.62 |
| A2 s902 / L6 s902 / Δ | 2.88 / 3.86 / −0.97 | 4.26 / 3.44 / +0.81 |
| 상대 변화(A2/L6 − 1) | s821 +10.4% · s902 −25.2% | s821 +60.6% · s902 +23.6% |
- clean mIoU 재계산 검산: A2 s821 58.6137 · s902 57.0924(등록 58.6138·57.0919), L6 s821 58.7706 · s902 56.3120. 원본 jarvis `/SSDb/jemo_maeng/analysis_logs/kd_axes_20261111/<ckpt>_test_pce/summary.json`(빈별 count·conf·acc 포함).

## ① test RMM 15조합 (진행 중 — 56 레인 중 27 완료)
- 부분집합 14개 × r 3개를 MM_PRESENT_ONLY 레인으로 나눠 실행(합산은 aggregate_missing 과 같은 식). 15조합 평균은 14 레인이 모두 끝나야 나온다.
- 지금까지 A2·L6 가 같은 시드에서 모두 끝난 부분집합 11개(s821 9 · s902 2)의 부분집합별 Δ(A2−L6), r=.75/.5/.25:
  - s821: depth −0.23/−0.15/−0.21 · depth+event+lidar +0.15/−0.03/−0.18 · event −0.26/−0.61/−0.45 · img+depth −0.23/−0.14/−0.18 · img+depth+event −0.18/−0.10/−0.14 · img+depth+lidar −0.18/−0.21/−0.20 · img+event −0.86/−0.67/−0.45 · img+event+lidar −0.85/−0.71/−0.44 · lidar −0.51/−0.62/−0.37
  - s902: img+depth+lidar +0.85/+0.84/+0.81 · img+event +1.14/+1.20/+1.33
- 남은 레인: jarvis 4090 GPU5·6·7 큐 · amphibious 5090 3 레인 · bengio 3090 5 레인(현재 레인 후 중단) · levine 1 레인 · jarvis 빈 GPU 디스패처 8 레인(기동 시점 빈 GPU 확인 후). **예상 완료 11-12 08:00~11:00**(jarvis GPU 가 비는 시점에 좌우).

## ③ val EMM15·val RMM (진행 중)
| | val clean | val EMM15 | val RMM .75/.5/.25 |
| --- | ---: | ---: | --- |
| A2 s821(robust_a2_20261019) | 69.59 | 57.53 | 60.75 / 62.89 / 65.07 |
| A2 s902(robust_a2_20261019) | 71.05 | 58.07 | 61.97 / 64.11 / 66.41 |
| L6 s821 | 70.30 | 58.67 | 미측정 |
| L6 s902 | — | 측정 중(jarvis GPU4, 약 7시간, 11-12 08:00 전후) | 미측정 |
- Δ val EMM15 s821 = −1.14. L6 val RMM 은 ① 이 끝난 뒤 같은 레인 분할로 측정(시드당 14 레인).
- 주의: A2 val 값은 10-19 실행(서버·torch 판이 다름), L6 은 jarvis torch 2.7 실행 — 서버 간 clean 차는 이번 측정들에서 ≤0.005 수준이었다.

## 운영 기록
- 11-11 14:35 levine GPU5·6, 11-12 01:18 jarvis GPU0~3 에 다른 사용자 점유 상태에서 레인을 띄워 OOM 으로 즉시 종료(각 1분 이내, 상대 프로세스 영향 미확인). 이후 jarvis 는 '기동 시점 빈 GPU 재확인' 디스패처로만 띄운다.
