착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 평가 해상도 상향이 왜 우리 모델만 크게 올리나 — 클래스·두께·조건 분해(DELIVER 우리 vs DGFusion, MUSES 1024/1536/2048)

- id: DRN-261016-03
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: Q1 이득은 어떤 클래스·물체 두께에서 나오나(MUSES 도 DELIVER 처럼 패치 이하 얇은 성분인가) Q2 같은 해상도 상향에서 DGFusion(+0.87)이 우리(E1 +1.56)보다 덜 오르는 이유가 같은 클래스·두께 구간 차이인가 Q3 MUSES 2048 에서 야간만 −3~4 떨어지는 원인(클래스·두께·조건 중 어디)
- 가설: -
- 바꾼 변수(change): -
- 기준선: -
- 성공 기준: Q1·Q2: 클래스별 Δ표 + 두께 구간별 '찾음률/찾았지만 못 그림' Δ표(1024→1536)로 이득의 70% 이상이 설명되는 구간을 특정. Q3: 2048 야간 하락의 클래스별 분해로 상위 3클래스가 하락의 절반 이상인지. 모두 기존 도구(thin_error_anatomy·eval_muses_official 조건별 hist) 재사용, 새 학습 없음
- 반증 조건: -
- configs: tools/baseline_failure/thin_error_anatomy.py, tools/eval_muses_official.py(report.json 조건별·클래스별), DELIVER 덤프: DGFusion 공식 1024/1536 원본 GT 축 예측(native_axis_eval 산출)

## 가설(사전 기록, 2026-10-16 MMSAM | 생각정리)
- H-a(Q1): 이득은 모델 입력 16px 패치보다 얇은 성분(두께 <1 패치)과 얇은 클래스(DELIVER Pole·Pedestrian·Static·TrafficLight / MUSES pole·traffic light·traffic sign·person·rider·bicycle)에 몰린다. 근거: DELIVER Q2 시드821 1024→1536 에서 두께 0.25~0.5 패치 "찾았지만 못 그림" .445→.333(analysis_logs/thin_anatomy_test_20261011_res/run_all/).
- H-b(Q2): DGFusion(Swin, 1/4 해상도 특징 보유)은 1024 에서 이미 얇은 성분을 그리므로 같은 구간의 증가폭이 작다. 반증: DGFusion 의 1024→1536 이득이 우리와 같은 두께 구간에서 비슷하게 나오면 H-b 기각(차이는 다른 데서 옴).
- H-c(Q3): MUSES 2048 야간 하락은 학습 때 보지 못한 스케일(학습 레터박스 1024, 2048 은 2배)에서 넓은 영역 클래스(road·sidewalk·building·sky·vegetation 등)의 맥락이 깨지는 것. 반증: 하락이 얇은 클래스에 몰리면 기각.

## 실행(analysis 역할)
1. DELIVER: DGFusion 공식 가중치 test 1024·1536 원본 GT 축 예측 PNG(native_axis_eval 이 남긴 것 — 없으면 같은 도구로 덤프)와 우리 Q2 3시드·E1 3시드 1024·1536 덤프로 `tools/baseline_failure/thin_error_anatomy.py` 실행(모델별 1024/1536 쌍). 산출 = 클래스별 IoU Δ, 두께 구간 Δ.
2. MUSES: `muses_resolution_chain` 의 report.json(태그별 per-class·조건별 hist)에서 1024/1536/2048 × 클래스 × 조건(day/night) Δ표. 두께 분해가 필요하면 val 250장 예측 덤프 후 thin_error_anatomy 의 MUSES 판(클래스 매핑만 바꿔) — 새 코드가 필요하면 build 항목으로 분리 요청.
3. analysis.md 에 사실(표) → 해석(검증 여부 표시) → 다음 결정 제안. 판정은 method 세션이 `lab-plan verdict`.

## 정정
