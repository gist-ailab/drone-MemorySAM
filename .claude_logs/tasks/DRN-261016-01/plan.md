착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 기준선 평가 해상도 반응(원본 1042 GT 축) — DGFusion 공식·CAFuser(b) val@1024/1280/1536 + 선택 해상도 test

- id: DRN-261016-01
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: 평가 해상도를 val 로 고르는 규칙을 기준선에도 적용하면 격차가 유지되는가(우리 E1·Q2 는 1536 에서 test +1.2~1.5)
- 가설: -
- 바꾼 변수(change): -
- 기준선: -
- 성공 기준: 같은 규칙(원본 GT·val 선택 해상도) 비교표 확정; 기준선 1024 재현 ±0.1(DGFusion 56.73·CAFuser(b) 55.79)
- 반증 조건: -
- configs: lecun native_axis_eval.py, analysis_logs/native_axis_baseline_20261016/

## 정정
