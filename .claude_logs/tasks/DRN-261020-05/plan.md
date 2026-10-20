착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# S2-x 탐색 1시드 — F_s 생성 전 s4 창 단위 센서 간 교차주의 1층(게이트 없음, 잔차 합)

- id: DRN-261020-05
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 센서별 고해상 격자끼리 경계를 맞추게 하면 S2-sum 보다 나아지는가(특히 MUSES 야간·희소 센서)
- 가설: DELIVER 는 모달 잉여라 Δ 작음(≤ +0.3), MUSES 야간에서 효과. 게이트·softmax 없음
- 바꾼 변수(change): S2-sum + HIRES_ADAPTER.XSENSOR_ATTN: window 8, 1층, 표준 초기화
- 기준선: S2-sum 같은 시드 s821
- 성공 기준: thin4 Δ ≥ +0.3 또는 MUSES 전이에서 야간 Δ ≥ +0.5 면 3시드 승격
- 반증 조건: 둘 다 미달이면 종료
- configs: -

## 정정
