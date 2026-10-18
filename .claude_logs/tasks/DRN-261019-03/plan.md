착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# A2 강건성 — 2시드(s821·s902) DELIVER test EMM15·RMM·NM + val EMM15·RMM(공개 벤치 정의), Q2 와 같은 프로토콜

- id: DRN-261019-03
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: 고해상 경로(A2)를 넣어도 Q2 의 결측·열화 강건성이 유지되는가(논문 강건성 주장을 최종 모델로)
- 가설: -
- 바꾼 변수(change): -
- 기준선: -
- 성공 기준: 같은 프로토콜 Q2 3시드 대비: test EMM15 평균 Δ ≥ −1.0 그리고 val EMM15 평균이 공개 최고 RMMSS 49.93 초과 유지. 2시드 표 확정
- 반증 조건: -
- configs: tools/missing_modality_eval.py(Q2 때와 같은 인자, robust_q2_20260929·robust_q2_val_20261008 와 동일), A2 ckpt NAS ckpts(a2 s821 f05b54d4… · s902 0aab4cae…)

## 정정
