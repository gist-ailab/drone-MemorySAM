착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L8 · A2 − router(재학습, MODEL.ROUTER.ENABLE false) — 최종 모델에서 router 의 역할

- id: DRN-261029-07
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: router 없이 처음부터 학습한 A2 는 clean·EMM·조건별 적응(야간 −RGB 완화, depth 의존 이동)이 A2 와 다른가. 결과는 원고 서술만 바꾼다(모델 구조는 A2 고정, user 2026-10-29)
- 가설: router 기여는 작고 조건별 적응은 두 패스 학습과 교차 attention 에서 주로 생긴다
- 바꾼 변수(change): MODEL.ROUTER.ENABLE true→false (router CE 0.4 도 함께 꺼짐), 나머지 A2 config 동일
- 기준선: A2 같은 시드(s821·s902), L5·L6 과 같은 평가 묶음 + 조건별 센서 하나 제거(test)
- 성공 기준: router 를 '적응 장치'로 주장 = 2시드 평균에서 router 제거 시 야간·저노출 −RGB 완화폭이 절반 이하로 줄거나 clean ≥ −0.5 또는 EMM15 ≥ −1.0 하락(두 시드 부호 일치). 아니면 구성요소로만 서술
- 반증 조건: 모든 지표 |Δ| < 0.3 이고 적응 이동 유지 → router 는 기여 없음, 원고에서 역할 주장 삭제(구조 설명·ablation 수치만)
- configs: -

## 정정
