착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L5 · A2 − 열화 패스(→ clean 패스 + 동결 E1 교사 증류만, DEGRADE_P 0)

- id: DRN-261020-13
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 열화 두 패스가 A2 의 강건성·clean 에 얼마나 기여하는가(Q2noDeg 의 A2 판)
- 가설: Q2noDeg(10-06) 재현: clean −0.9, EMM15 크게 하락(−8 이상), depth 제거 Δ 가 E1 수준(−24~−39)으로 복귀
- 바꾼 변수(change): TRAIN.QAF.DEGRADE_P 0 · MISSING_FRAC 0 (KD 유지), 그 외 A2 동일
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성
- 반증 조건: 해당 없음
- configs: -

## 정정
