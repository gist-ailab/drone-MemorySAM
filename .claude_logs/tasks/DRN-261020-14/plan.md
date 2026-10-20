착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L6 · A2 − 교사 증류(→ 열화 두 패스만, KD_W 0)

- id: DRN-261020-14
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 동결 clean 교사 증류가 A2 에 얼마나 기여하는가(Q2noKD 의 A2 판)
- 가설: Q2noKD(10-11) 재현: clean −1.0, 강건성 유지 → 증류는 clean 보존 역할
- 바꾼 변수(change): TRAIN.QAF.KD_W 0, 그 외 A2 동일
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성
- 반증 조건: 해당 없음
- configs: -

## 정정
