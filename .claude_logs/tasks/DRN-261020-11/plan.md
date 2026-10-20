착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L3 · A2 − 4탭(→ 마지막 블록 탭 1개)

- id: DRN-261020-11
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 중간층 4탭이 A2 에 얼마나 기여하는가(고해상 어댑터의 key/value 도 4탭이라 결합 효과 포함)
- 가설: 얇은 객체·경계에서 −0.5 이상
- 바꾼 변수(change): MODEL.TAPS.LAYERS: [24] (고해상 어댑터 탭 수 1 로 맞춤), 그 외 A2 동일
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성
- 반증 조건: 해당 없음
- configs: -

## 정정
