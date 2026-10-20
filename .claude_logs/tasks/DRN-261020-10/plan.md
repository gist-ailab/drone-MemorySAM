착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L2 · A2 − cross-modal attention 융합(→ 센서 토큰 평균 MeanFusionTrunk)

- id: DRN-261020-10
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 교차주의 융합이 A2 에 얼마나 기여하는가
- 가설: clean −0.5 이상, 조건별 의존도 이동 폭 감소
- 바꾼 변수(change): FUSION 트렁크를 MeanFusionTrunk(P43 식)로, 그 외 A2 동일(config 파생, 코드 변경 없음 예상)
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성
- 반증 조건: 평균 융합이 A2 와 ±0.2 안이면 '융합 구조 기여' 문장 삭제
- configs: -

## 정정
