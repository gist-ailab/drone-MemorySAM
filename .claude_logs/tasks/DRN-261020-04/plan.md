착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# S2-fused 대조 · DELIVER 40ep 3시드 — 센서 합산 고해상 임베딩 하나로 마스크 전용(센서별 분리의 효과를 분리)

- id: DRN-261020-04
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: S2 의 이득이 '마스크 전용 결선' 때문인가 '센서별 분리' 때문인가
- 가설: S2-sum > S2-fused (thin4·TrafficLight 방향 일치, 차이 ≥ 0.2) 이면 센서별 분리가 실체. 같으면 주장을 마스크 전용 결선으로 축소
- 바꾼 변수(change): MODEL.HIRES_ADAPTER.USE: mask_only_fused (센서 합산 F 하나, W_s 없음)
- 기준선: S2-sum 같은 시드
- 성공 기준: 대조군: 판정은 S2-sum 항목에서
- 반증 조건: 해당 없음(대조군)
- configs: -

## 정정
