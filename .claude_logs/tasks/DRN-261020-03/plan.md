착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# S2-sum · DELIVER 40ep 스크린 3시드(s821·s902·s903) — 고해상 특징을 센서별 마스크 로짓 전용으로(합산 제거), A2 같은 시드 짝 비교

- id: DRN-261020-03
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 고해상 특징의 쓰임을 FPN 합산에서 센서별 마스크 내적 전용으로 바꾸면 A2 의 clean 이득을 유지하면서 신호등 오탐·얇은 두께 못 그림이 줄어드는가
- 가설: H-S2a: 세부가 dense 분류에 들어가지 않으므로 TrafficLight FP 감소(Δ>0). H-S2b: 센서별 stride-4 격자에서 마스크를 그려 두께 0.25~0.5패치 못 그림 ≤ .35. H-S2c: 결측 강건성은 합 구조라 유지(EMM15 Δ ≥ −0.5). 기제 예측: 센서별 마스크 항 기여가 물체 질의→depth, 노면·질감 질의→RGB 로 갈린다(D4 클래스 분업과 일치)
- 바꾼 변수(change): MODEL.HIRES_ADAPTER.USE: fpn_sum → mask_only (그 외 A2 config 동일)
- 기준선: A2 같은 시드(DRN-261012-02 s821·s902, DRN-261018-01 s903), legal v2 test@1024
- 성공 기준: 3시드 평균 24클래스 Δ(S2−A2) ≥ 0 그리고 TrafficLight Δ > 0 그리고 thin4 못 그림 ≤ .35 그리고 test EMM15 Δ ≥ −0.5
- 반증 조건: 24클래스 Δ ≤ −0.5 → S2 기각(세부가 what 에도 필요). S2-fused(DRN 별도) 와 차이 < 0.2 → '센서별' 효과 없음 = 노벨티 축소 기록. ep20 trainer val 이 A2 같은 시드 ep20 대비 −1.0 이하이면 kill
- configs: -

## 정정
