착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# A2 실패-키 분석(학습 0): 구조 제안의 선행 조건 — TrafficLight FP 장면 단위·세부-의미 분리 증거·열화+야간 혼동행렬

- id: DRN-261020-01
- kind: analysis
- 문제(PRB): PRB-DRN-261020-01
- 질문: A2 가 아직 틀리는 세 곳이 같은 기제('세부는 늘었는데 의미 구분이 따라오지 못함')인가, 아니면 서로 다른 기제인가?
- 가설: H1 TrafficLight FP 는 고해상 경로 특징이 백본 의미 특징과 융합될 때 의미 쪽이 희석되어 생긴다(FP 픽셀에서 고해상 경로 기여 > 백본 기여). H2 '찾았지만 못 그림' 잔여는 1패치 미만 두께에서 FPN 가장 고해상 레벨의 수용영역 한계. H3 MUSES 열화+야간 넓은 영역 밀림은 열화 패스 학습이 야간 조건을 과소 표집해서(열화 분포 vs 실제 야간 분포 불일치).
- 바꾼 변수(change): 학습 없음. 기존 A2 2시드 ckpt(s821·s902) + Q2 2시드 ckpt 로 (1) TrafficLight FP 상위 20장면 시각화·FP 픽셀의 경로별 기여(dense vs query, hires on/off 추론 시 ablation) (2) TrafficLight 단독 두께 구간(thin_error_anatomy 클래스 단일 모드 추가) (3) MUSES 조건별 혼동행렬(night 3조건 분리) 및 DELIVER night/fog 조건별 혼동행렬
- 기준선: Q2 같은 시드(s821·s902) 같은 도구
- 성공 기준: 세 실패 각각에 대해 [사실] 등급의 기제 문장 1개와 반증된 가설 목록이 analysis.md 에 남는다. H1 은 FP 픽셀에서 hires-off 추론으로 FP 가 50% 이상 줄면 채택.
- 반증 조건: hires-off 추론으로 TrafficLight FP 가 20% 미만으로만 줄면 H1 기각(원인은 Q2 레시피 쪽).
- configs: -

## 정정
