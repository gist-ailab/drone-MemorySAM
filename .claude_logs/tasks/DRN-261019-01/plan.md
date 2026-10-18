착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MUSES 4모달 교사 E1M4 — 4탭 clean 200ep(+radar), A2-4모달의 동결 교사

- id: DRN-261019-01
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: MUSES 4모달(img·lidar·event·radar)에서 E1 구조를 clean 으로 학습해 Q2 레시피의 교사를 만든다(DELIVER E1 확정과 같은 역할)
- 가설: 교사 품질은 4모달 P39.1-rank(공식 test 79.571)와 비슷할 것
- 바꾼 변수(change): E1M full200 에 radar 추가(DATASET.MODALS)
- 기준선: MUSES 4모달 P39.1-rank 시드2 공식 test 79.571 / E1M 3모달 공식 val 81.70~81.93
- 성공 기준: 학습 완주 + 공식 val ≥ 81.0(교사로 쓸 수 있는 수준)
- 반증 조건: 공식 val < 80 이면 교사 품질 의심 — radar·4모달 경로 점검 후 진행
- configs: configs/hpca100-muses_rgbelr_P39_1_physaugoff_taps_full200_E1M4.yaml

## 정정
