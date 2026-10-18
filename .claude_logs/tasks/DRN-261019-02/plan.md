착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MUSES 4모달 A2 — E1 구조 + Q2 레시피(두 패스+E1M4 교사 증류) + 고해상 경로, 40ep 스크린 s2·s3

- id: DRN-261019-02
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: DELIVER 논문 모델(A2)을 같은 방식으로 MUSES 4모달에 학습했을 때 공식 val·test 가 4모달 기존 최고와 DGFusion 을 넘는가
- 가설: DELIVER 에서 A2 가 Q2 대비 +1.0, Q2 가 E1 대비 +1.3 이었던 것처럼 MUSES 4모달에서도 기존 최고보다 오른다
- 바꾼 변수(change): 4모달 E1M 스크린에 TRAIN.QAF(Q2 레시피) + MODEL.HIRES_ADAPTER
- 기준선: MUSES 4모달 P39.1-rank 공식 test 79.571 · DGFusion 79.5 · 교사 E1M4 공식 val
- 성공 기준: 2시드 공식 val 평균 > 교사 E1M4 공식 val, 그리고 test(user 제출) 평균 > 79.571
- 반증 조건: 2시드 공식 val 평균 ≤ 교사 − 0.3 이면 MUSES 이식 실패
- configs: configs/hpca100-muses_rgbelr_P39_1_physaugoff_taps_screen40_A2_s2.yaml,configs/hpca100-muses_rgbelr_P39_1_physaugoff_taps_screen40_A2_s3.yaml

## 정정
