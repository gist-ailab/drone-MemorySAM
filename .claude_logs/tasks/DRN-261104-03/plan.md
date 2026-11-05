착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MCubeS w/o C2D2 행 크롭 정합 · E1Mc + 고해상 경로(두 패스 없음) 768 학습 크롭 2시드 20260827·20260828 200ep — C2D2 768 런과의 clean 차이에서 크롭 교란 제거

- id: DRN-261104-03
- kind: experiment
- 문제(PRB): -
- 질문: MCubeS 에서 C2D2(768 크롭) 2시드 59.80/59.76 이 w/o C2D2(1024 크롭, s827 1런) 60.27 보다 낮은 것이 두 패스 학습의 비용인지 학습 크롭 차이(768 vs 1024)인지
- 가설: 같은 768 크롭에서 w/o C2D2 2시드 평균이 C2D2 2시드 평균(59.78)과 ±0.5 안이면 두 패스 학습은 MCubeS clean 에 중립, 0.5 넘게 높으면 비용, 낮으면 이득
- 바꾼 변수(change): DATASET.TRAIN_CROP/TRAIN_BASE 1024 → 768 (w/o C2D2 config 기준 한 변수). 구조·레시피·epoch·선택 규칙(test 102 val-best) 동일
- 기준선: C2D2 768 2시드(DRN-261017-02) 59.80/59.76 · w/o C2D2 1024 s827(DRN-261031-01) 60.27 · E1Mc 1024 s827/s828 58.94/58.61
- 성공 기준: 완주 + 학습기 top1(test 102 관행) + 하네스 재채점 → Table 3 에 'Ours w/o C2D2 training(768)' 행으로 2시드 평균 게재. C2D2 − (w/o C2D2 768) 의 부호·크기를 본문에 그대로 적는다(음수여도 게재)
- 반증 조건: 두 시드 부호가 다르면 '시드 노이즈 범위' 로 적고 평균만 게재
- configs: configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2hires_768.yaml, configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828_A2hires_768.yaml

## 정정
