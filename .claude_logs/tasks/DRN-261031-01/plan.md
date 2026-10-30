착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MCubeS w/o C2D2 비교 행 — E1Mc + 고해상 경로(두 패스 없음) 시드 20260827, hpca100 ep128/200 중단 런 이어 학습(AUTO_RESUME)

- id: DRN-261031-01
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: MCubeS 에서 고해상 경로만 더한 모델(두 패스 학습 없음)이 어디까지 가는가 — 논문 모델(C2D2 포함) 과의 차로 두 패스 학습의 MCubeS 기여를 분리
- 가설: C2D2 포함 런보다 낮고, E1Mc 보다는 높다(DELIVER 에서 고해상 경로 +0.9, 두 패스 +1.8 의 순서가 유지)
- 바꾼 변수(change): configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2hires.yaml (TRAIN.QAF 없음) 그대로, ep128 부터 AUTO_RESUME(paper 세션 스크립트 scripts/resume_mcubes_a2_s827.sh, hpca100 tmux mcubes_resume_s827). 중단 전 학습기 top1 ep70 59.84(test 102 선택)
- 기준선: E1Mc 시드 20260827 200ep · 논문 모델 C2D2(DRN-261017-02 같은 시드)
- 성공 기준: 완주 후 legal 평가 1건, ablation 표 'w/o C2D2' 행으로 기록(판정 없음)
- 반증 조건: 해당 없음(비교 행)
- configs: -

## 정정
