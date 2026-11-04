착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MCubeS 논문 모델 A2(C2D2 포함: E1Mc + 고해상 경로 + 두 패스 학습·동결 E1Mc 교사 증류) 2시드 20260827·20260828 — 200ep, 24GB 는 768 크롭 변형

- id: DRN-261017-02
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: DELIVER A2 의 이득이 MCubeS 에서도 나는가
- 가설: MCubeS 도 얇은 구조물·재질 경계가 있어 고해상 경로가 도움
- 바꾼 변수(change): E1Mc(4탭) config 에 MODEL.HIRES_ADAPTER on + TRAIN.QAF(두 패스 + KD, DELIVER A2 와 같은 값, 커리큘럼 [0/50/100]ep) + 교사 = 같은 시드 E1Mc clean ckpt(TEACHER_CFG 로 교사 config 지정). 24GB 카드는 DATASET.TRAIN_CROP/TRAIN_BASE 768. 열화 연산자는 MCubeS 모달 매핑(degrade.py TRAIN_OPS image/nir/aolp/dolp)
- 기준선: 같은 시드 E1Mc 200ep(커뮤니티 test 102장, 선택 규칙 = 관행대로 test 102 로 val-best 선택, user 2026-10-30) + w/o C2D2 행(E1Mc+고해상, DRN 별도)
- 성공 기준: 2시드 평균 Δ vs E1Mc ≥ +0.5 이고 두 시드 부호 일치 → 메인 표 3 의 Ours. 보조: w/o C2D2 행 대비 Δ 로 두 패스 학습의 MCubeS 기여 기록
- 반증 조건: 2시드 평균 Δ < +0.2 면 이식 실패
- configs: configs/yeon-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2c2d2_768.yaml, configs/yeon-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828_A2c2d2_768.yaml, configs/lecun-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2c2d2_768.yaml, configs/lecun-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828_A2c2d2_768.yaml, configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2c2d2.yaml, configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828_A2c2d2.yaml

## 정정
