착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L7 · A2 − 센서별 물리 열화(→ 결측 0채움만, 결측 노출량은 A2와 동일) — 두 패스 학습이 RobustSeg 'basic distillation'과 다른가

- id: DRN-261029-01
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: A2의 열화 패스에서 센서별 물리 열화 연산자(패치 드롭·블러·게인/감마·색 변화·depth 구멍·LiDAR 빔 드롭·event 해상도 저하)와 강도 커리큘럼을 빼고 완전 결측(0채움)만 남기면 clean·결측·노이즈 성능이 떨어지는가. 논문(IEEE Access, DRN-261027-01) contribution '두 패스 학습'이 선행(RobustSeg Tab.3 Basic-distillation = 동결 clean 교사 KD + 결측 학습)과 다르다는 것을 보이는 유일한 실험
- 가설: 물리 열화가 결측 학습을 넘어 기여한다: 결측만 변형은 A2 대비 clean 24클래스·test EMM15·NM(salt-and-pepper, 학습에 없음)이 모두 낮다
- 바꾼 변수(change): TRAIN.QAF.DEGRADE_P 0.5→0.1 + MISSING_FRAC 0.2→1.0 (센서당 결측 확률 0.1 은 A2 와 동일하게 유지하고 물리 열화만 0으로; 교사 증류·clean 패스·고해상 경로 등 나머지 A2 config 동일)
- 기준선: A2 같은 시드 s821·s902(legal v2 test@1024, EMM15 val·test, NM .05/.10/.20), L5·L6 과 같은 평가 묶음
- 성공 기준: 2시드(s821·s902) 짝 Δ(A2 − L7) 평균이 test clean 24클래스 ≥ +0.5 또는 test EMM15 ≥ +1.0 또는 NM S&P .20 ≥ +1.0 이고, 그 지표에서 두 시드 부호 일치 → 센서별 물리 열화가 basic distillation 과 다른 기여
- 반증 조건: 세 지표 모두 |Δ| < 0.3 → 기여 문장을 "증류+결측 학습의 상호작용" 으로 축소, 센서별 열화를 차별점으로 주장하지 않음
- configs: -

## 정정
