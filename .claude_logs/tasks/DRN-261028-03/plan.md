착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 고해상 신호 구동 토큰 적응 LoRA · DELIVER 40ep 1시드 s821 — 기준 모델(A2-shared 또는 A2) + 공유↔센서별 토큰 혼합(신호=고해상 통계), Q2 학습 불변

- id: DRN-261028-03
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 실제 열화가 고해상 통계로 감지되는 토큰에서만 센서별 적응을 끄면 clean 을 잃지 않고 실제 열화 케이스·부분 결측에서 오르는가(P56-C 가 신호 부재로 붕괴한 것과 달리 α 가 토큰별로 갈리는가)
- 가설: α 토큰 표준편차 ≥ 0.05, clean 24클래스 Δ ≥ −0.3, 실제 열화 파티션 평균 Δ ≥ +1.0 또는 depth/RGB RMM r0.5 저하 20% 감소, EMM15 Δ ≥ 0
- 바꾼 변수(change): MODEL.LORA_MODE state_routed + LORA_ROUTER.SIGNAL hires_stats (그 외 기준 모델 config 동일; 기준 = L1 강건성 통과 시 A2-shared, 아니면 A2)
- 기준선: 같은 시드 기준 모델(A2-shared s821 또는 A2 s821), legal v2 test@1024 + mm_eval_v2 + 조건·케이스 파티션
- 성공 기준: 4기준 전부 충족 시 시드 902·903 추가 + MUSES 전이(별도 등록). 초고 동결 11-10 전 3시드 완료 시 편입, 아니면 다음 논문
- 반증 조건: α std < 0.02(붕괴) 또는 clean Δ < −0.3 또는 EMM Δ < 0 → 기각, 토큰 적응 LoRA 축 종결 기록. ep20 val 이 기준 대비 −1.0 이면 kill
- configs: -

## 정정
