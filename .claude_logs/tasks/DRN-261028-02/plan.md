착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 고해상 신호 구동 토큰 적응 LoRA 구현: LORA_ROUTER.SIGNAL=hires_stats(P56-C 라우터 입력을 센서별 고해상 스템 통계로 교체) + 스모크

- id: DRN-261028-02
- kind: build
- 문제(PRB): PRB-DRN-261020-01
- 질문: P56-C 의 state_routed LoRA 에서 라우터 입력만 고해상 토큰 통계(노름·고주파 에너지·채널 분산, stop-grad)로 바꿔 off 계약을 지키며 구현할 수 있는가
- 가설: 기존 StateRoutedLoRAQKV·p56c_router 를 확장하면 SIGNAL 미지정 시 byte 동일, 지정 시 토큰별 α 가 입력 통계에 따라 변한다
- 바꾼 변수(change): encoder.py/p56c_router.py: SIGNAL hires_stats 분기(고해상 스템 센서별 출력 → 토큰 통계 3종 → stride-16 풀링 → MLP, bias 초기화 α≈0.5) · config 키 LORA_ROUTER.SIGNAL · 스모크(off 계약·grad·합성 블러 패치에서 α 저하)
- 기준선: P56-C 코드(develop)
- 성공 기준: 스모크 통과 + 검수. 착수 조건 = DRN-261028-01 프로브 통과
- 반증 조건: 프로브 미통과면 착수하지 않음
- configs: -

## 정정
