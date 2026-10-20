착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# S2 구현: 센서별 고해상 임베딩 F_s + 마스크 로짓 전용 결선(MODEL.HIRES_ADAPTER.USE fpn_sum|mask_only|mask_only_fused) + 스모크

- id: DRN-261020-06
- kind: build
- 문제(PRB): PRB-DRN-261020-01
- 질문: A2 의 고해상 특징을 FPN 합산 대신 센서별 마스크 내적 전용으로 쓰는 결선을 off 계약을 지키며 구현할 수 있는가(메모리 포함)
- 가설: 구현 가능. USE=fpn_sum 이면 A2 와 state_dict·forward byte 동일, ENABLE=false 면 Q2 와 byte 동일. 센서별 어댑터는 추출기 가중치 공유·센서 배치(B·N)로 돌리면 활성화 증가 1.6배 이내
- 바꾼 변수(change): hires_adapter.py 센서별 출력 모드(s4 레벨만 반환) · model.py USE 분기(_apply_hires_adapter 우회, [F_s] 전달) · m2f_head.py sensor_s4 인자 + 센서별 W_s 선형 + masks += Σ_s einsum · tools/smoke_hires_adapter.py 확장(등가 assert 2종·grad 흐름·메모리 프로파일) · 기제 판독용 센서별 마스크 항 기여 덤프 옵션
- 기준선: A2 코드(develop 1b029c5 이후)
- 성공 기준: 완료 조건: (1) USE=fpn_sum 출력 byte 동일 (2) ENABLE=false 출력 Q2 와 byte 동일 (3) mask_only 에서 W_s·어댑터로 grad 흐름 (4) 1024² 프로파일 활성화 ≤ A2 1.6배 (5) 코드 검수 파이프라인 통과 후 develop 병합
- 반증 조건: 메모리가 A2 의 2배를 넘으면 '공유 어댑터 1회 + 센서별 1×1 투영' 축소판으로 바꾸고 기록
- configs: -

## 정정
