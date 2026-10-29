착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 내용 기반 토큰 MoE-LoRA 구현: 센서 공유 전문가 K=4(r8) + 라우터(블록 12 토큰 stop-grad, softmax 혼합, 부하 균형 손실) + 라우팅 진단 덤프(전문가×클래스·센서별 분포)

- id: DRN-261029-04
- kind: build
- 문제(PRB): PRB-DRN-261020-01
- 질문: P56-C 의 state_routed 코드를 '열화 혼합' 에서 '내용 전문가 혼합' 으로 바꿔 off 계약을 지키며 구현할 수 있는가
- 가설: LORA_MODE: content_moe — 전문가 K 공유, 라우터 입력 = 블록 12 토큰(stop-grad), 센서 임베딩 없음, 부하 균형 손실(switch 식). 미지정 시 byte 동일
- 바꾼 변수(change): encoder.py/p56c_router.py 확장: 전문가 K 세트·softmax 라우터·균형 손실·eval 스태시(토큰별 전문가 분포) + 진단 도구(전문가×클래스 정규화 상호정보, 센서별 분포 거리, 사용률) + 스모크(off 계약·grad·균형 손실)
- 기준선: P56-C 코드(develop)
- 성공 기준: 스모크 통과 + 검수. 착수 시점 = 메인 3표(MCubeS·MUSES A2) 기동 뒤
- 반증 조건: 해당 없음(구현)
- configs: -

## 정정
