착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 논문 비용표(학습 0): A2·Q2·E1·DGFusion·CAFuser 의 params·FLOPs·추론 지연·GPU 메모리, DELIVER 1024 같은 조건

- id: DRN-261027-02
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: 우리 모델의 추론 비용이 기준선 대비 어느 수준인가(IEEE Sensors J 리뷰어 필수 요구)
- 가설: A2 는 DGFusion(Swin-T) 보다 파라미터 5배 이상·지연 2~3배(동결 포함 총량 기준). 학습 파라미터는 74M
- 바꾼 변수(change): 학습 없음. 같은 GPU(4090 1장)·같은 입력(1×4모달 1024²)·bf16, 100회 평균 지연·최대 메모리·fvcore FLOPs. 기준선은 재학습 킷 체크아웃 그대로
- 기준선: DGFusion·CAFuser 재학습 체크아웃(jarvis/lecun)
- 성공 기준: 표 1개(모델 | 총 params | 학습 params | GFLOPs | ms/img | GB) + 측정 스크립트·환경 기록
- 반증 조건: 해당 없음(측정)
- configs: -

## 정정
