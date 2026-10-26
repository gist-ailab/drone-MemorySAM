착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 벤치 교차 확인(학습 0): 객체 단위 클래스 오라클 상한 + 객체 풀링 선형 프로브 — DELIVER(A2·Q2) 와 MUSES(E1M 1024/1536, 4모달 A2 완주 후 재측정)

- id: DRN-261022-01
- kind: analysis
- 문제(PRB): PRB-DRN-261020-01
- 질문: ① '마스크는 맞는데 클래스가 틀린' 오류가 DELIVER 만의 현상인가, MUSES 에도 있는가 ② 예측 객체의 클래스만 정답으로 바꾸면(마스크 고정) mIoU 가 각 벤치에서 얼마나 오르는가(클래스 보완 방법의 여유 상한) ③ 포스터 vs 신호등 같은 혼동쌍이 저해상 의미 토큰(fused, stride-16)으로는 안 갈리고 고해상 경로 특징의 객체 풀링으로는 갈리는가
- 가설: H-X1: 객체 클래스 오라클 여유는 DELIVER A2 에서 ≥ +1.5, MUSES 1536 에서도 ≥ +1.0 (둘 다면 방법론 정당). H-X2: 혼동쌍(DELIVER TrafficLight↔Other/TrafficSign, MUSES traffic light↔traffic sign/pole)은 fused 풀링 선형 프로브 AUROC < 0.8, 고해상 풀링 AUROC ≥ 0.9
- 바꾼 변수(change): 학습 없음. (a) 객체 클래스 오라클: 예측 맵의 연결 성분마다 GT 다수 클래스로 라벨 교체 후 mIoU 재계산(DELIVER test A2 s821·s902 / Q2 s821 덤프, MUSES val E1M 3시드 1024·1536 — 예측 덤프 없으면 공식 val 평가기로 1회 재추론해 pred 저장) + 클래스별 Δ 와 혼동쌍 기여 (b) 선형 프로브: A2 val 에서 GT 객체별로 fused 토큰 평균과 고해상 경로 s4 특징의 마스크 내 평균을 추출(모델 eval 스태시 훅, 학습 영향 0), 혼동쌍 2진 로지스틱 회귀 5-fold AUROC (c) MUSES 4모달 A2(DRN-261019-02) 완주 후 (a)(b) 재측정
- 기준선: DELIVER: A2 vs Q2 같은 시드. MUSES: 1024 vs 1536 같은 ckpt
- 성공 기준: 표 2개(벤치×모델 오라클 Δ·클래스별, 혼동쌍×특징 AUROC) + [사실] 등급 문장: 'DELIVER·MUSES 모두 객체 클래스 여유 ≥ +1.0' 여부
- 반증 조건: MUSES 오라클 여유 < +0.5 이면 클래스 보완 방법은 DELIVER 한정 효과 → 논문 주 방법으로 올리지 않음. 고해상 풀링 AUROC 가 fused 와 차이 < 0.05 면 '고해상 질감이 클래스를 가른다' 가설 기각
- configs: -

## 정정
