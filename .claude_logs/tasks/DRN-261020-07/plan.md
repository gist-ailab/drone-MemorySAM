착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 백본 대조 1 · '우리 방법을 뺀 DINOv3': 동결 DINOv3-L + 센서별 LoRA + 단순 융합(센서 토큰 평균, 교차주의 없음) + 같은 헤드, clean 40ep 2시드(s821·s902)

- id: DRN-261020-07
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: DINOv3-L 을 꽂고 단순 융합만 해도 DGFusion(56.71) 급이 나오는가(리뷰어 '백본 덕' 반박)
- 가설: E1(교차주의 융합, 56.24) 보다 낮고 DGFusion 에 못 미친다. 과의존(depth 제거)은 E1·DGFusion 과 같은 −24~−39 패턴
- 바꾼 변수(change): FUSION 을 MeanFusionTrunk(P43 식 모달 평균)로 교체, 그 외 E1 스크린 config 동일(4탭·LoRA r8·헤드·clean 학습)
- 기준선: E1 같은 시드 스크린(s821·s902), legal v2 test@1024
- 성공 기준: 사다리 표 완성: RGB 단독 < 단순 융합 < E1 < Q2 < A2 가 test 24클래스·depth 제거 Δ 로 보고됨. 단순 융합이 E1 보다 ≥ +0.3 높으면 교차주의 융합의 기여 주장을 삭제
- 반증 조건: 단순 융합이 DGFusion 을 넘으면 '구조 기여' 문장을 빼고 '백본+학습' 으로 서술을 바꾼다(결과를 숨기지 않음)
- configs: -

## 정정
