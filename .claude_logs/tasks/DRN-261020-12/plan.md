착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L4 · A2 − 질의 헤드+arbiter(→ dense conv 헤드만)

- id: DRN-261020-12
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: Mask2Former-lite 질의 헤드와 arbiter 결합이 A2 에 얼마나 기여하는가(고해상 특징을 쓰는 헤드가 다른 구성요소임을 분리)
- 가설: clean −0.5~−1.0, 특히 큰 영역 클래스
- 바꾼 변수(change): MODEL.M2F.ENABLE false · P39.ARBITER false(dense 경로만), 그 외 A2 동일
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성
- 반증 조건: 해당 없음
- configs: -

## 정정
