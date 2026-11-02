착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 백본 맞춤 대조 2단계 · MCubeS C2D2-B(DINOv3-B 교사 + 고해상 경로 + 두 패스·증류) 2시드 — FuseAdapt(DINOv2-B 57.62) 직접 대조 행

- id: DRN-261102-02
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: B 급 백본에서 C2D2 전체 레시피가 FuseAdapt 를 넘는가(백본 크기를 맞춘 직접 비교)
- 가설: C2D2-B ≥ 57.62(FuseAdapt) 그리고 E1Mc-B 대비 Δ ≥ +0.5(레시피 이득이 백본 크기와 무관). 미달이면 '우리 우위는 백본 크기에 의존' 으로 솔직히 서술
- 바꾼 변수(change): 1단계 교사 ckpt 를 TEACHER_CKPT/TEACHER_CFG 로, 나머지 = …_A2c2d2 config 에 BACKBONE 3키만 B 로. 24GB 면 768 크롭
- 기준선: E1Mc-B 같은 시드 · FuseAdapt 57.62(arXiv 2610.21779 Tab.1, 같은 test 102 관행 — 선택 규칙 동일 여부 원문 확인 필요)
- 성공 기준: 표 행: Backbone·학습 params 열과 함께 Table 4 에 '백본 맞춤' 표시. 채택 = 2시드 평균 ≥ 57.62 그리고 Δ vs E1Mc-B ≥ +0.5
- 반증 조건: 2시드 평균 < 57.0 → 백본 의존 서술로 전환
- configs: -

## 정정
