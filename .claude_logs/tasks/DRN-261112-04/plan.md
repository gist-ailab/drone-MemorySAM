착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 2×2 빈칸 · DELIVER 고해상 경로 있음 + C2D2 없음(clean 학습만) 3시드 — 고해상 경로의 단독 정확도 이득과 C2D2 의 고해상 모델 위 효과 분리

- id: DRN-261112-04
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 원고 tab:abl_robust 의 2×2(C2D2 × 고해상 경로) 중 DELIVER '고해상 있음·C2D2 없음' 칸이 비어 있다(lab-plan check 유사 항목 없음). 이 칸이 있어야 (1) 고해상 경로 +1.15(56.59→57.74, C2D2 위에서 측정)가 C2D2 없이도 나오는지 (2) 고해상 모델에서 C2D2 가 clean 을 올리는지·내리는지(MCubeS 는 59.89→59.78) (3) C2D2 의 EMM 이득이 고해상 모델에서도 같은지를 가른다. 쌓아 올리기 표의 '+ 고해상 경로' 단계도 이 런이다
- 가설: 고해상 경로는 C2D2 없이도 clean 을 올리고(C2D2 없음 56.40 대비 ≥ +0.8), C2D2 는 고해상 모델에서 clean 을 시드 폭 안으로 유지하면서 EMM 을 크게 올린다(≥ +5)
- 바꾼 변수(change): TRAIN.QAF.ENABLE true→false(두 패스·열화·교사 증류 끔), MODEL.HIRES_ADAPTER.ENABLE true 유지, 나머지 A2 config 동일 — 즉 E1 구조 + 고해상 경로를 clean 으로만 학습
- 기준선: 같은 시드의 C2D2 없음·고해상 없음(E1 40ep 스크린, 2시드 56.40)과 A2(3시드 57.74), legal v2 test@1024 + EMM15(val·test) + RMM — tab:abl_robust 와 같은 평가 묶음
- 성공 기준: 측정 칸 채움(표 행). 해석 기준(사전): 고해상 단독 이득 = (이 런 − C2D2 없음·고해상 없음) clean ≥ +0.8 이고 세 시드 부호 일치면 '고해상 경로가 정확도를 올린다'를 독립 근거로 주장. C2D2 의 고해상 위 효과 = A2 − 이 런, clean |Δ| < 0.5 이고 EMM ≥ +5 면 '정확도 유지·강건성 획득' 주장
- 반증 조건: 고해상 단독 clean 이득 < +0.3 이면 '+1.15 는 C2D2 와 결합할 때만'으로 서술 수정. A2 − 이 런 clean ≤ −0.5(세 시드 부호 일치)면 'C2D2 가 고해상 모델의 clean 을 깎는다'를 한계로 명시
- configs: configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hiresNoC2D2.yaml, configs/jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2hiresNoC2D2.yaml, configs/jarvis-deliver_rgbdel_P46_c3only_seed20260903_screen40_A2hiresNoC2D2.yaml

## 정정
