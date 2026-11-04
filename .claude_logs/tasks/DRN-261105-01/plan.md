착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# DELIVER 백본 크기 대조 · DINOv3-B 로 같은 방법(E1-B 교사 clean 40ep → A2-B 두 패스+증류+고해상) s821 — '이득은 큰 백본 덕'이라는 반박 방어

- id: DRN-261105-01
- kind: experiment
- 문제(PRB): -
- 질문: DINOv3-L 을 DINOv3-B 로 바꿔도 두 패스 학습 + 고해상 경로의 이득(A2 − E1)이 유지되는가, 그리고 B 로도 DGFusion 56.7 수준에 닿는가
- 가설: A2-B − E1-B(같은 시드 s821, 24클래스 test legal v2) ≥ +1.0 이고 EMM15 Δ ≥ +5(DINOv3-L 에서 E1 55.94 → A2 58.61, EMM 40.01 → 49.12 였음). A2-B 절대값은 DGFusion 56.7 ±1 안으로 예상(미확정)
- 바꾼 변수(change): 백본 크기 하나(ViT-L/16 → ViT-B/16, 탭 [6,12,18,24] → [3,6,9,12]). 레시피·LoRA r8·고해상 경로·두 패스·증류 설정 모두 A2 와 동일. 교사도 같은 B 로 새로 학습(E1-B)
- 기준선: DINOv3-L 짝: E1 s821 55.94 · A2 s821 58.61 (EMM 40.01 → 49.12) · DGFusion 56.7(원문) · MCubeS 쪽 B 대조(DRN-261102-01/02) 와 서술 통일
- 성공 기준: 두 런 완주 + legal v2 test(24클래스) + EMM15 → 원고 §5 '백본 사다리' 표에 B 행 2개 추가(E1-B·A2-B). A2-B − E1-B ≥ +1.0 이면 '방법 이득은 백본 크기에 독립' 문장, < +0.5 면 '큰 백본과 결합할 때만' 으로 쓴다. A2-B 절대값은 결과 그대로 게재
- 반증 조건: A2-B − E1-B < +0.5 이거나 EMM Δ < +2 → 방법 이득이 백본 크기에 의존한다고 적고 한계에 넣는다
- configs: configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_E1B.yaml, configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2B.yaml

## 정정
