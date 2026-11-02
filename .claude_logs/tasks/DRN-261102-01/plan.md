착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 백본 맞춤 대조 1단계 · MCubeS 교사 E1Mc-B(동결 DINOv3-B + 센서별 LoRA r8 + 4탭, clean 200ep) 2시드 20260827·20260828 — FuseAdapt(DINOv2-B 57.62) 와 크기를 맞춘 행의 교사

- id: DRN-261102-01
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: 백본을 B 급(86M)으로 줄여도 우리 레시피가 FuseAdapt(DINOv2-B, MCubeS 57.62) 를 넘는가 — '이득이 큰 백본 덕' 반박용. 이 항목은 그 비교에 필요한 clean 교사
- 가설: E1Mc-B clean 은 E1Mc-L(58.94/58.61) 보다 낮지만(ProbeA2: B→L +5.9), C2D2 학생(2단계)이 FuseAdapt 57.62 근처 또는 이상
- 바꾼 변수(change): configs/hpca100-mcubes_…_E1Mc_seed{s}.yaml 에서 MODEL.BACKBONE ReliaDINO-ViTB16 · BACKBONE_TIMM vit_base_patch16_dinov3 · BACKBONE_FALLBACK vit_base_patch14_reg4_dinov2(미사용 기대) 만 변경(DIM 자동). 그 외 동일
- 기준선: E1Mc-L 같은 시드(교사 비교용) · FuseAdapt DINOv2-B 57.62(최종 비교는 2단계)
- 성공 기준: 완주 + legal 채점(test 102). 2단계 교사로 사용
- 반증 조건: 해당 없음(교사)
- configs: -

## 정정
