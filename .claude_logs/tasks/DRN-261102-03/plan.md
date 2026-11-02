착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# DINOv2-B(patch 14) 호환성 CPU 스모크(학습 0) — FuseAdapt 와 같은 백본으로 바꿀 때 4탭·고해상 경로·FPN(stride 16 가정)·1024 입력이 깨지는지

- id: DRN-261102-03
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: BACKBONE_TIMM 을 vit_base_patch14_reg4_dinov2 로 두면 1024 입력(1024/14 비정수)·FPN stride 16 가정·고해상 어댑터 참조점이 동작하는가
- 가설: 입력을 14 의 배수(예: 1022 또는 896)로 맞춰야 하고 FPN 레벨 크기 정렬이 깨질 수 있음 → 코드 수정 범위를 먼저 확인
- 바꾼 변수(change): CPU 에서 build_reliadino(B, patch14) + forward(1,3,1024,1024)/(896,896) + 교사 cfg 빌드, 실패 지점 기록
- 기준선: DINOv3-B(patch16) 빌드
- 성공 기준: 보고: 동작 여부·필요 수정 목록·추정 비용. 통과 시 DINOv2-B 쌍(교사+C2D2) 등록 여부를 method 가 결정
- 반증 조건: 해당 없음
- configs: -

## 정정
