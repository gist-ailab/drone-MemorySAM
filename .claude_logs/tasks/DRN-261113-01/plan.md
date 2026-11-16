착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation H1 · 고해상 경로 입력을 RGB 만으로(네 센서 → RGB) 2시드 — DINOv3 어댑터의 멀티센서 확장이 얇은 구조·악조건에 기여하는가

- id: DRN-261113-01
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 고해상 경로는 DINOv3 공식 DINOv3_Adapter 와 구조가 같고(2026-11-13 공식 코드 확인), 우리만의 차이는 네 센서를 공유 stem + 센서별 1×1 투영 합산·센서별 탭 투영 합산으로 받는 멀티센서 확장이다. 이 확장이 측정된 기여인지(RGB 만 넣은 경로 대비) 확인해 기여 불릿 2 'adapter extended to several sensors' 의 근거로 삼는다. 모델(A2)은 바꾸지 않는다(ablation 행)
- 가설: 네 센서 입력 경로가 RGB 만 경로보다 얇은 4클래스(Pole·TrafficSign·Pedestrian·Static) 평균과 야간·저노출 clean 에서 높다(depth·LiDAR 의 기하 경계가 고해상 경로에 들어가므로)
- 바꾼 변수(change): 고해상 경로(SPM stem 과 extractor 가 읽는 탭 합산)에 RGB 센서만 넣는다. 백본·융합·헤드·C2D2 는 A2 그대로. 🔴 코드 소폭 필요: hires_adapter 는 전 센서 x_list·탭을 받으므로 센서 부분집합 옵션(예 MODEL.HIRES_ADAPTER.MODALS: ['img']) 추가 — off 계약(기본값 = 전 센서 = A2 byte 동일) 스모크·검수 후 develop 병합
- 기준선: A2 같은 시드(s821 58.61 · s902 57.09), legal v2 test@1024, 얇은 4클래스 IoU, 조건별 clean(Table 6 축), EMM15 test
- 성공 기준: 채택(멀티센서 확장 기여) = 2시드 짝 Δ(A2 − H1) 평균이 얇은 4클래스 평균 ≥ +1.0 또는 test clean ≥ +0.5 이고 그 지표에서 두 시드 부호 일치
- 반증 조건: 모든 지표 |Δ| < 0.3 이면 멀티센서 확장은 기여 없음 → 기여 불릿 2 에서 'extended to several sensors' 를 근거 있는 기여로 쓰지 않고 구현 사실로만
- configs: configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hiresH1rgbonly.yaml, configs/jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2hiresH1rgbonly.yaml

## 정정
