착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation H2 · 고해상 경로 stem 공유 → 센서별 stem(MODEL.HIRES_ADAPTER.MODE per_modal) 2시드 — 공유 stem 선택의 근거

- id: DRN-261113-02
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 고해상 경로 SPM stem 을 네 센서가 공유하는 현재 설계(shared_stem)가 센서별 stem(per_modal, 코드 이미 있음)보다 낫거나 같은가. 같으면 공유 stem 이 파라미터를 아끼는 선택으로 정당화되고, 센서별이 낫다면 그 사실을 보고한다(모델은 A2 고정)
- 가설: 센서별 stem 은 파라미터만 늘리고 clean·얇은 4클래스에서 공유 stem 과 시드 폭 안(|Δ| < 0.5)
- 바꾼 변수(change): MODEL.HIRES_ADAPTER.MODE: shared_stem → per_modal (나머지 A2 config 동일, 코드 변경 없음)
- 기준선: A2 같은 시드(s821·s902), legal v2 test@1024, 얇은 4클래스 IoU, EMM15 test, 파라미터 수
- 성공 기준: 측정 칸 채움. 해석(사전): |Δ clean| < 0.5 이고 얇은 4클래스 |Δ| < 1.0 이면 '공유 stem 이 센서별과 같은 정확도를 더 적은 파라미터로' 주장. 센서별이 두 시드 모두 clean +0.5 이상이면 그 사실을 ablation 에 보고(모델 변경은 하지 않음)
- 반증 조건: 공유 stem 이 두 시드 모두 clean −0.5 이하로 나쁘면(센서별이 우월) 원고에 공유 선택의 한계로 명시
- configs: configs/bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hiresH2permodal.yaml, configs/bengio-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2hiresH2permodal.yaml

## 정정
