착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L9 · 우리 모델 + 단일 패스 augmentation(clean 패스·교사 없음) 2시드 — C2D2 가 '그냥 augmentation' 과 다른가

- id: DRN-261112-05
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: L7 기각(물리 열화 = 결측만과 같음)·L6 결론 없음(증류 독립 기여 미확인)으로 C2D2 의 강건성은 '열화 입력을 보는 것'에서 온다. 그렇다면 같은 열화를 배치에 섞는 단일 패스 augmentation(같은 샘플의 clean 짝도, 교사도 없음)만으로 같은 결과가 나오는가. tab:retrained 는 같은 단일 패스 augmentation 을 DGFusion·CAFuser 에 넣었지만 백본이 달라(Swin-T 대 DINOv3-L) 학습 방식과 구조를 가르지 못한다 — 이 런이 우리 모델에서 직접 가른다
- 가설: 두 패스(같은 샘플의 원본·손상본 짝)는 단일 패스 augmentation 보다 clean 을 덜 잃는다(augmentation 은 clean 을 깎는다: DGFusion 재학습 −0.9) — 즉 C2D2 의 차별점은 '정확도를 지키며 강건성 획득'
- 바꾼 변수(change): C2D2 의 두 패스를 단일 패스로: 매 배치 입력을 열화기(같은 TRAIN_OPS·확률 0.5·결측 0.2·커리큘럼)로 손상한 뒤 한 번만 forward, 모든 손실(OHEM·M2F·보조 CE·router·prototype·VICReg)을 손상 입력에서 계산, clean forward·교사 증류 없음. 🔴 기존 CLEAN_PASS false 는 clean forward 의 보조 손실이 남아(train_reliadino.py:1059) 순수 단일 패스가 아니므로 새 모드(예 TRAIN.QAF.SINGLE_PASS) 구현 필요 — 코드 위임 + 검수 + develop 병합 후 기동
- 기준선: A2 같은 시드(s821·s902, C2D2 = 두 패스+증류) 와 C2D2 없음(clean 학습), legal v2 test@1024 + EMM15 test·val + NM — L5·L6·L7·L8 과 같은 평가 묶음
- 성공 기준: 채택(C2D2 ≠ augmentation) = 2시드 짝 Δ(A2 − L9) 평균이 test clean ≥ +0.5 또는 test EMM15 ≥ +1.0 이고 그 지표에서 두 시드 부호 일치
- 반증 조건: clean·EMM 모두 |Δ| < 0.3 이면 C2D2 는 우리 모델에서 단일 패스 augmentation 과 같음 → 원고에서 두 패스 구조를 차별점으로 주장하지 않고 '강한 센서 손상 학습'으로 서술, 제목·기여 문장 재검토
- configs: configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L9singlepass.yaml, configs/jarvis-deliver_rgbdel_P46_c3only_seed20260902_screen40_A2L9singlepass.yaml

## 정정
