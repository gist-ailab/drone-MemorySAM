착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MCubeS 로더 학습 전용 크롭 옵션(DATASET.TRAIN_CROP, 기본 1024) + 평가 하네스 재동결 — A2 를 24GB 카드에서 돌리기 위함

- id: DRN-261027-03
- kind: build
- 문제(PRB): PRB-DRN-261016-01
- 질문: MCubeS A2(고해상 경로 + 두 패스)를 24GB GPU 에서 학습하려면 학습 크롭 768 이 필요한데, 평가 수치를 바꾸지 않고 로더에 옵션을 넣을 수 있는가
- 가설: 학습 변환(RandomScaleCrop)만 인자화하면 평가 경로(FixScaleCrop 1024)는 불변이고 기본값(1024)에서 기존과 동일
- 바꾼 변수(change): semseg/datasets/mcubes.py __init__ 에 train_crop/train_base 인자(기본 1024), train_reliadino.py ds_kwargs 에 DATASET.TRAIN_CROP 전달, 스모크 tools/smoke_mcubes_train_crop.py. 이후 tools/eval_harness_guard.py --freeze 재동결 + 대표 ckpt(P39.1 MCubeS) 재채점으로 수치 불변 확인
- 기준선: develop HEAD mcubes.py(해시 c2b2232e…)
- 성공 기준: 완료 조건: 스모크 통과(기본값 동일·768 동작·평가 함수 소스 불변) + --freeze 커밋(사유 명시) + 대표 ckpt 재채점 Δ = 0.00 (analysis 요청)
- 반증 조건: 재채점이 0.01 이상 달라지면 패치 철회
- configs: -

## 정정
