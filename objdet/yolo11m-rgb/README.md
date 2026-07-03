# yolo11m-rgb — E1.1 YOLO11-m RGB-only 기준점 (비교군)

`19_det_diagnosis_plan.md` Phase 1 E1.1: 공식 ultralytics 레포를 clone 하여
poongsan_v2 **v2 capture-holdout split** 에서 **RGB 단독**으로 YOLO11-m 을
100ep fine-tune → 우리 스택(P29/P30-Det)의 기준점.

**분기 규칙**: mAP50 ≥0.7 → 우리 스택 문제 확정. ~0.5 → 데이터/난이도 문제.
비교 대상: P29-Det bundle mAP50 0.4455 (ep9), 목표 0.85.

## 구성

- `ultralytics/` — 공식 레포 clone (학습은 hinton 에서 동일 clone 을 `pip install -e` 로 사용)
- `convert_poongsan_to_yolo.py` — COCO v2 split → YOLO 포맷 변환 (RGB 만 복사)
- `train_hinton.sh` — hinton GPU1 학습+최종 test 평가 스크립트 (100ep, imgsz 640, batch 16, seed 0)
- `splits/` — **jarvis 정본** `det_{train,test}_v2.json` + 로더가 실제 keep 한 프레임 목록
  `kept_{train,test}_v2.txt` (train 5,862 / test 1,772)

## 공정성 (우리 파이프라인과 동일한 데이터)

- split JSON 은 **jarvis(`/SSDd/jemo_maeng/dset/poongsan_v2/_det_splits/`) 정본**을 사용.
  `/ailab_mat2` 원본 어노테이션은 그 후 변경되어 (train +500장, 라벨 diff 759건)
  `build_det_splits.py` 재실행 결과와 다름 — 재생성본 쓰면 안 됨.
- `REQUIRE_ALL_MODALITIES` 필터 재현: rgb/thermal_aligned/depth_map_lidar 3개 파일이
  jarvis 사본에 모두 존재하는 프레임만 → train 5,862(18,020 box) / test 1,772(5,078 box).
  box 수까지 jarvis JSON 과 교차 검증 완료. RGB 픽셀은 /ailab_mat2 원본과 md5 동일.
- 클래스: id 정렬 후 0-base 리맵 (multimodal_det.py 와 동일), 10클래스.
- native 640×480 → imgsz 640 letterbox (E1.1 "native 해상도").

## hinton 실행 위치

- 데이터: `hinton:/SSDd/jemo_maeng/dset/poongsan_v2_yolo_rgb/` (+ `poongsan_v2_rgb.yaml`)
- 코드: `hinton:/SSDd/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM/objdet/yolo11m-rgb/`
- conda env `yolo` (ultralytics 8.4.84, torch 2.12.1+cu130) — 기존 MMSS_SAM 은 안 건드림
- 실행: tmux 세션 `jemo` 윈도우 `y11m_rgb`, 로그 `train_y11m_rgb_v2.log`,
  결과 `runs/y11m_rgb_v2_100ep/`

## hinton GPU 지정 함정 (재발 방지)

ultralytics 는 `device=N` 을 **절대 GPU 번호**로 취급해 런타임에
`CUDA_VISIBLE_DEVICES` 를 덮어쓴다 → CVD 로 GPU 를 고르면 무시됨.
반드시 `device=1` 처럼 직접 지정 + `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
(이걸 몰라서 GPU0(공유 중)에 올라가 OOM → ultralytics 가 batch 를 4까지 자동 축소했었음)

## 결과 (2026-07-03 완료)

**v2 test (1,772장 / 5,078 box), best.pt(ep12 부근, fitness 기준):**
mAP50 **0.821** / mAP50-95 0.526 / P 0.831 / R 0.831 (학습 2.6h, TITAN RTX 1장)

- 분기 규칙 0.7 초과 → **우리 스택 문제 확정** (P29-Det 3-modal 0.4455 대비 RGB 단독 +0.38)
- per-class 최악은 Doors 0.499, 나머지 9클래스는 0.75~0.96
- ep12 피크 후 과적합 (P29 ep9 피크와 동일 패턴 — v2 capture-holdout 특성)
- 산출물: `runs/y11m_rgb_v2_100ep/` (results.csv, confusion matrix, PR curves, weights/best.pt),
  `runs/y11m_rgb_v2_100ep_testeval/` — hinton 원본과 로컬 모두 보관

## label-v3 실험 (진행 중, 2026-07-03)

- 라벨 출처: **`v20260702_2303` 동결본** (`instances_v20260702_2303.json` == 빌드에 쓴 `instances.json`,
  8캡처 전부 md5 일치 확인; 매니페스트 `versions/v20260702_2303.json` 총합 15,153/43,386과 split 합계 일치)
- split: 동일 capture-holdout (test = 115206+114808). **test는 E1.1과 프레임·박스 동일**(1,772/5,078, 라벨 diff 2장뿐)
- train: 5,862→**6,766장**(+15.4%), 18,020→**19,220 boxes** — 신규 라벨 프레임 +904장 통과
- 학습: 50ep, 나머지 E1.1과 동일. run명 `yolo11m_rgb_labelv3_50ep`
- 주의: 레포 로그의 "v3"(시간순 split)와 다른 개념 — 여기서 v3는 **라벨 버전**. 시간순-v3로 잘못 돌린 run은 폐기함

## label-v3 결과 (2026-07-03)

**YOLO11-m, v2 test 동일(1,772장/5,078box): mAP50 0.864 / mAP50-95 0.581 / P 0.877 / R 0.883** (50ep, 1.5h)
- 구 레이블 대비 +0.043 (0.821→0.864) — **목표 0.85 돌파**. 원인: train +904장(신규 라벨)
- run: `runs/yolo11m_rgb_labelv3_50ep/`, 시각화: `vis_test_labelv3/` (전체 test GT|Pred)
- YOLOv5m(`yolov5mu.pt`) 동일 조건 교차검증: `runs/yolov5mu_rgb_labelv3_50ep/`
