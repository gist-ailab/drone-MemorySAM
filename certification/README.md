# D1 ViT-S+ — 공인인증 detection 패키지 (`26-drone-certificate`)

인증 대상 모델 **D1 ViT-S+** = ReliaDINO(frozen DINOv3 **ViT-S+/16** + per-modal LoRA
+ ReliabilityGatedFusion + SimpleFPN) **+ RF-DETR NMS-free head**. 3-modal
(RGB + LiDAR depth + Thermal), 10 classes, 768². 웨이트 = `det_D1_vitsp_20260723`
(epoch 11, md5 `b78f614cba1375bb54dfeacd5e58cef3`).

인증 수치 (poongsan_v2 final test, 3239장, predicted-scope):
mAP50 **0.9166** (night 0.8765 / normal 0.9418) · FPS 7.38 (3090) / ~10–12 (5080) · VRAM 0.76GB.

## 1. 인증 평가 (엔터 한 번)

```bash
bash certification/run_cert.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]
# 예) bash certification/run_cert.sh \
#       /ailab_mat2/.../submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth \
#       /SSDd/jemo_maeng/dset/poongsan_v2 0
```

동작:
1. **시작(데이터 로딩)** — 클래스 10종 + 활용 모달리티(img/lidar/thermal) + 모델·ckpt(md5)·장치·이미지 수 출력.
2. **ENTER** — 여기서 대기 (`--auto`로 생략).
3. **실행** — 이미지별 추론(ms) + 검출(클래스별) + GT 비교(TP/FP/FN) 스트리밍, 오버레이 PNG(예측=클래스색, GT=녹색) 저장, `runs/cert_D1/inference_log.txt` 기록.
4. **끝** — mAP/mAP50/mAP75 (전체+주야) + per-class AP50 + **`mAP50 = … | FPS = …`** 한 줄 + VRAM. `cert_report.json` 저장.

옵션: `--limit N`(빠른 확인) · `--stride N`(N장마다) · `--show`(X11 있으면 cv2 창) · `--score-thresh`(표시 임계, 기본 0.3) · `--auto`.

## 2. 학습/평가 분리 검증 (누수 없음 증명)

인증은 train과 eval(test)이 겹치지 않아야 한다. 모델·GPU 없이 annotation만 비교:

```bash
python certification/check_split.py --cfg configs/det/det_D1_vitsp_jarvis.yaml [--data-root <mount>]
# 또는: python certification/check_split.py --train <train.json> --test <test.json>
```

두 수준을 검사: **frame-level**(양쪽에 같은 이미지 파일 존재 여부 — 하드 누수) + **clip-level**(같은 캡처 세션 공유 여부 — 인접 프레임 near-duplicate 누수). PASS 조건 = frame-disjoint + 카테고리 일치. 리포트 `split_check.json` 저장, exit code 0/1.

현재 poongsan final split = **clean capture-holdout**: train 5클립(112051/113007/113534/115206/120059, 12681장) vs eval 3클립(114021/114808/115624, 3239장), **frame·clip 모두 겹침 0** → 인접프레임 누수조차 없음.

## 3. YOLO baseline 슬롯 (별도 대조군, 유지)

`objdet/yolo11m-rgb/` — RGB-only YOLO 학습/평가 경로:
- 변환: `convert_final_yolo.py` (poongsan final split → YOLO 포맷)
- 학습: `train_hinton.sh` (YOLO11m, RGB)
- 시각화/평가: `viz_test_set.py`, `viz_multimodal_det.py`
YOLO는 인증 모델과 독립된 baseline이며, 필요 시 여기서 학습·평가한다.

## 4. 환경

- repo `develop` 계열. conda `MMSS_SAM` (또는 torch+timm≥1.0+pycocotools+yaml).
- **DINOv3 백본은 timm≥1.0 필요** — 일부 서버는 `pylibs_p34`로 shadow (run_cert.sh가 자동 prepend).
- 🔴 **drone-demo GPU**: 현재 driver/library mismatch로 `torch.cuda`=False. 인퍼런스 전 드라이버 복구(재부팅) 또는 venv cuDNN `LD_LIBRARY_PATH` 프리픽스 필요.
- 데이터: poongsan_v2 (img/lidar/thermal + `_final_ann/instances_test_common.json`). 머신마다 `DATA_ROOT` 인자로 지정.

재현/배치 수치는 `tools/det_eval_breakdown.py`, FPS는 `tools/det_fps_bench.py` 참조 (동일 `_det_common` 기반, 수치 일치).
