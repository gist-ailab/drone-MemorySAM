# D1 ViT-S+ 공인인증 핸드오프 (다른 세션용) — 2026-07-25

> **목적**: 국가 R&D 공인인증 평가를 준비하는 **다른 세션**이 재조사 없이 바로 착수하도록, 확정된 인증 모델(**D1 ViT-S+**)의 **웨이트·코드·정보**를 한곳에 모은 자립 문서. (모델 정체는 user 확정: "reliadino+rfdetrhead가 맞아".)
> 대상 환경 = **drone-demo 컴퓨터, `26-drone-certificate` 브랜치**. 상세 스윕 근거는 [det-cert-D1-realtime.md](det-cert-D1-realtime.md).

## 1. 모델 정체 (인증 대상 = D1 ViT-S+)
- **구조**: ReliaDINO(frozen **DINOv3 ViT-S+/16** 백본 + per-modal LoRA + ReliabilityGatedFusion + SimpleFPN) **+ RF-DETR NMS-free head**. 순수 standalone RF-DETR 아님.
- **클래스**: `ReliaDINORFDETRDetector` (`objdet/models/det_model.py`), `tools/_det_common.py:build_detector`가 config만으로 생성.
- **모달리티**: 3-modal `['img','lidar','thermal']` (RGB + LiDAR depth + Thermal). **poongsan_v2 test도 멀티모달**(각 capture에 `rgb`+`depth_map_lidar`(+egofill)+`thermal_aligned`+`event_aligned` 물리 존재) — RGB-only 아님. eval은 `REQUIRE_ALL_MODALITIES: true`로 3-modal 실제 로드.
  - 🔴 **인증 스토리 필수 주의**: modality ablation(P29 det 실측)에서 **RGB-only ≥ 3-modal on mAP50**(RGB 0.7964 vs 3-modal 0.7895). 즉 lidar/thermal이 존재·사용되나 **목표지표 mAP50엔 이득 없음**(strict-IoU COCO mAP만 소폭↑). 멀티모달 모델·멀티모달 test가 맞되 **mAP50은 RGB가 사실상 캐리** → 인증 리포트 시 3-modal로 제출하되 이 ablation을 인지(또는 RGB-only가 동등하니 배포 단순화 옵션). 근거 = memory `det-final-ann-modality-ablation`.
- **클래스 수**: 10 (poongsan_v2 카테고리; config `N_CLASSES: null` → 데이터셋에서 유도).

## 2. 웨이트 (2곳 백업, md5 검증됨)
- `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth` (354M, **epoch 11**)
- `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth` (미러, 2026-07-25 백업)
- **md5** = `b78f614cba1375bb54dfeacd5e58cef3` (두 곳 일치 확인). 포맷 = `{'model_state_dict':..., 'metrics':..., 'epoch':11}`.

## 3. 코드 (전부 `develop` 브랜치에 추적됨 — 서버는 `git fetch local && checkout develop`)
- **Config**: `configs/det/det_D1_vitsp_jarvis.yaml` — `DET_MODEL: ReliaDINORFDETRDetector`, `BACKBONE_TIMM: vit_small_plus_patch16_dinov3`, `MODALS: [img,lidar,thermal]`, `NUM_QUERIES:300 / GROUP_DETR:13 / DEC_LAYERS:4`, `IMG_SIZE:[768,768]`. (drone-demo용은 ROOT/ANNOTATION 경로만 교체.)
- **모델 코드**: `objdet/models/det_model.py`(ReliaDINORFDETRDetector), `semseg/models/rfdetr_head/`(RF-DETR head 벤더링, `_vendor/`).
- **재사용 도구**(모델 무관, `--cfg`/`--ckpt`만 교체):
  - `tools/_det_common.py` — build_detector·load_det_checkpoint·build_loader·run_inference·**eval_overall/eval_per_class**(mAP/mAP50/mAP75 + per-class), split_ann_by_clip(야간/주간).
  - `tools/det_eval_breakdown.py` — per-condition(주/야) mAP 브레이크다운 러너.
  - `tools/det_fps_bench.py` — FPS 측정. `tools/measure_det_vram.py` — BS1 추론 VRAM.
  - `tools/det_viz_samples.py`(샘플 PNG 오버레이) · `tools/det_viz_video.py`(GT-vs-예측 side-by-side 영상).
- **제출 코드 패키지**: `/ailab_mat2/.../submission/code/det_cert/` = `run_cert_eval.sh` + `tools/` + `configs/` + `README.md`(재현 절차).

## 4. 성능 (인증 리포트 수치 — poongsan_v2 final test 3239장, predicted-scope)
| split | images | mAP | **mAP50** | mAP75 |
|---|---|---|---|---|
| all | 3239 | 0.6263 | **0.9166** | 0.7033 |
| night | 1768 | 0.6066 | **0.8765** | 0.6789 |
| normal | 1471 | 0.6495 | **0.9418** | 0.7199 |
- 주야 격차 mAP50 −0.065(야간에도 목표 0.85 상회). 학습시 best AP50 0.9205@ep11(annotation-scope, 참고).
- **FPS**(3090 실측 BS1 768²) ViT-S+ **7.38**, RTX 5080 추정 ~10–12fps(≥5fps 충족). **VRAM** BS1 추론 **0.76GB**(5080 16GB의 4.8%).
- 야간 취약 클래스: Obstacles(−0.34)·Windows·Fire Extinguishers.

## 5. 평가 데이터셋
- **poongsan final test split**. 원본 = jarvis `/SSDd/jemo_maeng/dset/poongsan_v2/`, ANNOTATION = `_final_ann/instances_test_common.json`. drone-demo로 옮길 경우 이미지+ann 동반 이송(3-modal: img/lidar/thermal).
- **eval-scope = predicted**(모달리티 결손으로 드롭된 프레임의 GT를 AP 분모에서 제외; `det_eval_breakdown.py --eval-scope predicted`). best_checkpoint = mAP(.50:.95) 기준.

## 6. 인증 eval 스크립트 요구사항 (user 지정 — 다른 세션이 만들 것)
엔터 한 번에 도는 통합 스크립트:
- **시작(데이터 로딩)**: 클래스 목록 + 활용 모달리티(img/lidar/thermal) 프린트.
- **실행**: 이미지별 시각화 + 인퍼런스 로그 + GT 비교 + 수치 기록.
- **끝**: **mAP50 리포트** + **FPS 리포트**.
- 재료: `_det_common`(eval_overall→mAP50) + `det_viz_samples`/`det_viz_video`(시각화·GT비교) 조합. 신규 통합 스크립트로 묶으면 됨.
- **YOLO 학습/평가 경로도 남겨둘 것**(user 지정 — 별도 baseline).

## 7. drone-demo 환경 주의
- 🔴 **drone-demo GPU 미인식**: `nvidia-smi` "Driver/library version mismatch". 인퍼런스 전 해결 필요 — venv 번들 cuDNN을 `LD_LIBRARY_PATH`에 프리픽스([[hpca100-cudnn-fix]] 유사) 또는 드라이버 리부트 시도.
- drone-demo는 원격·공유 머신(X11 이슈 주의). SAM2 코드 실행 시 `PYTHONPATH=<repo>/semseg/models/sam2`.

## 8. 착수 체크리스트 (다른 세션)
1. `26-drone-certificate` 브랜치 생성(develop 기준) — 아직 없음.
2. develop 최신 pull → 위 config/모델/도구 확보.
3. 웨이트 = §2 경로에서 로드(`--ckpt .../det_D1_vitsp_20260723/best_checkpoint.pth`).
4. §6 요구대로 통합 eval 스크립트 작성 + YOLO 슬롯.
5. drone-demo GPU(§7) 먼저 살리기.
