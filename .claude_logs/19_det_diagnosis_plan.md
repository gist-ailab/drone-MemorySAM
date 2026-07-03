# P29/P30-Det 성능 진단 계획 (실험 기반 분석 파이프라인)

> 작성: 2026-07-02 (브레인스토밍 세션 결과). 목표 mAP50 **0.85** vs 현재 최고 **0.4455**(P29-Det bundle ep9), P30-Det **0.2490**(ep34).
> 원칙: **뇌피셜 금지 — 모든 가설은 아래 실험으로 기각/확정한다.** 각 실험 완료 시 이 문서의 상태 표를 갱신할 것.
> 코드 위치 주의: det 코드는 main이 아니라 **worktree** — `.claude/worktrees/p30-det/`(최신) / `.claude/worktrees/p29-det/`. repo 루트 `objdet/`·`configs/det/det_P9_base.yaml`은 **stale**(구버전 stride [16,32,64] — 신뢰 금지).

---

## 0. 현재 사실관계 스냅샷 (2026-07-02)

| 항목 | P29-Det (bundle) | P30-Det |
|------|------|------|
| Backbone | SAM2 Hiera-B+ + SoftMoE-LoRA(r4) + RBMA mem-attn (`LoRA_Sam_P29_Det`) | + SDC (`LoRA_Sam_P30_Det`) |
| Fusion | **mean** (`_fuse_modalities`) | **ReliabilityAnchoredRouter** per-FPN-level (anchor λ=1.0) |
| Head (primary) | FCOS 3-scale P3/P4/P5 stride [4,8,16] + ATSS | **DETR식 query decoder 100q — P5(stride16, 64×64) 단일 스케일만** (`det_model.py:400`), FCOS는 aux |
| Recipe | 50ep, LR 2e-4, letterbox+aug, AMP | 40ep, WD 5e-4, **AMP off**(NaN 때문), batch 4 |
| 최고 성적 | **mAP50 0.4455 @ep9** → 이후 하락(~0.36) | **mAP50 0.2490 @ep34** (query head 수렴 극도로 느림: ep4 0.036) |

- 데이터: poongsan_v2 (실내 드론 RGB+LiDAR+Thermal, 10클래스, native 640×480→1024 letterbox). `REQUIRE_ALL_MODALITIES: true`로 **13,712장 중 train 5,862장만 사용**. split v2 = capture holdout (test 1,772).
- RBMA는 s16 `mem` feature에만 주입됨 (fpn0 s4 / fpn1 s8은 raw encoder feature) — `extract_det_features()` (`sam_lora_image_encoder_seg.py:7647`).
- eval: pycocotools COCOeval. **best ckpt 선택 기준이 mAP50이 아니라 COCO mAP** (알려진 quirk). 로그가 "eval이 query/FCOS 어느 head를 보고하는지 모호" 플래그 (`15_training_monitor_log.md:207`).
- **외부 baseline(YOLO/RT-DETR/RF-DETR) 측정치 전무.** 목표 0.85는 측정 근거 없는 희망값.
- 과거 이슈: v1 데이터 버그(빈 프레임 52% → AP≈0, `17_p29det_data_fix.md`), stride 4× 정렬 버그(수정됨), P30 AMP NaN(fp32로 해소), 양 run 공통 ep9 피크 후 하락(LR/스케줄 의심, 미해결).

## 0.5 핵심 진단 (실험으로 확정할 가설)

1. **P30-Det 하락(0.249)은 Router(모듈)의 유죄 증거가 아직 아님** — head 교체(FCOS→단일스케일 query decoder)와 fusion 교체(mean→Router)가 **동시에** 들어간 confound. 단일 s16 query decoder는 소물체에 구조적으로 불리(SAM3-RBMA seg ~24 plateau와 동일 병리)하고, DETR류 head에 40ep는 태부족.
2. "YOLO RGB-only가 더 잘 나올 것" 직감은 아마 사실 — 우리 스택은 from-scratch head + 데이터 절반 폐기 + 미해결 스케줄 문제라는 핸디캡을 안고 있음. **측정으로 기준점을 세워야 함.**
3. novelty는 reliability **fusion**이지 det head가 아님 → head는 빌려오고(fusion만 우리 것), 주장은 absolute mAP가 아니라 **동일 head에서 RGB-only 대비 악조건 delta**로 가는 것이 정공법 (research_vault MM-SAM-adapter 스쿠프 대응과도 정합).

---

## 1. Phase 0 — 학습 없이 기존 ckpt 분석 (GPU 수분~수시간) 🔴 최우선

| # | 실험 | 방법 | 기각되는 가설 | 상태 |
|---|------|------|------|------|
| E0.1 | **P30 ckpt의 FCOS aux head 단독 eval** | P30 ep34 ckpt 로드, query 대신 aux FCOS 출력으로 COCOeval | aux≈0.44면 Router 무죄·query head 유죄 확정. **공짜인데 가장 결정적** | ✅ **판정 완료(2026-07-02, ep39): aux AP50 0.431 ≈ P29 0.446 → Router 무죄·query head 유죄 확정.** AP_small 0.111(query 0.014→복원), Lighting 0.033→0.431, EmExit 0.054→0.443. hinton `out_p30_ep39_fcosaux/`, 패치=`det_model.py` EVAL_FCOS_AUX=1(백업 .bak_pre_e01). 상세=`/mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/E01_AUX_HEAD_VERDICT.md` |
| E0.2 | COCOeval 전체 분해 + TIDE | AP_s/m/l, AR@100 로깅 (P29 ep9 vs P30 ep34) + TIDE(cls/loc/bkg/miss/dup) | recall 부족(feature/scale) vs precision 부족(head/혼동) 분리 | ☐ |
| E0.3 | per-class AP + confusion matrix + FP/FN top-50 시각화 | score>0.3, 클래스쌍별 오분류 카운트 | "혼동" 실체 (Doors↔Windows↔EmExit?) | ☐ |
| E0.4 | GT 박스 크기 히스토그램 | native 640×480 기준 px, <16px/<32px 비율 | AP_small 물리 상한 → 목표 0.85 현실성 | ☐ |
| E0.5 | FCOS proposal recall@1000 per level | NMS 전 raw 후보의 GT 커버리지 | 후보 미생성(feature 문제) vs 분류 실패 | ☐ |

## 2. Phase 1 — 외부 baseline 브래킷 (GPU 반나절)

| # | 실험 | 방법 | 상태 |
|---|------|------|------|
| E1.1 | **YOLO11-m RGB-only fine-tune** | 같은 v2 split, 100ep, native 해상도 → **모든 논의의 기준점** | ✅ **완료(2026-07-03): test mAP50 0.821 / mAP50-95 0.526** (best@ep12, P 0.831 / R 0.831) → 분기 기준 0.7 초과, **우리 스택 문제 확정** |
| E1.2 | YOLO thermal-only / lidar-only | 각 모달리티 단독 정보량 측정 → fusion 이득 상한 추정 | ☐ |

**분기 규칙**: YOLO-RGB mAP50 ≥0.7 → 우리 스택 문제 확정, Phase 2 진행. ~0.5 → 데이터/난이도 문제 → 라벨 재감사 + 목표 재조정 후 진행.

## 3. Phase 2 — 우리 스택 통제 ablation (변인 1개씩)

| # | 실험 | 분리되는 변인 | 상태 |
|---|------|------|------|
| E2.1 | 우리 파이프라인 `MODALS=['img']` vs 3-modal | fusion 자체의 득실 | ☐ |
| E2.2 | **같은 FCOS head**에서 mean vs ReliabilityAnchoredRouter | P30의 ①(router)·②(head) confound 해체 | ☐ |
| E2.3 | frozen SAM2+FPN+FCOS vs frozen DINOv2+동일 head | SAM2 feature의 det 적합성 | ☐ |
| E2.4 | RBMA λ=0 vs on | RBMA의 det 기여 (s16 mem에만 주입되므로 원래 제한적일 수 있음) | ☐ |

## 4. Phase 3 — Breakthrough 후보 (Phase 0–2 결과로 선택)

1. **B1 (유력)**: COCO-pretrained head 이식 — RT-DETR/Deformable-DETR(multi-scale deformable attn) 또는 YOLO head를 RBMA-fused multimodal feature 위에. det 확장을 "잘하는" 대신 **빌린다**. 논문 주장 = 동일 head, RGB-only 대비 악조건(야간=thermal) delta.
2. **B2**: `REQUIRE_ALL_MODALITIES` 폐지 → missing-modality dropout → train 5,862→13,712장 복원 (robustness 스토리 정합).
3. **B3**: recipe 수리 — best-ckpt 기준 mAP50로 변경, EMA, ep9 피크 후 하락 원인(LR) 수정, 소물체용 mosaic/copy-paste aug.
4. **B4**: query head 유지 시 전제조건 — multi-scale deformable cross-attn + DN-denoising + 150ep+. 단일 s16 유지 불가.

## 5. 권장 실행 순서

**즉시: E0.1 + E0.2** (공짜, P30 모듈 유무죄 판가름) → **다음: E1.1 YOLO 기준점** → 결과 따라 Phase 2 범위 결정 → Phase 3 선택.

## 6. 실험 결과 기록 (실험 완료 시 여기에 append)

| 날짜 | 실험 | 결과 | 결론/기각된 가설 |
|------|------|------|------|
| 2026-07-03 | E1.1 YOLO11-m RGB-only (hinton GPU1, 100ep, batch16, imgsz640, seed0, 2.6h) | **test mAP50 0.821 / mAP50-95 0.526 / P 0.831 / R 0.831** (best@ep12). per-class mAP50: Allies 0.941, LandingMarkers 0.960, Casualties 0.941, Windows 0.881, Enemies 0.826, Lighting 0.811, FireExt 0.803, EmExits 0.802, Obstacles 0.747, **Doors 0.499(최악)**. ep12 피크 후 val_cls loss 상승(과적합) — P29 ep9 피크와 동일 병리. 산출물: `objdet/yolo11m-rgb/runs/` (로컬 동기화됨, best.pt 포함) | **"데이터/난이도 문제" 기각 — 우리 스택 문제 확정** (RGB 단독 0.821 vs 우리 3-modal 0.4455). 목표 0.85는 RGB-only 기성 모델로도 근접 가능한 수치 → 데이터 무죄. Doors가 유일한 난제 클래스. Phase 2 (E2.x) 진행 근거 확보, B1(head 이식) 방향 지지 |
| 2026-07-03 | E1.1b YOLO11-m RGB-only **label-v3**(=어노테이션 v20260702_2303, train 6,766장 +15.4%; test는 E1.1과 동일 1,772장/5,078box) 50ep | **test mAP50 0.864 / mAP50-95 0.581 / P 0.877 / R 0.883** (1.5h). 산출물: `objdet/yolo11m-rgb/runs/yolo11m_rgb_labelv3_50ep/`, 전체 test GT\|Pred 시각화 `vis_test_labelv3/` | **신규 라벨 +904장으로 mAP50 +0.043 (0.821→0.864) — 목표 0.85 돌파.** RGB-only로도 목표 달성 가능 확정. 데이터 추가 라벨링의 한계효용 확인. YOLOv5m 교차검증 완료(아래 행) |
| 2026-07-03 | E1.1c YOLOv5m(u) RGB-only label-v3 50ep (교차검증) | **test mAP50 0.866 / mAP50-95 0.577 / P 0.860 / R 0.873** (1.3h). `objdet/yolo11m-rgb/runs/yolov5mu_rgb_labelv3_50ep/` | **YOLO11m(0.864)과 동률 — 0.86 수준은 모델 아키텍처가 아니라 라벨/데이터가 결정.** 목표 0.85는 세대 무관 기성 RGB 모델로 재현 가능 확정 |
