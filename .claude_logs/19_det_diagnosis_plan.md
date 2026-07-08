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
| 2026-07-03 | E1.1d YOLOv5m(u) RGB-only **split-v3**(시간순 80/20, 라벨 v20260702_2303, train 6,663/test 1,807) 50ep | **test mAP50 0.866 / mAP50-95 0.605 / P 0.886 / R 0.854** (1.3h). `objdet/yolo11m-rgb/runs/yolov5mu_rgb_v3split_50ep/` | **mAP50은 capture-holdout(0.866)과 동률, mAP50-95만 +0.028** — 예상한 시간순 split 낙관 편향이 mAP50 수준에선 미미. 0.86~0.87이 이 데이터의 라벨/난이도 상한으로 보임 (잔여 오류는 일반화가 아니라 태스크 고유: Doors·소물체). 단 test 프레임이 서로 달라 직접 비교는 주의 |
| 2026-07-04 | **E2.5 P29-Det × v20260703_egofill** (rgb+**lidar(egofill)**+thermal; 신규 라벨 + egofill lidar 복원 → train 5,862→11,799장 2.01배; 레시피 원본 동일, bengio 5×3090, 50ep 완주; eval=원본 lidar 1,772 프레임) | ✅ **완료(2026-07-05): best AP50 0.8501 @ep9 (AP 0.513 / AP75 0.551)**. ep9 피크 후 완만 하락(ep49 0.812) — P29 관례 곡선, best ckpt 보존. best_ckpt=`bengio:.../outputs/det_egofill/det_P29_egofill_bengio/best_checkpoint.pth` | **동일 스택·동일 레시피에서 데이터만으로 0.4455→0.850 (목표 0.85 도달).** "우리 스택 문제"의 주범이 데이터(구라벨+절반 폐기)였음이 확정. 라벨 효과는 E1.1b(+0.043)로 분리 참조 → egofill 데이터 복원 기여가 지배적. 상세=21_egofill_dataset.md |
| 2026-07-05 | **E2.6 P29-Det 모달리티 ablation: lidar→event** (rgb+**event**+thermal; train/eval을 E2.5와 동일한 11,799/1,772 프레임으로 고정 → 유일 변인=모달리티; bengio 5×3090) | 🔄 **학습 중**: ep0 진행(최초 launch가 egofill 종료 직후 GPU 메모리 해제 지연으로 OOM→좀비 정리 후 clean GPU 재실행). config `det_P29_event_bengio.yaml` | (예정) event vs lidar(egofill) 검출 기여 직접 비교. event_aligned 커버리지 100%(결손 0) |

## 7. split-v3 구조 검증 (2026-07-03, 에이전트 실증 분석)

**v3 = 캡처 내 시간순 프레임 분할** (클립 holdout 아님): 8개 캡처 각각 타임스탬프 순 앞 80% train / 뒤 20% test(연속 블록), 경계 15프레임 drop — 스크립트 의도와 실제 JSON 완전 일치 (`min(test ts) > max(train ts)` 8/8, 재구성식 8/8 성립).
**낙관 편향 주의**: 경계 gap 대부분 1~2.7초(15fps) → 장면·객체·조명이 train/test 공유. v3 수치는 상한선 성격, 일반화 측정은 v2(캡처 holdout)가 정본. 보고 시 병기 권장.

## 8. best weights 모음 위치 (2026-07-07)

**전 서버 학습 best 웨이트 → `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/weights/`**
- `yolo_rgb_baselines/` — YOLO RGB 기준점 5종 (v2/labelv3/v3split/final)
- `p29det_multimodal/` — P29-Det 멀티모달 (egofill best/peak, event, 릴레이 final_*)
- `README.md` — 파일별 실험·데이터·mAP50 매니페스트
- 진행 중 run(Y1/event/릴레이)은 각 서버 수집기가 완료 시 자동 복사.

## 9. final 저조도 split 실험 (릴레이, 2026-07-07~)

**데이터**: final split (train 5클립 / test 3클립 중 114021🌙·115624🌙 저조도). egofill 통합 lidar.
평가 = 저조도(1,768~1,769) vs 정상(1,654~1,671) 분해. **멀티모달 robustness 검증 목적**.

| run | 입력 | 전체 mAP50/AP50 | 저조도 | 정상 | 저조도 delta |
|-----|------|------|------|------|------|
| **Y1** YOLOv5m (완료) | RGB | 0.907 | **0.865** | **0.935** | **−0.070** (RGB 저조도 취약 확인) |
| M3 P29 full (학습중) | RGB+LiDAR+Thermal | - | - | - | (관건: <0.070이면 fusion robustness 승) |
| M1 P29 (대기) | RGB | - | - | - | - |
| M2 P29 (대기) | RGB+Thermal | - | - | - | - |

- **event ablation(E2.6) 완료**: best AP50 0.8427 (COCO AP peak 0.5174, ep14) — lidar egofill(0.850)과 동등. **3번째 모달 lidar≈event 확정**.
- Y1 all-test 0.907(v2/v3 0.866보다 높음): final test 클립 구성이 RGB에 유리한 면. 핵심은 저조도-정상 delta.
