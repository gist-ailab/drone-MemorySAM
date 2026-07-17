---
created: 2026-07-08
---

# 🧾 실험 레지스트리 (Experiment Registry Hub)

> **역할**: 트랙 전체(seg/det)의 핵심 실험을 한 표로 추적하는 허브. 새 실험 launch 시 행 추가, 상태 변화(🟢best/🟡active/🔴dead) 시 갱신.
> 상세 서사는 [log.md](log.md)(결과 canonical) · [monitor-log.md](monitor-log.md)(RUN-N 실시간) · [../status/current.md](../status/current.md)(현재 스냅샷) 참조.
> 상태 범례: 🟢 best(트랙 최고/기준점) · 🟡 active(학습 중·완주 후 참조) · 🔴 dead(사망/취소/회귀 확정).
> (repo 루트 상대경로. seed 수치 기준일 = 2026-07-08, 최신 수치는 monitor-log 확인.)

## Seg — MULTIAQUA (MACVi Challenge, M-score)

| 실험ID(config명) | 트랙 | 데이터셋 | 모델버전 | config 경로 | 서버 | ckpt 경로 | 상태 | 핵심 수치 | 관련 문서 |
|---|---|---|---|---|---|---|---|---|---|
| levine_multiaqua_rgbtl_P9_hardaug8_physaug (ep131) | seg | MULTIAQUA (RGB+T+L) | P9 | `configs/multiaqua/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml` | levine | `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth` | 🟢 best (공동 1위) | **M-score 82.10** (Val 93.29 / Test 70.91, 재제출 #16710) | [log.md](log.md) §요약표, [../models/explain/p09-cross-modal-fusion-head.md](../models/explain/p09-cross-modal-fusion-head.md) |
| levine_multiaqua_rgbtl_P22_hardaug8_physaug (ep120) | seg | MULTIAQUA (RGB+T+L) | P22 | `configs/multiaqua/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml` | levine | `outputs/MMSamP22/levine_multiaqua_rgbtl_P22_hardaug8_physaug/` | 🟢 best (공동 1위) | **M-score 82.10** (Val 93.42 / Test 70.77, #16932) | [log.md](log.md) §요약표 |
| levine_multiaqua_rgbtl_P21_hardaug8_physaug (ep94) | seg | MULTIAQUA (RGB+T+L) | P21 | `configs/archive/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml` | levine | `outputs/MMSamP21/...` (ep94 best) | 🟡 완료 (P21 best) | M-score 81.77 (Val 93.17 / Test 70.36, #16792) | [log.md](log.md), [../models/explain/p21-deba-fp.md](../models/explain/p21-deba-fp.md) |

## Seg — DELIVER (논문 트랙, 목표 val ≥66.51 / test ≥56.71)

| 실험ID(config명) | 트랙 | 데이터셋 | 모델버전 | config 경로 | 서버 | ckpt 경로 | 상태 | 핵심 수치 | 관련 문서 |
|---|---|---|---|---|---|---|---|---|---|
| b200_deliver_rgbdel_P31_physaug | seg | DELIVER (img/depth/event/lidar) | P31 (P31.1) | `configs/deliver/b200-deliver_rgbdel_P31_physaug.yaml` | B200 (GPU4-7) | `outputs/MMSamP31/b200_deliver_rgbdel_P31_physaug/DELIVER_CMNeXt-B2_idel/` | 🟡 active (학습 중, **현 최선 DELIVER**) | ep162 기준 Day-Val **63.20**@ep106(=P29 동률) / Test **54.75**@ep158(P29 +0.41). 목표 갭 −3.31/−1.96 | monitor RUN-9, [../decisions/2026-07-02-p31-redesign-proposal.md](../decisions/2026-07-02-p31-redesign-proposal.md) |
| **ANALYSIS: P29·P31·P32·P34 표준분석 (lecun)** | seg-analysis | DELIVER test 5-cond | P29 ep146 + P31 ep182 + P32 ep158 + **P34 ep40(스냅샷)** | `tools/seg_analysis_pipeline.py` (D1,D2,D2N,D3,D3B,D4,D5) | lecun GPU0/1 | ckpt `/SSDb/jemo_maeng/ckpt/{P31,P32}/` | ✅ 완료 (07-12 4모델 + 07-13 P34 ep140 재분석) | 산출: NAS `/drone_nas/drone/analysis_logs/` (HDD2 ISSUE-023 재발로 대체). 종합=[analysis/2026-07-12-p29-p34-standard-analysis.md](analysis/2026-07-12-p29-p34-standard-analysis.md). P34 ep140 mean **55.65** (Test 57.60=목표+0.89 달성 ckpt), retarget 클래스 전부 1위 | [tools/README_seg_analysis.md](../../tools/README_seg_analysis.md) (항목1-4 매핑) |
| b200_deliver_rgbdel_P29_physaug | seg | DELIVER | P29 (SDC) | `configs/deliver/b200-deliver_rgbdel_P29_physaug.yaml` | B200 | `outputs/MMSamP29/b200_deliver_rgbdel_P29_physaug/DELIVER_CMNeXt-B2_idel/` | 🟡 종료 (비교 기준선) | Val **63.20**@ep100 / Test **54.34**@ep146 (목표 갭 −3.31/−2.37) | monitor RUN-2, [analysis/2026-06-30-p28-p29-failure-analysis.md](analysis/2026-06-30-p28-p29-failure-analysis.md) |
| b200_deliver_rgbdel_P30_physaug | seg | DELIVER | P30 | `configs/deliver/b200-deliver_rgbdel_P30_physaug.yaml` | B200 | `outputs/MMSamP30/b200_deliver_rgbdel_P30_physaug/DELIVER_CMNeXt-B2_idel/` | 🔴 dead (회귀 확정) | Val 49.76@ep136 / Test 44.10@ep146 (P29 대비 −13.4/−10.2) | monitor RUN-4 |
| P28 RBMA seg (B200 RUN-1) | seg | DELIVER | P28 (RBMA) | — (monitor RUN-1 참조) | B200 | `last_checkpoint.pth` 보존 | 🔴 dead (ep16 사망) | best Val 57.87@ep12 / Test 50.61@ep12 | monitor RUN-1, [analysis/2026-06-30-p28-p29-failure-analysis.md](analysis/2026-06-30-p28-p29-failure-analysis.md) |
| **ANALYSIS: P37a-CEFR MUSES 출력분석 (yeon)** | seg-analysis | MUSES val | P37a ep110 (val-best 81.16) | `tools/probe_cefr_routing.py` + `tools/module_ablation.py` | yeon GPU0 (worktree dm_analysis) | `/SSDb/jemo_maeng/ckpt/P37a/` (yeon·lecun) | ✅ 완료 (07-18) | σ(a)=0.121(채택), **라우팅 분화 실패 0/19 committed**(전 클래스 event 0.385 틸트), cefr_off Δ+0.16=no-op, router_off Δ+34.66=의존(비기여), gate/calib no-op 재현 | [analysis/2026-07-18-p37a-muses-cefr-output-analysis.md](analysis/2026-07-18-p37a-muses-cefr-output-analysis.md) |
| SAM3-RBMA (DELIVER 25cls) | seg | DELIVER 25cls | SAM3-RBMA | — ([../decisions/2026-06-16-sam3-porting-plan.md](../decisions/2026-06-16-sam3-porting-plan.md) 참조) | — | — | 🟡 학습/디버깅 중 | decoder repurpose로 class-collapse 돌파: val 8.49→16.27@ep22 (상승 중) | [../status/current.md](../status/current.md) |

## Det — poongsan indoor (국가 R&D, 목표 mAP50 0.85)

| 실험ID(config명) | 트랙 | 데이터셋 | 모델버전 | config 경로 | 서버 | ckpt 경로 | 상태 | 핵심 수치 | 관련 문서 |
|---|---|---|---|---|---|---|---|---|---|
| det_P29_egofill_bengio | det | poongsan_v2 + egofill lidar (train 11,799, val=v2 test 1772) | P29-Det | `configs/det/det_P29_egofill_bengio.yaml` | bengio (egofill 체크아웃) | `outputs/det_egofill/` | 🟢 **best** | **mAP50 0.8501**@ep9 (공식 v2 test) — 🎯 목표 0.85 달성 | monitor RUN-11, [../datasets/lidar-egofill.md](../datasets/lidar-egofill.md) |
| det_P29_event_bengio | det | poongsan egofill_common11799, MODALS img/**event**/thermal | P29-Det | `configs/det/det_P29_event_bengio.yaml` | bengio | `outputs/det_event/det_P29_event_bengio/` | 🟡 완주 (모달 ablation) | mAP50 **0.8427**@ep14 — event ≈ egofill-lidar(−0.008) | monitor RUN-14 |
| det_P29_final_full | det | poongsan 최종 annotation (`_final_ann/instances_train_egofill.json`) | P29-Det | `configs/det/det_P29_final_full.yaml` | bengio | `outputs/det_final_full/` | 🟡 active (2026-07-08~ 학습 중, EPOCHS 50) | ep0 — 제출용 final 수치 확보 목적 (egofill 0.8501의 최종 annotation 재학습) | monitor RUN-15 |
| det_P29_v2 (재학습) | det | poongsan_v2 (clean label) | P29-Det | — ([../det/p29det-data-fix.md](../det/p29det-data-fix.md) 참조) | jarvis | — | 🟡 완료 (스택 기준선) | mAP50 **0.446**@ep9 (v2 공식) — ep9 피크 후 하락 | [../det/p29det-data-fix.md](../det/p29det-data-fix.md), [../det/diagnosis-plan.md](../det/diagnosis-plan.md) |
| det_P31_v3clip_jarvis | det | poongsan v3clip split (비공식) | P31.1-Det (calibrated rel. + decisive router + FCOS) | `configs/det/det_P31_v3clip_jarvis.yaml` | jarvis (GPU1-4) | `outputs/det/det_P31_v3clip_jarvis/` | 🟡 완료 (⚠️ v2와 직접비교 불가 — v2 재평가 필요, 태스크 D2) | mAP50 **0.4724** (v3clip) | monitor RUN-10, [../det/diagnosis-plan.md](../det/diagnosis-plan.md) §7 |
| det_P30_v2 | det | poongsan_v2 | P30-Det (router+query decoder) | — | jarvis | 비교 리포트 `/mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/` | 🔴 dead (소물체 붕괴) | mAP50 0.256 (P29 0.446 대비 하락; router↔query-head confound) | [../det/diagnosis-plan.md](../det/diagnosis-plan.md) |
| YOLO11m RGB-only 기준점 (E1.1b/c) | det | poongsan label-v3 | YOLO11m (외부 head, RGB-only) | `objdet/yolo11m-rgb/` | hinton | — | 🟢 외부 기준점 | **mAP50 0.864** (label-v3) — "데이터 무죄, 스택 유죄" 판정 근거 | [../det/diagnosis-plan.md](../det/diagnosis-plan.md) E1.1, [../meta/taskboard.md](../meta/taskboard.md) §0 |
