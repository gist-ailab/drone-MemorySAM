---
legacy_id: 17
legacy_file: 17_p29det_data_fix.md
moved: 2026-07-08
---

# P29-Det 학습실패 진단 → 깨끗한 라벨셋 재학습 (v2/v3 split)

> 작성: 2026-06-30 (백그라운드 세션). 대상: P29-Det = RBMA(P29) 백본 + FPN/FCOS 객체탐지, poongsan 실내 RGB+LiDAR+Thermal.
> 코드/설정 = branch `worktree-p29-det`(commit b6c7c47~). 학습 서버 = jarvis(8×RTX4090).

---

## 1. 증상
- jarvis 첫 학습(`det_P29_indoor_jarvis.yaml`, 구 `_det_splits`): **COCO AP ≈ 0** (ep4 AP=0.0032 → ep9 AP=0.0020). "4ep>8ep"은 의미 없는 0 근처 노이즈.
- epoch 평균 loss가 ~0.88에서 **평탄**(ep1 0.95 → ep13 0.86), 사실상 학습 정체.

## 2. 근본 원인 (진단 확정)
- 학습 로그상 **전체 step의 53.1%가 `n_pos=0`**(FCOS positive 타겟 0개) — reg/centerness 신호 없이 background cls loss(≈0.04~0.13)만 기여.
- 원인: 구 `_det_splits/det_train.json`(11,677장)의 **52.1%(6,081장)가 어노테이션 0개**. 데이터가 **연속 비디오(20Hz raw, 4 caps)** 인데 COCO `images`엔 전 프레임이 들어있고 라벨은 일부 프레임에만 존재.
- **빈 프레임 = 진짜 배경이 아님.** 캡처별 빈 프레임 6장 육안검증 → 전부 사람(Allies)·문(Doors)·창문(Windows)·비상구(EmExit)·소화기(FireExt) 등 타겟 객체 포함된 **미레이블(false negative)** 프레임.
- 영향: batch=1 학습에서 절반 step이 "여기엔 아무것도 없다"고 가르치며 **정탐을 적극 억제** → AP≈0. (단순 낭비가 아니라 파괴적 지도)
- 부차 원인: ① `SEG_CHECKPOINT:''`(학습된 RBMA seg 백본 미로드, SAM2 pretrained+랜덤만) ② 640×480→1024² **비등방 stretch**(x1.6/y2.13) 형태 왜곡 ③ warmup 5/50 → "8ep"은 full-LR 3ep.

## 3. 조치 — 깨끗한 라벨셋으로 교체
- 새 라벨셋: `/ailab_mat2/Projects/Drone/DATA/260618_poongsan/` (8 captures, 캡처별 `annotations/instances.json`). **전 캡처 empty=0%**, 총 13,712장 / 42,108 박스 / 10클래스.
- jarvis로 rsync: `/SSDd/jemo_maeng/dset/poongsan_v2/capture_*/{rgb,depth_map_lidar,thermal_aligned}` (필요 3모달 6.1GB). COCO modalities 경로에 `capture_XXX/` prefix, ROOT=poongsan_v2.
- **lidar 부분 커버**: depth_map_lidar 7,634/13,712장만 존재 → `REQUIRE_ALL_MODALITIES:true`라 lidar 없는 프레임 drop(설계상 동일). **남는 이미지는 100% 어노테이션 보유**(검증: 모든 split에서 kept_all3modal == kept_with_ann).

## 4. Split 2종 (사용자 지시: 둘 다 생성, v2 먼저 학습 / 안되면 v3)
연속 비디오라 split 방식이 test 수치 해석을 좌우(인접 프레임 near-duplicate leakage). 빌더: `scripts/build_det_splits.py`(provenance), 산출물 `poongsan_v2/_det_splits/`.

| split | 방식 | train(전체→kept all-3-modal) | test(전체→kept) | 비고 |
|------|------|------|------|------|
| **v2** | 캡처 holdout: test=cap_115206+114808 | 10,535 → **5,862** | 3,177 → **1,772** | temporal leakage 0, 양쪽 10클래스. **정직한 일반화.** |
| **v3** | 캡처 내 시간 80/20(gap=15) | 10,967 → 5,933 | 2,625 → 1,628 | 장면 일치·클래스 균형, v2보다 낙관적. fallback용. |

- config: `configs/det/det_P29_indoor_jarvis_v2.yaml`, `..._v3.yaml`. **데이터 외 하이퍼파라미터는 구 config와 동일**(변수 격리 — split 효과만 측정).

## 5. v2 재학습 (진행 중, 2026-06-30 20:49 launch)
- jarvis tmux `jemo:p29det_clean`, GPU 1,2,3,4(구 고장난 run kill 후 회수), torchrun nproc=4 port 29531, `WANDB_MODE=offline`.
- log: `logs/det_P29_indoor_jarvis_v2_<ts>.log`, 출력: `outputs/det/det_P29_indoor_jarvis_v2/`.
- **검증 성공**: dataset kept=5862(train)/1772(test); **n_pos==0 = 0%**(1294 step, min48/median814/max3106) — 구 53.1% → 0%. 모든 step이 positive 타겟 보유.
- 런처 재사용: `bash scripts/_launch_p29det_clean.sh <cfg> <gpus> <port>` (jarvis).

## 6. 남은 레버 (v2가 부족할 때)
1. v3(시간분할)로 전환. 2. `SEG_CHECKPOINT`에 학습된 RBMA seg ckpt 로드(백본 의미있는 init). 3. letterbox(AR보존+패딩)로 stretch 왜곡 제거. 4. empty-aware sampling 불필요(이미 0%). 5. effective batch↑(grad-accum).
