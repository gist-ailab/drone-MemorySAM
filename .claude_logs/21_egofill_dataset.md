# 20. lidar egofill 데이터셋 (v20260703_egofill)

> 2026-07-03. 목적: RGB 15Hz vs LiDAR 10Hz 주기 차이로 라벨 프레임의 37%가
> `depth_map_lidar` 없음 → `REQUIRE_ALL_MODALITIES`로 대량 폐기되는 문제를
> **ego-motion 보정 최근접 스캔 재투영**으로 해소. intensity_map_lidar 도 동일 스캔에서 생성.

## 결과 요약

- 버전: **v20260703_egofill** (base: v20260702_2303). `/ailab_mat2/Projects/Drone/DATA/260618_poongsan/`
  - 각 capture 에 `depth_map_lidar_egofill/`, `intensity_map_lidar_egofill/` 추가 (원본 무변경)
  - `instances.json` 갱신 + `instances_v20260703_egofill.json` 동결 + `versions/` 매니페스트 + `VERSIONS.md`
  - 합성 프레임은 image 에 `lidar_egofill: true` 표식
- **fill 6,403 / 결측 유지 212** (lidar 공백 >300ms 인 프레임 — 억지 fill 은 노이즈 판단)
  → lidar 커버리지 8,538(56%) → **14,941/15,153 (98.6%)**
- v2 capture-holdout split 재생성 시: **train 5,862 → 11,799 (2.01배, egofill 5,033 포함)**,
  test full 3,142 / **평가용 제한 test 1,772 (원본 lidar 만, 기존 모든 실험과 동일 프레임)**
  = `_det_splits_egofill/det_test_v2_orig1772.json`

## 파이프라인 (sensors/drone_multimodal_capture/scripts/egofill/)

`imu_extract.py` → `gap_extract.py` → `egofill_render.py` → `freeze_egofill_version.py`

1. **IMU**: bag `/livox/imu` (Mid-360 내장, 실효 125~173Hz) 추출 (rosbags, ROS 불필요)
2. **스캔 소스**: 앵커별 매칭 스캔 = `sync_index.csv`(`lidar_stamp_us`). 66%는 이미 export 된
   `lidar_aligned/*.pcd` 재사용, 34%(sync 미매칭 공백)는 bag 에서 직접 추출(±300ms 최근접)
3. **ego 보정**: 자이로를 카메라 프레임으로 축변환 후 [t_scan→t_rgb] 적분한 회전 R 적용
   (`P @ R`). 병진은 무시 (dt 중앙값 36ms, gap 프레임 ≤300ms)
4. **렌더**: 기존 depth/intensity 맵과 동일 인코딩 — 640×480, JET, 점반경 1, z-buffer,
   프레임별 min/max 정규화

## 역공학으로 확정한 사실 (재발 방지용 — calib yaml 믿지 말 것)

- `lidar_aligned/*.pcd` 는 **이미 RGB 카메라 좌표계** (raw bag 스캔과 index 1:1 대응)
- 투영 intrinsics = **RealSense 공장값** (camera_info: fx 614.0, fy 614.05, cx 320.2, cy 243.0, D=0)
  — `calibration/calib/intrinsics_rgb.yaml` (fx 752 재추정값) **아님**
- 실제 lidar→camera 변환 (Kabsch 역산, 잔차 <1µm, 2개 캡처 교차검증; Mid-360 ~71° 틸트 장착):
  `R=[[0.99993,-0.00282,0.011456],[0.011745,0.329708,-0.94401],[-0.001115,0.944079,0.329718]]`,
  `t=[0.032481,-0.138011,-0.093521]` — `extrinsics_lidar_to_rgb.yaml`(reliable:false) **아님**
- 검증: 자기 스캔 재투영 시 원본 맵과 점 개수 일치(9,348 vs 9,350), 잔차 1~2px
  (원본은 per-point offset_time 보정 추정). 자이로 보정 방향은 이웃스캔 깊이 일관성
  (|dz| 2.7cm(R) vs 2.9(무보정) vs 3.5(역방향))으로 확정
- lidar 결손의 근본 원인 = RGB 15Hz vs LiDAR 10Hz + 1:1 배정 정렬 (04_issues 참조)

## 멀티모달 학습 실험 (진행 중)

- **P29-Det (LoRA_Sam_P29_Det, mean fusion, FCOS), 레시피 원본 동일** (50ep, LR 2e-4, batch1/GPU, AMP)
- 서버: **bengio GPU 5장 DDP** (원본 jarvis 6×4090 → 5×3090; jarvis 만석)
- 코드: p29-det worktree 를 `bengio:/SSDb/.../drone-MemorySAM-p29det-egofill/` 로 동기화
  (bengio 본 레포 det 코드는 구버전 — 건드리지 않음)
- config: `configs/det/det_P29_egofill_bengio.yaml`
- 비교점: P29-Det 0.4455 (구 라벨, train 5,862) vs 이번 run (신 라벨+egofill, train 11,799).
  라벨효과는 E1.1b(YOLO +0.043)로 분리 참조. 평가는 동일 1,772 프레임

## 알려진 한계: 동적 객체 (2026-07-03 검증)

ego 보정은 **드론 자기움직임만** 보상 — 사람이 스캔↔rgb 사이에 움직인 것은 보상 불가.
사람 박스 내 lidar 점 깊이가 주변보다 가까운("사람 감지") 비율 (capture_115206, 동일 지표):
**원본 71% → near-fill(≤53ms) 47% → gap-fill(≤300ms) 대부분 실패**(뒤 배경 깊이가 찍힘).
- 정적 7클래스는 정상, 사람 3클래스(Allies/Enemies/Casualties)는 egofill 프레임에서 lidar 단서 열화
- 평가는 원본-lidar 1,772 프레임만 사용하므로 평가 오염 없음. train 에서의 영향은
  P29-Det egofill run 결과로 판단 → 필요 시 gap-fill 제외 ablation (train ~10,000장)
