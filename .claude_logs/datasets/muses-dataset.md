# MUSES 데이터셋 — 경로/구성/panoptic·uncertainty GT

> 목적: MUSES 데이터셋의 원본 zip·해제본 경로, panoptic/uncertainty GT 구성 확인 기록.
> 2026-08-06 확인: `gt_panoptic`/`gt_uncertainty`는 **이미 mid-July부터 완전히 존재**함
> (인계 문서가 "미다운로드"라 기록했던 것은 오기 — PQ 측정 블로커는 이 데이터가 아니라
> 다른 곳에 있었다).

## 원본 zip

`/ailab_mat2/dataset/MUSES_zips/`:

| 파일 | 크기 |
|---|---|
| frame_camera_trainvaltest.zip | 5.3G |
| event_camera_trainvaltest.zip | 1.1G |
| lidar_trainvaltest.zip | 2.2G |
| radar_trainvaltest.zip | 5.1G |
| reference_frame_trainvaltest.zip | 4.4G |
| gnss_trainvaltest.zip | 1.0M |
| gt_semantic_trainval.zip | 84M |
| gt_detection_trainval.zip | 6.0M |
| **gt_panoptic_trainval.zip** | **36M** |
| **gt_uncertainty_trainval.zip** | **12M** |

출처: `https://muses.ethz.ch/MUSES_packages/`(평문 Apache 인덱스, 계정/토큰 불필요).
zip들은 2026-07-14 스테이징됨(파일 자체의 내부 mtime은 2024-04-12 — MUSES 원 제작 시점).

## 해제본 구성

`/ailab_mat2/dataset/MUSES/`:

- `gt_panoptic/{train,val}/{clear,fog,rain,snow}/{day,night}/*_gt_panoptic.png`
  — train 1500장 / val 250장(test는 GT 비공개라 없음, 정상).
  - `gt_panoptic/train.json`, `gt_panoptic/val.json` — COCO-panoptic 스타일 매니페스트
    (`images`/`annotations`/`categories`, val.json 기준 images=250·annotations=250·categories=19).
  - `gt_panoptic/test_image_info.json` — test는 GT 없이 이미지 메타만.
  - `gt_panoptic/gt_panoptic_by_condition/*.json`(42개) — 날씨/주야 조건별 분할 매니페스트(편의용).
- `gt_uncertainty/{train,val}/{clear,fog,rain,snow}/{day,night}/*_gt_uncertainty.png`
  — train 1500장 / val 250장.

파일명 규약 확인됨: `{sequence}_frame_{frame:06d}_gt_panoptic.png` /
`{sequence}_frame_{frame:06d}_gt_uncertainty.png` — MUSES 공통 스템 규약과 일치
(예: `REC0241_frame_499918_gt_panoptic.png`, `REC0006_frame_043620_gt_uncertainty.png`).

## 결론

**PQ(Panoptic Quality) 측정을 위한 GT 데이터는 이미 전부 준비돼 있다.** 별도 다운로드 불필요.
남은 블로커가 있다면 데이터가 아니라 **평가 코드/파이프라인 쪽**이다(다음 조사 필요).
