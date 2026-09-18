# baseline_failure — DGFusion·CAFuser·ReliaDINO 실패 사례 분석 킷

DELIVER **val 2005 + test 1897** 전 이미지 위에서 세 모델(DGFusion·CAFuser·우리
ReliaDINO)의 예측을 이미지별·클래스별·조건별로 대조해 실패 구조를 찾는다. 설계서 =
`.claude_logs/experiments/analysis/2026-09-17-baseline-failure-analysis-plan.md` (D1~D6).

- **우리 모델**: `val.py` 경로(1024·BS1·native 1042² GT)를 그대로 import 해서 덤프한다.
- **기준선**: 재추론 대신, `--eval-only` 가 이미 남긴 `inference/sem_seg_predictions.json`
  (COCO RLE 예측)을 `rle_json_to_png.py` 로 PNG 복원한다. (RLE JSON 이 없으면 이 킷의
  `d2_dump_evaluator.py` + `d2_dump.patch` 로 재추론하며 덤프하는 경로도 그대로 있다.)

## 🔴 저장 규약 (2차 보강 — single source = `common.py`)
1차 덤프는 파일명을 `000050_rgb_front.png` 로 **평탄화**해, DELIVER 의 조건/케이스 하위
폴더마다 반복되는 같은 basename 이 서로 덮여 test 1270/1897·val 1733/2005 만 남는 결함이
있었다. 2차부터 **모든 덤프·변환기가 원본 이미지의 데이터셋 루트 기준 상대 경로를
그대로 중첩 디렉터리로 보존**한다:
- **image_id = 데이터셋 루트 기준 상대 경로**(확장자 제외) = `img/<condition>/<split>/<scene>/<stem>_rgb_front`.
  우리 덤프(RGB 경로)와 기준선 변환(JSON `file_name`)이 같은 RGB 경로에서 같은 id 를
  내야 join 이 성립한다(스모크가 동치성을 assert). `common.image_id_from_rel` 이 단일 출처.
- 저장 경로 = `<out>/<split>/pred/<image_id>.png` (중첩). 반복 basename 이 덮어쓰지 않는다.
- 저장이 끝나면 **장수를 세어 기대값(val 2005 / test 1897)과 다르면 assert 로 멈춘다**
  (`common.assert_expected_count`). `per_image_metrics` 는 평탄 덤프를 만나면 명확한 에러.
- trainID 0~24 유효, 255 = ignore. 혼동행렬 `hist[gt, pred]` (`semseg/metrics.py` 재현).
- 전역 mIoU = 25클래스 IoU 평균(부재 클래스 0 포함). 이미지별 IoU 는 GT 부재 클래스 NaN.
- 조건 ∈ {cloud,fog,night,rain,sun}, 센서 고장 케이스는 scene 폴더명 접미사
  (`_motionblur|_overexposure|_underexposure|_lidarjitter|_eventlowres`), 없으면 `none`.

## 파이프라인
```
D1 덤프 ─┬─ 우리:      dump_preds_ours.py                     (상대 경로 보존 + 장수 assert)
         ├─ 기준선(권장): rle_json_to_png.py  ← sem_seg_predictions.json 복원
         └─ 기준선(대안): train_net.py --eval-only (+ d2_dump.patch, BF_DUMP_DIR)
C  규약대조 check_label_convention.py  → category_id 이동량(0 이어야 정상) + JSON 히스토그램
D2 지표     per_image_metrics.py   → per_image_*.csv, joined_<split>.csv (중첩 image_id join)
D3 채굴     failure_mining.py      → mining/{d31..d36, confusion/, d32_top_lists.json}
D5 시각화   viz_panels.py          → mining/panels/<list>/<image_id>.png
D4 기제     modality_zero_ablation.py (우리) · probe_dgfusion.py · probe_cafuser.py (기준선)
```
산출물 루트(예): `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/baseline_failure_20260917/{model}/{split}/{pred,conf,gt}/`.

## D1 — 우리 모델 덤프
```bash
conda activate MMSS_SAM
python tools/baseline_failure/dump_preds_ours.py \
  --cfg configs/eval/<ours>.yaml --model_path <ckpt.pth> \
  --split test --out <ROOT>/ours \
  --expected-miou 56.99          # ±0.05 밖이면 exit 2 (덤프 무효)
# val 도 동일하게 --split val (E1 시드1 값으로 검증)
```
`val.py` 는 수정하지 않는다(추론 함수만 import). 저장물 = `<out>/<split>/{pred,conf,gt}` +
`summary.json`. image_id = 상대 경로(`img/<cond>/<split>/<scene>/<stem>_rgb_front`)로 중첩
저장되고, 끝에 장수(val 2005 / test 1897)를 assert 한다(`--limit` 디버그 실행은 예외).

## D1 — 기준선 덤프 (권장: RLE JSON → PNG 복원)

기준선 `--eval-only` 로그가 남긴 `inference/sem_seg_predictions.json` 을 그대로 PNG 로
복원한다(재추론 불필요). **먼저 라벨 규약을 대조**한 뒤 변환한다.

```bash
conda activate MMSS_SAM   # pycocotools 필요
# (C) 규약 대조 — category_id 히스토그램 + (가능하면) 이동량 탐색
python tools/baseline_failure/check_label_convention.py \
  --json <RUN>/inference/sem_seg_predictions.json          # 방법 (ii): min/max/히스토그램
# 변환(기본 offset 0, 보정 없음). --dataset_root 로 절대 file_name 을 잘라 상대 경로화.
python tools/baseline_failure/rle_json_to_png.py \
  --json <RUN>/inference/sem_seg_predictions.json \
  --dataset_root $PWD/datasets/DELIVER --split test \
  --out <ROOT>/dgffinal --expected_count 1897 \
  --gt_dir <ROOT>/ours/test/gt --expected_miou 55.5605     # (D) 재현 검산 → summary.reproduced
# 변환 뒤 (C) 방법 (i): 변환 PNG vs 우리 GT 로 이동량이 0 인지 최종 확인
python tools/baseline_failure/check_label_convention.py \
  --pred_dir <ROOT>/dgffinal/test/pred --gt_dir <ROOT>/ours/test/gt
```
- **이동량이 0 이 아니면**(예: JSON 이 1~25 로 1-based) `check_label_convention` 이 exit 3
  으로 멈추고 필요한 `--label_offset` 을 알려 준다. 그 값을 변환기에 넘겨 다시 변환한다
  (추측 보정 금지). offset 적용 후 category_id 가 0~24 밖이면 변환기도 에러로 멈춘다.
- **재현 검산(D1 유효 판정)**: `--gt_dir`+`--expected_miou` 를 주면 변환 PNG 로 전역 mIoU 를
  재계산해 로그값(DGFusion test 80k 55.6807 / final 55.5605, CAFuser test final 55.3761,
  val 66.0405)과 ±0.05 안이면 `summary.reproduced=true`. GT 해상도가 달라 리샘플했으면
  summary 에 남으니, 기준선 evaluator 의 GT 읽기 방식(native 1042 vs 리사이즈)을 확인해 보고.
- summary.json = 장수·`overlap_pixels`(겹침, argmax 면 0)·`ignore_pixel_ratio`(255 비율)·
  category_id min/max·(검산 시) reproduce_miou/delta/note.

## D1 — 기준선 덤프 (대안: 저장소 안에서 재추론하며 덤프)
1. 이 킷의 `d2_dump_evaluator.py` 를 저장소 루트에 복사하고, `d2_dump.patch` 를 적용한다:
   ```bash
   cp <repo_hub>/tools/baseline_failure/d2_dump_evaluator.py .
   git apply tools_or_local/d2_dump.patch   # 실패 시 build_evaluator 의 DELIVER 분기를 수동 편집
   ```
   패치는 `BF_DUMP_DIR` 이 있을 때만 `DumpSemSegEvaluator` 를 끼운다(없으면 공식 경로 그대로).
   > 이 evaluator 도 상대 경로 image_id 로 저장하며, file_name 이 절대 경로면
   > `BF_DATASET_ROOT=$PWD/datasets/DELIVER` 로 잘라낸다. `BF_SPLIT=val|test` 를 주면
   > `evaluate()` 끝에서 장수(2005/1897)를 assert 한다.
2. **규약 점검**(필수): 기준선 GT 가 우리와 같은 trainID 인지 먼저 단언한다.
   ```bash
   python d2_dump_evaluator.py --check-gt datasets/DELIVER/semantic/<...>   # ok=False면 remap 필요
   ```
3. **DGFusion** (yeon, `/SSDb/jemo_maeng/dgfusion_train`, env `dgfusion`):
   ```bash
   export PYTHONPATH=$PWD/OneFormer:$PWD WANDB_MODE=offline DETECTRON2_DATASETS=$PWD/datasets
   RUN=output/dgfusion_swin_tiny_bs8_200k_deliver_clde
   # test
   BF_DUMP_DIR=<ROOT>/dgffinal/test CUDA_VISIBLE_DEVICES=<gpu> python train_net.py \
     --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
     --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS $RUN/model_final.pth \
     DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" MODEL.TEST.DEPTH_ON False \
     OUTPUT_DIR $RUN/dump_test
   # val 은 DATASETS.TEST_SEMANTIC 기본값(등록된 val)으로, BF_DUMP_DIR=<ROOT>/dgffinal/val
   # 80k 체크포인트도 별도로: MODEL.WEIGHTS $RUN/model_0079999.pth, BF_DUMP_DIR=<ROOT>/dgf80k/<split>
   ```
   ⚠️ 90k 이후 체크포인트는 fp16 추론 시 NaN(README 상위 "추론 주의") — 공식 test 명령은 fp32라 무관.
4. **CAFuser** (lecun, `/SSDb/jemo_maeng/cafuser_train`, env `dgfusion`): 위와 동일하되
   `--config-file configs/deliver/swin/cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml`,
   `MODEL.WEIGHTS $RUN/model_final.pth`, `BF_DUMP_DIR=<ROOT>/caf/<split>`. (bs6·3GPU 파생본 — 보고 의무.)

> 덤프 유효 조건: 공식 로그의 mIoU 가 등록 수치(DGFusion test 55.68/55.56, CAFuser val 66.04)와
> 재현되어야 한다(`DumpSemSegEvaluator` 는 공식 `SemSegEvaluator.process` 를 그대로 호출하므로
> mIoU 는 변하지 않는다). 재현 안 되면 덤프 무효.

## D2 — 이미지별 지표 + join
```bash
python tools/baseline_failure/per_image_metrics.py \
  --gt <ROOT>/ours/test/gt --split test --out <ROOT>/analysis/test \
  --models ours=<ROOT>/ours/test/pred dgf80k=<ROOT>/dgf80k/test/pred \
           dgffinal=<ROOT>/dgffinal/test/pred caf=<ROOT>/caf/test/pred
```
각 모델의 전역 재계산 mIoU 를 `<pred>/../summary.json`(우리·RLE 변환 모두 생성) 과 대조한다
(불일치 시 exit 1). 재추론 경로처럼 summary.json 이 없으면 대조를 건너뛰고 공식 로그 mIoU 와
손대조하라. image_id 는 **중첩 상대 경로**로 join 하며, 평탄 덤프를 만나면 즉시 에러로 멈춘다.

## D3 — 실패 채굴
```bash
python tools/baseline_failure/failure_mining.py \
  --joined <ROOT>/analysis/test/joined_test.csv --gt <ROOT>/ours/test/gt \
  --models ours=... dgf80k=... dgffinal=... caf=... \
  --per-image ours=<ROOT>/analysis/test/per_image_ours_test.csv dgf80k=... caf=... \
  --split test --out <ROOT>/analysis/test
# val→test 델타(D3-6): val 을 먼저 돌려 perclass_val.json 을 만든 뒤
#   test 실행에 --per-class-other <...>/mining/perclass_val.json 를 준다.
```

## D5 — 패널
```bash
python tools/baseline_failure/viz_panels.py \
  --list-json <ROOT>/analysis/test/mining/d32_top_lists.json --topn 12 \
  --deliver-root /ailab_mat2/dataset/DELIVER --split test \
  --gt <ROOT>/ours/test/gt --models ours=... dgf80k=... caf=... \
  --out <ROOT>/analysis/test
```

## D4 — 기제 측정
- **우리 모달 zero-out**:
  ```bash
  python tools/baseline_failure/modality_zero_ablation.py \
    --cfg configs/eval/<ours>.yaml --model_path <ckpt> --split test \
    --out <ROOT>/analysis/test/mining/modal_zero_ours.json
  ```
  라우터/트렁크 등 우리 모듈 진단은 `tools/module_diagnostics.py` · `tools/viz_features.py`.
- **기준선 모달 zero-out**: `d2_zero_modality.patch` 적용 후 위 test 명령에
  `BF_ZERO_MODAL=CAMERA|LIDAR|EVENT|DEPTH` 를 앞에 붙여 4회 실행(각 결과 mIoU 를 base 와
  비교). 입력 dict 의 모달 키는 `CAMERA`,`LIDAR`,`EVENT`,`DEPTH`(+주 모달 중복 `image`)이며
  `CAMERA` 지정 시 `image` 까지 함께 0 으로 채운다.
- **DGFusion/CAFuser 내부 프로브**(저장소 안에서):
  ```bash
  # 먼저 모듈 이름 확인 후 정규식을 맞춘다(추측 금지)
  python <repo_hub>/tools/baseline_failure/probe_dgfusion.py \
    --config-file <cfg> --weights <ckpt> --list-modules | less
  python .../probe_dgfusion.py --config-file <cfg> --weights <ckpt> \
    --depth-head-regex '<...>' --depth-token-regex '<...>' \
    --xattn-regex '<...>' --split test --out probe_dgf_test.csv
  # depth 토큰 0 치환 대조: BF_ZERO_DEPTH_TOKEN=1 python .../probe_dgfusion.py ...
  # 모달 zero-out 기제 측정(A4): --zero-modal CAMERA|LIDAR|EVENT|DEPTH — 해당 모달 입력을
  # 0 으로 채운 추론에서 이미지별 depth AbsRel·delta1(+분할 mIoU)를 같은 CSV 행에 기록.
  # depth GT 는 DELIVER depth/ 원본, 로그 스케일 여부는 MODEL.DEPTH_HEAD.LOSS.LOG_SCALE 를 읽어 처리.
  python .../probe_dgfusion.py --config-file <cfg> --weights <ckpt> \
    --zero-modal DEPTH --split test --out probe_dgf_zero_depth.csv
  python .../probe_cafuser.py --config-file <cfg> --weights <ckpt> --out probe_caf_test.csv
  ```

## D2b — 채점 프로토콜 대조(A2) · 체크포인트 안정성(A5) · depth 구간 IoU(A6)
```bash
# A2: 기준선(1024² 채점)과 우리(native 1042² GT)를 같은 축에서 재채점. 파일명에 프로토콜이
# 붙어 두 결과가 덮어쓰지 않는다(native 는 기존 이름 그대로). 요약 JSON 에 프로토콜·한계 note 기록.
python tools/baseline_failure/per_image_metrics.py --gt <ROOT>/ours/test/gt \
  --split test --out <ROOT>/analysis/test --gt_protocol resized1024 \
  --models ours=<ROOT>/ours/test/pred dgffinal=<ROOT>/dgffinal/test/pred

# A5: 체크포인트별 예측 덤프로 이미지가 일관되게 나쁜지(always_fail)/좋은지/흔들리는지 분류.
# 라벨 규칙 = 이미지별 mIoU 가 그 체크포인트의 중앙값 미만인 비율 ≥ 9/11(≈0.818, --fail_ratio).
python tools/baseline_failure/ckpt_stability.py --gt <ROOT>/ours/test/gt \
  --split test --out <ROOT>/analysis/test \
  --preds ep40=<ROOT>/ours_ep40/test/pred ep55=<ROOT>/ours_ep55/test/pred ep70=<ROOT>/ours_ep70/test/pred

# A6: 원본 depth 의 로그 5분위 구간별 IoU·pixel acc + 모델 간 차이(dgf−caf 등).
# depth==0 또는 GT==255 픽셀 제외, 구간 경계는 split 전체 유효 depth 픽셀에서 계산해 JSON 기록.
python tools/baseline_failure/depth_bin_iou.py --gt <ROOT>/ours/test/gt \
  --depth_root /ailab_mat2/dataset/DELIVER --split test --out <ROOT>/analysis/test \
  --models ours=<ROOT>/ours/test/pred dgf=<ROOT>/dgffinal/test/pred caf=<ROOT>/caf/test/pred
```

## 스모크
```bash
python tools/smoke_baseline_failure.py   # CPU, ~수초. 파일 3·4·5 end-to-end + 1·2·6·7 import/argparse
```

## 미확인 가정 (실행 전 확인 요망)
1. **RLE JSON 스키마·경로**: `sem_seg_predictions.json` 이 이미지별 `{file_name, category_id,
   segmentation(COCO RLE, size=[h,w])}` 목록이라고 가정한다. `file_name` 이 RGB(`img/...`)
   경로이며 우리 val.py 와 같은 RGB 경로를 가리켜야 image_id 가 정합한다. 절대 경로면
   `--dataset_root` 로 잘라내고, 다르면 `--path-sub OLD NEW`. counts 가 str 이면 bytes 로
   되돌려 디코드한다(정상). **실제 JSON 의 한 레코드를 먼저 열어 스키마를 확인**하라.
2. **기준선 category_id ↔ trainID 이동량**: 0 이라고 가정하지 말고 `check_label_convention`
   두 방법(JSON 히스토그램 + 변환 PNG vs 우리 GT 이동량)으로 확정한다. 0 이 아니면
   `rle_json_to_png --label_offset` 로만 보정(추측 금지, 기본은 에러).
3. **GT 해상도(재현 검산)**: detectron2 DELIVER 로더(`create_deliver_gt_sem_seg_loading_fn`)가
   raw 1~25 → 0~24 로 변환하므로 evaluator 가 보는 GT 는 0~24 여야 한다. 우리 GT 덤프는
   native 1042² 다. 기준선 evaluator 가 GT/pred 를 1024²로 리사이즈해 채점했다면 재현 mIoU
   가 어긋날 수 있다 — 이때 `rle_json_to_png` 는 pred 를 GT 해상도로 리샘플하고 그 사실을
   summary 에 남기니, **기준선의 GT 읽기 해상도를 확인해 보고**하라.
4. **probe 모듈 이름**(depth head/token, cond token, cross-attn, CAA)은 저장소마다 다르다 —
   `--list-modules` 로 확인 후 정규식을 지정. 못 찾으면 명확한 에러로 멈춘다(임의 층 선택 금지).
5. **`d2_zero_modality.patch` 의 batched_inputs key 이름**(image/depth/hha/lidar/event)은 미확인 —
   첫 실행 print 로 실제 key 를 확인하고 `_bf_keys` 를 조정하라.
6. **기대 장수**: val 2005 / test 1897 을 상수로 둔다(설계서 D2). 데이터셋 등록이 다르면
   `--expected_count` 로 덮어쓴다. 실제 RGB 파일 수와 대조해 확인하라.
7. 이 작업은 **코드·스모크만** 작성했다 — 학습·원격 서버 실행·git 커밋은 하지 않았다.
