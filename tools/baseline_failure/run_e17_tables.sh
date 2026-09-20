#!/bin/bash
# E17(고해상도 세부 가지) 예측을 회수해 표 네 벌을 6 모델로 다시 만든다.
# 🔴 반드시 순차로 돈다. 앞서 두 작업을 동시에 띄웠다가 시스템 메모리 부족으로 둘 다
# 정리된 적이 있다(2026-09-20). 도구 하나가 12~14GB 를 쓴다.
# E17 은 `test/pred`(1042)를 쓴다. GT 가 1042 이고 다른 우리 모델 예측도 1042 라 축이 맞는다.
# (`pred1024` 는 기준선 공식 프로토콜 쪽 좌표계이므로 여기서는 쓰지 않는다.)
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
HP=/tmp/jemo_scratch/dump_E17screen
LOG=/home/jemo/.claude/jobs/f92837ca/tmp/e17_tables.log
cd "$REPO"

{
echo "$(date '+%F %T') E17 예측 회수 시작"
mkdir -p "$R/preds/ours_E17screen_s1/test"
rsync -rt --no-perms --no-owner --no-group "hpca100:$HP/test/pred/" \
      "$R/preds/ours_E17screen_s1/test/pred/" || exit 1
N=$(find "$R/preds/ours_E17screen_s1/test/pred" -name '*.png' | wc -l)
echo "$(date '+%F %T') 회수 완료 — $N 장"
[ "$N" -ge 1897 ] || { echo "1,897 장에 못 미침 — 중단"; exit 1; }

M=(ours="$R/preds/ours_E1conf_s1/test/pred"
   ours_e13s3="$R/preds/ours_E13conf_s3/test/pred"
   ours_e17="$R/preds/ours_E17screen_s1/test/pred"
   ours_p46base="$R/preds/ours_P46base/test/pred"
   dgf="$R/preds/dgf80k/test/pred"
   caf="$R/preds/caf/test/pred")

A6OUT=$R/reports/depth_bin_iou_20260919
CELLOUT=$R/reports/cell_map_20260919
FAROUT=$R/reports/far_range_diag_20260919

echo "$(date '+%F %T') [1/3] A6(구간별 + 구간×클래스) 시작"
"$PY" -u tools/baseline_failure/depth_bin_iou.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" --split test --out "$A6OUT" --bins 5
echo "$(date '+%F %T') [1/3] 종료 exit=$?"

echo "$(date '+%F %T') [2/3] 셀 지도 시작"
"$PY" -u tools/baseline_failure/cell_map.py \
  --gt "$R/preds/ours_gt/test/gt" --models "${M[@]}" \
  --ref ours --baselines dgf caf --split test --out "$CELLOUT"
echo "$(date '+%F %T') [2/3] 종료 exit=$?"

echo "$(date '+%F %T') [3/3] 원거리 진단 시작"
"$PY" -u tools/baseline_failure/far_range_diag.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" --edges_json "$A6OUT/depth_bin_iou_test.json" \
  --split test --out "$FAROUT"
echo "$(date '+%F %T') [3/3] 종료 exit=$?"
echo "$(date '+%F %T') 전체 종료"
} > "$LOG" 2>&1
tail -12 "$LOG"
