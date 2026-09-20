#!/bin/bash
# 원거리 손해의 원인 진단 세 표. GPU 불필요, 저장된 예측 PNG·GT·원본 depth 만 읽는다.
# E17 덤프가 이미 회수돼 있으면 6모델, 아니면 5모델로 돈다.
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
OUT=$R/reports/far_range_diag_20260919
LOG=/home/jemo/.claude/jobs/f92837ca/tmp/far_diag.log
mkdir -p "$OUT"
cd "$REPO"

M=(ours="$R/preds/ours_E1conf_s1/test/pred"
   ours_e13s3="$R/preds/ours_E13conf_s3/test/pred"
   ours_p46base="$R/preds/ours_P46base/test/pred"
   dgf="$R/preds/dgf80k/test/pred"
   caf="$R/preds/caf/test/pred")

E17DIR=$(find "$R/preds/ours_E17screen_s1" -type d -name pred 2>/dev/null | head -1)
if [ -n "${E17DIR:-}" ] && [ "$(find "$E17DIR" -name '*.png' 2>/dev/null | wc -l)" -ge 1897 ]; then
  M+=(ours_e17="$E17DIR")
  echo "E17 포함(6모델)"
else
  echo "E17 아직 없음(5모델)"
fi

{
  echo "$(date '+%F %T') 시작 — 모델 ${#M[@]}개"
  "$PY" -u tools/baseline_failure/far_range_diag.py \
    --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
    --models "${M[@]}" \
    --edges_json "$R/reports/depth_bin_iou_20260919/depth_bin_iou_test.json" \
    --split test --out "$OUT"
  echo "$(date '+%F %T') 종료 exit=$?"
} > "$LOG" 2>&1
tail -8 "$LOG"
ls -la "$OUT"
