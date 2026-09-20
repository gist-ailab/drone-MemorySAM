#!/bin/bash
# 시스템 메모리가 넉넉해지면 E17 6모델 표의 남은 두 단계를 돌린다.
#
# 왜 기다리는가. 이 도구들은 하나가 12~14GB 를 쓴다. 앞서 두 번 메모리 부족으로 정리됐다
# (2026-09-20). 다른 세션 작업과 겹치면 또 죽으므로, 여유가 충분히 생긴 뒤에 들어간다.
# 1 단계(A6 구간별·구간×클래스)는 이미 6 모델로 끝나 있어 다시 돌리지 않는다.
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
LOG=/home/jemo/.claude/jobs/f92837ca/tmp/e17_tables_resume.log
NEED_GB=${NEED_GB:-24}      # 도구 하나가 14GB 까지 올라가므로 여유를 두고 잡는다
STREAK=${STREAK:-3}         # 일시적 반등에 속지 않도록 연속 확인

avail() { free -g | awk 'NR==2{print $7}'; }

{
echo "$(date '+%F %T') 메모리 여유 ${NEED_GB}GB 가 ${STREAK}회 연속 확인될 때까지 대기"
ok=0
for i in $(seq 1 720); do     # 최대 24 시간
  a=$(avail)
  if [ "${a:-0}" -ge "$NEED_GB" ]; then
    ok=$((ok + 1))
    echo "$(date '+%F %T') 여유 ${a}GB (연속 ${ok}/${STREAK})"
    [ "$ok" -ge "$STREAK" ] && break
  else
    [ "$ok" -gt 0 ] && echo "$(date '+%F %T') 여유 ${a}GB — 연속 끊김"
    ok=0
  fi
  sleep 120
done
if [ "$ok" -lt "$STREAK" ]; then
  echo "$(date '+%F %T') 24 시간 안에 여유가 확보되지 않았다 — 실행하지 않는다"
  exit 1
fi

cd "$REPO"
M=(ours="$R/preds/ours_E1conf_s1/test/pred"
   ours_e13s3="$R/preds/ours_E13conf_s3/test/pred"
   ours_e17="$R/preds/ours_E17screen_s1/test/pred"
   ours_p46base="$R/preds/ours_P46base/test/pred"
   dgf="$R/preds/dgf80k/test/pred"
   caf="$R/preds/caf/test/pred")

echo "$(date '+%F %T') [2/3] 셀 지도 시작 (6 모델)"
"$PY" -u tools/baseline_failure/cell_map.py \
  --gt "$R/preds/ours_gt/test/gt" --models "${M[@]}" \
  --ref ours --baselines dgf caf --split test \
  --out "$R/reports/cell_map_20260919"
echo "$(date '+%F %T') [2/3] 종료 exit=$?"

echo "$(date '+%F %T') [3/3] 원거리 진단 시작 (6 모델)"
"$PY" -u tools/baseline_failure/far_range_diag.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" \
  --edges_json "$R/reports/depth_bin_iou_20260919/depth_bin_iou_test.json" \
  --split test --out "$R/reports/far_range_diag_20260919"
echo "$(date '+%F %T') [3/3] 종료 exit=$?"
echo "$(date '+%F %T') 전체 종료"
} > "$LOG" 2>&1
tail -12 "$LOG"
