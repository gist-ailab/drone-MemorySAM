#!/bin/bash
# 세 표(구간×클래스 · 셀 지도/연결 성분 · 원거리 진단)를 임의의 모델 묶음으로 만든다.
# R1(depth 경계 prior)·R2(성분 손실) 스크린 덤프가 오면 이 스크립트에 모델만 더해 부른다.
#
# 사용:
#   MODELS="ours=<경로> r1=<경로> r2=<경로>" OUT_TAG=r1r2_40ep bash run_three_tables.sh
# 모델 인자는 "이름=예측PNG루트" 를 공백으로 이어 붙인 한 덩어리다.
#
# 🔴 반드시 순차로 돈다. 도구 하나가 12~14GB 를 쓰고, 둘을 동시에 띄웠다가 시스템 메모리
# 부족으로 정리된 적이 있다(2026-09-20). 시작 전에 여유도 확인한다.
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
GT=$R/preds/ours_gt/test/gt
DEPTH_ROOT=/ailab_mat2/dataset/DELIVER
NEED_GB=${NEED_GB:-24}
OUT_TAG=${OUT_TAG:?OUT_TAG 를 지정하라 (예: r1r2_40ep)}
MODELS=${MODELS:?MODELS 를 지정하라 (예: "ours=/경로/pred r1=/경로/pred")}
REF=${REF:-ours}
BASELINES=${BASELINES:-"dgf caf"}
LOG=/home/jemo/.claude/jobs/f92837ca/tmp/three_tables_${OUT_TAG}.log

read -r -a M <<< "$MODELS"
A6OUT=$R/reports/depth_bin_iou_${OUT_TAG}
CELLOUT=$R/reports/cell_map_${OUT_TAG}
FAROUT=$R/reports/far_range_diag_${OUT_TAG}
mkdir -p "$A6OUT" "$CELLOUT" "$FAROUT"
cd "$REPO"

avail() { free -g | awk 'NR==2{print $7}'; }

{
echo "$(date '+%F %T') 모델 ${#M[@]}개: ${M[*]%%=*}"
echo "$(date '+%F %T') 메모리 여유 ${NEED_GB}GB 대기"
ok=0
for i in $(seq 1 720); do
  a=$(avail)
  if [ "${a:-0}" -ge "$NEED_GB" ]; then
    ok=$((ok + 1)); [ "$ok" -ge 3 ] && break
  else
    ok=0
  fi
  sleep 120
done
[ "$ok" -ge 3 ] || { echo "여유 확보 실패 — 실행하지 않는다"; exit 1; }

echo "$(date '+%F %T') [1/3] 구간별 + 구간×클래스"
"$PY" -u tools/baseline_failure/depth_bin_iou.py \
  --gt "$GT" --depth_root "$DEPTH_ROOT" --models "${M[@]}" \
  --split test --out "$A6OUT" --bins 5
echo "$(date '+%F %T') [1/3] exit=$?"

echo "$(date '+%F %T') [2/3] 셀 지도 + 연결 성분 분리"
"$PY" -u tools/baseline_failure/cell_map.py \
  --gt "$GT" --models "${M[@]}" --ref "$REF" --baselines $BASELINES \
  --split test --out "$CELLOUT"
echo "$(date '+%F %T') [2/3] exit=$?"

echo "$(date '+%F %T') [3/3] 원거리 진단(성분 크기×거리 · 경계 F · 오분류 전이)"
"$PY" -u tools/baseline_failure/far_range_diag.py \
  --gt "$GT" --depth_root "$DEPTH_ROOT" --models "${M[@]}" \
  --edges_json "$A6OUT/depth_bin_iou_test.json" --split test --out "$FAROUT"
echo "$(date '+%F %T') [3/3] exit=$?"
echo "$(date '+%F %T') 전체 종료 — 산출물: $A6OUT, $CELLOUT, $FAROUT"
} > "$LOG" 2>&1
tail -12 "$LOG"
