#!/bin/bash
# E17 덤프가 끝나면 예측을 공유 루트로 회수하고, A6(구간×클래스)와 셀 지도를 E17 을 넣어
# 다시 만든다. 종료 판정은 프로세스 이름이 아니라 산출 장수로 한다(1,897 장).
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
HP=/tmp/jemo_scratch/dump_E17screen   # 감시 세션이 돌린 덤프(중복 방지로 내 것은 중단)

echo "$(date '+%F %T') E17 덤프 완료 대기(1,897장)"
for i in $(seq 1 300); do
  n=$(ssh hpca100 "find $HP -name '*.png' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
  [ "${n:-0}" -ge 1897 ] && break
  sleep 60
done
n=$(ssh hpca100 "find $HP -name '*.png' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
echo "$(date '+%F %T') 덤프 장수 = ${n:-0}"
if [ "${n:-0}" -lt 1897 ]; then echo "1,897 장에 못 미쳐 중단"; exit 1; fi

echo "$(date '+%F %T') 예측 회수: hpca100 -> 공유 루트"
mkdir -p "$R/preds/ours_E17screen_s1"
rsync -rt --no-perms --no-owner --no-group \
  "hpca100:$HP/" "$R/preds/ours_E17screen_s1/" || exit 1
E17DIR=$(find "$R/preds/ours_E17screen_s1" -type d -name pred | head -1)
echo "$(date '+%F %T') E17 예측 디렉터리 = $E17DIR ($(find "$E17DIR" -name '*.png' | wc -l) 장)"

M=(ours="$R/preds/ours_E1conf_s1/test/pred"
   ours_e13s3="$R/preds/ours_E13conf_s3/test/pred"
   ours_e17="$E17DIR"
   ours_p46base="$R/preds/ours_P46base/test/pred"
   dgf="$R/preds/dgf80k/test/pred"
   caf="$R/preds/caf/test/pred")

cd "$REPO"
A6OUT=$R/reports/depth_bin_iou_20260919
CELLOUT=$R/reports/cell_map_20260919

echo "$(date '+%F %T') A6 재실행(E17 포함)"
"$PY" -u tools/baseline_failure/depth_bin_iou.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" --split test --out "$A6OUT" --bins 5 2>&1 | tail -4

echo "$(date '+%F %T') 셀 지도 재실행(E17 포함)"
"$PY" -u tools/baseline_failure/cell_map.py \
  --gt "$R/preds/ours_gt/test/gt" --models "${M[@]}" \
  --ref ours --baselines dgf caf --split test --out "$CELLOUT" 2>&1 | tail -6
echo "$(date '+%F %T') 전체 종료"
