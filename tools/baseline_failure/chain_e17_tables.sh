#!/bin/bash
# E17(고해상도 세부 가지) 덤프가 끝나면 예측을 공유 루트로 회수하고, E17 을 넣어 표 네 벌을
# 다시 만든다: A6 구간별·구간×클래스, 셀 지도(셀×클래스·묶음 요약·연결 성분), 그리고
# 원거리 진단 세 표.
# 종료 판정은 프로세스 이름이 아니라 산출 장수로 한다(1,897 장). 이름으로 판정하면 원격에
# 보내는 명령줄 자체가 잡혀 자기 자신을 세게 된다(2026-09-19 실제 사고).
# 덤프는 감시 세션이 돌린 것을 쓴다. 같은 체크포인트(md5 eedbd198…)로 둘이 겹쳐 돌아
# 진행이 뒤진 내 쪽을 중단했다.
set -uo pipefail
PY=/home/jemo/anaconda3/envs/MMSS_SAM/bin/python
REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize
R=/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis/baseline_failure_20260917
HP=/tmp/jemo_scratch/dump_E17screen

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
echo "$(date '+%F %T') E17 예측 = $E17DIR ($(find "$E17DIR" -name '*.png' | wc -l) 장)"

M=(ours="$R/preds/ours_E1conf_s1/test/pred"
   ours_e13s3="$R/preds/ours_E13conf_s3/test/pred"
   ours_e17="$E17DIR"
   ours_p46base="$R/preds/ours_P46base/test/pred"
   dgf="$R/preds/dgf80k/test/pred"
   caf="$R/preds/caf/test/pred")

cd "$REPO"
A6OUT=$R/reports/depth_bin_iou_20260919
CELLOUT=$R/reports/cell_map_20260919
FAROUT=$R/reports/far_range_diag_20260919

echo "$(date '+%F %T') A6 재실행(E17 포함)"
"$PY" -u tools/baseline_failure/depth_bin_iou.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" --split test --out "$A6OUT" --bins 5 2>&1 | tail -4

echo "$(date '+%F %T') 셀 지도 재실행(E17 포함)"
"$PY" -u tools/baseline_failure/cell_map.py \
  --gt "$R/preds/ours_gt/test/gt" --models "${M[@]}" \
  --ref ours --baselines dgf caf --split test --out "$CELLOUT" 2>&1 | tail -6

# 원거리 진단은 앞선 5모델 실행이 아직 돌고 있으면 끝나기를 기다린다. 같은 출력 디렉터리를
# 두 프로세스가 함께 쓰면 표가 섞인다.
echo "$(date '+%F %T') 원거리 진단 5모델 실행 종료 대기"
for i in $(seq 1 240); do
  pgrep -f "[f]ar_range_diag.py" > /dev/null 2>&1 || break
  sleep 60
done
echo "$(date '+%F %T') 원거리 진단 재실행(E17 포함)"
"$PY" -u tools/baseline_failure/far_range_diag.py \
  --gt "$R/preds/ours_gt/test/gt" --depth_root /ailab_mat2/dataset/DELIVER \
  --models "${M[@]}" --edges_json "$A6OUT/depth_bin_iou_test.json" \
  --split test --out "$FAROUT" 2>&1 | tail -6
echo "$(date '+%F %T') 전체 종료"
