#!/bin/bash
# 61케이스 x 체크포인트 (a)(b) = 8샤드를 bengio GPU 8장(0~7)에 배치한다.
# (a) s1..s4 -> GPU 0,1,2,3 / (b) s1..s4 -> GPU 4,5,6,7 (한 번에 8장).
# 샤드 = 난수 그룹 단위(s1 clean+EMM+NM 19 · s2 RMM.25 · s3 RMM.5 · s4 RMM.75, 각 14) — 그룹을 쪼개면 마스크가 달라진다.
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
mapfile -t LINES < tools_bf/shards.txt
run() {  # WHICH IDX GPU
  echo "GPU=$3 WHICH=$1 TAG=full_$1_s$2 ONLY=\"${LINES[$(($2-1))]}\" bash tools_bf/run_rb_shard.sh 2>&1 | tee robust_out/full_$1_s$2_master.log"
}
start() {  # 세션이름 명령...
  local s=$1; shift
  tmux new-session -d -s "$s" "cd /SSDb/jemo_maeng/dgfusion_train && $*"
  echo "$(date '+%F %T') 기동 $s"
}
start rb_a_s1 "$(run a 1 0)"
start rb_a_s2 "$(run a 2 1)"
start rb_a_s3 "$(run a 3 2)"
start rb_a_s4 "$(run a 4 3)"
start rb_b_s1 "$(run b 1 4)"
start rb_b_s2 "$(run b 2 5)"
start rb_b_s3 "$(run b 3 6)"
start rb_b_s4 "$(run b 4 7)"
tmux ls
