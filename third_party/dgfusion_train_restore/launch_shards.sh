#!/bin/bash
# 61케이스를 난수 그룹 4개(clean+EMM+NM 19 · RMM.25 14 · RMM.5 14 · RMM.75 14)로 나눠 각각 tmux 세션에서 돌린다.
# RMM 은 비율마다 생성기 하나를 14조합이 이어 쓰므로 한 그룹을 쪼개면 마스크가 단일 프로세스 실행과 달라진다 → 그룹 단위 샤딩.
# 사용: bash launch_shards.sh <a|b> <GPU 4개를 공백으로>   예) bash launch_shards.sh a 1 2 4 5
set -uo pipefail
WHICH=${1:?a|b}; shift
GPUS=("$@")
[ "${#GPUS[@]}" -eq 4 ] || { echo "GPU 4개를 줘라(그룹 4개)"; exit 2; }
cd /SSDb/jemo_maeng/dgfusion_train
i=0
while IFS= read -r line; do
  i=$((i+1)); g=${GPUS[$((i-1))]}
  S="rb_${WHICH}_s${i}"
  tmux new-session -d -s "$S" "cd /SSDb/jemo_maeng/dgfusion_train && GPU=$g WHICH=$WHICH TAG=full_${WHICH}_s${i} ONLY=\"$line\" bash tools_bf/run_rb_shard.sh 2>&1 | tee robust_out/full_${WHICH}_s${i}_master.log"
  echo "$(date '+%F %T') 기동 $S GPU=$g ($(echo "$line" | tr ';' '\n' | wc -l)케이스)"
done < tools_bf/shards.txt
