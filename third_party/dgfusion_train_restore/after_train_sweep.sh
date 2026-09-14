#!/bin/bash
# Wait for a detectron2 training run to finish, then test-evaluate the given checkpoints in parallel
# on whatever GPUs are free at that moment (one tmux window per GPU, round-robin tag assignment).
# Usage: bash after_train_sweep.sh <repo_dir> <run_dir> <train_cfg_pattern> <sweep_script> <tag> [<tag> ...]
#   run_dir           : output dir holding model_<tag>.pth and model_final.pth (relative to repo_dir)
#   train_cfg_pattern : string that appears in the training process command line (to detect exit)
#   sweep_script      : bash script called as `<sweep_script> <gpu> <tag> ...`
set -uo pipefail
REPO=$1; RUN=$2; PAT=$3; SWEEP=$4; shift 4
TAGS="$*"
cd "$REPO"
echo "$(date '+%F %T') waiting for training ($PAT) to finish; tags: $TAGS"
while [ ! -f "$RUN/model_final.pth" ] || pgrep -f "train_net.py.*$PAT" >/dev/null; do sleep 15; done
echo "$(date '+%F %T') training finished"
FREE=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
         | awk -F', ' '$2 <= 2000 && $3 <= 10 {print $1}' | head -4)
N=$(echo "$FREE" | grep -c .)
if [ "$N" -eq 0 ]; then echo "$(date '+%F %T') no free GPU, aborting"; exit 1; fi
declare -A Q
i=0
for t in $TAGS; do
  g=$(echo "$FREE" | sed -n "$((i % N + 1))p")
  Q[$g]="${Q[$g]:-} $t"
  i=$((i + 1))
done
for g in "${!Q[@]}"; do
  tmux new-window -d -t jemo -n "sweep_g$g" "bash $SWEEP $g ${Q[$g]}; exec bash"
  echo "$(date '+%F %T') GPU $g:${Q[$g]}"
done
