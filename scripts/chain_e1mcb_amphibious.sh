#!/bin/bash
# DRN-261102-01 (user 2026-11-02): on amphibious-demo, wait until the running L8 ablation frees GPU0, then train the
# MCubeS DINOv3-B teacher (E1Mc-B) for seeds 20260827 and 20260828 one after the other, and after each run dump the
# whole 102-image MCubeS test split (loader split 'val') with tools/qualbank_dump.py for the paper's qualbank.
# Start (from the repo root on amphibious-demo; the host has no tmux/screen, user systemd has Linger=no):
#   setsid bash scripts/chain_e1mcb_amphibious.sh > logs/chain_e1mcb.log 2>&1 < /dev/null &
# State: logs/chain_e1mcb.status (one line per step), logs/chain_e1mcb.pid. Stop: kill $(cat logs/chain_e1mcb.pid)
set -u
cd "$(dirname "$0")/.."
PY=/home/ailab/workspace/jemo_maeng/venvs/dms/bin/python
ROOT=/home/ailab/workspace/jemo_maeng/dset/MCubeS
QB=/home/ailab/workspace/jemo_maeng/qualbank/mcubes_test
ST=logs/chain_e1mcb.status
mkdir -p logs "$QB"
echo $$ > logs/chain_e1mcb.pid
st() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
export PYTHONPATH="$PWD/semseg/models/sam2:$PWD"

free_gpu() {   # GPU0 empty by the project rule (<= 2000 MiB and <= 10 %)
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 |
    awk -F', ' '{exit !($1 <= 2000 && $2 <= 10)}'
}
st "WAIT for GPU0 (L8 ablation running)"
while true; do
  if free_gpu; then sleep 120; free_gpu && break; fi
  sleep 300
done
st "GPU0 free"

for seed in 20260827 20260828; do
  cfg=configs/amphibious-mcubes_rgbadn_P39_1_rank_E1McB_seed${seed}.yaml
  name=$(basename "$cfg" .yaml)
  save=outputs/ReliaDINO/amphibious_mcubes_rgbadn_P39_1_rank_E1McB_seed${seed}
  mkdir -p "logs/$name"
  for attempt in 1 2 3; do              # first run + at most 2 AUTO_RESUME retries (project rule)
    log="logs/$name/${name}_$(date +%Y%m%d_%H%M%S).log"
    st "TRAIN seed=$seed attempt=$attempt log=$log"
    # TRAIN.DDP is true in the config (same recipe as the hpca100 E1Mc teacher) -> launch through torchrun with
    # one process; plain `python train_reliadino.py` fails with "Default process group has not been initialized"
    # (2026-11-04 04:41, all attempts).
    CUDA_VISIBLE_DEVICES=0 $PY -m torch.distributed.run --standalone --nproc_per_node=1 \
      train_reliadino.py --cfg "$cfg" > "$log" 2>&1
    rc=$?
    tl=$(ls "$save"/*/train.log 2>/dev/null | head -1)
    if [ -n "$tl" ] && grep -q "Total Training Time" "$tl"; then st "TRAIN DONE seed=$seed rc=$rc"; break; fi
    st "TRAIN EXIT seed=$seed rc=$rc without 'Total Training Time'"
    [ "$attempt" = 3 ] && { st "GIVE UP seed=$seed after 2 retries"; continue 2; }
    sleep 60
  done
  ck=$(ls "$save"/*/epoch*_top1_checkpoint.pth 2>/dev/null | head -1)
  if [ -z "$ck" ]; then st "NO top1 checkpoint seed=$seed"; continue; fi
  exp=$(basename "$ck" | sed -E 's/^epoch[0-9]+_([0-9.]+)_top1.*/\1/')
  md5sum "$ck" | tee -a "$ST"
  st "DUMP seed=$seed ckpt=$ck expected=$exp"
  CUDA_VISIBLE_DEVICES=0 $PY tools/qualbank_dump.py --cfg "$cfg" --model_path "$ck" --split val --root "$ROOT" \
     --out "$QB/E1McB_s${seed}" --expected-miou "$exp" > "logs/$name/qualbank_dump.log" 2>&1
  st "DUMP rc=$? seed=$seed -> $QB/E1McB_s${seed}"
done
st "CHAIN END"
