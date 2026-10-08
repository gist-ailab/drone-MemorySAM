#!/usr/bin/env bash
# MUSES x P34-ReliaDINO — DGFusion-aligned projection + 4 modalities, bengio 8x RTX 3090.
# Written locally and scp'd: ssh heredoc writes have silently failed on this box before.
set -euo pipefail

REPO=/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM
CFG=configs/muses_rgbelr_P34_dgf_4modal.yaml
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="$REPO/logs/muses_rgbelr_P34_dgf_4modal"
LOG="$LOGDIR/muses_rgbelr_P34_dgf_4modal_${TS}.log"

mkdir -p "$LOGDIR"
cd "$REPO"

# dinov3 lives in timm 1.0.24 here; the env's timm 0.4.12 cannot build the backbone.
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:${PYTHONPATH:-}
# wandb + tensorboard protobuf clash on this box
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PY=/home/jemo_maeng/anaconda3/envs/MMSS_SAM/bin

echo "log -> $LOG"
setsid nohup "$PY/torchrun" --standalone --nproc_per_node=8 --master_port=29688 \
    train_reliadino.py --cfg "$CFG" > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$LOGDIR/run.pid"
echo "launched pid=$PID  ts=$TS"
echo "$LOG" > "$LOGDIR/latest.log.path"
