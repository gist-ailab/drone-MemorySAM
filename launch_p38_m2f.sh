#!/bin/bash
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export HF_HUB_DISABLE_XET=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p logs/hpca100-deliver_rgbdel_P38_m2f
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/hpca100-deliver_rgbdel_P38_m2f/run_${TS}.log
echo "$LOG" > /tmp/p38_m2f_logpath.txt
exec /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/torchrun --nproc_per_node=4 --master_port=29534 train_reliadino.py --cfg configs/hpca100-deliver_rgbdel_P38_m2f.yaml > $LOG 2>&1 < /dev/null
