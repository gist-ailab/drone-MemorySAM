#!/bin/bash
# Wait until 4 GPUs are free on this server, then resume DGFusion from the latest checkpoint.
# Free = memory.used <= 2000 MiB and utilization <= 10 %, on two checks 20 s apart (same set).
set -uo pipefail
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd /SSDb/jemo_maeng/dgfusion_train
export PYTHONPATH=$PWD/OneFormer:$PWD
export WANDB_MODE=offline
export DETECTRON2_DATASETS=$PWD/datasets
mkdir -p logs

free_set() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 <= 2000 && $3 <= 10 {print $1}' | head -4 | paste -sd, -
}

echo "$(date '+%F %T') waiting for 4 free GPUs..."
while true; do
  A=$(free_set)
  if [ "$(echo "$A" | tr ',' '\n' | grep -c .)" -ge 4 ]; then
    sleep 20
    B=$(free_set)
    if [ "$A" = "$B" ]; then break; fi
  fi
  sleep 30
done

TS=$(date +%Y%m%d_%H%M%S)
echo "$(date '+%F %T') launching DGFusion resume on GPUs $A (log logs/dgfusion_deliver_resume_yeon_$TS.log)"
CUDA_VISIBLE_DEVICES=$A python train_net.py --dist-url tcp://127.0.0.1:50368 --num-gpus 4 \
    --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
    --resume OUTPUT_DIR output/dgfusion_swin_tiny_bs8_200k_deliver_clde \
    2>&1 | tee logs/dgfusion_deliver_resume_yeon_$TS.log
