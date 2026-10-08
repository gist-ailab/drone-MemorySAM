#!/usr/bin/env bash
cd /SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
TS=$(date +%Y%m%d_%H%M%S)
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,3,5,6,7 \
torchrun --nproc_per_node=6 --master_port=29532 \
  train_det.py --cfg configs/det/det_P31_egofill_bengio.yaml \
  > logs/det_P31_egofill_6gpu_${TS}.log 2>&1
