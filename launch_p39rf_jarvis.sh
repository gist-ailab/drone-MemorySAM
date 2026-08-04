#!/bin/bash
set -x
source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
cd /home/jemo_maeng/src/drone-MemorySAM-p39rf
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6,7
export DET_GRAD_CLIP=0.1
nohup torchrun --standalone --nproc_per_node=2 --master_port=29770 train_det.py \
  --cfg configs/det/det_P39rf_trunkexp_jarvis.yaml > logs/p39rf_jarvis.log 2>&1 &
echo "LAUNCHED PID: $!"
disown
