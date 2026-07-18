#!/bin/bash
cd /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,5,6,7
export DET_GRAD_CLIP=0.1                              # DETR-standard clip (collapse fix)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 22.9GB peak on 24GB — anti-frag
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
echo "launch p37a $TS on GPUs $CUDA_VISIBLE_DEVICES" >> logs/p37a_launch.log
torchrun --standalone --nproc_per_node=4 --master_port=29711 train_det.py \
    --cfg configs/det/det_P37a_cefr_yeon.yaml > logs/p37a_cefr_${TS}.log 2>&1
