#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
# Q2noKD(DRN-261005-01) 시드821 — Q2 레시피에서 교사 증류만 끈 원천분리 ablation, yeon GPU1
set -uo pipefail
cd /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
export PYTHONPATH=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop:/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop/semseg/models/sam2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

/home/jemo_maeng/anaconda3/envs/MMSS_SAM/bin/torchrun --standalone --nproc_per_node=1 --master_port=29913 \
  train_reliadino.py --cfg configs/yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD.yaml \
  > logs/q2nokd_s821_launch_20261006.log 2>&1
