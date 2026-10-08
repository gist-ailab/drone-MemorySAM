#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# P56-B(거리 조건화 attention) 시드821 ep40 완주, val-best top1(epoch25_66.9) legal v2 val+test
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/p56b_s821_eval1024_hpca100.yaml
CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56B/DELIVER_ReliaDINO-ViTL16_idel/epoch25_66.9_top1_checkpoint.pth
mkdir -p logs

echo "=== legal v2 val (P56-B s821) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode val --model_path "$CKPT" 2>&1 | tee logs/p56b_s821_legalv2_val_20261005.log

echo "=== legal v2 test (P56-B s821) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/p56b_s821_legalv2_test_20261005.log

echo "LEGALV2_DONE exit=$?"
