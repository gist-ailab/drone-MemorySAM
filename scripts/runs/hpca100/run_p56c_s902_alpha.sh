#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/hpca100-deliver_rgbdel_P46_c3only_seed20260902_screen40_P56C.yaml
CKPT=outputs/ReliaDINO/p56c_s902_judge20261007/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.41_top1_checkpoint.pth
mkdir -p logs analysis_logs/p56c_eval_20261007
echo "=== harness guard check ==="
$PY tools/eval_harness_guard.py --check
echo "=== p56c alpha stats seed902 ==="
$PY tools/p56c_alpha_stats.py --cfg "$CFG" --model_path "$CKPT" --split test --subset_every 5 --max_images 400 --out analysis_logs/p56c_eval_20261007/alpha_s902.json
echo "P56C_S902_ALPHA_DONE exit=$?"
