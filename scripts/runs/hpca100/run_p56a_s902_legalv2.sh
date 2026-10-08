#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# P56-A(모달 충돌 학습) 시드821 — ep40 val-best top1 legal v2 재채점(val+test)
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/p56a_s902_eval1024_hpca100.yaml
CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_P56A/DELIVER_ReliaDINO-ViTL16_idel/epoch20_66.73_top1_checkpoint.pth
mkdir -p logs

echo "=== harness guard check ==="
$PY tools/eval_harness_guard.py --check

echo "=== legal v2 val ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode val --model_path "$CKPT" 2>&1 | tee logs/p56a_s902_legalv2_val_20261004.log

echo "=== legal v2 test ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/p56a_s902_legalv2_test_20261004.log

echo "LEGALV2_DONE exit=$?"
