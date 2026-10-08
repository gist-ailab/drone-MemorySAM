#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# Q2 시드821(대조군) ep40 top1 legal v2 test만 — P56-B 시드821과 클래스별 비교용(판정 세션 지시 2026-10-05)
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/q2_s821_eval1024_hpca100.yaml
CKPT=/home/jovyan/SSDb/jemo_maeng/robust_0923/ckpts/Q2_s821_epoch40_67.0.pth
mkdir -p logs

echo "=== legal v2 test (Q2 s821) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/q2_s821_legalv2_test_20261005.log

echo "LEGALV2_TEST_DONE exit=$?"
