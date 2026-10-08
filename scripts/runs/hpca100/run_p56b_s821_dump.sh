#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# 원거리 진단(판정 세션 지시 2026-10-05): P56-B 시드821 val-best(ep25 top1) test 예측 덤프, GPU2
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/p56b_s821_eval1024_hpca100.yaml
CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56B/DELIVER_ReliaDINO-ViTL16_idel/epoch25_66.9_top1_checkpoint.pth
OUT=analysis_out/p56b_s821_dump_20261005
mkdir -p "$OUT" logs

$PY tools/baseline_failure/dump_preds_ours.py --cfg "$CFG" --model_path "$CKPT" --split test --out "$OUT" --batch 8 2>&1 | tee logs/p56b_s821_dump_20261005.log

echo "DUMP_DONE exit=$?"
