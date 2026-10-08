#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# P56-A(모달 충돌 학습) 시드821 ep40 val-best top1 — 강건성(EMM/RMM/NM) + DELIVER 조건별 breakdown
# oracle_headroom_probe.py --by_condition (windows/null/emm_max_missing 미사용 → 순수 mme.main() 전달)
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop/semseg/models/sam2
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
CFG=configs/p56a_s821_eval1024_hpca100.yaml
CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.05_top1_checkpoint.pth
OUT=analysis_out/p56a_s821_robust_cond_20261004
mkdir -p "$OUT" logs

$PY tools/oracle_headroom_probe.py \
  --cfg "$CFG" --model_path "$CKPT" --split test --out "$OUT" \
  --protocol all --nm_gaussian --nm_gaussian_std 0.2 --seed 0 \
  --by_condition \
  2>&1 | tee logs/p56a_s821_robust_cond_20261004.log

echo "ROBUST_COND_DONE exit=$?"
