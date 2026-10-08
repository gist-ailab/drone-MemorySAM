#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# ③ 실제 케이스별(DELIVER 10 조건) clean mIoU — P56-A s821 vs Q2 s821, 같은 도구 한 번에 비교
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
A_CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.05_top1_checkpoint.pth
Q2_CKPT=/home/jovyan/SSDb/jemo_maeng/robust_0923/ckpts/Q2_s821_epoch40_67.0.pth
OUT=analysis_out/p56a_q2_s821_percase_20261004
mkdir -p "$OUT" logs

$PY tools/eval_per_domain.py \
  --cfg "$CFG" \
  --ckpt P56A=$A_CKPT \
  --ckpt Q2=$Q2_CKPT \
  --dataset-root /home/jovyan/SSDb/jemo_maeng/dset/DELIVER \
  --conditions cloud,fog,night,rain,sun,motionblur,overexposure,underexposure,lidarjitter,eventlowres \
  --split test --gpu 1 --batch 8 --out-dir "$OUT" --repo . \
  --val-script tools/legal_rescore_v2.py \
  2>&1 | tee logs/p56a_q2_s821_percase_20261004.log

echo "PERCASE_DONE exit=$?"
