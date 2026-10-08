#!/bin/bash
# 서버 전용 실행 기록(jarvis, 2026-10-08)
# Q2noDeg(DRN-260926-56 원천분리: DEGRADE_P=0, 증류만) 시드821 ep40 val-best top1 legal v2 val+test
set -uo pipefail
cd /SSDb/jemo_maeng/src/drone-MemorySAM
export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:/SSDb/jemo_maeng/src/drone-MemorySAM/semseg/models/sam2:/SSDb/jemo_maeng/src/drone-MemorySAM
PY=/home/jemo_maeng/miniconda3/envs/MMSS_SAM/bin/python3.10
CFG=configs/q2nodeg_s821_eval1024_jarvis.yaml
CKPT=outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noDeg/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.87_top1_checkpoint.pth
mkdir -p logs

echo "=== legal v2 val (Q2noDeg s821) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode val --model_path "$CKPT" 2>&1 | tee logs/q2nodeg_s821_legalv2_val_20261006.log

echo "=== legal v2 test (Q2noDeg s821) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/q2nodeg_s821_legalv2_test_20261006.log

echo "LEGALV2_DONE exit=$?"
