#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
# P56-C(환경조건부 LoRA 전문가 혼합) 시드821 ep40 완주, val-best top1(epoch30_66.49) legal v2 val+test — yeon GPU0
set -uo pipefail
R=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
cd "$R" || exit 1
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$R:$R/semseg/models/sam2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

CFG=configs/p56c_s821_eval1024_yeon.yaml
CKPT=outputs/ReliaDINO/yeon_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56C/DELIVER_ReliaDINO-ViTL16_idel/epoch30_66.49_top1_checkpoint.pth

echo "=== legal v2 val (P56-C s821) ==="
python tools/legal_rescore_v2.py --cfg "$CFG" --mode val --model_path "$CKPT" 2>&1 | tee logs/p56c_s821_legalv2_val_20261006.log

echo "=== legal v2 test (P56-C s821) ==="
python tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/p56c_s821_legalv2_test_20261006.log

echo "LEGALV2_DONE exit=$?"
