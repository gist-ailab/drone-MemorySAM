#!/bin/bash
# 서버 전용 실행 기록(jarvis, 2026-10-08)
# P56-C(환경조건부 LoRA 전문가 혼합) 시드902 ep40 val-best top1 legal v2(nearest-exact) val+test
# ckpt = yeon 완주분을 jarvis로 임포트(md5 34b8a4c101410546d48e6d5a8ef88a5f 확인).
set -uo pipefail
cd /SSDb/jemo_maeng/src/drone-MemorySAM
export CUDA_VISIBLE_DEVICES=__GPU__
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:/SSDb/jemo_maeng/src/drone-MemorySAM/semseg/models/sam2:/SSDb/jemo_maeng/src/drone-MemorySAM
PY=/home/jemo_maeng/miniconda3/envs/MMSS_SAM/bin/python3.10
CFG=configs/p56c_s902_eval1024_jarvis.yaml
CKPT=outputs/ReliaDINO/p56c_s902_imported/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.41_top1_checkpoint.pth
mkdir -p logs

echo "=== legal v2 val (P56-C s902) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode val --model_path "$CKPT" 2>&1 | tee logs/p56c_s902_legalv2_val_20261006.log

echo "=== legal v2 test (P56-C s902) ==="
$PY tools/legal_rescore_v2.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee logs/p56c_s902_legalv2_test_20261006.log

echo "LEGALV2_DONE exit=$?"
