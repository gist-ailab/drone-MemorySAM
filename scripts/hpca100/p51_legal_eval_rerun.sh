#!/bin/bash
# P51-CMLC on/off — full legal re-eval (overall + per-class + per-condition), v2.
# v1 crashed 100% (ModuleNotFoundError: sam2) — PYTHONPATH was missing on hpca100.
# Usage: CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1> bash p51_legal_eval_rerun.sh
set -uo pipefail

REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
VENV_PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
export PYTHONPATH="$REPO/semseg/models/sam2:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$REPO"

ON_CKPT="$REPO/outputs/ReliaDINO/hpca100_deliver_rgbdel_P51_cmlc_on/DELIVER_ReliaDINO-ViTL16_idel/epoch30_66.41_top1_checkpoint.pth"
OFF_CKPT="$REPO/outputs/ReliaDINO/hpca100_deliver_rgbdel_P51_cmlc_off/DELIVER_ReliaDINO-ViTL16_idel/epoch54_67.15_top1_checkpoint.pth"
ON_CFG=configs/eval/hpca100-deliver_rgbdel_P51_cmlc_on_eval1024.yaml
OFF_CFG=configs/eval/hpca100-deliver_rgbdel_P51_cmlc_off_eval1024.yaml
OUT=$REPO/logs/p51_legal_eval_v2
mkdir -p "$OUT"

echo "[p51-eval-v2] (a) overall legal test for on"
"$VENV_PY" val.py --cfg "$ON_CFG" --mode test --model_path "$ON_CKPT" > "$OUT/overall_on.log" 2>&1
grep -A100 "Class" "$OUT/overall_on.log" | tail -40

echo "[p51-eval-v2] (a) overall legal test for off"
"$VENV_PY" val.py --cfg "$OFF_CFG" --mode test --model_path "$OFF_CKPT" > "$OUT/overall_off.log" 2>&1
grep -A100 "Class" "$OUT/overall_off.log" | tail -40

echo "[p51-eval-v2] (b) per-condition legal test for on (cloud/fog/night/rain/sun)"
"$VENV_PY" tools/eval_per_domain.py --cfg "$ON_CFG" --ckpt "best=$ON_CKPT" \
  --dataset-root /home/jovyan/SSDb/jemo_maeng/dset/DELIVER --split test \
  --out-dir "$OUT/perdomain_on" > "$OUT/perdomain_on.log" 2>&1
tail -20 "$OUT/perdomain_on.log"

echo "[p51-eval-v2] (b) per-condition legal test for off (cloud/fog/night/rain/sun)"
"$VENV_PY" tools/eval_per_domain.py --cfg "$OFF_CFG" --ckpt "best=$OFF_CKPT" \
  --dataset-root /home/jovyan/SSDb/jemo_maeng/dset/DELIVER --split test \
  --out-dir "$OUT/perdomain_off" > "$OUT/perdomain_off.log" 2>&1
tail -20 "$OUT/perdomain_off.log"

echo "[p51-eval-v2] === DONE ==="
