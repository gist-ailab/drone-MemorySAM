#!/usr/bin/env bash
# One-command certification evaluation for the D1 ViT-S+ real-time detector.
# Prints classes + modalities, waits for ENTER, then streams per-image inference +
# GT comparison, and ends with the mAP50 and FPS report.
#
#   bash certification/run_cert.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]
#
# DATA_ROOT = the poongsan_v2 mount on THIS machine (default = jarvis path).
# GPU       = CUDA device index (default 0).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CKPT="${1:?usage: run_cert.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]}"
DATA_ROOT="${2:-/SSDd/jemo_maeng/dset/poongsan_v2}"
GPU="${3:-0}"

# DINOv3 backbone needs timm>=1.0 (this fleet ships it under pylibs_p34 on some boxes)
[ -d /SSDb/jemo_maeng/pylibs_p34 ] && export PYTHONPATH="/SSDb/jemo_maeng/pylibs_p34:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # tensorboard/protobuf guard

cd "$REPO"
python certification/cert_eval.py \
    --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt "$CKPT" --data-root "$DATA_ROOT" \
    --out runs/cert_D1 --gpu "$GPU"
