#!/usr/bin/env bash
# RBMA-on-SAM3 launcher (avoids multi-line copy/paste breaking the command).
#
# Single GPU (e.g. GPU 2):
#   CUDA_VISIBLE_DEVICES=2 bash run_sam3_rbma.sh
# Other config:
#   CUDA_VISIBLE_DEVICES=2 bash run_sam3_rbma.sh configs/b200-multiaqua_rgbtl_SAM3RBMA_hardaug8_physaug.yaml
# Multi-GPU (DDP):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash run_sam3_rbma.sh <cfg>
set -e

CFG="${1:-configs/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="semseg/models/sam3:${PYTHONPATH}"
NPROC="${NPROC:-1}"

# GPU 선택: CUDA_VISIBLE_DEVICES를 직접 주지 않으면 빈 GPU NPROC장을 자동 배정.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(bash "$(dirname "$0")/scripts/pick_free_gpus.sh" "$NPROC")" || {
    echo "[run_sam3_rbma] 빈 GPU ${NPROC}장을 찾지 못했습니다. nvidia-smi 확인 후 NPROC를 낮추거나 CUDA_VISIBLE_DEVICES를 직접 지정하세요." >&2
    exit 1
  }
  export CUDA_VISIBLE_DEVICES
fi

echo "[run_sam3_rbma] cfg=${CFG} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} | NPROC=${NPROC}"

if [ "${NPROC}" -gt 1 ]; then
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-21613}" \
    train_sam3_rbma.py --cfg "${CFG}"
else
  python train_sam3_rbma.py --cfg "${CFG}"
fi
