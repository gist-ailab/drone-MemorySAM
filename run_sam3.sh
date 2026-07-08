#!/usr/bin/env bash
# Unified SAM3(RBMA) train launcher — 구 run_sam3_train.sh + run_sam3_rbma.sh 통합
# (두 스크립트 모두 동일 진입점 train_sam3_rbma.py를 실행했음)
#
# Usage:
#   NGPU=1 bash run_sam3.sh [cfg]                       # 빈 GPU 자동 선택 (NGPU/NPROC 동의어)
#   CUDA_VISIBLE_DEVICES=0,1 bash run_sam3.sh [cfg]     # 직접 지정 시 그대로 존중 (nproc=지정 GPU 수)
#   MASTER_PORT=12345 bash run_sam3.sh [cfg]            # torchrun 포트 오버라이드
#   DRY_RUN=1 bash run_sam3.sh [cfg]                    # 커맨드만 출력(실행 안 함)
set -e

CFG="${1:-configs/deliver/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml}"
NPROC="${NPROC:-${NGPU:-1}}"

# GPU 선택: CUDA_VISIBLE_DEVICES를 직접 주면 존중, 없으면 빈 GPU NPROC장 자동 배정.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(bash "$(dirname "$0")/scripts/pick_free_gpus.sh" "$NPROC")" || {
    echo "[run_sam3] 빈 GPU ${NPROC}장을 찾지 못했습니다. nvidia-smi 확인 후 NGPU/CUDA_VISIBLE_DEVICES를 조정하세요." >&2
    exit 1
  }
  export CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES
  NPROC="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi

# OMP_NUM_THREADS 경고 방지
export OMP_NUM_THREADS=1
# 공유 GPU 메모리 단편화 완화 (OOM "reserved but unallocated" 대응)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SAM3 전용 환경 (필수)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="semseg/models/sam3:${PYTHONPATH}"

cfg_name=$(basename "${CFG}" .yaml)
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_PATH="logs/${cfg_name}/${cfg_name}_${timestamp}.log"

CMD=(torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-21613}" \
     train_sam3_rbma.py --cfg "${CFG}")

echo "[run_sam3] cfg=${CFG} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | nproc=${NPROC}"

if [ -n "${DRY_RUN:-}" ]; then
  echo "[run_sam3][dry-run] ${CMD[*]} 2>&1 | tee ${LOG_PATH}"
  exit 0
fi

mkdir -p "logs/${cfg_name}"
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
