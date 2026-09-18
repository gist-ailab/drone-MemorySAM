#!/bin/bash
# A4-(iv) — DGFusion 80k: HHA(DEPTH) 입력을 0 으로 채웠을 때 depth 보조 헤드 정확도와
# 같은 조건의 이미지별 분할 mIoU 를 함께 기록한다. 기준(개입 없음)과 짝지어 돌린다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml
CKPT=output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth
OUT=$REPO/probe_out
LIMIT=${LIMIT:-0}
GPU=${GPU:-1}
mkdir -p "$OUT"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd "$REPO"
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

RX=(--depth-head-regex '^depth_head$'
    --depth-token-regex '^fusion_module\.depth_proj$'
    --cond-token-regex '^fusion_module\.weather_proj$'
    --xattn-regex '^fusion_module\.fusion\.res[2-5]\.attn$')

run() {  # run <tag> [extra args...]
  local tag=$1; shift
  echo "=== $tag ($(date '+%F %T')) ==="
  CUDA_VISIBLE_DEVICES=$GPU python tools/baseline_failure/probe_dgfusion.py \
    --config-file "$CFG" --weights "$CKPT" --limit "$LIMIT" \
    --out "$OUT/probe_${tag}.csv" "${RX[@]}" "$@" \
    --opts DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \
    > "$OUT/probe_${tag}.log" 2>&1
  echo "exit=$? rows=$(($(wc -l < "$OUT/probe_${tag}.csv" 2>/dev/null || echo 1) - 1))"
}

run base
run zero_depth --zero-modal DEPTH --zero-mode normalized
echo "완료 $(date '+%F %T')"
