#!/bin/bash
# DGFusion checkpoint test-set sweep (official README test command, one GPU per process).
# Usage: bash dgfusion_test_sweep.sh <gpu_id> <iter_tag> [<iter_tag> ...]
#   iter_tag = 0009999, 0019999, ...  -> output/.../model_<iter_tag>.pth
# Results: output/test_sweep/iter_<tag>/ + logs/test_sweep_iter_<tag>.log
set -uo pipefail
GPU=$1; shift
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd /SSDb/jemo_maeng/dgfusion_train
export PYTHONPATH=$PWD/OneFormer:$PWD
export WANDB_MODE=offline
export DETECTRON2_DATASETS=$PWD/datasets
RUN=output/dgfusion_swin_tiny_bs8_200k_deliver_clde
mkdir -p logs output/test_sweep
for TAG in "$@"; do
  CKPT=$RUN/model_${TAG}.pth
  OUT=output/test_sweep/iter_${TAG}
  if [ -f "$OUT/done" ]; then echo "skip $TAG (done)"; continue; fi
  echo "=== test eval $TAG on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU python test_net.py \
      --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
      --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS "$CKPT" \
      DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \
      MODEL.TEST.DEPTH_ON False OUTPUT_DIR "$OUT" \
      2>&1 | tee logs/test_sweep_iter_${TAG}.log
  if grep -q "'mIoU'" logs/test_sweep_iter_${TAG}.log; then touch "$OUT/done"; fi
done
echo "SWEEP_DONE gpu$GPU"
