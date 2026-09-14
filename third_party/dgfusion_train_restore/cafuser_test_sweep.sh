#!/bin/bash
# CAFuser checkpoint test-set sweep (official README test command, one GPU per process).
# Usage: bash cafuser_test_sweep.sh <gpu_id> <iter_tag|final> [...]
set -uo pipefail
GPU=$1; shift
source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd /SSDb/jemo_maeng/cafuser_train
export PYTHONPATH=$PWD/OneFormer:$PWD
export WANDB_MODE=offline
export DETECTRON2_DATASETS=$PWD/datasets
RUN=output/cafuser_swin_tiny_bs6_267k_deliver_clde
mkdir -p logs output/test_sweep
for TAG in "$@"; do
  OUT=output/test_sweep/iter_${TAG}
  if [ -f "$OUT/done" ]; then echo "skip $TAG (done)"; continue; fi
  echo "=== test eval $TAG on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU python train_net.py \
      --config-file configs/deliver/swin/cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml \
      --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS "$RUN/model_${TAG}.pth" \
      DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" OUTPUT_DIR "$OUT" \
      2>&1 | tee logs/test_sweep_iter_${TAG}.log
  if grep -q "'mIoU'" logs/test_sweep_iter_${TAG}.log; then touch "$OUT/done"; fi
done
echo "SWEEP_DONE gpu$GPU"
