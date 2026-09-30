#!/bin/bash
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
PY=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
export PYTHONPATH=$PWD:$PWD/OneFormer DETECTRON2_DATASETS=$PWD/datasets WANDB_MODE=offline CUDA_VISIBLE_DEVICES=${GPU:-1}
OUT=/SSDb/jemo_maeng/dgfusion_train/robust_out/smoke_a
mkdir -p $OUT
$PY tools/baseline_failure/robust_bench_eval.py \
  --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
  --weights output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth \
  --split test --out $OUT --limit ${LIMIT:-30} \
  --only clean 'emm|present=CAMERA+LIDAR+EVENT' nm0.1 \
  --dump_hists $OUT/shard_smoke.npz \
  --opts MODEL.TEST.DEPTH_ON False > $OUT/smoke.log 2>&1
echo "exit=$?"
tail -15 $OUT/smoke.log
