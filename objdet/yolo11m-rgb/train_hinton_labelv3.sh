#!/usr/bin/env bash
# poongsan label-v3 (2026-07-02 23시 신규 레이블) + capture-holdout split, RGB-only, 50ep.
# 사용: GPU=1 MODEL=yolo11m.pt bash train_hinton_labelv3.sh   (MODEL=yolov5mu.pt 로 v5m)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # ultralytics device=N 은 절대 GPU 번호 (README 참조)
MODEL=${MODEL:-yolo11m.pt}
TAG=$(basename "$MODEL" .pt | tr -d .)
RUNS="$(pwd)/runs"
NAME="${TAG}_rgb_labelv3_50ep"
yolo detect train \
  model="$MODEL" \
  data=/SSDd/jemo_maeng/dset/poongsan_labelv3_yolo_rgb/poongsan_labelv3_rgb.yaml \
  epochs=50 imgsz=640 batch=16 device=${GPU:-1} workers=8 seed=0 \
  project="$RUNS" name="$NAME" exist_ok=True
yolo detect val \
  model="$RUNS/$NAME/weights/best.pt" \
  data=/SSDd/jemo_maeng/dset/poongsan_labelv3_yolo_rgb/poongsan_labelv3_rgb.yaml \
  split=test device=${GPU:-1} \
  project="$RUNS" name="${NAME}_testeval" exist_ok=True
