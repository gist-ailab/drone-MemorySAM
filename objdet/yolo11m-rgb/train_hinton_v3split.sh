#!/usr/bin/env bash
# poongsan split-v3 (캡처 내 시간순 80/20 + 15프레임 gap), 라벨 v20260702_2303, RGB-only, 50ep.
# 사용: GPU=1 MODEL=yolov5mu.pt bash train_hinton_v3split.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # ultralytics device=N 은 절대 GPU 번호 (README 참조)
MODEL=${MODEL:-yolov5mu.pt}
TAG=$(basename "$MODEL" .pt | tr -d .)
RUNS="$(pwd)/runs"
NAME="${TAG}_rgb_v3split_50ep"
yolo detect train \
  model="$MODEL" \
  data=/SSDd/jemo_maeng/dset/poongsan_v3split_yolo_rgb/poongsan_v3_rgb.yaml \
  epochs=50 imgsz=640 batch=16 device=${GPU:-1} workers=8 seed=0 \
  project="$RUNS" name="$NAME" exist_ok=True
yolo detect val \
  model="$RUNS/$NAME/weights/best.pt" \
  data=/SSDd/jemo_maeng/dset/poongsan_v3split_yolo_rgb/poongsan_v3_rgb.yaml \
  split=test device=${GPU:-1} \
  project="$RUNS" name="${NAME}_testeval" exist_ok=True
