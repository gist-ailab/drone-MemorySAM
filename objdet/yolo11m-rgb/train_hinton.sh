#!/usr/bin/env bash
# E1.1 YOLO11-m RGB-only 기준점 (poongsan_v2, v2 capture-holdout split, 100ep)
# hinton GPU 1 에서 실행: bash train_hinton.sh   (GPU=<n> 으로 변경 가능)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo
cd "$(dirname "$0")"
# 주의: ultralytics 는 device=N 을 절대 GPU 번호로 취급하며 런타임에
# CUDA_VISIBLE_DEVICES 를 덮어쓴다 → CVD 로 GPU 를 고르지 말고 device= 로 지정.
# PCI_BUS_ID: CUDA 열거 순서를 nvidia-smi 와 일치시킴.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
RUNS="$(pwd)/runs"
yolo detect train \
  model=yolo11m.pt \
  data=/SSDd/jemo_maeng/dset/poongsan_v2_yolo_rgb/poongsan_v2_rgb.yaml \
  epochs=100 imgsz=640 batch=16 device=${GPU:-1} workers=8 seed=0 \
  project="$RUNS" name=y11m_rgb_v2_100ep exist_ok=True
# 학습 종료 후 test split(=val과 동일, v2 test) 최종 평가
yolo detect val \
  model="$RUNS/y11m_rgb_v2_100ep/weights/best.pt" \
  data=/SSDd/jemo_maeng/dset/poongsan_v2_yolo_rgb/poongsan_v2_rgb.yaml \
  split=test device=${GPU:-1} \
  project="$RUNS" name=y11m_rgb_v2_100ep_testeval exist_ok=True
