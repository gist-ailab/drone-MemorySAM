#!/usr/bin/env bash
# YOLOv5m low-light ablation ladder — architecture frozen, train-recipe only.
#
#   b0  baseline        stock YOLOv5m, stock hyp                 (control)
#   b1  +night-aug      + dark-tail low-light augmentation        (isolates aug)
#   b2  +loss           b1 + EIoU + focal(fl_gamma) + label-smooth (isolates loss)
#   b3  +mix            b2 + mixup + copy_paste (rare classes)     (full recipe)
#
# Each rung differs ONLY in env flags / hyp / --label-smoothing. The exported
# graph is identical across all rungs, so every one stays i.MX-portable.
#
# Prereq (once, on the GPU box):
#   conda activate yolo                        # ultralytics deps present
#   git clone https://github.com/ultralytics/yolov5   # classic anchor-based v5
#   export YOLOV5_DIR=<abs path to that clone>
#   python night_aug.py <a_train_rgb.png>      # confirm dark-tail calibration
#   python eiou.py                             # confirm EIoU sanity
#
# Usage:  bash train_ladder.sh <b0|b1|b2|b3> [GPU] [EPOCHS]
set -euo pipefail
RUNG="${1:?usage: train_ladder.sh <b0|b1|b2|b3> [GPU] [EPOCHS]}"
GPU="${2:-0}"
EPOCHS="${3:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${YOLOV5_DIR:?set YOLOV5_DIR to a classic ultralytics/yolov5 clone}"
: "${DATA_YAML:=/SSDd/jemo_maeng/dset/poongsan_v2_yolo_rgb/poongsan_v2_rgb.yaml}"
WEIGHTS="${WEIGHTS:-yolov5m.pt}"    # classic anchor-based (NOT yolov5mu.pt)
IMG=640; BATCH="${BATCH:-16}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

STOCK_HYP="$YOLOV5_DIR/data/hyps/hyp.scratch-low.yaml"
COMMON=(--data "$DATA_YAML" --weights "$WEIGHTS" --img "$IMG" --batch "$BATCH"
        --epochs "$EPOCHS" --device "$GPU" --workers 8 --seed 0
        --project "$HERE/runs" --exist-ok)

unset YOLO_NIGHTAUG YOLO_EIOU
case "$RUNG" in
  b0) HYP="$STOCK_HYP";               LS=(); NAME="y5m_b0_baseline" ;;
  b1) HYP="$STOCK_HYP";               LS=(); NAME="y5m_b1_nightaug"
      export YOLO_NIGHTAUG=1 ;;
  b2) HYP="$HERE/hyp.b2_loss.yaml";   LS=(--label-smoothing 0.1); NAME="y5m_b2_loss"
      export YOLO_NIGHTAUG=1 YOLO_EIOU=1 ;;
  b3) HYP="$HERE/hyp.lowlight.yaml";  LS=(--label-smoothing 0.1); NAME="y5m_b3_full"
      export YOLO_NIGHTAUG=1 YOLO_EIOU=1 ;;
  *)  echo "unknown rung: $RUNG (want b0|b1|b2|b3)"; exit 2 ;;
esac

echo "[ladder] rung=$RUNG  name=$NAME  gpu=$GPU  epochs=$EPOCHS"
echo "[ladder] NIGHTAUG=${YOLO_NIGHTAUG:-0}  EIOU=${YOLO_EIOU:-0}  hyp=$(basename "$HYP")  LS=${LS[*]:-none}"

python "$HERE/train_lowlight.py" "${COMMON[@]}" --hyp "$HYP" --name "$NAME" "${LS[@]}"

# final test-split eval (mAP50); night/normal breakdown via eval_lowlight.sh
python "$YOLOV5_DIR/val.py" --data "$DATA_YAML" --img "$IMG" --device "$GPU" \
  --weights "$HERE/runs/$NAME/weights/best.pt" --task test \
  --project "$HERE/runs" --name "${NAME}_testeval" --exist-ok
