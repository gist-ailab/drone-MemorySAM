#!/usr/bin/env bash
# night / normal mAP50 breakdown for a trained YOLOv5m — the money slide.
# Runs stock yolov5 val.py three times (all / lowlight / normal) by swapping the
# `val:` image list, so the presentation can show the low-light gap shrink.
#
# Usage: bash eval_lowlight.sh <best.pt> <base_data.yaml> <OUT_DIR> [GPU]
#   OUT_DIR must contain test_all.txt / test_lowlight.txt / test_normal.txt
#   (produced by convert_final_yolo.py).
set -euo pipefail
WEIGHTS="${1:?best.pt}"; BASE="${2:?base data yaml}"; OUT="${3:?dir with test_*.txt}"; GPU="${4:-0}"
: "${YOLOV5_DIR:?set YOLOV5_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TMP="$(mktemp -d)"

for subset in all lowlight normal; do
  LIST="$OUT/test_${subset}.txt"
  [ -s "$LIST" ] || { echo "skip $subset (no $LIST)"; continue; }
  YAML="$TMP/${subset}.yaml"
  # copy base yaml, force val: to this subset list
  grep -v -E '^\s*(val|test)\s*:' "$BASE" > "$YAML"
  echo "val: $LIST"  >> "$YAML"
  echo "test: $LIST" >> "$YAML"
  echo "==================== $subset ($(wc -l < "$LIST") imgs) ===================="
  python "$YOLOV5_DIR/val.py" --data "$YAML" --weights "$WEIGHTS" --img 640 \
    --device "$GPU" --task test --verbose 2>&1 | grep -E "^\s+(all|Class)" | head -20
done
rm -rf "$TMP"
