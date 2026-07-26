#!/usr/bin/env bash
# Pull the shared RGB YOLO dataset from ailab_mat2 to a LOCAL SSD cache.
# Training must NOT read the dataset straight off the sshfs NAS (per-epoch small
# -file reads over sshfs cripple throughput) — cache locally, then train.
#
#   bash fetch_dataset.sh [LOCAL_DIR]   (default: ./poongsan_v2_yolo_rgb)
#
# After it runs, use  $LOCAL_DIR/poongsan_v2_rgb.yaml  (path rewritten to local).
set -euo pipefail
SHARED="/ailab_mat2/personal/jemo_maeng/dset/poongsan_v2_yolo_rgb"
TAR="$SHARED/poongsan_v2_yolo_rgb.tar"
LOCAL="${1:-$PWD/poongsan_v2_yolo_rgb}"
PARENT="$(dirname "$LOCAL")"

[ -f "$TAR" ] || { echo "shared tar not found: $TAR (is ailab_mat2 mounted?)"; exit 1; }
mkdir -p "$PARENT"
echo "[fetch] extracting $TAR -> $PARENT (local SSD)"
tar xf "$TAR" -C "$PARENT"          # tar root dir = poongsan_v2_yolo_rgb
# rewrite yaml path: to the local cache so ultralytics resolves images locally
YAML="$LOCAL/poongsan_v2_rgb.yaml"
sed -i "s#^path:.*#path: $LOCAL#" "$YAML"
echo "[fetch] ready: $YAML"
echo "         train=$(find "$LOCAL/images/train" -type f | wc -l)  test=$(find "$LOCAL/images/test" -type f | wc -l)"
