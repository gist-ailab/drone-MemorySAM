#!/bin/bash
# DRN-261001-01: DGFusion 공식 MUSES 가중치(180k clre)로 MUSES val 250장 추론 — README 공식 명령 그대로.
#   GPU=n  TAG=이름  [SAVE=1 이면 panoptic 예측을 저장]
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
PY=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
export PYTHONPATH=$PWD:$PWD/OneFormer WANDB_MODE=offline
export DETECTRON2_DATASETS=/SSDf/jemo_maeng/dset/dgf_datasets
export CUDA_VISIBLE_DEVICES=${GPU:?GPU 를 지정하라}
TAG=${TAG:-official_val}
OUT=/SSDf/jemo_maeng/dgf_muses_out/$TAG
mkdir -p "$OUT"
EXTRA=()
if [ "${SAVE:-0}" = 1 ]; then
  EXTRA+=(MODEL.TEST.SAVE_PREDICTIONS.PANOPTIC True MODEL.TEST.SAVE_PREDICTIONS.CITYSCAPES_COLORS False)
fi
echo "$(date '+%F %T') [muses-official] $TAG 시작 GPU=$GPU"
$PY train_net.py \
  --config-file configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml \
  --eval-only MODEL.IS_TRAIN False \
  MODEL.WEIGHTS /SSDf/jemo_maeng/dgf_official/dgfusion_swin_tiny_bs8_180k_muses_clre.pth \
  OUTPUT_DIR "$OUT" \
  DATASETS.TEST_PANOPTIC "('muses_panoptic_val',)" \
  MODEL.TEST.PANOPTIC_ON True MODEL.TEST.SEMANTIC_ON True MODEL.TEST.DEPTH_ON False \
  "${EXTRA[@]}" > "$OUT/run.log" 2>&1
echo "$(date '+%F %T') [muses-official] $TAG 종료 exit=$?"
grep -E "mIoU|PQ|Traceback|Error" "$OUT/run.log" | tail -15 | cut -c1-300
