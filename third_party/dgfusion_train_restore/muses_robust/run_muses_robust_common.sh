#!/usr/bin/env bash
set -euo pipefail

model_kind=$1
weights=$2
out=$3
protocol=$4
setting=$5
gpu=$6
repo=/SSDb/jemo_maeng/dgfusion_train
python_bin=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[[ -f "$weights" ]] || { echo "Weights missing: $weights" >&2; exit 2; }
[[ -d "$repo/datasets/muses" ]] || { echo "MUSES dataset link missing: $repo/datasets/muses" >&2; exit 2; }
check_gpu() {
  local gpu_state mem util
  gpu_state=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  IFS=',' read -r mem util <<< "$gpu_state"
  mem=${mem//[[:space:]]/}
  util=${util//[[:space:]]/}
  if (( mem > 2000 || util > 10 )); then
      echo "GPU $gpu is busy (memory=${mem}MiB, util=${util}%). Require <=2000MiB and <=10%." >&2
      exit 3
  fi
}

mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="$gpu" DETECTRON2_DATASETS="$repo/datasets"
unset BF_ZERO_MODAL BF_ZERO_MODALS BF_NM_TYPE
export MUSES_ROBUST_PROTOCOL="$protocol" MUSES_ROBUST_SEED="${MUSES_ROBUST_SEED:-0}"
case "$model_kind" in
    dgf) cfg=configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml ;;
    caf) cfg=configs/muses/swin/cafuser_swin_tiny_bs8_180k_muses_clre.yaml ;;
    *) echo "Unknown model $model_kind" >&2; exit 2 ;;
esac
[[ -f "$repo/$cfg" ]] || { echo "Config missing: $repo/$cfg" >&2; exit 2; }

subsets=(
  img,lidar,event,radar
  img,lidar,event img,lidar,radar img,event,radar lidar,event,radar
  img,lidar img,event img,radar lidar,event lidar,radar event,radar
  img lidar event radar
)
settings=()
case "$protocol" in
  clean) settings=(clean) ;;
  emm) if [[ "$setting" == all ]]; then settings=("${subsets[@]}"); else settings=("$setting"); fi ;;
  rmm) [[ "$setting" == 0.25 || "$setting" == 0.5 || "$setting" == 0.75 ]] || { echo 'RMM ratio: 0.25|0.5|0.75' >&2; exit 2; }
       export MUSES_ROBUST_RATIO="$setting"
       settings=("${subsets[@]}") ;;
  nm) [[ "$setting" == low || "$setting" == mid || "$setting" == high ]] || { echo 'NM level: low|mid|high' >&2; exit 2; }
      settings=("$setting") ;;
  *) echo 'Protocol: clean|emm|rmm|nm' >&2; exit 2 ;;
esac

for case_name in "${settings[@]}"; do
  check_gpu
  export MUSES_ROBUST_SUBSET="$case_name" MUSES_ROBUST_LEVEL="$case_name"
  safe_name=${case_name//, /+}
  safe_name=${safe_name//,/+}
  if (( ${#settings[@]} == 1 )); then
    case_out="$out"
  else
    case_out="$out/$protocol/$setting/$safe_name"
  fi
  mkdir -p "$case_out"
  cd "$repo"
  "$python_bin" train_net.py --config-file "$cfg" --eval-only --num-gpus 1 \
    MODEL.IS_TRAIN False MODEL.WEIGHTS "$weights" OUTPUT_DIR "$case_out" \
    DATASETS.TEST_PANOPTIC "('muses_panoptic_val',)" \
    MODEL.TEST.PANOPTIC_ON True MODEL.TEST.SEMANTIC_ON True MODEL.TEST.DEPTH_ON False \
    MODEL.TEST.SAVE_PREDICTIONS.PANOPTIC True \
    MODEL.TEST.SAVE_PREDICTIONS.CITYSCAPES_COLORS False \
    > "$case_out/run.log" 2>&1
  "$python_bin" "$script_dir/score_dgf_muses.py" \
    "$case_out/inference/sem_seg_predictions.json" \
    "$repo/datasets/muses/gt_semantic" "$case_out" "$case_out" \
    > "$case_out/score.log" 2>&1
  echo "Completed $protocol $case_name: $case_out"
done
if (( ${#settings[@]} > 1 )); then
  "$python_bin" "$script_dir/summarize.py" "$out/$protocol/$setting"
fi
