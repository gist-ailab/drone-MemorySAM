#!/usr/bin/env bash
set -euo pipefail

model_kind=$1
weights=$2
out=$3
protocol=$4
setting=$5
gpu=$6
repo=/SSDb/jemo_maeng/dgfusion_train
python_bin=${DGF_PYTHON:-/home/jemo_maeng/miniconda3/envs/dgfusion/bin/python}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[[ -x "$python_bin" ]] || { echo "DGFusion Python is not executable: $python_bin (set DGF_PYTHON to the dgfusion environment's Python)" >&2; exit 2; }
[[ -f "$weights" ]] || { echo "Weights missing: $weights" >&2; exit 2; }
[[ -d "$repo/datasets/muses" ]] || { echo "MUSES dataset link missing: $repo/datasets/muses" >&2; exit 2; }
export PYTHONPATH="$repo/OneFormer:$repo:$(dirname "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"

caf_source="$script_dir/configs/cafuser_swin_tiny_bs8_180k_muses_clre.yaml"
caf_target="$repo/configs/muses/swin/cafuser_swin_tiny_bs8_180k_muses_clre.yaml"
[[ -f "$caf_source" ]] || { echo "Bundled CAFuser config missing: $caf_source" >&2; exit 2; }
if [[ -e "$caf_target" || -L "$caf_target" ]]; then
  cmp -s "$caf_source" "$caf_target" || { echo "CAFuser config differs from bundled file; refusing to overwrite: $caf_target" >&2; exit 2; }
else
  mkdir -p "$(dirname "$caf_target")"
  cp "$caf_source" "$caf_target"
fi
for base in "$repo/configs/muses/oneformer_R50_bs8_90k_muses.yaml" "$repo/configs/muses/Base-MUSES-UnifiedSegmentation.yaml"; do
  [[ -f "$base" ]] || { echo "CAFuser config base missing: $base" >&2; exit 2; }
done

patch_file="$script_dir/train_net_muses_robust.patch"
[[ -f "$repo/train_net.py" ]] || { echo "DGFusion train_net.py missing: $repo/train_net.py" >&2; exit 2; }
if ! grep -Fq '_install_muses_robust' "$repo/train_net.py"; then
  if ! (cd "$repo" && patch -p1 --dry-run < "$patch_file" && patch -p1 < "$patch_file"); then
    echo "Failed to apply MUSES robustness patch to $repo/train_net.py" >&2
    exit 2
  fi
fi

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
    --dist-url "tcp://127.0.0.1:$((50000 + RANDOM % 10000))" \
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
