#!/bin/bash
# 기준선 둘(DGFusion·CAFuser)에 depth·RGB 부분 열화(RMM r=0.5)를 적용한 test 측정.
# 완전 제거 Δ 와 나란히 놓고 "열화에 얼마나 버티는가" 를 보기 위한 것이다.
# 개입 규약은 우리 rmm_mask 와 같다: 원소별(픽셀·채널 독립) rand<r 자리를 모달 평균으로
# 바꾼다(정규화 후 0). 시드는 고정하고 보고에 명기한다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
OUT=$REPO/rmm_baselines
RATIO=${RATIO:-0.5}
SEED=${SEED:-0}
BATCH=${BATCH:-8}
ALLOWED=${ALLOWED:-"4 5 6"}
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

DGF_CFG=configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml
DGF_CKPT=output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth
CAF_CFG=configs/deliver/swin/cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml
CAF_CKPT=/SSDb/jemo_maeng/cafuser_eval/ckpts/model_0169999.pth

miou_of() { grep -o "'mIoU': [0-9.]*" "$1" 2>/dev/null | tail -1 | grep -o "[0-9.]*$"; }

free_gpu() {
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

run() {  # run <tag> <cfg> <ckpt> <modal>
  local tag=$1 cfg=$2 ckpt=$3 modal=$4
  local log="$OUT/eval_${tag}.log"
  grep -q "copypaste" "$log" 2>/dev/null && { echo "skip $tag" >> "$OUT/runner.log"; return 0; }
  local g; g=$(free_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
  echo "$(date '+%F %T') start $tag (GPU $g, ratio $RATIO, seed $SEED)" >> "$OUT/runner.log"
  BF_ZERO_MODAL=$modal BF_ZERO_MODE=normalized BF_ZERO_RATIO=$RATIO BF_ZERO_SEED=$SEED \
  BF_EVAL_BATCH=$BATCH CUDA_VISIBLE_DEVICES=$g python train_net.py \
    --config-file "$cfg" --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS "$ckpt" \
    DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" MODEL.TEST.DEPTH_ON False \
    OUTPUT_DIR "$OUT/$tag" > "$log" 2>&1
  echo "$(date '+%F %T') done $tag mIoU=$(miou_of "$log")" >> "$OUT/runner.log"
}

echo "$(date '+%F %T') 시작 (ratio=$RATIO seed=$SEED batch=$BATCH)" > "$OUT/runner.log"
run dgf_rmm_DEPTH  "$DGF_CFG" "$DGF_CKPT" DEPTH
run dgf_rmm_CAMERA "$DGF_CFG" "$DGF_CKPT" CAMERA
run caf_rmm_DEPTH  "$CAF_CFG" "$CAF_CKPT" DEPTH
run caf_rmm_CAMERA "$CAF_CFG" "$CAF_CKPT" CAMERA
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
cat "$OUT/runner.log"
