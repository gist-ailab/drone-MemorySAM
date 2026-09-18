#!/bin/bash
# CAFuser 대조 측정 — DGFusion 과 같은 축(정규화 후 0)으로 모달을 하나씩 지우고
# DELIVER test 1,897 장 공식 채점을 돌린다. 체크포인트는 lecun 학습본의 val-best(170k).
# 빈 GPU 2 장(6,7)에 두 개씩 붙여 순차 처리한다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml
CKPT=/SSDb/jemo_maeng/cafuser_eval/ckpts/model_0169999.pth
OUT=$REPO/cafuser_modal_zero
BATCH=${BATCH:-8}
mkdir -p "$OUT"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd "$REPO"
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

miou_of() { grep -o "'mIoU': [0-9.]*" "$1" 2>/dev/null | tail -1 | grep -o "[0-9.]*$"; }

run() {  # run <gpu> <tag> [env 이름=값 ...]
  local gpu=$1 tag=$2; shift 2
  local log="$OUT/eval_${tag}.log"
  if grep -q "copypaste" "$log" 2>/dev/null; then echo "skip $tag" >> "$OUT/runner.log"; return 0; fi
  # BF_EVAL_BATCH 를 먼저 두어야 호출부가 준 값이 뒤에서 덮어쓴다(env 는 뒤가 이긴다).
  env BF_EVAL_BATCH=$BATCH "$@" CUDA_VISIBLE_DEVICES=$gpu python train_net.py \
    --config-file "$CFG" --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS "$CKPT" \
    DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" MODEL.TEST.DEPTH_ON False \
    OUTPUT_DIR "$OUT/$tag" > "$log" 2>&1
  echo "$(date '+%F %T') done $tag mIoU=$(miou_of "$log")" >> "$OUT/runner.log"
}

echo "$(date '+%F %T') 시작 (batch=$BATCH)" > "$OUT/runner.log"
run 6 base & run 7 base_bs1 BF_EVAL_BATCH=1
wait
run 6 normalized_DEPTH  BF_ZERO_MODAL=DEPTH  BF_ZERO_MODE=normalized &
run 7 normalized_CAMERA BF_ZERO_MODAL=CAMERA BF_ZERO_MODE=normalized
wait
run 6 normalized_EVENT BF_ZERO_MODAL=EVENT BF_ZERO_MODE=normalized &
run 7 normalized_LIDAR BF_ZERO_MODAL=LIDAR BF_ZERO_MODE=normalized
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
cat "$OUT/runner.log"
