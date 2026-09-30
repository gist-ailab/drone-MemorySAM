#!/bin/bash
# 등가검증 기준값(process-per-case 방식): DGFusion (a) val-best 80k 를 test 1897장에서
# clean / BF_ZERO_MODALS=DEPTH(EMM 1개) / BF_NM_TYPE=sp d=0.1(NM) 세 케이스로 각각 재측정한다.
# 새 단일 순회기(robust_bench_eval.py)와 mIoU 차를 대조하는 기준이다(DRN-260930-02 사전기준 ①).
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
PY=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${GPU:-1}
CKPT=output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth
CFG=configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml
OUTROOT=/SSDb/jemo_maeng/dgfusion_train/robust_out/ref_a3
mkdir -p "$OUTROOT"

run_case() {   # $1=이름, 나머지=환경변수 KEY=VAL ...
  local name=$1; shift
  echo "$(date '+%F %T') [ref] $name 시작"
  env "$@" $PY train_net_bfref.py --config-file $CFG --eval-only --num-gpus 1 \
    MODEL.IS_TRAIN False MODEL.WEIGHTS "$CKPT" \
    DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \
    MODEL.TEST.DEPTH_ON False OUTPUT_DIR "$OUTROOT/$name" \
    > "$OUTROOT/$name.log" 2>&1
  echo "$(date '+%F %T') [ref] $name exit=$? mIoU=$(grep -o "'mIoU': [0-9.]*" "$OUTROOT/$name.log" | tail -1)"
}

run_case nm_sp0.1       BF_NM_TYPE=sp BF_NM_DENSITY=0.1 BF_NM_SEED=0
echo "$(date '+%F %T') [ref] 전체 종료"
