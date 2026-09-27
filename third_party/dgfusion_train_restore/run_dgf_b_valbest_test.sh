#!/bin/bash
# DGFusion (b) val-best(90k) test 평가. 앞선 시도는 원격 명령에 \x27 이스케이프가
# 그대로 문자열로 들어가 config 파싱이 깨졌다(따옴표 처리 오류). 스크립트 파일로 옮겨
# 인용 문제를 없앤다.
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline
RUN=output/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade
mkdir -p logs
GPU=${GPU:-3}
CUDA_VISIBLE_DEVICES=$GPU python train_net.py \
  --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade.yaml \
  --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS "$RUN/model_0089999.pth" \
  DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \
  MODEL.TEST.DEPTH_ON False OUTPUT_DIR "$RUN/test_valbest_90k" \
  > logs/dgf_b_valbest90k_test.log 2>&1
echo "exit=$?"
grep -o "'mIoU': [0-9.]*" logs/dgf_b_valbest90k_test.log | tail -1
