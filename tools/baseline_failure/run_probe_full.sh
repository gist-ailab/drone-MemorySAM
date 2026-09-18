#!/bin/bash
# A4-(iv) 본실행 — DELIVER test 1,897 장 전부.
# 다섯 조건을 서로 다른 빈 GPU 에서 동시에 돌린다(프로브는 이미지 단위라 배치를 못 키운다).
#   base        개입 없음
#   zero_depth  HHA(DEPTH) 입력을 정규화 후 0 이 되게 채움
#   zero_camera RGB 입력을 같은 방식으로 채움(대조군)
#   zero_event / zero_lidar  분할 성능에 영향이 없던 두 모달(귀무 대조군)
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml
CKPT=output/dgfusion_swin_tiny_bs8_200k_deliver_clde/model_0079999.pth
OUT=$REPO/probe_out_full
mkdir -p "$OUT"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
cd "$REPO"
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

RX=(--depth-head-regex '^depth_head$'
    --depth-token-regex '^fusion_module\.depth_proj$'
    --cond-token-regex '^fusion_module\.weather_proj$'
    --xattn-regex '^fusion_module\.fusion\.res[2-5]\.attn$')

run() {  # run <gpu> <tag> [extra args...]
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python tools/baseline_failure/probe_dgfusion.py \
    --config-file "$CFG" --weights "$CKPT" \
    --out "$OUT/probe_${tag}.csv" "${RX[@]}" "$@" \
    --opts DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \
    > "$OUT/probe_${tag}.log" 2>&1
  echo "$(date '+%F %T') done $tag exit=$?" >> "$OUT/runner.log"
}

echo "$(date '+%F %T') 시작" > "$OUT/runner.log"
run 1 base &
run 2 zero_depth  --zero-modal DEPTH  --zero-mode normalized &
run 3 zero_camera --zero-modal CAMERA --zero-mode normalized &
run 4 zero_event  --zero-modal EVENT  --zero-mode normalized &
run 5 zero_lidar  --zero-modal LIDAR  --zero-mode normalized &
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
