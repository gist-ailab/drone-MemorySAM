#!/bin/bash
# CAFuser 이미지별 mIoU — 조건별 Δ 산출용. DGFusion 프로브를 그대로 쓰되 depth 관련 감시는
# 끈다(CAFuser 에는 depth 헤드가 없다). 기준·depth 제거·RGB 제거 세 조건만 잰다.
# 판정 세션 지시(2026-09-19)에 따라 GPU 는 4,5,6 만 쓴다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/cafuser_swin_tiny_bs6_267k_deliver_clde_lecun.yaml
CKPT=/SSDb/jemo_maeng/cafuser_eval/ckpts/model_0169999.pth
OUT=$REPO/probe_cafuser_perimage
ALLOWED=${ALLOWED:-"4 5 6"}
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

RX=(--depth-head-regex none --depth-token-regex none --cond-token-regex none
    --xattn-regex '^fusion_module\.fusion\.res[2-5]\.attn$')

free_gpu() {
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

run() {  # run <tag> [extra args...]
  local tag=$1; shift
  [ -s "$OUT/probe_${tag}.csv" ] && { echo "skip $tag" >> "$OUT/runner.log"; return 0; }
  local g; g=$(free_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
  echo "$(date '+%F %T') start $tag (GPU $g)" >> "$OUT/runner.log"
  CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/probe_dgfusion.py \
    --config-file "$CFG" --weights "$CKPT" --out "$OUT/probe_${tag}.csv" "${RX[@]}" "$@" \
    --opts DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" MODEL.TEST.DEPTH_ON False \
    > "$OUT/probe_${tag}.log" 2>&1 &
  sleep 90
}

echo "$(date '+%F %T') 시작 (허용 GPU: $ALLOWED)" > "$OUT/runner.log"
run base
run zero_depth  --zero-modal DEPTH  --zero-mode normalized
run zero_camera --zero-modal CAMERA --zero-mode normalized
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
for f in "$OUT"/probe_*.csv; do
  printf "%-22s %s행\n" "$(basename "$f" .csv)" "$(($(wc -l < "$f") - 1))"
done | tee -a "$OUT/runner.log"
