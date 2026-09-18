#!/bin/bash
# 우리 모델 E1 확정 시드1 — 이미지별 mIoU 를 남기는 재실행. 앞선 측정은 전역 수치만 남겨
# 조건별 Δ 를 낼 수 없었다. 기준·depth 제거·RGB 제거 세 조건만 돌린다(event·lidar 는
# 전역 Δ 가 ±0.5 안이라 조건별로 갈라도 읽을 것이 없다).
set -uo pipefail
REPO=/SSDb/jemo_maeng/src/drone-MemorySAM
CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
CKPT=$REPO/outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200/DELIVER_ReliaDINO-ViTL16_idel/epoch140_68.9_top1_checkpoint.pth
OUT=$REPO/modal_zero_ours_seed1_perimage
BATCH=${BATCH:-8}
ALLOWED=${ALLOWED:-"4 5 6"}
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$REPO/semseg/models/sam2:$REPO
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

free_gpu() {
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

echo "$(date '+%F %T') 시작 (batch=$BATCH)" > "$OUT/runner.log"
for c in base zero_depth zero_img; do
  [ -s "$OUT/$c.csv" ] && { echo "skip $c" >> "$OUT/runner.log"; continue; }
  g=$(free_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
  echo "$(date '+%F %T') start $c (GPU $g)" >> "$OUT/runner.log"
  CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/modality_zero_ablation.py \
    --cfg "$CFG" --model_path "$CKPT" --split test --batch "$BATCH" \
    --only "$c" --out "$OUT/$c.json" --per_image_csv "$OUT/$c.csv" \
    > "$OUT/$c.log" 2>&1 &
  sleep 90
done
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
