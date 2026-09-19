#!/bin/bash
# 우리 모델(E1 확정 시드1) depth·RGB 부분 열화(RMM r=0.5) test 측정.
# 기준선과 같은 규약이다: 정규화된 값에서 원소별 rand<r 자리를 0 으로 둔다.
# 조건을 하나씩 따로 돌려 두 모달이 같은 시드에서 시작하게 한다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/src/drone-MemorySAM
CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
CKPT=$REPO/outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200/DELIVER_ReliaDINO-ViTL16_idel/epoch140_68.9_top1_checkpoint.pth
OUT=$REPO/rmm_ours
RATIO=${RATIO:-0.5}
SEED=${SEED:-0}
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

echo "$(date '+%F %T') 시작 (ratio=$RATIO seed=$SEED batch=$BATCH)" > "$OUT/runner.log"
for c in zero_depth zero_img; do
  tag="rmm_${c}"
  [ -s "$OUT/$tag.json" ] && { echo "skip $tag" >> "$OUT/runner.log"; continue; }
  g=$(free_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
  echo "$(date '+%F %T') start $tag (GPU $g)" >> "$OUT/runner.log"
  CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/modality_zero_ablation.py \
    --cfg "$CFG" --model_path "$CKPT" --split test --batch "$BATCH" \
    --only "$c" --ratio "$RATIO" --ratio_seed "$SEED" \
    --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 &
  sleep 90
done
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
for f in "$OUT"/rmm_*.json; do
  printf "%-16s %s\n" "$(basename "$f" .json)" "$(grep -o '[0-9]\+\.[0-9]\+' "$f" | head -1)"
done | tee -a "$OUT/runner.log"
