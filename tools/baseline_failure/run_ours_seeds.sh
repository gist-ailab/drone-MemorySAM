#!/bin/bash
# 우리 모델 모달 제거 — 시드 확장판. E1 시드2·3 과 E13 시드1 을 같은 개입 축으로 잰다.
# 조건 하나가 프로세스 하나이고, 허용된 GPU 집합에서 빈 장을 골라 붙인다.
# 판정 세션 지시(2026-09-19): jarvis 4,5,6 만 쓴다(0~3·7 은 감시 세션이 쓴다).
# 이미지별 mIoU 도 같이 남겨 조건별 Δ 산출에 쓴다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/src/drone-MemorySAM
OUT=$REPO/modal_zero_ours_seeds
BATCH=${BATCH:-8}
ALLOWED=${ALLOWED:-"4 5 6"}
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$REPO/semseg/models/sam2:$REPO
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

O=outputs/ReliaDINO
C=configs/eval
# 이름 | 평가 config | 체크포인트(학습기 val-best top1)
RUNS=(
"E1_s2|$C/jarvis-deliver_rgbdel_P46_c3only_seed20260902_eval1024_E1_confirm200_s2.yaml|$O/jarvis_deliver_rgbdel_P46_c3only_seed20260902_E1_confirm200_s2/DELIVER_ReliaDINO-ViTL16_idel/epoch134_68.76_top1_checkpoint.pth"
"E1_s3|$C/jarvis-deliver_rgbdel_P46_c3only_seed20260903_eval1024_E1_confirm200_s3.yaml|$O/jarvis_deliver_rgbdel_P46_c3only_seed20260903_E1_confirm200_s3/DELIVER_ReliaDINO-ViTL16_idel/epoch70_68.53_top1_checkpoint.pth"
"E13_s1|$C/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E13_confirm200.yaml|$O/jarvis_deliver_rgbdel_P46_c3only_seed20260821_E13_confirm200/DELIVER_ReliaDINO-ViTL16_idel/epoch100_67.84_top1_checkpoint.pth"
)
CONDS=(base zero_img zero_depth zero_event zero_lidar)

free_gpu() {  # 허용 집합 안에서 빈 장 하나
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

echo "$(date '+%F %T') 시작 (batch=$BATCH, 허용 GPU: $ALLOWED)" > "$OUT/runner.log"
for r in "${RUNS[@]}"; do
  name=${r%%|*}; rest=${r#*|}; cfg=${rest%%|*}; ckpt=${rest#*|}
  if [ ! -f "$ckpt" ]; then echo "체크포인트 없음: $ckpt" >> "$OUT/runner.log"; continue; fi
  for c in "${CONDS[@]}"; do
    tag="${name}_${c}"
    [ -s "$OUT/$tag.json" ] && { echo "skip $tag" >> "$OUT/runner.log"; continue; }
    g=$(free_gpu)
    while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
    echo "$(date '+%F %T') start $tag (GPU $g)" >> "$OUT/runner.log"
    CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/modality_zero_ablation.py \
      --cfg "$cfg" --model_path "$ckpt" --split test --batch "$BATCH" \
      --only "$c" --out "$OUT/$tag.json" --per_image_csv "$OUT/$tag.csv" \
      > "$OUT/$tag.log" 2>&1 &
    sleep 90   # 이 프로세스의 메모리 점유가 nvidia-smi 에 반영된 뒤 다음 장을 고른다
  done
done
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
for f in "$OUT"/*.json; do
  printf "%-22s %s\n" "$(basename "$f" .json)" "$(grep -o '[0-9]\+\.[0-9]\+' "$f" | head -1)"
done | tee -a "$OUT/runner.log"
