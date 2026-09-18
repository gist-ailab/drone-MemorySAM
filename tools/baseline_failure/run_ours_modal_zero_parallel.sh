#!/bin/bash
# 우리 모델(E1 확정 시드1, P46 C3-only) 모달 제거 측정 — 조건 하나를 GPU 한 장에 맡겨
# 동시에 돌린다. 한 프로세스로 다섯 조건을 직렬 처리하면 두 시간이 넘어가기 때문이다.
# 개입 축은 기준선과 같다: 정규화를 마친 입력 텐서를 0 으로 둔다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/src/drone-MemorySAM
CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
CKPT=$REPO/outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200/DELIVER_ReliaDINO-ViTL16_idel/epoch140_68.9_top1_checkpoint.pth
OUT=$REPO/modal_zero_ours_par
BATCH=${BATCH:-8}
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
# jarvis 기본 환경의 timm 0.4.12 는 DINOv3 백본 이름을 모른다. servers.conf jarvis 행 참조.
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$REPO/semseg/models/sam2:$REPO
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

free_gpus() {  # GPU0 은 사용자 예약이라 제외
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$1 != 0 && $2 <= 2000 && $3 <= 10 {print $1}'
}

CONDS=(base zero_img zero_depth zero_event zero_lidar)

echo "$(date '+%F %T') 시작 (batch=$BATCH)" > "$OUT/runner.log"
i=0
for c in "${CONDS[@]}"; do
  if [ -s "$OUT/$c.json" ]; then echo "skip $c" >> "$OUT/runner.log"; continue; fi
  GPUS=($(free_gpus))
  while [ "${#GPUS[@]}" -eq 0 ]; do sleep 60; GPUS=($(free_gpus)); done
  g=${GPUS[0]}
  echo "$(date '+%F %T') start $c (GPU $g)" >> "$OUT/runner.log"
  CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/modality_zero_ablation.py \
    --cfg "$CFG" --model_path "$CKPT" --split test --batch "$BATCH" \
    --only "$c" --out "$OUT/$c.json" > "$OUT/$c.log" 2>&1 &
  i=$((i+1))
  sleep 90   # 다음 조건을 고르기 전에 이 프로세스의 메모리 점유가 nvidia-smi 에 반영되게 둔다
done
wait
echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
for c in "${CONDS[@]}"; do
  printf "%-12s %s\n" "$c" "$(grep -o '\"'"$c"'\": [0-9.]*' "$OUT/$c.json" 2>/dev/null | tail -1)"
done | tee -a "$OUT/runner.log"
