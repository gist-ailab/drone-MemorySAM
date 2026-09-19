#!/bin/bash
# hpca100 에서 E17 40ep 스크린 체크포인트의 test 예측을 1024 좌표로 덤프한다.
# --save_1024 는 기준선과 같은 채점 좌표계(GT 1024 축소)로 저장하기 위한 것이다.
set -uo pipefail
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
WORK=/home/jovyan/SSDb/jemo_maeng/d4_ours
CFG=$WORK/cfg/hpca100-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E17.yaml
CKPT=$WORK/ckpts/E17_s1_epoch35.pth
OUT=$WORK/preds_e17
BATCH=${BATCH:-8}
ALLOWED=${ALLOWED:-"1 2 3"}
mkdir -p "$OUT"
cd "$REPO"

source /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/activate
export PYTHONPATH=$REPO/semseg/models/sam2:$REPO
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
_CUDNN=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
[ -n "$_CUDNN" ] && export LD_LIBRARY_PATH="$_CUDNN:${LD_LIBRARY_PATH:-}"

free_gpu() {
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

G=$(free_gpu)
while [ -z "$G" ]; do sleep 60; G=$(free_gpu); done
echo "$(date '+%F %T') GPU $G 에서 덤프 시작 (batch $BATCH)"
CUDA_VISIBLE_DEVICES=$G python tools/baseline_failure/dump_preds_ours.py \
  --cfg "$CFG" --model_path "$CKPT" --split test --out "$OUT" \
  --batch "$BATCH" --save_1024 2>&1 | tail -20
echo "$(date '+%F %T') 덤프 종료 exit=$?"
find "$OUT" -name '*.png' | wc -l
