#!/bin/bash
# CAFuser (b) 재개 기동: output/..._degrade/model_0089999.pth(iter 90k)에서 --resume 으로 이어 간다.
# 판정 세션 옵션 A: 열화 커리큘럼 [1.0,1.0,1.0] 명시, MAX_ITER 200000, bs8, 4 GPU.
# 판정 세션이 기동을 지시할 때만 실행한다. 빈 GPU 4장이 3회 연속 유지될 때까지 기다린 뒤 시작한다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade_resume.yaml
OUT=output/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade
NEED=${NEED:-4}
STREAK=${STREAK:-3}
TS=$(date +%Y%m%d_%H%M%S)
LOG=$REPO/logs/cafuser_deliver_degrade_resume_${TS}.log
mkdir -p "$REPO/logs"
cd "$REPO"

# 재개 전제: last_checkpoint 가 iter 90k 파일을 가리켜야 한다.
if [ "$(cat $OUT/last_checkpoint)" != "model_0089999.pth" ]; then
  echo "last_checkpoint 가 model_0089999.pth 가 아니다: $(cat $OUT/last_checkpoint) — 기동하지 않는다"
  exit 2
fi

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 <= 2000 && $3 <= 10 {print $1}'
}

CANDIDATE=()
ok=0
echo "$(date '+%F %T') 빈 GPU ${NEED}장이 ${STREAK}회 연속 유지될 때까지 대기"
for i in $(seq 1 480); do
  CUR=($(free_gpus))
  if [ "${#CUR[@]}" -ge "$NEED" ]; then
    if [ "$ok" -eq 0 ]; then
      CANDIDATE=("${CUR[@]}")
    else
      INTER=($(comm -12 <(printf '%s\n' "${CANDIDATE[@]}" | sort) <(printf '%s\n' "${CUR[@]}" | sort)))
      CANDIDATE=("${INTER[@]}")
    fi
    if [ "${#CANDIDATE[@]}" -ge "$NEED" ]; then
      ok=$((ok + 1))
      echo "$(date '+%F %T') 후보 ${CANDIDATE[*]} 연속 ${ok}/${STREAK}"
    else
      ok=0
      CANDIDATE=("${CUR[@]}")
    fi
  else
    ok=0
    CANDIDATE=()
  fi
  [ "$ok" -ge "$STREAK" ] && break
  sleep 60
done
if [ "$ok" -lt "$STREAK" ]; then
  echo "$(date '+%F %T') ${NEED}장을 ${STREAK}회 연속 못 얻었다 — 기동하지 않는다"
  exit 1
fi
SEL=$(IFS=,; echo "${CANDIDATE[*]:0:$NEED}")
echo "$(date '+%F %T') GPU $SEL 확보 — 재개 기동"

source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=$SEL nohup python train_net.py \
  --config-file "$CFG" --num-gpus "$NEED" --resume \
  OUTPUT_DIR "$OUT" \
  > "$LOG" 2>&1 &
echo "$(date '+%F %T') 기동 pid=$! log=$LOG"
