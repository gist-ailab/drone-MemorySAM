#!/bin/bash
# DGFusion (b) 재학습 기동 — 모달 드롭 0.2 + 열화 커리큘럼.
#
# 🔴 배치 결정의 근거. 이 저장소에는 gradient accumulation 이 없다(train_net.py·config.py·
# dgfusion/ 전체에 accum 관련 코드 없음, 2026-09-20 확인). 따라서 "장당 2샘플 + 누적 2회"
# 는 코드를 고치지 않고는 불가능하다. 대신 빈 GPU 4 장을 기다려 (a) 발표 레시피를 그대로
# 쓴다(IMS_PER_BATCH 8 = 장당 2). 이렇게 해야 (a) 와 (b) 가 배치·LR·스케줄이 같아 비교가
# 성립하고, 4 장이면 30 시간으로 2 장 60 시간보다도 빠르다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade.yaml
NEED=${NEED:-4}
TS=$(date +%Y%m%d_%H%M%S)
LOG=$REPO/logs/dgfusion_deliver_degrade_${TS}.log
mkdir -p "$REPO/logs"
cd "$REPO"

free_gpus() {  # 빈 GPU 목록. GPU0 은 사용자 예약이라 제외한다.
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$1 != 0 && $2 <= 2000 && $3 <= 10 {print $1}'
}

echo "$(date '+%F %T') 빈 GPU ${NEED}장 대기"
for i in $(seq 1 480); do
  G=($(free_gpus))
  [ "${#G[@]}" -ge "$NEED" ] && break
  sleep 60
done
G=($(free_gpus))
if [ "${#G[@]}" -lt "$NEED" ]; then
  echo "$(date '+%F %T') ${NEED}장을 못 얻었다(현재 ${#G[@]}장: ${G[*]:-없음}) — 기동하지 않는다"
  exit 1
fi
SEL=$(IFS=,; echo "${G[*]:0:$NEED}")
echo "$(date '+%F %T') GPU $SEL 확보 — 기동"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline
# 80k 이후 fp16 이 task_mlp 에서 넘치는 것이 확인돼 있어 bf16 으로 돈다(ISSUE 기록 참조).
export DGFUSION_AMP_BF16=1

CUDA_VISIBLE_DEVICES=$SEL nohup python train_net.py \
  --config-file "$CFG" --num-gpus "$NEED" \
  OUTPUT_DIR output/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade \
  > "$LOG" 2>&1 &
echo "$(date '+%F %T') 기동 pid=$! log=$LOG"
echo "$LOG" > /tmp/dgf_degrade_logpath
