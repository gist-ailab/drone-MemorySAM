#!/bin/bash
# [2026-10-31 paper 세션, user 지시 "GPU 비는 대로 이어 학습해"] MCubeS E1Mc+고해상 경로(C2D2 없음) 시드 20260827 재개 감시.
# 10-26 03:56 UTC 128/200에폭에서 외부 종료된 런을 같은 체크아웃·config 로 AUTO_RESUME 한다(last_checkpoint.pth md5 e870b2eb…).
# 빈 GPU 기준 = memory.used <= 2000MiB && util <= 10% 를 연속 2회(2분 간격) 만족. 1장만 쓴다(원 런 world_size 1).
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
CFG=configs/hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2hires.yaml
NAME=hpca100-mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2hires
VENV=/home/jovyan/SSDb/jemo_maeng/venv/p34
cd "$REPO" || exit 1
pick() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2<=2000 && $3<=10 {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}
echo "[watch] start $(date -u +%F' '%T) UTC"
while true; do
  g1=$(pick)
  if [ -n "$g1" ]; then
    sleep 120
    g2=$(pick)
    if [ "$g1" = "$g2" ]; then break; fi
  fi
  sleep 120
done
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/$NAME/${NAME}_resume_${TS}.log
echo "[watch] GPU $g1 free -> launch $(date -u +%F' '%T) UTC, log $LOG"
export PYTHONPATH=$REPO/semseg/models/sam2:$REPO
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=$g1 $VENV/bin/torchrun --standalone --nproc_per_node=1 train_reliadino.py --cfg $CFG > "$LOG" 2>&1
echo "[watch] train exited code $? $(date -u +%F' '%T) UTC"
