#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# hpca100에서 실행: P56-A 시드821(2GPU) 완주 감지 → 즉시 시드902를 같은 GPU1,2로 2GPU DDP 재개.
# 로컬 세션의 네트워크·메모리 상태와 무관하게 서버 쪽에서 자체적으로 완주 감지+후속 기동을 처리한다.
set -u
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
S821_SESSION=p56a_s821_2gpu
S821_LOG=$REPO/logs/p56a_s821_2gpu_launch_20261001.log
S902_LOG=$REPO/logs/p56a_s902_2gpu_launch_20261001.log
HANDOFF_LOG=$REPO/logs/p56a_handoff.log

echo "HANDOFF_WATCH_START $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$HANDOFF_LOG"

while tmux has-session -t "$S821_SESSION" 2>/dev/null; do
  sleep 60
done

sleep 5
if grep -qE "Total Training Time|epoch:40" "$S821_LOG" 2>/dev/null; then
  STATUS=completed
else
  STATUS=ended_without_epoch40_marker
fi
echo "HANDOFF_SESSION_ENDED $(date -u '+%Y-%m-%d %H:%M:%S UTC') status=$STATUS" >> "$HANDOFF_LOG"

if [ "$STATUS" = "completed" ]; then
  cd "$REPO" || exit 1
  export CUDA_VISIBLE_DEVICES=1,2
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export HF_HOME=/home/jovyan/.cache/huggingface
  export HF_HUB_OFFLINE=1
  export PYTHONPATH=$REPO:$REPO/semseg/models/sam2
  export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
  tmux new-session -d -s p56a_s902_2gpu \
    "cd $REPO && export CUDA_VISIBLE_DEVICES=1,2 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/home/jovyan/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=$REPO:$REPO/semseg/models/sam2 LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 && /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/torchrun --standalone --nproc_per_node=2 --master_port=29802 train_reliadino.py --cfg configs/hpca100-deliver_rgbdel_P46_c3only_seed20260902_screen40_P56A.yaml 2>&1 | tee -a $S902_LOG"
  echo "HANDOFF_LAUNCHED p56a_s902_2gpu $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$HANDOFF_LOG"
else
  echo "HANDOFF_SKIPPED_NOT_COMPLETED — 수동 확인 필요" >> "$HANDOFF_LOG"
fi
