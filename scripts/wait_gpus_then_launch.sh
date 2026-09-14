#!/bin/bash
# 사용법: bash wait_gpus_then_launch.sh <기다릴 tmux 세션> <GPU 목록(쉼표)> <기동 스크립트> <새 tmux 세션명> [최대 대기 분=600]
#   예: bash wait_gpus_then_launch.sh rescore_e1conf_s2 0,3 /SSDb/jemo_maeng/run_elorac_s903_jarvis.sh elora_c_s903
# 앞 작업(재채점 연쇄 등)의 tmux 세션이 사라지고 **지정 GPU 가 모두 2000MiB 이하로 빈 뒤에만** 새 세션으로 기동한다.
# 세션이 사라져도 GPU 가 안 비면(다른 작업이 들어왔으면) 기동하지 않고 계속 기다린다 — 남의 GPU 위에 얹지 않는다.
# 최대 대기 시간을 넘기면 기동하지 않고 WAIT_TIMEOUT 을 찍고 끝낸다. 마커: WAIT_* / LAUNCHED / LAUNCH_ABORT.
set -u
PREV="$1"; GPUS="$2"; RUN="$3"; NEW="$4"; MAXMIN="${5:-600}"
[ -f "$RUN" ] || { echo "LAUNCH_ABORT: 기동 스크립트 없음 ($RUN)"; exit 1; }
tmux has-session -t "=$NEW" 2>/dev/null && { echo "LAUNCH_ABORT: 세션 $NEW 이 이미 있다"; exit 1; }
echo "WAIT_START $(date '+%F %T') 앞 세션=$PREV GPU=$GPUS → $NEW"
free() { for g in ${GPUS//,/ }; do m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$m" -le 2000 ] || return 1; done; return 0; }
i=0
while [ $i -lt "$MAXMIN" ]; do
  if ! tmux has-session -t "=$PREV" 2>/dev/null && free; then
    sleep 30; free || { i=$((i+1)); sleep 30; continue; }   # 30초 뒤 한 번 더 확인(해제 중 순간값 배제)
    tmux new-session -d -s "$NEW" "bash $RUN"
    sleep 3; tmux has-session -t "=$NEW" 2>/dev/null && echo "LAUNCHED $(date '+%F %T') $NEW (GPU $GPUS)" || echo "LAUNCH_ABORT: $NEW 세션이 뜨지 않았다"
    exit 0
  fi
  i=$((i+1)); sleep 60
done
echo "WAIT_TIMEOUT $(date '+%F %T') — ${MAXMIN}분 동안 $PREV 종료+GPU 해제가 오지 않아 기동하지 않았다"
