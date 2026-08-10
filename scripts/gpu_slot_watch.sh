#!/usr/bin/env bash
# gpu_slot_watch.sh — 대형 GPU(A100/B200) 슬롯 감시 (2026-08-10 도입)
#
# 목적: hpca100(A100 40GB×4)·elice-b200(B200 공유박스)은 타인 점유로 뺏긴 상태다.
#   빈 슬롯이 "생기는 순간"을 놓치면 다시 뺏긴다(GPU-never-idle의 역방향).
#   cron이 주기적으로 훑어 빈 GPU 수의 **상태 전이(부족→충분)** 시에만 알린다.
#
# 판정 기준: CLAUDE.md 빈 GPU 규칙 그대로 — memory.used ≤ 2000MiB && util ≤ 10%.
# 알림: watchdog 규약 재사용 — $HUB/.watchdog/alerts.log append + notify-send.
# A100 슬롯 확보 시 투입 우선순위(plan.md "A100 대기열" 절 참조):
#   ① C2-MCC 순기여 (2장) ② ProbeA2-7B (1장, 반나절) ③ P49 @1024 대조 (4장)
#
# cron 예: */10 * * * * bash <repo>/scripts/gpu_slot_watch.sh scan
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH_DIR="$REPO_DIR/.watchdog"
STATE_FILE="$WATCH_DIR/gpu_slot_state"
ALERTS_LOG="$WATCH_DIR/alerts.log"
mkdir -p "$WATCH_DIR"

MAXMEM="${GPU_MAXMEM:-2000}"   # MiB
MAXUTIL="${GPU_MAXUTIL:-10}"   # %
SERVERS="${SLOT_SERVERS:-hpca100 elice-b200}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10"

_notify() { # <title> <body>
    command -v notify-send >/dev/null 2>&1 && \
        DISPLAY="${DISPLAY:-:0}" notify-send -u critical "$1" "$2" >/dev/null 2>&1 || true
}

count_free() { # <server> -> "free_count/total" (실패 시 "ssh-fail")
    local out
    out=$(ssh $SSH_OPTS "$1" \
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null) \
        || { echo "ssh-fail"; return; }
    local total=0 free=0 mem util
    while IFS=', ' read -r mem util; do
        [ -z "${mem:-}" ] && continue
        total=$((total+1))
        if [ "$mem" -le "$MAXMEM" ] && [ "$util" -le "$MAXUTIL" ]; then free=$((free+1)); fi
    done <<< "$out"
    echo "${free}/${total}"
}

scan() {
    local now line="" prev=""
    now=$(date '+%Y-%m-%d %H:%M:%S')
    [ -f "$STATE_FILE" ] && prev=$(cat "$STATE_FILE")
    for s in $SERVERS; do
        line="$line $s=$(count_free "$s")"
    done
    line="${line# }"
    echo "$line" > "$STATE_FILE"

    # 상태 전이 감지: 어느 서버든 free 수가 이전보다 늘고 1 이상이면 알림
    for s in $SERVERS; do
        local cur_f prev_f
        cur_f=$(echo "$line" | grep -o "$s=[0-9]*" | cut -d= -f2)
        prev_f=$(echo "$prev" | grep -o "$s=[0-9]*" | cut -d= -f2)
        [ -z "${cur_f:-}" ] && continue
        prev_f="${prev_f:-0}"
        if [ "$cur_f" -gt "$prev_f" ] && [ "$cur_f" -ge 1 ]; then
            local msg="[slot-watch] $s free GPU ${prev_f}->${cur_f} @ $now — A100 대기열 투입 검토(①C2 ②7B ③P49@1024)"
            echo "$now $msg" >> "$ALERTS_LOG"
            _notify "GPU slot opened: $s" "free ${prev_f}->${cur_f} — plan.md A100 대기열 확인"
        fi
    done
    echo "$now $line"
}

case "${1:-scan}" in
    scan)   scan ;;
    status) [ -f "$STATE_FILE" ] && cat "$STATE_FILE" || echo "no state yet" ;;
    *) echo "usage: $0 {scan|status}"; exit 1 ;;
esac
