#!/bin/bash
# Unified watchdog — polls every job's log for crash signatures AND staleness
# (no log growth in STALE_MIN minutes while it should be actively running).
# Run per-server via ssh from the hub. Each line printed is one event.
# Args: <log_path1> [<log_path2> ...]
STALE_MIN=25
declare -A last_size
declare -A last_check

while true; do
  for log in "$@"; do
    if [ ! -f "$log" ]; then
      echo "ALERT missing_log log=$log"
      continue
    fi
    if grep -qE 'OutOfMemoryError|ChildFailedError|Traceback \(most recent call last\)' "$log"; then
      # only alert once per log per crash — track via a sentinel file
      sentinel="${log}.watchdog_alerted"
      if [ ! -f "$sentinel" ]; then
        echo "ALERT crash log=$log"
        touch "$sentinel"
      fi
    fi
    cur_size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    now=$(date +%s)
    prev_size=${last_size[$log]:-0}
    prev_check=${last_check[$log]:-0}
    if [ "$prev_check" -ne 0 ]; then
      elapsed=$(( (now - prev_check) / 60 ))
      if [ "$cur_size" = "$prev_size" ] && [ "$elapsed" -ge "$STALE_MIN" ]; then
        sentinel="${log}.watchdog_stale_alerted"
        if [ ! -f "$sentinel" ]; then
          echo "ALERT stale log=$log no_growth_min=$elapsed"
          touch "$sentinel"
        fi
      else
        rm -f "${log}.watchdog_stale_alerted" 2>/dev/null
      fi
    fi
    last_size[$log]=$cur_size
    last_check[$log]=$now
  done
  sleep 300
done
