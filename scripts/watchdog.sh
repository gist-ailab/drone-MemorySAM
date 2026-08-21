#!/usr/bin/env bash
# watchdog.sh — babysit remote MemorySAM training runs so nobody has to poll them by hand.
#
# A run is registered by `scripts/remote_exp.sh run` (or manually, see `register`) as
#   .watchdog/runs/<run_id>/manifest.json
# and this script periodically ssh's to the server ONCE per run to collect log tail +
# GPU stats + process liveness, then decides a state and fires desktop/log alerts.
#
# Usage:
#   bash scripts/watchdog.sh [scan]                        # one sweep over active runs (default)
#   bash scripts/watchdog.sh status                        # table of active runs
#   bash scripts/watchdog.sh register <server> <config.yaml> <gpus_csv> <remote_log_path> [--repo PATH]
#   bash scripts/watchdog.sh close <run_id> [reason]       # stop watching a run
#   bash scripts/watchdog.sh install-cron | uninstall-cron
#
# States: launching -> running -> {completed | dead | stalled | failed_startup}
#   terminal = completed | dead | failed_startup | closed  (skipped by later scans)
#
# Tunables (env):
#   WATCHDOG_DIR           state root                       (default <repo>/.watchdog)
#   WATCHDOG_STARTUP_MIN   launching grace, minutes         (15)
#   WATCHDOG_STALL_MIN     no-iter-progress tolerance, min  (30)
#   WATCHDOG_STALL_UTIL    "GPU is idle" util threshold, %  (20)
#   WATCHDOG_SSH_TIMEOUT   ssh ConnectTimeout, seconds      (10)
#   WATCHDOG_TAIL_BYTES    log tail size                    (20000)
#   WATCHDOG_ITER_REGEX    ERE override for iteration parsing (must contain "<cur>/<total>")
#   WATCHDOG_ONALERT_CMD   hook: "$CMD <run_id> <state> <manifest_path>" on every alert
#
# NTFS repo: the exec bit does not stick — always invoke as `bash scripts/watchdog.sh ...`.
# `WATCHDOG_LIB=1 source scripts/watchdog.sh` loads the functions without running anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONF="${REMOTE_EXP_CONF:-$SCRIPT_DIR/servers.conf}"
WATCHDOG_DIR="${WATCHDOG_DIR:-$REPO_ROOT/.watchdog}"
RUNS_DIR="$WATCHDOG_DIR/runs"
ALERTS_LOG="$WATCHDOG_DIR/alerts.log"

WATCHDOG_STARTUP_MIN="${WATCHDOG_STARTUP_MIN:-15}"
WATCHDOG_STALL_MIN="${WATCHDOG_STALL_MIN:-30}"
WATCHDOG_STALL_UTIL="${WATCHDOG_STALL_UTIL:-20}"
WATCHDOG_SSH_TIMEOUT="${WATCHDOG_SSH_TIMEOUT:-10}"
WATCHDOG_TAIL_BYTES="${WATCHDOG_TAIL_BYTES:-20000}"

CRON_TAG="# memorysam-watchdog"
CRON_LOCK="/tmp/memorysam-watchdog.lock"

die() { echo "ERROR: $*" >&2; exit 1; }
now_iso() { date -Iseconds; }
now_s()   { date +%s; }

# ---------------------------------------------------------------------------
# pure helpers (no ssh, no filesystem) — unit-tested by tests/test_watchdog.sh
# ---------------------------------------------------------------------------

# tqdm redraws a bar with \r, so a raw tail is one giant line. Split it back up.
wd_normalize() { tr '\r' '\n'; }

# wd_parse_iter <log text> -> "<cur> <total>"  (rc 1 if nothing recognisable)
#
# Formats actually emitted by this repo's trainers (verified in the sources):
#   train_sam2_lora_paper.py   tqdm desc  "Epoch: [12/200] Iter: [73/187] LR: ... Loss: ..."
#   train_reliadino.py         tqdm desc  "Epoch [12/200] Loss 0.1 ..." + bar " 45%|##| 84/187 ["
#   train_det.py               tqdm desc  "Epoch [12]" + bar " 45%|##| 84/187 ["
#   train_sam3_rbma.py         plain      "ep12 it40/187 loss=0.4 (...)"
# WATCHDOG_ITER_REGEX overrides all of them; its match must contain "<cur>/<total>".
wd_parse_iter() {
  local text="$1" m="" re
  if [ -n "${WATCHDOG_ITER_REGEX:-}" ]; then
    m="$(grep -oE "$WATCHDOG_ITER_REGEX" <<<"$text" | tail -1 || true)"
  else
    for re in 'Iter:?[[:space:]]*\[[0-9]+/[0-9]+\]' \
              '\|[[:space:]]*[0-9]+/[0-9]+[[:space:]]*\[' \
              '(^|[[:space:]])it[0-9]+/[0-9]+([^0-9]|$)'; do
      m="$(grep -oE "$re" <<<"$text" | tail -1 || true)"
      [ -n "$m" ] && break
    done
  fi
  [ -n "$m" ] || return 1
  grep -oE '[0-9]+/[0-9]+' <<<"$m" | tail -1 | tr '/' ' '
}

# wd_parse_epoch <log text> -> "<cur> <total>"  (total 0 when the trainer omits it)
wd_parse_epoch() {
  local text="$1" m
  m="$(grep -oE 'Epoch:?[[:space:]]*\[[0-9]+/[0-9]+\]' <<<"$text" | tail -1 || true)"
  if [ -n "$m" ]; then grep -oE '[0-9]+/[0-9]+' <<<"$m" | tr '/' ' '; return 0; fi
  m="$(grep -oE 'Epoch:?[[:space:]]*\[[0-9]+\]' <<<"$text" | tail -1 || true)"
  if [ -n "$m" ]; then echo "$(grep -oE '[0-9]+' <<<"$m") 0"; return 0; fi
  m="$(grep -oE '(^|[[:space:]]|\[)ep[0-9]+([][:space:]])' <<<"$text" | tail -1 || true)"
  if [ -n "$m" ]; then echo "$(grep -oE '[0-9]+' <<<"$m") 0"; return 0; fi
  return 1
}

# wd_scan_errors <log text> -> space separated signature tokens (possibly empty)
#   RANDOM_INIT : semseg/models/reliadino/encoder.py falls back to an untrained backbone
#                 when the HF download fails (the hpca100 trap). Fatal, always alert.
#   NAN_LOSS    : nan/inf but only in a loss context, so "inference"/"info" can't trip it.
# (here-strings, not pipes: `grep -q` exits on the first hit and a pipe under
#  `set -o pipefail` would then report the writer's SIGPIPE as a failure.)
wd_scan_errors() {
  local text="$1" out=""
  grep -q  'RANDOM INIT'                       <<<"$text" && out="$out RANDOM_INIT"
  grep -q  'Traceback (most recent call last)' <<<"$text" && out="$out TRACEBACK"
  grep -qi 'CUDA out of memory'                <<<"$text" && out="$out OOM"
  grep -qE 'NCCL.*(error|Error|ERROR|timeout|timed out|abort)|ncclInternalError|Watchdog caught collective' \
           <<<"$text" && out="$out NCCL"
  grep -qiE '[Ll]oss[A-Za-z_]*[[:space:]]*[:=][[:space:]]*[-+]?(nan|inf)\b' \
           <<<"$text" && out="$out NAN_LOSS"
  echo "${out# }"
}

# wd_has_completion <log text> -> rc 0 when the trainer printed its normal end-of-run banner.
#   train_sam2_lora_paper.py / train_reliadino.py : tabulate row "Total Training Time"
#   train_det.py                                  : "Training complete. Best AP:"
# (train_sam3_rbma.py prints no end banner — see the epoch==total fallback in wd_decide_state.)
wd_has_completion() {
  grep -qE 'Total Training Time|Training complete\. Best AP' <<<"$1"
}

wd_is_terminal() {
  case "$1" in completed|dead|failed_startup|closed) return 0 ;; *) return 1 ;; esac
}

# wd_max <space separated ints> -> max (0 when empty)
wd_max() {
  local best=0 v
  for v in $1; do [ "$v" -gt "$best" ] 2>/dev/null && best="$v"; done
  echo "$best"
}

# wd_decide_state <cur_state> <alive> <progressed> <secs_since_launch> <secs_since_progress> \
#                 <max_util> <errors> <completed> [epoch_done]
# -> "<new_state> <alert_kind|->".  Mechanises CLAUDE.md §1.6's launch-verification rules.
wd_decide_state() {
  local st="$1" alive="$2" prog="$3" since_launch="$4" since_prog="$5" \
        util="$6" errs="$7" done_marker="$8" epoch_done="${9:-0}"
  local startup_s=$(( WATCHDOG_STARTUP_MIN * 60 )) stall_s=$(( WATCHDOG_STALL_MIN * 60 ))

  if wd_is_terminal "$st"; then echo "$st -"; return 0; fi

  # RANDOM INIT means the backbone never loaded its pretrained weights. Iterations may well
  # be advancing, but the run is worthless — condemn it immediately regardless of progress.
  case " $errs " in
    *" RANDOM_INIT "*) echo "failed_startup failed_startup"; return 0 ;;
  esac

  if [ "$alive" -eq 0 ]; then
    # normal end-of-run banner, or (sam3-style trainers with no banner) the last epoch finished
    if [ "$done_marker" -eq 1 ] || [ "$epoch_done" -eq 1 ]; then echo "completed completed"; return 0; fi
    # never got going -> a startup failure, not a mid-training death
    if [ "$st" = "launching" ]; then echo "failed_startup failed_startup"; return 0; fi
    echo "dead dead"; return 0
  fi

  case "$st" in
    launching)
      if [ "$prog" -eq 1 ]; then echo "running -"; return 0; fi
      case " $errs " in
        *" TRACEBACK "*) echo "failed_startup failed_startup"; return 0 ;;
      esac
      # 15 min with zero iteration progress. The classic shape is rank0 pinned at 0% util
      # inside a collective (NCCL deadlock) while the other ranks spin — see CLAUDE.md §1.6.
      if [ "$since_launch" -ge "$startup_s" ]; then echo "failed_startup failed_startup"; return 0; fi
      echo "launching -"; return 0
      ;;
    running|stalled|unreachable)
      if [ "$prog" -eq 1 ]; then
        if [ "$st" = "stalled" ]; then echo "running recovered"; else echo "running -"; fi
        return 0
      fi
      # AND of "no progress" and "GPUs idle": eval passes freeze the iter counter for many
      # minutes while the GPUs stay busy, and those must not be reported as stalls.
      if [ "$since_prog" -ge "$stall_s" ] && [ "$util" -le "$WATCHDOG_STALL_UTIL" ]; then
        if [ "$st" = "stalled" ]; then echo "stalled -"; else echo "stalled stalled"; fi
        return 0
      fi
      echo "${st/unreachable/running} -"; return 0
      ;;
    *) echo "$st -"; return 0 ;;
  esac
}

# ---------------------------------------------------------------------------
# notification
# ---------------------------------------------------------------------------

# _notify <urgency> <title> <body>
# cron has no DISPLAY/DBUS_SESSION_BUS_ADDRESS, so lift them out of the running
# gnome-shell's /proc/<pid>/environ (this machine's documented convention).
# Never fatal: a failed notification must not abort the scan.
_notify() {
  local urgency="$1" title="$2" body="$3"
  (
    set +e
    command -v notify-send >/dev/null 2>&1 || exit 0
    if [ -z "${DISPLAY:-}" ] || [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
      local pid env_disp env_dbus
      pid="$(pgrep -u "$(id -u)" -n gnome-shell 2>/dev/null | head -1)"
      if [ -n "$pid" ] && [ -r "/proc/$pid/environ" ]; then
        env_disp="$(tr '\0' '\n' < "/proc/$pid/environ" | grep -m1 '^DISPLAY=')"
        env_dbus="$(tr '\0' '\n' < "/proc/$pid/environ" | grep -m1 '^DBUS_SESSION_BUS_ADDRESS=')"
        [ -n "$env_disp" ] && export "${env_disp?}"
        [ -n "$env_dbus" ] && export "${env_dbus?}"
      fi
    fi
    notify-send -u "$urgency" "$title" "$body" >/dev/null 2>&1
  ) || true
}

# wd_alert <run_id> <kind> <state> <manifest_path> <urgency> <message>
wd_alert() {
  local rid="$1" kind="$2" state="$3" manifest="$4" urgency="$5" msg="$6" icon="🔴"
  [ "$urgency" = "low" ] && icon="🟢"
  mkdir -p "$WATCHDOG_DIR"
  printf '%s\t%s\t%s\t%s\n' "$(now_iso)" "$rid" "$kind" "$msg" >> "$ALERTS_LOG"
  echo "$icon ALERT [$kind] $rid — $msg"
  _notify "$urgency" "$icon MemorySAM: $rid" "$msg"
  if [ -n "${WATCHDOG_ONALERT_CMD:-}" ]; then
    ( set +e; $WATCHDOG_ONALERT_CMD "$rid" "$state" "$manifest" >/dev/null 2>&1 ) || true
  fi
}

# ---------------------------------------------------------------------------
# json / run bookkeeping
# ---------------------------------------------------------------------------

wd_json_get() {  # wd_json_get <file> <key> [default]
  WD_F="$1" WD_K="$2" WD_D="${3:-}" python3 -c '
import json, os, sys
try:
    with open(os.environ["WD_F"]) as f: d = json.load(f)
except Exception:
    d = {}
v = d.get(os.environ["WD_K"], os.environ["WD_D"])
sys.stdout.write("" if v is None else str(v))
'
}

# wd_run_state <run_dir> -> current state (status.json wins over manifest.json)
wd_run_state() {
  local d="$1" s=""
  [ -f "$d/status.json" ] && s="$(wd_json_get "$d/status.json" state)"
  [ -n "$s" ] || s="$(wd_json_get "$d/manifest.json" state launching)"
  echo "${s:-launching}"
}

# wd_active_runs -> run_ids whose state is not terminal
wd_active_runs() {
  local d rid
  [ -d "$RUNS_DIR" ] || return 0
  for d in "$RUNS_DIR"/*/; do
    [ -f "$d/manifest.json" ] || continue
    rid="$(basename "$d")"
    wd_is_terminal "$(wd_run_state "$d")" && continue
    echo "$rid"
  done
}

# ---------------------------------------------------------------------------
# remote probe (exactly one ssh per run)
# ---------------------------------------------------------------------------

# wd_remote_probe <server> <remote_log_path> <gpus_csv> <cfg_name> <tail_bytes>
# Prints a sectioned payload; stubbed out by the tests.
wd_remote_probe() {
  ssh -o BatchMode=yes -o "ConnectTimeout=$WATCHDOG_SSH_TIMEOUT" "$1" \
      bash -s -- "$2" "$3" "$4" "$5" <<'REMOTE'
LOG="$1"; GPUS="$2"; CN="$3"; NBYTES="$4"
echo "===WD_TMUX==="
tmux list-windows -t jemo 2>/dev/null || true
echo "===WD_PROC==="
# our own probe shell carries $CN in its argv — exclude it, and only count real trainers
ps -eo pid=,args= 2>/dev/null | grep -F -- "$CN" | grep -vw "$$" | grep -cE 'python|torchrun' || true
echo "===WD_GPU==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null || true
echo "===WD_LOG==="
tail -c "$NBYTES" "$LOG" 2>/dev/null || true
REMOTE
}

# wd_section <NAME> <file> -> the lines of that section
wd_section() {
  awk -v s="===WD_$1===" '
    $0==s {p=1; next}
    /^===WD_[A-Z]+===$/ {if (p) exit; next}
    p {print}' "$2"
}

# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

wd_scan_one() {
  local rid="$1" d="$RUNS_DIR/$1" mf="$RUNS_DIR/$1/manifest.json"
  local server cfg_name gpus log_path launched_at
  server="$(wd_json_get "$mf" server)"
  cfg_name="$(wd_json_get "$mf" cfg_name)"
  gpus="$(wd_json_get "$mf" gpus)"
  log_path="$(wd_json_get "$mf" log_path)"
  launched_at="$(wd_json_get "$mf" launched_at)"
  [ -n "$server" ] && [ -n "$log_path" ] || { echo "WARN: $rid — manifest incomplete, skipping" >&2; return 0; }

  local st prev_iter prev_epoch prev_prog_at prev_unreach prev_alerted had_prev=0
  st="$(wd_run_state "$d")"
  [ -f "$d/status.json" ] && had_prev=1
  prev_iter="$(wd_json_get "$d/status.json" last_iter 0)";     prev_iter="${prev_iter:-0}"
  prev_epoch="$(wd_json_get "$d/status.json" last_epoch 0)";   prev_epoch="${prev_epoch:-0}"
  prev_prog_at="$(wd_json_get "$d/status.json" progress_at 0)"; prev_prog_at="${prev_prog_at:-0}"
  prev_unreach="$(wd_json_get "$d/status.json" unreachable_count 0)"; prev_unreach="${prev_unreach:-0}"
  prev_alerted="$(wd_json_get "$d/status.json" alerted "")"

  local launched_s now t_probe rc=0
  now="$(now_s)"
  launched_s="$(date -d "$launched_at" +%s 2>/dev/null || echo "$now")"
  [ "$prev_prog_at" -gt 0 ] 2>/dev/null || prev_prog_at="$launched_s"

  t_probe="$(mktemp "${TMPDIR:-/tmp}/wd_probe.XXXXXX")"
  wd_remote_probe "$server" "$log_path" "$gpus" "$cfg_name" "$WATCHDOG_TAIL_BYTES" \
    > "$t_probe" 2>/dev/null || rc=$?

  if [ "$rc" -ne 0 ]; then
    rm -f "$t_probe"
    local unreach=$(( prev_unreach + 1 ))
    echo "  $rid: ssh FAILED (consecutive=$unreach)"
    WD_STATE="unreachable"; WD_REACHABLE=0; WD_UNREACH="$unreach"; WD_ALIVE=0
    WD_LAST_ITER="$prev_iter"; WD_ITERS_TOTAL=0; WD_LAST_EPOCH="$prev_epoch"; WD_EPOCHS_TOTAL=0
    WD_PROGRESSED=0; WD_GPU_UTIL=""; WD_GPU_MEM=""; WD_PROGRESS_AT="$prev_prog_at"
    WD_ALERTED="$prev_alerted"; WD_NEW_ALERT=""; WD_ERRORS=""
    wd_write_status "$d" "$rid"
    # hpca100's MTU flapping produces isolated ssh failures — only shout on the 2nd in a row
    if [ "$unreach" -ge 2 ]; then
      wd_alert "$rid" unreachable unreachable "$mf" critical \
        "$server 에 ${unreach}회 연속 ssh 실패 — 서버/네트워크 확인 필요 (log: $log_path)"
    fi
    return 0
  fi

  local proc_n log_txt gpu_lines util_list mem_list max_util alive
  proc_n="$(wd_section PROC "$t_probe" | tr -dc '0-9\n' | tail -1)"; proc_n="${proc_n:-0}"
  gpu_lines="$(wd_section GPU "$t_probe")"
  log_txt="$(wd_section LOG "$t_probe" | wd_normalize)"
  util_list="$(printf '%s\n' "$gpu_lines" | awk -F',' 'NF>=3 {gsub(/ /,"",$2); print $2}')"
  mem_list="$(printf '%s\n' "$gpu_lines"  | awk -F',' 'NF>=3 {gsub(/ /,"",$3); print $3}')"
  max_util="$(wd_max "$(printf '%s\n' "$util_list" | tr '\n' ' ')")"
  alive=0; [ "$proc_n" -gt 0 ] 2>/dev/null && alive=1

  local iter_pair epoch_pair cur_iter tot_iter cur_epoch tot_epoch
  iter_pair="$(wd_parse_iter "$log_txt" || true)"
  epoch_pair="$(wd_parse_epoch "$log_txt" || true)"
  read -r cur_iter tot_iter <<<"${iter_pair:-0 0}"
  read -r cur_epoch tot_epoch <<<"${epoch_pair:-0 0}"

  local errs done_marker=0 epoch_done=0
  errs="$(wd_scan_errors "$log_txt")"
  wd_has_completion "$log_txt" && done_marker=1
  [ "$tot_epoch" -gt 0 ] && [ "$cur_epoch" -ge "$tot_epoch" ] && [ "$tot_iter" -gt 0 ] \
    && [ "$cur_iter" -ge "$tot_iter" ] && epoch_done=1

  # progress = a strictly larger (epoch, iter) than the PREVIOUS SCAN. The very first scan
  # has no baseline, so it only records one — CLAUDE.md §1.6 wants two observations before
  # calling a run alive (that is exactly what the "73/187 → 92/187 after 25s" rule means).
  local key prev_key progressed=0 prog_at="$prev_prog_at"
  key=$(( cur_epoch * 10000000 + cur_iter ))
  prev_key=$(( prev_epoch * 10000000 + prev_iter ))
  if [ "$had_prev" -eq 0 ]; then
    prog_at="$now"
  elif [ "$key" -gt "$prev_key" ]; then
    progressed=1; prog_at="$now"
  fi

  local since_launch=$(( now - launched_s )) since_prog=$(( now - prog_at ))
  local decision new_state alert_kind
  decision="$(wd_decide_state "$st" "$alive" "$progressed" "$since_launch" "$since_prog" \
                              "$max_util" "$errs" "$done_marker" "$epoch_done")"
  read -r new_state alert_kind <<<"$decision"

  printf '  %s: %s  iter=%s/%s ep=%s/%s util=[%s] alive=%s%s\n' \
    "$rid" "$new_state" "$cur_iter" "$tot_iter" "$cur_epoch" "$tot_epoch" \
    "$(printf '%s' "$util_list" | tr '\n' ' ' | sed 's/ $//')" "$alive" \
    "${errs:+  errors=$errs}"

  # keep the last 50 lines around so a dead-run alert can point somewhere useful
  printf '%s\n' "$log_txt" | tail -n 50 > "$d/last50.log"

  # ---- alerts (each kind fires at most once per run) ----
  local new_alerted="$prev_alerted" fired=""
  _already() { case ",$new_alerted," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
  _mark()    { new_alerted="${new_alerted:+$new_alerted,}$1"; fired="$fired $1"; }

  case " $errs " in
    *" RANDOM_INIT "*)
      if ! _already RANDOM_INIT; then
        _mark RANDOM_INIT
        wd_alert "$rid" RANDOM_INIT "$new_state" "$mf" critical \
          "백본이 RANDOM INIT 으로 폴백됐다 — 사전학습 가중치 미로드. 즉시 kill 후 재기동할 것. (log: $log_path)"
      fi ;;
  esac
  case " $errs " in
    *" OOM "*) if ! _already OOM; then _mark OOM
        wd_alert "$rid" OOM "$new_state" "$mf" critical "CUDA out of memory 발견 (log: $log_path)"; fi ;;
  esac
  case " $errs " in
    *" NAN_LOSS "*) if ! _already NAN_LOSS; then _mark NAN_LOSS
        wd_alert "$rid" NAN_LOSS "$new_state" "$mf" critical "loss 가 nan/inf (log: $log_path)"; fi ;;
  esac
  case " $errs " in
    *" NCCL "*) if ! _already NCCL; then _mark NCCL
        wd_alert "$rid" NCCL "$new_state" "$mf" critical "NCCL 에러/타임아웃 발견 (log: $log_path)"; fi ;;
  esac

  if [ "$alert_kind" != "-" ] && ! _already "$alert_kind"; then
    _mark "$alert_kind"
    case "$alert_kind" in
      failed_startup)
        wd_alert "$rid" failed_startup "$new_state" "$mf" critical \
          "기동 실패 — ${WATCHDOG_STARTUP_MIN}분 내 iter 전진 없음(현재 ${cur_iter}/${tot_iter}, max util ${max_util}%${errs:+, $errs}). rank0 util 0% 면 NCCL 데드락 패턴. 로그: $log_path" ;;
      stalled)
        wd_alert "$rid" stalled "$new_state" "$mf" critical \
          "정체 — ${WATCHDOG_STALL_MIN}분 이상 iter 불변(${cur_iter}/${tot_iter}) + GPU util ${max_util}% 저조. 로그: $log_path" ;;
      dead)
        wd_alert "$rid" dead "$new_state" "$mf" critical \
          "프로세스 사망 + 정상 종료 마커 없음 (iter ${cur_iter}/${tot_iter}, ep ${cur_epoch}/${tot_epoch}). 마지막 50줄: $d/last50.log" ;;
      completed)
        wd_alert "$rid" completed "$new_state" "$mf" low \
          "정상 완주 (ep ${cur_epoch}/${tot_epoch}). 슬롯 ${server}:${gpus} 비었음 — .claude_logs/experiments/plan.md 대기열 확인" ;;
      recovered)
        wd_alert "$rid" recovered "$new_state" "$mf" low "정체에서 회복 — iter 다시 전진 중 (${cur_iter}/${tot_iter})" ;;
    esac
  fi
  unset -f _already _mark

  WD_STATE="$new_state"; WD_REACHABLE=1; WD_UNREACH=0; WD_ALIVE="$alive"
  WD_LAST_ITER="$cur_iter"; WD_ITERS_TOTAL="$tot_iter"
  WD_LAST_EPOCH="$cur_epoch"; WD_EPOCHS_TOTAL="$tot_epoch"
  WD_PROGRESSED="$progressed"
  WD_GPU_UTIL="$(printf '%s' "$util_list" | tr '\n' ',' | sed 's/,$//')"
  WD_GPU_MEM="$(printf '%s' "$mem_list" | tr '\n' ',' | sed 's/,$//')"
  WD_PROGRESS_AT="$prog_at"; WD_ALERTED="$new_alerted"; WD_NEW_ALERT="${fired# }"
  WD_ERRORS="$errs"
  wd_write_status "$d" "$rid"

  rm -f "$t_probe"
}

# wd_write_status <run_dir> <run_id> — everything else comes in through WD_* variables.
# They are re-exported here so the python heredoc sees them regardless of how the caller
# set them (a `VAR=x wd_write_status` prefix on a *function* is not portably exported).
wd_write_status() {
  export WD_STATE WD_REACHABLE WD_UNREACH WD_ALIVE WD_LAST_ITER WD_ITERS_TOTAL \
         WD_LAST_EPOCH WD_EPOCHS_TOTAL WD_PROGRESSED WD_PROGRESS_AT \
         WD_GPU_UTIL WD_GPU_MEM WD_ALERTED WD_NEW_ALERT
  export WD_ERRORS="${WD_ERRORS:-}"
  WD_DIR="$1" WD_RID="$2" WD_AT="$(now_iso)" python3 - <<'PY'
import json, os, time

d = os.environ["WD_DIR"]
path = os.path.join(d, "status.json")
try:
    with open(path) as f:
        prev = json.load(f)
except Exception:
    prev = {}


def ints(name):
    raw = os.environ.get(name, "")
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


def i(name, default=0):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


alerts = prev.get("alerts", [])
for kind in os.environ.get("WD_NEW_ALERT", "").split():
    alerts.append({"at": os.environ["WD_AT"], "kind": kind})

alerted = [a for a in os.environ.get("WD_ALERTED", "").split(",") if a]

st = {
    "run_id":            os.environ["WD_RID"],
    "checked_at":        os.environ["WD_AT"],
    "reachable":         bool(i("WD_REACHABLE")),
    "unreachable_count": i("WD_UNREACH"),
    "alive":             bool(i("WD_ALIVE")),
    "last_iter":         i("WD_LAST_ITER"),
    "iters_total":       i("WD_ITERS_TOTAL"),
    "last_epoch":        i("WD_LAST_EPOCH"),
    "epochs_total":      i("WD_EPOCHS_TOTAL"),
    "progressed":        bool(i("WD_PROGRESSED")),
    "progress_at":       i("WD_PROGRESS_AT"),
    "gpu_util":          ints("WD_GPU_UTIL"),
    "gpu_mem":           ints("WD_GPU_MEM"),
    "errors":            os.environ.get("WD_ERRORS", "").split(),
    "state":             os.environ["WD_STATE"],
    # de-dup flags: a kind listed here never alerts again for this run
    "alerted":           ",".join(alerted),
    "alerts":            alerts[-50:],
    "updated_epoch_s":   int(time.time()),
}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(st, f, indent=2, ensure_ascii=False)
    f.write("\n")
os.replace(tmp, path)
PY
}

cmd_scan() {
  mkdir -p "$RUNS_DIR"
  local runs rid n=0
  runs="$(wd_active_runs)"
  if [ -z "$runs" ]; then echo "watchdog: 활성 run 없음 ($RUNS_DIR)"; return 0; fi
  echo "watchdog scan @ $(now_iso)"
  while read -r rid; do
    [ -n "$rid" ] || continue
    n=$(( n + 1 ))
    wd_scan_one "$rid" || echo "  $rid: scan error (계속 진행)" >&2
  done <<<"$runs"
  echo "watchdog: ${n} run 점검 완료"
}

cmd_status() {
  mkdir -p "$RUNS_DIR"
  printf "%-46s %-9s %-8s %-14s %-9s %-9s %-10s %s\n" \
    RUN_ID SERVER GPUS STATE ITER EPOCH UTIL CHECKED_AT
  local rid d
  while read -r rid; do
    [ -n "$rid" ] || continue
    d="$RUNS_DIR/$rid"
    printf "%-46s %-9s %-8s %-14s %-9s %-9s %-10s %s\n" \
      "$rid" \
      "$(wd_json_get "$d/manifest.json" server)" \
      "$(wd_json_get "$d/manifest.json" gpus)" \
      "$(wd_run_state "$d")" \
      "$(wd_json_get "$d/status.json" last_iter -)/$(wd_json_get "$d/status.json" iters_total -)" \
      "$(wd_json_get "$d/status.json" last_epoch -)/$(wd_json_get "$d/status.json" epochs_total -)" \
      "$(wd_json_get "$d/status.json" gpu_util -)" \
      "$(wd_json_get "$d/status.json" checked_at never)"
  done <<<"$(wd_active_runs)"
}

# register <server> <config.yaml> <gpus_csv> <remote_log_path> [--repo PATH]
# For runs remote_exp.sh cannot launch (e.g. B200) — same manifest shape, so scan works.
cmd_register() {
  local server="${1:-}" config="${2:-}" gpus="${3:-}" log_path="${4:-}"
  [ -n "$server" ] && [ -n "$config" ] && [ -n "$gpus" ] && [ -n "$log_path" ] \
    || die "usage: register <server> <config.yaml> <gpus_csv> <remote_log_path> [--repo PATH]"
  shift 4
  local repo=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --repo) repo="${2:-}"; shift 2 ;;
      *) die "register: unknown option '$1'" ;;
    esac
  done
  if [ -z "$repo" ]; then
    # reuse the registry parser rather than duplicating it
    # shellcheck source=/dev/null
    REMOTE_EXP_LIB=1 source "$SCRIPT_DIR/remote_exp.sh"
    resolve "$server"
    repo="$REPO"
    [ "$repo" != "FILL_ME" ] || die "repo_path for '$server' is FILL_ME in $CONF — pass --repo <path>."
  fi
  local cfg_name ts run_id d
  cfg_name="$(basename "$config" .yaml)"
  # prefer the timestamp already embedded in the log filename so re-registering is stable
  ts="$(basename "$log_path" .log | grep -oE '[0-9]{8}_[0-9]{6}$' || true)"
  [ -n "$ts" ] || ts="$(date +%Y%m%d_%H%M%S)"
  run_id="${server}_${cfg_name}_${ts}"
  d="$RUNS_DIR/$run_id"
  mkdir -p "$d"
  RID="$run_id" SRV="$server" RP="$repo" CFG="$config" CN="$cfg_name" G="$gpus" \
  NP="$(awk -F',' '{print NF}' <<<"$gpus")" EN="manual" \
  WN="$(echo "$cfg_name" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)" \
  LP="$log_path" LA="$(now_iso)" python3 - "$d/manifest.json" <<'PY'
import json, os, sys
m = {
    "run_id":      os.environ["RID"],
    "server":      os.environ["SRV"],
    "repo":        os.environ["RP"],
    "config":      os.environ["CFG"],
    "cfg_name":    os.environ["CN"],
    "gpus":        os.environ["G"],
    "nproc":       int(os.environ["NP"]),
    "entry":       os.environ["EN"],
    "tmux_window": os.environ["WN"],
    "log_path":    os.environ["LP"],
    "launched_at": os.environ["LA"],
    "state":       "launching",
}
with open(sys.argv[1], "w") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  echo "registered run_id=$run_id -> $d/manifest.json"
}

cmd_close() {
  local rid="${1:-}"; shift || true
  local reason="${*:-manual close}"
  [ -n "$rid" ] || die "usage: close <run_id> [reason]"
  local d="$RUNS_DIR/$rid"
  [ -d "$d" ] || die "unknown run_id '$rid' (see: watchdog.sh status)"
  WD_STATE="closed"; WD_REACHABLE=1; WD_UNREACH=0; WD_ALIVE=0
  WD_LAST_ITER="$(wd_json_get "$d/status.json" last_iter 0)"
  WD_ITERS_TOTAL="$(wd_json_get "$d/status.json" iters_total 0)"
  WD_LAST_EPOCH="$(wd_json_get "$d/status.json" last_epoch 0)"
  WD_EPOCHS_TOTAL="$(wd_json_get "$d/status.json" epochs_total 0)"
  WD_PROGRESSED=0; WD_GPU_UTIL=""; WD_GPU_MEM=""
  WD_PROGRESS_AT="$(wd_json_get "$d/status.json" progress_at 0)"
  WD_ALERTED="$(wd_json_get "$d/status.json" alerted "")"; WD_NEW_ALERT=""; WD_ERRORS=""
  wd_write_status "$d" "$rid"
  mkdir -p "$WATCHDOG_DIR"
  printf '%s\t%s\t%s\t%s\n' "$(now_iso)" "$rid" closed "$reason" >> "$ALERTS_LOG"
  echo "closed $rid ($reason)"
}

# ---------------------------------------------------------------------------
# cron
# ---------------------------------------------------------------------------

wd_cron_line() {
  echo "*/5 * * * * flock -n $CRON_LOCK bash $REPO_ROOT/scripts/watchdog.sh scan >> $WATCHDOG_DIR/cron.log 2>&1 $CRON_TAG"
}

# rewrite the crontab keeping every foreign entry byte-for-byte; only our tagged line moves
wd_cron_apply() {
  local keep_line="${1:-}" tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/wd_cron.XXXXXX")"
  crontab -l 2>/dev/null | grep -vF "$CRON_TAG" > "$tmp" || true
  [ -n "$keep_line" ] && echo "$keep_line" >> "$tmp"
  crontab "$tmp"
  rm -f "$tmp"
}

cmd_install_cron() {
  command -v crontab >/dev/null || die "crontab not found"
  mkdir -p "$WATCHDOG_DIR"
  wd_cron_apply "$(wd_cron_line)"
  echo "installed (idempotent):"; wd_cron_line
}

cmd_uninstall_cron() {
  command -v crontab >/dev/null || die "crontab not found"
  wd_cron_apply ""
  echo "removed any '$CRON_TAG' crontab line (other entries untouched)"
}

# ---------------------------------------------------------------------------

if [ -n "${WATCHDOG_LIB:-}" ]; then return 0; fi

cmd="${1:-scan}"; shift || true
case "$cmd" in
  scan|"")        cmd_scan "$@" ;;
  status)         cmd_status "$@" ;;
  register)       cmd_register "$@" ;;
  close)          cmd_close "$@" ;;
  install-cron)   cmd_install_cron "$@" ;;
  uninstall-cron) cmd_uninstall_cron "$@" ;;
  -h|--help|help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown subcommand '$cmd' (scan|status|register|close|install-cron|uninstall-cron)" ;;
esac
