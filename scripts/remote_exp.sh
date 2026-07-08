#!/usr/bin/env bash
# remote_exp.sh — launch & track MemorySAM experiments on lab servers over SSH.
#
# Registry: scripts/servers.conf  (alias | repo_path | conda_env | default_gpus | notes)
# Requires passwordless SSH to each alias (see ~/.ssh/config). tmux must exist on the remote.
#
# Usage:
#   scripts/remote_exp.sh run    <server> <config.yaml> [gpus] [nproc] [entry]
#   scripts/remote_exp.sh log    <server> <config|cfg_name> [follow]
#   scripts/remote_exp.sh status <server>
#   scripts/remote_exp.sh list   <server>
#   scripts/remote_exp.sh servers
#
# Examples:
#   scripts/remote_exp.sh run bengio configs/multiaqua/bengio-multiaqua_rgbtl_P9_hardaug6.yaml 0,1,2,3
#   scripts/remote_exp.sh log bengio bengio-multiaqua_rgbtl_P9_hardaug6
#   scripts/remote_exp.sh status bengio
#
# Behaviour of `run`:
#   - ensures tmux session 'jemo' exists on the server (creates if missing)
#   - opens a NEW window named after the config and runs training there (survives disconnect)
#   - gpus     : CUDA_VISIBLE_DEVICES; defaults to the registry default_gpus.
#                use "auto" (1 free GPU) or "auto:N" (N free GPUs) to auto-pick idle GPUs on the remote.
#   - nproc    : torchrun --nproc_per_node; defaults to the GPU count
#   - entry    : train script; 'auto' (default) -> train_sam3_rbma.py for SAM3/RBMA configs,
#                else train_sam2_lora_paper.py. Pass a filename to override.
#   - logs to  logs/<cfg_name>/<cfg_name>_<timestamp>.log  (tracked by the `log` subcommand)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$SCRIPT_DIR/servers.conf"
SESSION="jemo"

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$CONF" ] || die "registry not found: $CONF"

# lookup <alias> -> prints "repo<TAB>env<TAB>gpus<TAB>notes" or nothing
lookup() {
  awk -F'|' -v a="$1" '
    /^[[:space:]]*#/ {next} /^[[:space:]]*$/ {next}
    { gsub(/^[ \t]+|[ \t]+$/,"",$1)
      if ($1==a) { for(i=2;i<=5;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i)
                   print $2"\t"$3"\t"$4"\t"$5; exit } }' "$CONF"
}

resolve() {  # sets REPO ENV GPUS NOTES from alias
  local row; row="$(lookup "$1")"
  [ -n "$row" ] || die "unknown server '$1'. Known: $(grep -vE '^[[:space:]]*(#|$)' "$CONF" | awk -F'|' '{gsub(/ /,"",$1);printf "%s ",$1}')"
  IFS=$'\t' read -r REPO ENV GPUS NOTES <<<"$row"
}

cmd="${1:-}"; shift || true

case "$cmd" in
  servers)
    printf "%-10s %-55s %-10s %-16s %s\n" ALIAS REPO_PATH ENV DEFAULT_GPUS NOTES
    grep -vE '^[[:space:]]*(#|$)' "$CONF" | while IFS='|' read -r a r e g n; do
      printf "%-10s %-55s %-10s %-16s %s\n" \
        "$(echo "$a"|xargs)" "$(echo "$r"|xargs)" "$(echo "$e"|xargs)" "$(echo "$g"|xargs)" "$(echo "$n"|xargs)"
    done
    ;;

  run)
    server="${1:-}"; config="${2:-}"; gpus="${3:-}"; nproc="${4:-}"; entry="${5:-auto}"
    [ -n "$server" ] && [ -n "$config" ] || die "usage: run <server> <config.yaml> [gpus] [nproc] [entry]"
    resolve "$server"
    [ "$REPO" != "FILL_ME" ] || die "repo_path for '$server' is FILL_ME — edit scripts/servers.conf first."
    [ -n "$gpus" ] || gpus="$GPUS"
    # auto GPU selection: gpus="auto" or "auto:N" → pick N free GPUs on the remote
    # (a GPU is free when memory.used<=2000MiB AND util<=10%, lowest-memory-first).
    if [ "$gpus" = "auto" ] || [[ "$gpus" == auto:* ]]; then
      want="${gpus#auto:}"; [[ "$want" =~ ^[0-9]+$ ]] || want=1
      free="$(ssh -o BatchMode=yes "$server" \
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" \
        | awk -F',' '{ gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);
                       if ($2+0<=2000 && $3+0<=10) print ($2+0)"\t"$1 }' \
        | sort -n | awk -F"\t" '{print $2}')"
      nfree="$(printf '%s\n' "$free" | sed '/^$/d' | wc -l | tr -d ' ')"
      [ "$nfree" -ge "$want" ] || die "auto GPU: '$server'에 빈 GPU ${want}장이 없음(가용 ${nfree}). 'status $server'로 확인하세요."
      gpus="$(printf '%s\n' "$free" | sed '/^$/d' | head -n "$want" | paste -sd, -)"
      echo ">> $server : auto-selected free GPUs = $gpus"
    fi
    [ "$gpus" != "FILL_ME" ] && [ -n "$gpus" ] || die "no gpus given and no default_gpus in registry for '$server' (try: run $server $config auto:N)."
    if [ -z "$nproc" ]; then nproc="$(awk -F',' '{print NF}' <<<"$gpus")"; fi
    cfg_name="$(basename "$config" .yaml)"
    win="$(echo "$cfg_name" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)"
    echo ">> $server : launching '$cfg_name' | gpus=$gpus nproc=$nproc entry=$entry"
    ssh -o BatchMode=yes "$server" bash -s -- "$REPO" "$ENV" "$gpus" "$nproc" "$config" "$win" "$entry" <<'REMOTE'
set -e
REPO="$1"; ENV="$2"; GPUS="$3"; NPROC="$4"; CFG="$5"; WIN="$6"; ENTRY="$7"
cfg_name="$(basename "$CFG" .yaml)"
[ -d "$REPO" ] || { echo "REMOTE ERROR: repo not found: $REPO" >&2; exit 1; }
[ -f "$REPO/$CFG" ] || { echo "REMOTE ERROR: config not found: $REPO/$CFG" >&2; exit 1; }
# entry auto-detect
if [ "$ENTRY" = "auto" ]; then
  case "$cfg_name" in
    *SAM3*|*sam3*|*RBMA*|*rbma*) ENTRY="train_sam3_rbma.py" ;;
    *) ENTRY="train_sam2_lora_paper.py" ;;
  esac
fi
PRE=""
case "$ENTRY" in
  *sam3*) PRE="export HF_HUB_OFFLINE=1 PYTHONPATH=semseg/models/sam3:\$PYTHONPATH &&" ;;
esac
command -v tmux >/dev/null || { echo "REMOTE ERROR: tmux not installed" >&2; exit 1; }
tmux has-session -t jemo 2>/dev/null || tmux new-session -d -s jemo
IDX="$(tmux new-window -P -F '#{window_index}' -t jemo -n "$WIN")"
PORT=$((21600 + RANDOM % 300))
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${cfg_name}/${cfg_name}_${TS}.log"
RUN="cd '$REPO' && mkdir -p 'logs/${cfg_name}' && conda activate '$ENV' && $PRE export CUDA_VISIBLE_DEVICES='$GPUS' OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && echo '[remote_exp] PORT=$PORT NPROC=$NPROC ENTRY=$ENTRY' && torchrun --nproc_per_node=$NPROC --master_port=$PORT $ENTRY --cfg '$CFG' 2>&1 | tee '$LOG'"
tmux send-keys -t "jemo:$IDX" "$RUN" C-m
echo "LAUNCHED session=jemo window=$WIN(idx $IDX) port=$PORT"
echo "LOG=$REPO/$LOG"
echo "TRACK: scripts/remote_exp.sh log <server> $cfg_name"
REMOTE
    ;;

  log)
    server="${1:-}"; what="${2:-}"; follow="${3:-}"
    [ -n "$server" ] && [ -n "$what" ] || die "usage: log <server> <config|cfg_name> [follow]"
    resolve "$server"
    [ "$REPO" != "FILL_ME" ] || die "repo_path for '$server' is FILL_ME — edit scripts/servers.conf first."
    cfg_name="$(basename "$what" .yaml)"
    ssh -o BatchMode=yes "$server" bash -s -- "$REPO" "$cfg_name" "$follow" <<'REMOTE'
REPO="$1"; CN="$2"; FOLLOW="$3"
cd "$REPO" 2>/dev/null || { echo "no repo $REPO" >&2; exit 1; }
F="$(ls -t logs/"$CN"/*.log 2>/dev/null | head -1)"
[ -n "$F" ] || { echo "no log found under logs/$CN/ in $REPO" >&2; exit 1; }
echo "== $REPO/$F =="
if [ "$FOLLOW" = "follow" ] || [ "$FOLLOW" = "-f" ]; then tail -n 80 -f "$F"; else tail -n 80 "$F"; fi
REMOTE
    ;;

  status)
    server="${1:-}"; [ -n "$server" ] || die "usage: status <server>"
    resolve "$server"
    ssh -o BatchMode=yes "$server" bash -s <<'REMOTE'
echo "== tmux session 'jemo' =="
tmux list-windows -t jemo 2>/dev/null || echo "(no 'jemo' session)"
echo; echo "== GPU =="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || echo "(nvidia-smi unavailable)"
REMOTE
    ;;

  list|ls)
    server="${1:-}"; [ -n "$server" ] || die "usage: list <server>"
    resolve "$server"
    ssh -o BatchMode=yes "$server" "tmux list-windows -t jemo 2>/dev/null || echo '(no jemo session)'"
    ;;

  ""|-h|--help|help)
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;

  *) die "unknown subcommand '$cmd' (run|log|status|list|servers|help)";;
esac
