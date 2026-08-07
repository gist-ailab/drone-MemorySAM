#!/usr/bin/env bash
# remote_exp.sh — launch & track MemorySAM experiments on lab servers over SSH.
#
# Registry: scripts/servers.conf  (alias | repo_path | conda_env | default_gpus | notes | policy)
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
#   - refuses to launch on a server whose registry policy is `off`, or on any GPU listed
#     in its `ban:` policy (both auto-pick and an explicit GPU list are checked)
#   - writes a run manifest to .watchdog/runs/<run_id>/manifest.json on the hub so that
#     scripts/watchdog.sh can babysit the run without a human polling it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONF="${REMOTE_EXP_CONF:-$SCRIPT_DIR/servers.conf}"
SESSION="jemo"
WATCHDOG_DIR="${WATCHDOG_DIR:-$REPO_ROOT/.watchdog}"

die() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$CONF" ] || die "registry not found: $CONF"

# lookup <alias> -> prints repo/env/gpus/notes/policy joined by US (0x1f), or nothing.
# Field 6 (policy) is optional: legacy 5-field lines yield an empty policy.
# US rather than TAB because TAB is IFS whitespace, and `read` would then collapse an
# empty notes column into the policy one.
lookup() {
  awk -F'|' -v a="$1" -v S=$'\037' '
    /^[[:space:]]*#/ {next} /^[[:space:]]*$/ {next}
    { gsub(/^[ \t]+|[ \t]+$/,"",$1)
      if ($1==a) { for(i=2;i<=6;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i)
                   print $2 S $3 S $4 S $5 S $6; exit } }' "$CONF"
}

# parse_policy <policy field> -> sets POLICY_OFF (0/1) and POLICY_BANNED (csv, may be empty)
parse_policy() {
  local p="${1:-}"
  POLICY_OFF=0; POLICY_BANNED=""
  p="$(echo "$p" | tr -d '[:space:]')"
  case "$p" in
    ""|-|none) return 0 ;;
    off)       POLICY_OFF=1 ;;
    ban:*)     POLICY_BANNED="${p#ban:}" ;;
    *)         echo "WARN: unknown policy '$p' in $CONF — ignoring" >&2 ;;
  esac
}

resolve() {  # sets REPO ENV GPUS NOTES POLICY POLICY_OFF POLICY_BANNED from alias
  local row; row="$(lookup "$1")"
  [ -n "$row" ] || die "unknown server '$1'. Known: $(grep -vE '^[[:space:]]*(#|$)' "$CONF" | awk -F'|' '{gsub(/ /,"",$1);printf "%s ",$1}')"
  IFS=$'\037' read -r REPO ENV GPUS NOTES POLICY <<<"$row"
  parse_policy "${POLICY:-}"
}

# policy_desc -> one-line human summary of the resolved policy
policy_desc() {
  if [ "${POLICY_OFF:-0}" = "1" ]; then echo "OFF (launching disabled)"
  elif [ -n "${POLICY_BANNED:-}" ]; then echo "ban ${POLICY_BANNED}"
  else echo "none"; fi
}

# banned_hits <gpus_csv> -> the banned indices present in that list (space separated, may be empty)
banned_hits() {
  local want="$1" hits="" g b
  [ -n "${POLICY_BANNED:-}" ] || return 0
  for g in ${want//,/ }; do
    for b in ${POLICY_BANNED//,/ }; do
      [ "$g" = "$b" ] && hits="$hits $g"
    done
  done
  echo "${hits# }"
}

# filter_banned  (stdin: one GPU index per line) -> the same list minus banned indices
filter_banned() {
  local banned="${POLICY_BANNED:-}"
  if [ -z "$banned" ]; then cat; return 0; fi
  awk -v ban="$banned" 'BEGIN{n=split(ban,b,","); for(i=1;i<=n;i++) skip[b[i]]=1}
                        NF && !($1 in skip) {print $1}'
}

# remote_gpu_query <server> -> "index, memory.used, utilization.gpu" csv lines.
# Split out as a function so tests can stub it (unit tests must never ssh anywhere).
remote_gpu_query() {
  ssh -o BatchMode=yes "$1" \
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
}

# pick_free_gpus <server> <want> -> csv of <want> free, non-banned GPU indices (dies if short).
# free == memory.used<=2000MiB AND util<=10%, lowest-memory-first.
pick_free_gpus() {
  local server="$1" want="$2" free nfree
  free="$(remote_gpu_query "$server" \
    | awk -F',' '{ gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);
                   if ($2+0<=2000 && $3+0<=10) print ($2+0)"\t"$1 }' \
    | sort -n | awk -F"\t" '{print $2}' | filter_banned)"
  nfree="$(printf '%s\n' "$free" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "$nfree" -lt "$want" ]; then
    local extra=""
    [ -n "${POLICY_BANNED:-}" ] && extra=" (policy ban:${POLICY_BANNED} 제외 후)"
    die "auto GPU: '$server'에 빈 GPU ${want}장이 없음(가용 ${nfree}${extra}). 'status $server'로 확인하세요."
  fi
  printf '%s\n' "$free" | sed '/^$/d' | head -n "$want" | paste -sd, -
}

# write_manifest — record the launched run on the hub for scripts/watchdog.sh.
# args: run_id server repo config cfg_name gpus nproc entry tmux_window log_path launched_at
write_manifest() {
  local dir="$WATCHDOG_DIR/runs/$1"
  mkdir -p "$dir"
  RID="$1" SRV="$2" RP="$3" CFG="$4" CN="$5" G="$6" NP="$7" EN="$8" WN="$9" LP="${10}" LA="${11}" \
  python3 - "$dir/manifest.json" <<'PY'
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
  echo "$dir/manifest.json"
}

# remote_launch — the ssh half of `run`. A function so tests can stub it.
# args: server repo env gpus nproc config win entry ts
remote_launch() {
  ssh -o BatchMode=yes "$1" bash -s -- "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" <<'REMOTE'
set -e
REPO="$1"; ENV="$2"; GPUS="$3"; NPROC="$4"; CFG="$5"; WIN="$6"; ENTRY="$7"; TS="$8"
cfg_name="$(basename "$CFG" .yaml)"
[ -d "$REPO" ] || { echo "REMOTE ERROR: repo not found: $REPO" >&2; exit 1; }
[ -f "$REPO/$CFG" ] || { echo "REMOTE ERROR: config not found: $REPO/$CFG" >&2; exit 1; }
# entry auto-detect
# Decide from the config's CONTENT, not its filename. The old filename globs only
# matched P34/P35/P36, so every later ReliaDINO config (P37+) and every det config
# silently fell through to train_sam2_lora_paper.py — the wrong trainer.
# Order matters: det configs carry DET_MODEL: ReliaDINO*Detector, so DET wins first.
if [ "$ENTRY" = "auto" ]; then
  if grep -qE '^[[:space:]]*DET_MODEL[[:space:]]*:' "$REPO/$CFG"; then
    ENTRY="train_det.py"
  elif grep -qE '^[[:space:]]*NAME[[:space:]]*:[[:space:]]*ReliaDINO[[:space:]]*(#.*)?$' "$REPO/$CFG"; then
    ENTRY="train_reliadino.py"
  else
    case "$cfg_name" in
      *SAM3*|*sam3*|*RBMA*|*rbma*) ENTRY="train_sam3_rbma.py" ;;
      *reliadino*|*ReliaDINO*|*P34*|*P35*|*P36*) ENTRY="train_reliadino.py" ;;
      *) ENTRY="train_sam2_lora_paper.py" ;;
    esac
  fi
  echo "[remote_exp] entry auto-detect -> $ENTRY (cfg=$cfg_name)" >&2
fi
# activation: absolute path => venv (e.g. hpca100), bare name => conda env (all existing servers)
case "$ENV" in
  /*) # venv: torch bundles its own cuDNN, but a login shell's LD_LIBRARY_PATH can
      # shadow it with an older system cuDNN. libcudnn_cnn_train.so.8 is dlopen'd
      # lazily on the FIRST CONV BACKWARD, so a mismatch passes forward and then
      # dies with "GET was unable to find an engine to execute this computation".
      # Prepend the venv's cuDNN so it wins. (hpca100: system 8.9.0 vs torch 8.9.2)
      ACT="source '$ENV/bin/activate'"
      CUDNN_LIB="$("$ENV/bin/python" -c 'import nvidia.cudnn,os;print(os.path.join(os.path.dirname(nvidia.cudnn.__file__),"lib"))' 2>/dev/null || true)"
      [ -n "$CUDNN_LIB" ] && ACT="$ACT && export LD_LIBRARY_PATH='$CUDNN_LIB':\$LD_LIBRARY_PATH"
      # wandb telemetry (sentry) can be unreachable on egress-restricted boxes (e.g. the
      # hpca100 K8s pod). rank0 then blocks in a futex on wandb's network thread while
      # ranks 1..N-1 spin forever in an all-reduce -> silent NCCL deadlock at 0/187 iters,
      # GPUs pinned at 100% with only weights resident. Disable telemetry on venv servers.
      ACT="$ACT && export WANDB_MODE=disabled"
      ;;
  *)  ACT="conda activate '$ENV'" ;;
esac
PRE=""
case "$ENTRY" in
  *sam3*) PRE="export HF_HUB_OFFLINE=1 PYTHONPATH=semseg/models/sam3:\$PYTHONPATH &&" ;;
esac
command -v tmux >/dev/null || { echo "REMOTE ERROR: tmux not installed" >&2; exit 1; }
tmux has-session -t jemo 2>/dev/null || tmux new-session -d -s jemo
IDX="$(tmux new-window -P -F '#{window_index}' -t jemo -n "$WIN")"
PORT=$((21600 + RANDOM % 300))
# TS is generated on the hub and passed in (it used to be `date` here) so the hub knows
# the log path deterministically and can point a watchdog manifest at the right file.
LOG="logs/${cfg_name}/${cfg_name}_${TS}.log"
RUN="cd '$REPO' && mkdir -p 'logs/${cfg_name}' && $ACT && $PRE export CUDA_VISIBLE_DEVICES='$GPUS' OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 && echo '[remote_exp] PORT=$PORT NPROC=$NPROC ENTRY=$ENTRY' && torchrun --nproc_per_node=$NPROC --master_port=$PORT $ENTRY --cfg '$CFG' 2>&1 | tee '$LOG'"
tmux send-keys -t "jemo:$IDX" "$RUN" C-m
echo "LAUNCHED session=jemo window=$WIN(idx $IDX) port=$PORT"
echo "LOG=$REPO/$LOG"
echo "TRACK: scripts/remote_exp.sh log <server> $cfg_name"
REMOTE
}

# `REMOTE_EXP_LIB=1 source scripts/remote_exp.sh` loads the functions above without
# dispatching a subcommand — used by tests/test_watchdog.sh.
if [ -n "${REMOTE_EXP_LIB:-}" ]; then return 0; fi

cmd="${1:-}"; shift || true

case "$cmd" in
  servers)
    printf "%-10s %-55s %-10s %-16s %-10s %s\n" ALIAS REPO_PATH ENV DEFAULT_GPUS POLICY NOTES
    grep -vE '^[[:space:]]*(#|$)' "$CONF" | while IFS='|' read -r a r e g n p; do
      printf "%-10s %-55s %-10s %-16s %-10s %s\n" \
        "$(echo "$a"|xargs)" "$(echo "$r"|xargs)" "$(echo "$e"|xargs)" "$(echo "$g"|xargs)" \
        "$(echo "${p:--}"|xargs)" "$(echo "$n"|xargs)"
    done
    ;;

  run)
    server="${1:-}"; config="${2:-}"; gpus="${3:-}"; nproc="${4:-}"; entry="${5:-auto}"
    [ -n "$server" ] && [ -n "$config" ] || die "usage: run <server> <config.yaml> [gpus] [nproc] [entry]"
    resolve "$server"
    [ "$REPO" != "FILL_ME" ] || die "repo_path for '$server' is FILL_ME — edit scripts/servers.conf first."
    # policy gate 1: a server marked `off` in the registry never launches
    [ "${POLICY_OFF:-0}" != "1" ] || die "'$server' is disabled by registry policy=off (scripts/servers.conf) — pick another server or clear the policy field."
    [ -n "$gpus" ] || gpus="$GPUS"
    # auto GPU selection: gpus="auto" or "auto:N" → pick N free, non-banned GPUs
    # (free == memory.used<=2000MiB AND util<=10%, lowest-memory-first; see pick_free_gpus)
    if [ "$gpus" = "auto" ] || [[ "$gpus" == auto:* ]]; then
      want="${gpus#auto:}"; [[ "$want" =~ ^[0-9]+$ ]] || want=1
      gpus="$(pick_free_gpus "$server" "$want")"
      echo ">> $server : auto-selected free GPUs = $gpus"
    fi
    [ "$gpus" != "FILL_ME" ] && [ -n "$gpus" ] || die "no gpus given and no default_gpus in registry for '$server' (try: run $server $config auto:N)."
    # policy gate 2: an explicit (or default_gpus) list must not touch a banned GPU
    hits="$(banned_hits "$gpus")"
    [ -z "$hits" ] || die "'$server' policy ban:${POLICY_BANNED} — requested GPUs '$gpus' include reserved GPU(s): ${hits}. Those indices are off-limits (user-reserved / someone else's). Use only the allowed indices, or 'run $server $config auto:N'."
    if [ -z "$nproc" ]; then nproc="$(awk -F',' '{print NF}' <<<"$gpus")"; fi
    cfg_name="$(basename "$config" .yaml)"
    win="$(echo "$cfg_name" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)"
    # the timestamp is minted HERE (not on the remote) so the hub knows the remote log path
    TS="$(date +%Y%m%d_%H%M%S)"
    echo ">> $server : launching '$cfg_name' | gpus=$gpus nproc=$nproc entry=$entry ts=$TS"
    remote_launch "$server" "$REPO" "$ENV" "$gpus" "$nproc" "$config" "$win" "$entry" "$TS"
    # launch succeeded (set -e would have aborted otherwise) -> register with the watchdog
    run_id="${server}_${cfg_name}_${TS}"
    mf="$(write_manifest "$run_id" "$server" "$REPO" "$config" "$cfg_name" "$gpus" "$nproc" \
                         "$entry" "$win" "$REPO/logs/${cfg_name}/${cfg_name}_${TS}.log" \
                         "$(date -Iseconds)")"
    echo "WATCHDOG: registered run_id=$run_id ($mf)"
    echo "WATCHDOG: 'bash scripts/watchdog.sh status' 로 현황, 'scan' 으로 1회 점검"
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
    echo "== policy ($server) =="
    echo "policy: $(policy_desc)"
    echo
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
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;

  *) die "unknown subcommand '$cmd' (run|log|status|list|servers|help)";;
esac
