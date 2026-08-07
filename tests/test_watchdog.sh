#!/usr/bin/env bash
# tests/test_watchdog.sh — offline test suite for scripts/watchdog.sh + the
# servers.conf policy / run-manifest additions in scripts/remote_exp.sh.
#
#   bash tests/test_watchdog.sh
#
# NOTHING here touches a real server: `ssh` is shadowed by a stub on PATH, the
# registry is a throwaway conf in a temp dir, and no cron entry is ever installed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WD="$ROOT/scripts/watchdog.sh"
RE="$ROOT/scripts/remote_exp.sh"

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n     %s\n' "$1" "${2:-}"; }
check(){ # check <name> <expected> <actual>
  if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "expected [$2] got [$3]"; fi; }
contains(){ # contains <name> <needle> <haystack>
  case "$3" in *"$2"*) ok "$1";; *) bad "$1" "[$2] not found in: $3";; esac; }
section(){ printf '\n\033[1m== %s\033[0m\n' "$1"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/wdtest.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/wd"

# ---------------------------------------------------------------- ssh stub
# Behaviour selected by FAKE_SSH_MODE; it never leaves this machine.
cat > "$TMP/bin/ssh" <<'STUB'
#!/usr/bin/env bash
case "${FAKE_SSH_MODE:-ok}" in
  fail) echo "ssh: connect to host: Connection timed out" >&2; exit 255 ;;
esac
for a in "$@"; do
  case "$a" in
    *nvidia-smi*--query-gpu*) cat "$FAKE_GPU_CSV"; exit 0 ;;
  esac
done
# `bash -s` invocations: either remote_exp's launcher or watchdog's probe
if [ -n "${FAKE_PROBE_PAYLOAD:-}" ]; then cat "$FAKE_PROBE_PAYLOAD"; exit 0; fi
cat >/dev/null            # swallow the here-doc script
echo "LAUNCHED session=jemo window=stub(idx 9) port=21999"
echo "LOG=/fake/repo/logs/stub/stub.log"
STUB
chmod +x "$TMP/bin/ssh" 2>/dev/null || true
PATH="$TMP/bin:$PATH"; export PATH

# --------------------------------------------------------------- temp registry
CONF="$TMP/servers.conf"
cat > "$CONF" <<'CONF'
# alias | repo_path | conda_env | default_gpus | notes | policy
legacy5   | /srv/legacy   | MMSS_SAM | 0,1     | five field legacy line, no policy column
dashpol   | /srv/dash     | MMSS_SAM | 0,1     | explicit no-restriction | -
offsrv    | /srv/off      | MMSS_SAM | 0,1     | closed for launching | off
bansrv    | /srv/ban      | MMSS_SAM | 0,1,2,3 | GPU1,2 reserved | ban:1,2
autosrv   | /srv/auto     | MMSS_SAM | FILL_ME | auto-pick only | ban:0
emptynote | /srv/empty    | MMSS_SAM | 0,1     |  | off
CONF
export REMOTE_EXP_CONF="$CONF"
export WATCHDOG_DIR="$TMP/wd"

section "1. syntax"
if bash -n "$RE"; then ok "bash -n remote_exp.sh"; else bad "bash -n remote_exp.sh"; fi
if bash -n "$WD"; then ok "bash -n watchdog.sh";   else bad "bash -n watchdog.sh"; fi
if command -v shellcheck >/dev/null 2>&1; then
  if shellcheck -S warning "$RE" "$WD"; then ok "shellcheck (severity>=warning)"
  else bad "shellcheck (severity>=warning)"; fi
else
  echo "  --   shellcheck not installed, skipped"
fi

# Load the libs in function-only mode. The unsets matter: if REMOTE_EXP_LIB leaked into the
# environment, the `bash "$RE" run ...` subprocesses below would return before dispatching.
export REMOTE_EXP_LIB=1
# shellcheck source=/dev/null
source "$RE"
unset REMOTE_EXP_LIB
export WATCHDOG_LIB=1
# shellcheck source=/dev/null
source "$WD"
unset WATCHDOG_LIB
# both scripts run `set -euo pipefail`, which the source pulled into this shell —
# the assertions below deliberately run commands that fail, so turn -e back off.
set +e

section "2. backward compatibility (5-field lines)"
resolve legacy5
check "legacy repo"     "/srv/legacy" "$REPO"
check "legacy env"      "MMSS_SAM"    "$ENV"
check "legacy gpus"     "0,1"         "$GPUS"
check "legacy policy off flag" "0"    "$POLICY_OFF"
check "legacy policy banned"   ""     "$POLICY_BANNED"
check "legacy policy_desc"     "none" "$(policy_desc)"
check "legacy servers listing has the row" "1" \
  "$(bash "$RE" servers | grep -c '^legacy5 ')"

section "3. policy parsing"
resolve dashpol; check "'-' means unrestricted" "none" "$(policy_desc)"
resolve offsrv;  check "off flag"  "1"          "$POLICY_OFF"
                 check "off desc"  "OFF (launching disabled)" "$(policy_desc)"
resolve bansrv;  check "ban list"  "1,2"        "$POLICY_BANNED"
                 check "ban desc"  "ban 1,2"    "$(policy_desc)"
resolve emptynote
check "empty notes column does not shift the policy field" "1" "$POLICY_OFF"
check "empty notes column keeps default_gpus"              "0,1" "$GPUS"
resolve bansrv
check "banned_hits(0,1)"   "1"   "$(banned_hits 0,1)"
check "banned_hits(1,2)"   "1 2" "$(banned_hits 1,2)"
check "banned_hits(0,3)"   ""    "$(banned_hits 0,3)"
check "filter_banned"      "0
3" "$(printf '0\n1\n2\n3\n' | filter_banned)"

section '4. policy enforcement in run (mocked ssh)'
out="$(bash "$RE" run offsrv configs/x.yaml 0 2>&1)"; rc=$?
check "off server: exit != 0" "1" "$([ $rc -ne 0 ] && echo 1 || echo 0)"
contains "off server: message names the policy" "policy=off" "$out"

out="$(bash "$RE" run bansrv configs/x.yaml 0,1 2>&1)"; rc=$?
check "explicit banned GPU: exit != 0" "1" "$([ $rc -ne 0 ] && echo 1 || echo 0)"
contains "explicit banned GPU: names the offending index" "reserved GPU(s): 1" "$out"
contains "explicit banned GPU: names the policy"          "ban:1,2"           "$out"

out="$(bash "$RE" run bansrv configs/x.yaml 0,3 2>&1)"; rc=$?
check "allowed explicit GPUs launch (rc 0)" "0" "$rc"
contains "allowed explicit GPUs reach the launcher" "LAUNCHED" "$out"

# auto-pick must skip banned GPUs even when they are the idlest
FAKE_GPU_CSV="$TMP/gpu.csv"; export FAKE_GPU_CSV
cat > "$FAKE_GPU_CSV" <<'CSV'
0, 5, 2
1, 10, 0
2, 12, 0
3, 900, 3
4, 31000, 98
CSV
out="$(bash "$RE" run bansrv configs/x.yaml auto:2 2>&1)"; rc=$?
check "auto-pick rc" "0" "$rc"
contains "auto-pick excludes banned 1,2" "auto-selected free GPUs = 0,3" "$out"
# only GPU3 is free once 0 is banned and 4 is busy -> asking for 2 must fail
out="$(bash "$RE" run autosrv configs/x.yaml auto:4 2>&1)"; rc=$?
check "auto-pick shortage: exit != 0" "1" "$([ $rc -ne 0 ] && echo 1 || echo 0)"
contains "auto-pick shortage mentions the ban" "ban:0" "$out"

section "5. run manifest"
rm -rf "$TMP/wd/runs"
out="$(bash "$RE" run bansrv configs/deliver/foo_P46.yaml 0,3 2>&1)"
contains "run prints the registered run_id" "WATCHDOG: registered run_id=bansrv_foo_P46_" "$out"
mf="$(find "$TMP/wd/runs" -name manifest.json | head -1)"
if [ -n "$mf" ]; then ok "manifest.json created"; else bad "manifest.json created"; fi
if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$mf"; then
  ok "manifest.json is valid JSON"; else bad "manifest.json is valid JSON"; fi
mget(){ python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$mf" "$1"; }
check "manifest.server"   "bansrv"        "$(mget server)"
check "manifest.repo"     "/srv/ban"      "$(mget repo)"
check "manifest.cfg_name" "foo_P46"       "$(mget cfg_name)"
check "manifest.gpus"     "0,3"           "$(mget gpus)"
check "manifest.nproc"    "2"             "$(mget nproc)"
check "manifest.state"    "launching"     "$(mget state)"
RUN_TS="$(basename "$(dirname "$mf")" | grep -oE '[0-9]{8}_[0-9]{6}$')"
check "manifest.log_path is deterministic from the hub timestamp" \
  "/srv/ban/logs/foo_P46/foo_P46_${RUN_TS}.log" "$(mget log_path)"

section "6. iteration/epoch parsing against the real trainer formats"
SAM2_LINE='Epoch: [12/200] Iter: [73/187] LR: 0.00006000 Loss: 0.412345 Proto: 0.010000:  39%|###       | 73/187 [00:32<00:49,  2.30it/s]'
RELIA_LINE='Epoch [7/300] Loss 0.4123 cal 0.0100 auxCE 0.0200:  45%|####      | 84/187 [00:32<00:39,  2.61it/s]'
DET_LINE='Epoch [3]:  10%|#         | 12/120 [00:05<00:48,  2.20it/s]'
SAM3_LINE='ep5 it40/187 loss=0.4123 (main=0.4000 aux=0.0123) lr=6.00e-05'
check "sam2  iter"  "73 187" "$(wd_parse_iter "$SAM2_LINE")"
check "sam2  epoch" "12 200" "$(wd_parse_epoch "$SAM2_LINE")"
check "relia iter"  "84 187" "$(wd_parse_iter "$RELIA_LINE")"
check "relia epoch" "7 300"  "$(wd_parse_epoch "$RELIA_LINE")"
check "det   iter"  "12 120" "$(wd_parse_iter "$DET_LINE")"
check "det   epoch" "3 0"    "$(wd_parse_epoch "$DET_LINE")"
check "sam3  iter"  "40 187" "$(wd_parse_iter "$SAM3_LINE")"
check "sam3  epoch" "5 0"    "$(wd_parse_epoch "$SAM3_LINE")"
# a tqdm bar is one \r-joined line; the last redraw must win
CR_LOG="$(printf 'Epoch: [1/200] Iter: [10/187] Loss: 1.0\rEpoch: [1/200] Iter: [92/187] Loss: 0.9')"
check "carriage-return tail takes the LAST redraw" "92 187" \
  "$(wd_parse_iter "$(printf '%s' "$CR_LOG" | wd_normalize)")"
check "no iteration line -> rc 1" "1" \
  "$(wd_parse_iter 'loading checkpoint...' >/dev/null 2>&1; echo $?)"
check "WATCHDOG_ITER_REGEX override" "5 9" \
  "$(WATCHDOG_ITER_REGEX='STEP [0-9]+/[0-9]+' wd_parse_iter 'foo STEP 5/9 bar')"

section "7. error signatures"
check "clean log"    ""            "$(wd_scan_errors 'Epoch: [1/200] Iter: [3/187] Loss: 0.5')"
check "RANDOM INIT"  "RANDOM_INIT" "$(wd_scan_errors "[ReliaDINO] all pretrained loads failed — falling back to RANDOM INIT 'dinov3_vitl16'. Do NOT train a real run like this.")"
check "traceback"    "TRACEBACK"   "$(wd_scan_errors 'Traceback (most recent call last):')"
check "oom"          "OOM"         "$(wd_scan_errors 'torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB')"
check "nccl"         "NCCL"        "$(wd_scan_errors '[E ProcessGroupNCCL.cpp:828] Watchdog caught collective operation timeout')"
check "nan loss"     "NAN_LOSS"    "$(wd_scan_errors 'Epoch: [1/200] Iter: [3/187] Loss: nan')"
check "loss=inf"     "NAN_LOSS"    "$(wd_scan_errors 'ep1 it20/187 loss=inf (main=inf aux=0.0)')"
check "'inference' is not a nan hit" "" "$(wd_scan_errors 'running inference on val split')"
if wd_has_completion 'Total Training Time  01:23:45'; then ok "completion marker (sam2/reliadino)"; else bad "completion marker (sam2/reliadino)"; fi
if wd_has_completion 'Training complete. Best AP: 0.4212';  then ok "completion marker (det)"; else bad "completion marker (det)"; fi
if wd_has_completion 'Epoch: [1/200]'; then bad "no false completion"; else ok "no false completion"; fi

section "8. state machine"
# wd_decide_state <state> <alive> <progressed> <since_launch> <since_prog> <util> <errs> <done> [epoch_done]
check "launching + progress -> running" "running -" \
  "$(wd_decide_state launching 1 1 60 0 90 '' 0)"
check "launching, still early -> launching" "launching -" \
  "$(wd_decide_state launching 1 0 300 300 0 '' 0)"
check "launching, 15min no progress -> failed_startup" "failed_startup failed_startup" \
  "$(wd_decide_state launching 1 0 1000 1000 0 '' 0)"
check "launching + RANDOM INIT -> failed_startup" "failed_startup failed_startup" \
  "$(wd_decide_state launching 1 0 60 60 90 'RANDOM_INIT' 0)"
check "launching + traceback -> failed_startup" "failed_startup failed_startup" \
  "$(wd_decide_state launching 1 0 60 60 90 'TRACEBACK' 0)"
check "launching, process gone -> failed_startup" "failed_startup failed_startup" \
  "$(wd_decide_state launching 0 0 60 60 0 '' 0)"
check "running, 30min frozen but GPUs busy -> NOT stalled (eval)" "running -" \
  "$(wd_decide_state running 1 0 9999 3600 95 '' 0)"
check "running, 30min frozen and GPUs idle -> stalled" "stalled stalled" \
  "$(wd_decide_state running 1 0 9999 3600 0 '' 0)"
check "stalled alerts only once" "stalled -" \
  "$(wd_decide_state stalled 1 0 9999 3600 0 '' 0)"
check "stalled -> running on progress" "running recovered" \
  "$(wd_decide_state stalled 1 1 9999 0 90 '' 0)"
check "dead: process gone, no marker" "dead dead" \
  "$(wd_decide_state running 0 0 9999 100 0 '' 0)"
check "completed: process gone + marker" "completed completed" \
  "$(wd_decide_state running 0 0 9999 100 0 '' 1)"
check "completed: no banner but last epoch/iter done" "completed completed" \
  "$(wd_decide_state running 0 0 9999 100 0 '' 0 1)"
for t in completed dead failed_startup closed; do
  check "terminal '$t' is frozen" "$t -" "$(wd_decide_state "$t" 1 1 10 10 90 '' 0)"
done

section "9. scan (mocked probe)"
export WATCHDOG_DIR="$TMP/wd2"; RUNS_DIR="$WATCHDOG_DIR/runs"; ALERTS_LOG="$WATCHDOG_DIR/alerts.log"
rm -rf "$WATCHDOG_DIR"; mkdir -p "$RUNS_DIR"
RID="bansrv_scanme_20260808_010203"; RD="$RUNS_DIR/$RID"
mkdir -p "$RD"
cat > "$RD/manifest.json" <<JSON
{
  "run_id": "$RID", "server": "bansrv", "repo": "/srv/ban",
  "config": "configs/scanme.yaml", "cfg_name": "scanme", "gpus": "0,3",
  "nproc": 2, "entry": "auto", "tmux_window": "scanme",
  "log_path": "/srv/ban/logs/scanme/scanme_20260808_010203.log",
  "launched_at": "$(date -Iseconds)", "state": "launching"
}
JSON

mkprobe(){ # mkprobe <procs> <utils "0,95 3,90"> <log text>
  local f="$TMP/probe.txt" p
  { echo "===WD_TMUX==="; echo "9: scanme* (1 panes)"
    echo "===WD_PROC==="; echo "$1"
    echo "===WD_GPU==="; for p in $2; do echo "${p%%,*}, ${p##*,}, 15000"; done
    echo "===WD_LOG==="; printf '%s\n' "$3"; } > "$f"
  echo "$f"
}

FAKE_PROBE_PAYLOAD="$(mkprobe 2 "0,95 3,92" 'Epoch: [1/200] Iter: [30/187] Loss: 0.9')"
export FAKE_PROBE_PAYLOAD
out="$(cmd_scan 2>&1)"
contains "scan reports the run" "$RID" "$out"
check "1st scan keeps launching (no prior iter)" "launching" "$(wd_run_state "$RD")"
if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$RD/status.json"; then
  ok "status.json is valid JSON"; else bad "status.json is valid JSON"; fi
sget(){ python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$RD/status.json" "$1"; }
check "status.last_iter"  "30"          "$(sget last_iter)"
check "status.gpu_util"   "[95, 92]"    "$(sget gpu_util)"
check "status.gpu_mem"    "[15000, 15000]" "$(sget gpu_mem)"
check "status.alive"      "True"        "$(sget alive)"

FAKE_PROBE_PAYLOAD="$(mkprobe 2 "0,95 3,92" 'Epoch: [1/200] Iter: [92/187] Loss: 0.8')"
cmd_scan >/dev/null 2>&1
check "iter advanced -> running" "running"  "$(wd_run_state "$RD")"
check "progressed flag"          "True"     "$(sget progressed)"
check "last_iter updated"        "92"       "$(sget last_iter)"

# process gone with the trainer's real end banner -> completed + free-slot notice
FAKE_PROBE_PAYLOAD="$(mkprobe 0 "0,0 3,0" 'Epoch: [200/200] Iter: [187/187] Loss: 0.2
+----------------------+------------------+
| Total Training Time  | 21:13:45         |')"
out="$(cmd_scan 2>&1)"
check "completed" "completed" "$(wd_run_state "$RD")"
contains "completed alert frees the slot" "슬롯 bansrv:0,3 비었음" "$out"
contains "alerts.log written" "completed" "$(cat "$ALERTS_LOG")"
check "terminal run drops out of the active list" "" "$(wd_active_runs)"

# --- a run that dies without a banner
RID2="bansrv_dying_20260808_020304"; RD2="$RUNS_DIR/$RID2"; mkdir -p "$RD2"
sed "s/$RID/$RID2/g; s/scanme/dying/g" "$RD/manifest.json" > "$RD2/manifest.json"
python3 - "$RD2/manifest.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p)); d["state"] = "running"
json.dump(d, open(p, "w"))
PY
FAKE_PROBE_PAYLOAD="$(mkprobe 0 "0,0 3,0" 'Epoch: [3/200] Iter: [12/187] Loss: 0.9
Traceback (most recent call last):
RuntimeError: CUDA error')"
out="$(cmd_scan 2>&1)"
check "no banner + process gone -> dead" "dead" "$(wd_run_state "$RD2")"
contains "dead alert points at the saved tail" "last50.log" "$out"
if [ -s "$RD2/last50.log" ]; then ok "last50.log captured"; else bad "last50.log captured"; fi

# --- RANDOM INIT is fatal on sight
RID3="bansrv_randinit_20260808_030405"; RD3="$RUNS_DIR/$RID3"; mkdir -p "$RD3"
sed "s/$RID/$RID3/g; s/scanme/randinit/g" "$RD/manifest.json" > "$RD3/manifest.json"
FAKE_PROBE_PAYLOAD="$(mkprobe 2 "0,95 3,92" "[ReliaDINO] all pretrained loads failed — falling back to RANDOM INIT 'dinov3_vitl16'.
Epoch: [1/200] Iter: [4/187] Loss: 2.9")"
out="$(cmd_scan 2>&1)"
contains "RANDOM INIT alert fires"      "RANDOM_INIT"    "$out"
check    "RANDOM INIT -> failed_startup" "failed_startup" "$(wd_run_state "$RD3")"

# --- alert de-duplication + the WATCHDOG_ONALERT_CMD hook
RID4="bansrv_dedup_20260808_040506"; RD4="$RUNS_DIR/$RID4"; mkdir -p "$RD4"
sed "s/$RID/$RID4/g; s/scanme/dedup/g" "$RD/manifest.json" > "$RD4/manifest.json"
export WATCHDOG_ONALERT_CMD="$TMP/bin/hook.sh"
cat > "$TMP/bin/hook.sh" <<'HOOK'
#!/usr/bin/env bash
echo "$1 $2 $3" >> "$HOOK_OUT"
HOOK
chmod +x "$TMP/bin/hook.sh" 2>/dev/null || true
export HOOK_OUT="$TMP/hook.out"; : > "$HOOK_OUT"
FAKE_PROBE_PAYLOAD="$(mkprobe 2 "0,95 3,92" 'Epoch: [1/200] Iter: [4/187] Loss: nan')"
cmd_scan >/dev/null 2>&1
cmd_scan >/dev/null 2>&1
check "nan alert fires exactly once over two scans" "1" \
  "$(grep -c "$RID4" "$ALERTS_LOG")"
check "onalert hook ran once with the run_id" "1" "$(grep -c "$RID4" "$HOOK_OUT")"
contains "hook received the manifest path" "manifest.json" "$(cat "$HOOK_OUT")"
unset WATCHDOG_ONALERT_CMD

# --- unreachable: alert only on the 2nd consecutive failure
RID5="bansrv_unreach_20260808_050607"; RD5="$RUNS_DIR/$RID5"; mkdir -p "$RD5"
sed "s/$RID/$RID5/g; s/scanme/unreach/g" "$RD/manifest.json" > "$RD5/manifest.json"
: > "$ALERTS_LOG"
export FAKE_SSH_MODE=fail
cmd_scan >/dev/null 2>&1
check "1st ssh failure is silent"  "0" "$(grep -c "$RID5" "$ALERTS_LOG")"
check "1st failure recorded"       "unreachable" "$(wd_run_state "$RD5")"
cmd_scan >/dev/null 2>&1
check "2nd consecutive failure alerts" "1" "$(grep -c "$RID5" "$ALERTS_LOG")"
unset FAKE_SSH_MODE

section "10. register / close"
export WATCHDOG_DIR="$TMP/wd3"; RUNS_DIR="$WATCHDOG_DIR/runs"; ALERTS_LOG="$WATCHDOG_DIR/alerts.log"
out="$(bash "$WD" register bansrv configs/b200_P50.yaml 4,5,6,7 \
        /srv/ban/logs/b200_P50/b200_P50_20260808_101112.log 2>&1)"
contains "register echoes the run_id" "bansrv_b200_P50_20260808_101112" "$out"
RMF="$RUNS_DIR/bansrv_b200_P50_20260808_101112/manifest.json"
if python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$RMF"; then
  ok "registered manifest is valid JSON"; else bad "registered manifest is valid JSON"; fi
check "register resolved repo from servers.conf" "/srv/ban" \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repo"])' "$RMF")"
check "register nproc from gpu csv" "4" \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["nproc"])' "$RMF")"
out="$(bash "$WD" register unknownhost configs/x.yaml 0 /l/x.log --repo /explicit/repo 2>&1)"
check "register --repo bypasses the registry" "/explicit/repo" \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["repo"])' \
     "$(find "$RUNS_DIR" -path '*unknownhost*' -name manifest.json | head -1)")"
out="$(bash "$WD" status 2>&1)"
contains "status lists the registered run" "bansrv_b200_P50_20260808_101112" "$out"
contains "status has a header"             "RUN_ID"                          "$out"
bash "$WD" close bansrv_b200_P50_20260808_101112 "superseded" >/dev/null 2>&1
check "close makes the run terminal" "closed" \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' \
     "$RUNS_DIR/bansrv_b200_P50_20260808_101112/status.json")"
if bash "$WD" status 2>&1 | grep -q b200_P50; then bad "closed run hidden from status"; else ok "closed run hidden from status"; fi

section "11. cron line (generated only — never installed)"
CRONLINE="$(wd_cron_line)"
contains "every 5 minutes"        '*/5 * * * *'                "$CRONLINE"
contains "flock guards overlap"   'flock -n /tmp/memorysam-watchdog.lock' "$CRONLINE"
contains "absolute script path"   "$ROOT/scripts/watchdog.sh scan"        "$CRONLINE"
contains "log redirect"           "cron.log 2>&1"              "$CRONLINE"
contains "removable tag"          "# memorysam-watchdog"       "$CRONLINE"

printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
