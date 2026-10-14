#!/bin/bash
# move_files_to_nas.sh — verified MOVE of an explicit file list from one source host to drone_nas.
#   usage: move_files_to_nas.sh <tag> <src_ssh_host> <list.txt> <workdir> [batch_GB]
#   <tag>     : name used under the NAS archive root (e.g. lecun, ailab_mat2)
#   list.txt  : absolute file paths on <src_ssh_host>, one per line
# Per batch (~batch_GB): precheck (exists, not open by my processes) -> src md5 (parallel)
#   -> tar stream src -> NAS (via this hub) -> md5 -c on NAS -> recheck src size+mtime -> rm on src.
# Nothing is deleted unless every file of the batch passed md5 on the NAS side.
# Ledger: <workdir>/<tag>_MANIFEST.tsv  (tag  src_path  bytes  mtime  md5  nas_path  status  when)
set -uo pipefail
TAG=$1; SRC=$2; LIST=$3; W=$4; BGB=${5:-40}
NASH=jemo.maeng@172.27.190.182
NASROOT=/volume1/drone/personal/jemo_maeng/nonbest_weights_archive_20261015
DEST=$NASROOT/$TAG
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
mkdir -p "$W/$TAG"; WD=$W/$TAG; MAN=$W/${TAG}_MANIFEST.tsv; DONE=$WD/done_paths.txt; touch "$MAN" "$DONE"
log(){ echo "$(date '+%F %T') [$TAG] $*"; }
$SSH $NASH "mkdir -p '$DEST'" || { log "NAS mkdir failed"; exit 1; }
# remaining = list - done
grep -vxF -f "$DONE" "$LIST" > "$WD/remaining.txt" || true
log "remaining files: $(wc -l < "$WD/remaining.txt")"
# stat all remaining on source (size mtime path); missing ones are reported
$SSH "$SRC" "while IFS= read -r f; do if [ -f \"\$f\" ]; then stat -c '%s	%Y	%n' -- \"\$f\"; else printf 'MISSING	0	%s\n' \"\$f\"; fi; done" < "$WD/remaining.txt" > "$WD/stat.tsv" || { log "stat failed"; exit 1; }
awk -F'\t' '$1=="MISSING"{print $3}' "$WD/stat.tsv" > "$WD/missing.txt"
[ -s "$WD/missing.txt" ] && log "MISSING on source (skipped): $(wc -l < "$WD/missing.txt")"
awk -F'\t' '$1!="MISSING"' "$WD/stat.tsv" > "$WD/todo.tsv"
# split into batches by cumulative size
awk -F'\t' -v lim=$((BGB*1000000000)) -v d="$WD" 'BEGIN{b=1;s=0} {if(s>0 && s+$1>lim){b++;s=0} s+=$1; print > (d"/batch_" sprintf("%03d",b) ".tsv")}' "$WD/todo.tsv"
for B in $(ls "$WD"/batch_*.tsv 2>/dev/null | sort); do
  bn=$(basename "$B" .tsv); n=$(wc -l < "$B"); gb=$(awk -F'\t' '{s+=$1}END{printf "%.1f",s/1e9}' "$B")
  cut -f3 "$B" > "$WD/$bn.paths"
  # in-use check: any of my processes holding these files open, or cwd inside run dir with recent write
  busy=$($SSH "$SRC" "for p in \$(pgrep -u \$(whoami)); do ls -l /proc/\$p/fd 2>/dev/null | awk '{print \$NF}'; done | sort -u" | grep -xF -f "$WD/$bn.paths" | head -3)
  if [ -n "$busy" ]; then log "$bn SKIP: open by a process: $busy"; continue; fi
  log "$bn start: $n files $gb GB"
  sed 's#^/##' "$WD/$bn.paths" > "$WD/$bn.rel"
  # src md5 in parallel
  ( $SSH "$SRC" "cd / && xargs -d '\n' -r md5sum --" < "$WD/$bn.rel" > "$WD/$bn.srcmd5" 2> "$WD/$bn.srcmd5.err"; echo $? > "$WD/$bn.srcmd5.rc" ) &
  mp=$!
  ok=0
  for t in 1 2 3; do
    $SSH "$SRC" "tar -C / -cf - --files-from=-" < "$WD/$bn.rel" 2> "$WD/$bn.tar.err" | $SSH $NASH "tar -C '$DEST' -xf -" 2>> "$WD/$bn.tar.err"
    rc=("${PIPESTATUS[@]}")
    if [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]; then ok=1; break; fi
    log "$bn tar stream failed (rc=${rc[*]}) try $t: $(tail -2 "$WD/$bn.tar.err" | tr '\n' ' ')"; sleep 60
  done
  wait $mp
  if [ $ok != 1 ]; then log "$bn FAILED transfer — source untouched"; continue; fi
  if [ "$(cat "$WD/$bn.srcmd5.rc")" != 0 ]; then log "$bn FAILED src md5 — source untouched"; continue; fi
  # verify on NAS
  $SSH $NASH "cd '$DEST' && md5sum -c --quiet -" < "$WD/$bn.srcmd5" > "$WD/$bn.nascheck" 2>&1; vrc=$?
  nsrc=$(wc -l < "$WD/$bn.srcmd5")
  if [ $vrc != 0 ] || [ "$nsrc" != "$n" ]; then log "$bn VERIFY FAILED (rc=$vrc, md5 lines $nsrc/$n) — source untouched"; head -5 "$WD/$bn.nascheck"; continue; fi
  # recheck size+mtime on source unchanged since stat
  $SSH "$SRC" "while IFS= read -r f; do stat -c '%s	%Y	%n' -- \"\$f\" 2>/dev/null; done" < "$WD/$bn.paths" > "$WD/$bn.restat"
  if ! diff -q <(sort -t$'\t' -k3 "$B") <(sort -t$'\t' -k3 "$WD/$bn.restat") >/dev/null; then log "$bn source changed during copy — NOT deleting"; continue; fi
  # ledger then delete
  now=$(date '+%F %T')
  awk -v tag="$TAG" -v dest="$DEST" -v now="$now" 'NR==FNR{m["/"substr($0,35)]=substr($0,1,32);next} {split($0,a,"\t"); printf "%s\t%s\t%s\t%s\t%s\t%s%s\tverified_deleted\t%s\n", tag, a[3], a[1], strftime("%F %T",a[2]), m[a[3]], dest, a[3], now}' "$WD/$bn.srcmd5" "$B" >> "$MAN"
  $SSH "$SRC" "xargs -d '\n' -r rm -f --" < "$WD/$bn.paths"
  left=$($SSH "$SRC" "n=0; while IFS= read -r f; do [ -e \"\$f\" ] && n=\$((n+1)); done; echo \$n" < "$WD/$bn.paths")
  cat "$WD/$bn.paths" >> "$DONE"
  log "$bn DONE: verified $n files $gb GB, deleted on source (left=$left)"
done
log "ALL BATCHES FINISHED"
