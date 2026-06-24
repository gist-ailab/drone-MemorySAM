#!/usr/bin/env bash
# Pick free GPU indices and print them comma-separated (e.g. "2,5").
# A GPU is "free" when memory.used <= MAXMEM MiB AND utilization.gpu <= MAXUTIL %.
# Free GPUs are returned lowest-memory-first.
#
# Usage:   scripts/pick_free_gpus.sh [N] [MAXMEM_MiB] [MAXUTIL_pct]
#   N        number of GPUs needed        (default 1; "all" = every free GPU)
#   MAXMEM   memory-used threshold, MiB    (default 2000, or $GPU_MAXMEM)
#   MAXUTIL  utilization threshold, %      (default 10,   or $GPU_MAXUTIL)
#
# Output:  the chosen indices on stdout; exits 1 (nothing on stdout) if < N free.
# Override: if CUDA_VISIBLE_DEVICES is already set, it is echoed back unchanged
#           (explicit user choice always wins — no second-guessing).
set -euo pipefail

# Respect an explicit override.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "$CUDA_VISIBLE_DEVICES"
  exit 0
fi

N="${1:-1}"
MAXMEM="${2:-${GPU_MAXMEM:-2000}}"
MAXUTIL="${3:-${GPU_MAXUTIL:-10}}"

command -v nvidia-smi >/dev/null 2>&1 || { echo "pick_free_gpus: nvidia-smi not found" >&2; exit 2; }

# index | memory.used(MiB) | util(%) → keep free ones, sort by memory ascending, emit index.
mapfile -t free < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v mm="$MAXMEM" -v mu="$MAXUTIL" '
        { gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3);
          if ($2+0 <= mm && $3+0 <= mu) print ($2+0)"\t"$1 }' \
    | sort -n | awk -F'\t' '{print $2}'
)

total_free="${#free[@]}"
[ "$N" = "all" ] && N="$total_free"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ] || [ "$total_free" -lt "$N" ]; then
  echo "pick_free_gpus: need $N free GPU(s) (<=${MAXMEM}MiB & <=${MAXUTIL}% util), found ${total_free}: ${free[*]:-none}" >&2
  exit 1
fi

sel=("${free[@]:0:$N}")
( IFS=','; printf '%s\n' "${sel[*]}" )
