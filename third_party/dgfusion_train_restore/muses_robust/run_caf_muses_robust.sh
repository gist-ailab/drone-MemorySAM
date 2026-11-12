#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo 'Usage: run_caf_muses_robust.sh WEIGHTS OUT PROTOCOL SUBSET_OR_RATIO_OR_LEVEL GPU' >&2; exit 2; }
exec bash "$(dirname "$0")/run_muses_robust_common.sh" caf "$@"
