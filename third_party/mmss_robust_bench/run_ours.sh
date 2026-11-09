#!/bin/bash
# Run benchmark jobs for one seed sequentially on one GPU.
# usage: OURS_REPO=.. PY=.. LOGDIR=.. HOSTCFG=bengio [PYLIBS=..] bash run_ours.sh <gpu> <seed> <job>...
#   job = emm | nm0 | nm1 | nm2 | rmm0.75 | rmm0.5 | rmm0.25
set -u
GPU=$1; SEED=$2; shift 2
KIT=$(cd "$(dirname "$0")" && pwd)
: "${OURS_REPO:?}" "${PY:?}" "${LOGDIR:?}"
export OURS_REPO PYTHONPATH="${PYLIBS:+$PYLIBS:}$OURS_REPO/semseg/models/sam2"
MODEL=${MODELTAG:-a2}
CFG="$KIT/configs/${MODEL}_s${SEED}_${HOSTCFG:-bengio}.yaml"
for job in "$@"; do
  case $job in
    emm)  script=ours_val_mm_EMM.py; envs=() ;;
    nm*)  script=ours_val_mm_NM.py;  envs=("MMSSB_NM_LEVEL=${job#nm}") ;;
    rmm*) script=ours_val_mm_RMM.py; envs=("MMSSB_RMM_R=${job#rmm}") ;;
    *) echo "unknown job $job"; exit 2 ;;
  esac
  log="$LOGDIR/${MODEL}_s${SEED}_${job}.log"
  echo "[lane gpu$GPU] s$SEED $job start $(date '+%F %T')" | tee -a "$log"
  env "${envs[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$KIT/ours/$script" --cfg "$CFG" 2>&1 | tee -a "$log"
  echo "[lane gpu$GPU] s$SEED $job end rc=${PIPESTATUS[0]} $(date '+%F %T')" | tee -a "$log"
done
