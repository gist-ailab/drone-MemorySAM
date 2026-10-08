#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
set -u
PROC_NAME=rmm_nm
export CUDA_VISIBLE_DEVICES=6
source /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop/yeon_e1screen821_v3_common.sh
python tools/eval_harness_guard.py --check > "$LOGD/guard_${CHAIN}_rmm_nm.log" 2>&1 || { clog "ABORT 가드 실패"; exit 1; }
export MM_PRESENT_ONLY=depth+event+lidar
run_case rmm_img --protocol rmm --rmm_ratios 0.25 0.5 0.75
unset MM_PRESENT_ONLY
run_case nm --protocol nm --nm_density 0.05 0.1 0.2
clog "RMM_NM_DONE"
