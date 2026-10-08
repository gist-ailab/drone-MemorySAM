#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
set -u
PROC_NAME=emm
export CUDA_VISIBLE_DEVICES=5
source /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop/yeon_e1screen821_v3_common.sh
python tools/eval_harness_guard.py --check > "$LOGD/guard_${CHAIN}_emm.log" 2>&1 || { clog "ABORT 가드 실패"; exit 1; }
run_case emm --protocol emm
clog "EMM_DONE"
