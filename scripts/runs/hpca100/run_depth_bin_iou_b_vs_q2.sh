#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# 원거리 진단(판정 세션 지시 2026-10-05): P56-B 시드821 vs Q2 시드821, 4구간(log-분위) depth bin IoU
set -uo pipefail
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
export CUDA_VISIBLE_DEVICES=""
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
GT=analysis_out/p56b_s821_dump_20261005/test/gt
OUT=analysis_out/depth_bin_b_vs_q2_s821_20261005
mkdir -p "$OUT" logs

$PY tools/baseline_failure/depth_bin_iou.py \
  --gt "$GT" \
  --depth_root /home/jovyan/SSDb/jemo_maeng/dset/DELIVER \
  --split test --out "$OUT" --bins 4 \
  --models B=analysis_out/p56b_s821_dump_20261005/test/pred Q2=analysis_out/q2_s821_dump_20261005/test/pred \
  2>&1 | tee logs/depth_bin_b_vs_q2_s821_20261005.log

echo "DEPTHBIN_DONE exit=$?"
