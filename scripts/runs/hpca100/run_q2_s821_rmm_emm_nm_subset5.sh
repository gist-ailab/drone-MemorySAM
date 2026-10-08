#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# ②-b 대조군(판정 세션 지시 2026-10-04): P56-A s821 subset5 체인과 같은 부분집합·같은 명령으로
# Q2 시드821(비교 기준) RMM->EMM->NM, GPU2. 전수값(robust_q2_20260929)과 섞지 않음 — 부분집합 전용.
set -u
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
cd "$REPO" || exit 1
source /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/activate
export PYTHONPATH=$REPO:$REPO/semseg/models/sam2
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=2

CFG=configs/q2_s821_eval1024_hpca100.yaml
CKPT=/home/jovyan/SSDb/jemo_maeng/robust_0923/ckpts/Q2_s821_epoch40_67.0.pth
OUT=analysis_out/q2_s821_robust_subset5_20261004
LOGD=logs
mkdir -p "$OUT" "$LOGD"

clog() { echo "[$(date -u '+%F %T')UTC] $*" | tee -a "$LOGD/q2_s821_robust_subset5_chain_20261004.log"; }

[ -f "$CKPT" ] || { clog "ABORT ckpt 없음 $CKPT"; exit 1; }
python tools/eval_harness_guard.py --check > "$LOGD/guard_q2_s821_robust_subset5.log" 2>&1 || { clog "ABORT 하네스 가드 실패"; exit 1; }
clog "시작(subset_every 5, 380장) ckpt md5 $(md5sum "$CKPT" | cut -d' ' -f1)"

clog "=== RMM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol rmm --batch 8 --seed 0 \
  --subset_every 5 \
  --out "$OUT/rmm" 2>&1 | tee "$LOGD/q2_s821_rmm_subset5_20261004.log"
clog "RMM rc=$? DONE"

clog "=== EMM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol emm --batch 8 --seed 0 \
  --subset_every 5 \
  --out "$OUT/emm" 2>&1 | tee "$LOGD/q2_s821_emm_subset5_20261004.log"
clog "EMM rc=$? DONE"

clog "=== NM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol nm --nm_gaussian --nm_gaussian_std 0.2 --batch 8 --seed 0 \
  --subset_every 5 \
  --out "$OUT/nm" 2>&1 | tee "$LOGD/q2_s821_nm_subset5_20261004.log"
clog "NM rc=$? DONE"

clog "CHAIN_SUBSET5_ALL_DONE"
