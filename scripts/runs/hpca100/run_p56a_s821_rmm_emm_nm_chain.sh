#!/bin/bash
# 서버 전용 실행 기록(hpca100, 2026-10-08)
# ② 강건성 체인(P56-A 모달 충돌 학습 시드821, ep40 val-best top1) — mm_eval_v2.py(legal v2 nearest-exact
# 몽키패치, hpca_gpu1_q3emm.sh 선례와 같은 하네스)로 RMM(42케이스, 3부분집합×3비율+clean)
# -> EMM(15케이스) -> NM 순서, GPU1 순차 실행(생각정리 지시: ③ percase 끝난 뒤 이 GPU에서 이어감).
set -u
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM-develop
cd "$REPO" || exit 1
source /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/activate
export PYTHONPATH=$REPO:$REPO/semseg/models/sam2
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=1

CFG=configs/p56a_s821_eval1024_hpca100.yaml
CKPT=outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A/DELIVER_ReliaDINO-ViTL16_idel/epoch40_67.05_top1_checkpoint.pth
OUT=analysis_out/p56a_s821_robust_20261004
LOGD=logs
mkdir -p "$OUT" "$LOGD"

clog() { echo "[$(date -u '+%F %T')UTC] $*" | tee -a "$LOGD/p56a_s821_robust_chain_20261004.log"; }

[ -f "$CKPT" ] || { clog "ABORT ckpt 없음 $CKPT"; exit 1; }
python tools/eval_harness_guard.py --check > "$LOGD/guard_p56a_s821_robust.log" 2>&1 || { clog "ABORT 하네스 가드 실패"; exit 1; }
clog "시작 ckpt md5 $(md5sum "$CKPT" | cut -d' ' -f1)"

clog "=== RMM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol rmm --batch 8 --seed 0 \
  --expected_clean_miou 55.82 --tol 0.1 --out "$OUT/rmm" 2>&1 | tee "$LOGD/p56a_s821_rmm_v2_20261004.log"
clog "RMM rc=$? DONE"

clog "=== EMM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol emm --batch 8 --seed 0 \
  --expected_clean_miou 55.82 --tol 0.1 --out "$OUT/emm" 2>&1 | tee "$LOGD/p56a_s821_emm_v2_20261004.log"
clog "EMM rc=$? DONE"

clog "=== NM 시작 ==="
python tools/mm_eval_v2.py --cfg "$CFG" --model_path "$CKPT" --split test --protocol nm --nm_gaussian --nm_gaussian_std 0.2 --batch 8 --seed 0 \
  --expected_clean_miou 55.82 --tol 0.1 --out "$OUT/nm" 2>&1 | tee "$LOGD/p56a_s821_nm_v2_20261004.log"
clog "NM rc=$? DONE"

clog "CHAIN_ALL_DONE"
