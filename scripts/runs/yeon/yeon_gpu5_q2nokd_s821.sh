#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
# yeon GPU5: Q2noKD(KD_W=0, 시드821) 재시도 — jarvis에서 09-24 11:26:28 rc=1로 실패, 재시도 안 됨.
set -u
R=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
C=$R/configs/yeon-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD.yaml
LOGD=$R/logs
mkdir -p $LOGD
clog() { echo "[$(date '+%F %T')] $*" | tee -a $LOGD/q2noKD_yeon_20260925.log; }
[ -f "$C" ] || { clog "ABORT cfg 없음 $C"; exit 1; }
set +u
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM || exit 1
set -u
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$R:$R/semseg/models/sam2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=5
cd $R
clog "Q2noKD 시드821 시작(yeon GPU5 재시도) cfg=$C"
torchrun --standalone --nproc_per_node=1 --master_port=29881 train_reliadino.py --cfg $C >> $LOGD/q2noKD_yeon_20260925.log 2>&1
clog "Q2noKD_YEON_DONE rc=$?"
