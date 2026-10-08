#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
set -u
R=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
CK=$R/ckpts_external/e1screen821_epoch35_67.25_top1_checkpoint.pth
C=$R/configs/eval/yeon-e1screen821_eval1024.yaml
LOGD=$R/logs
mkdir -p $LOGD
clog() { echo "[$(date '+%F %T')] $*" | tee -a $LOGD/e1screen821_yeon_gpu2_chain.log; }
[ -f "$CK" ] || { clog "ABORT ckpt 없음"; exit 1; }
[ -f "$C" ] || { clog "ABORT cfg 없음"; exit 1; }
set +u
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM || exit 1
set -u
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$R:$R/semseg/models/sam2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=2
cd $R
python tools/eval_harness_guard.py --check > $LOGD/guard_e1screen821_yeon_gpu2.log 2>&1 || { clog "ABORT 가드 실패"; exit 1; }
clog "E1스크린821 GPU2 시작 ckpt md5 $(md5sum $CK | cut -d' ' -f1)"
N=e1screen821_gpu2_emm
python tools/mm_eval_v2.py --cfg $C --model_path $CK --split test --protocol emm --batch 8 --seed 0 --expected_clean_miou 55.94 --tol 0.1 --out $R/robust_out/$N > $LOGD/$N.log 2>&1
clog "$N rc=$? $(grep -a 'clean_mIoU\|clean 등록수치' $LOGD/$N.log | tail -1 | cut -c1-160)"
for P in depth+event+lidar img+depth+lidar img+depth+event img+event+lidar; do
  N=e1screen821_gpu2_rmm_${P//+/_}
  MM_PRESENT_ONLY=$P python tools/mm_eval_v2.py --cfg $C --model_path $CK --split test --protocol rmm --rmm_ratios 0.25 0.5 0.75 --batch 8 --seed 0 --expected_clean_miou 55.94 --tol 0.1 --out $R/robust_out/$N > $LOGD/$N.log 2>&1
  clog "$N rc=$? $(grep -a 'clean_mIoU\|clean 등록수치' $LOGD/$N.log | tail -1 | cut -c1-160)"
done
N=e1screen821_gpu2_nm
python tools/mm_eval_v2.py --cfg $C --model_path $CK --split test --protocol nm --nm_density 0.05 0.1 0.2 --batch 8 --seed 0 --expected_clean_miou 55.94 --tol 0.1 --out $R/robust_out/$N > $LOGD/$N.log 2>&1
clog "$N rc=$? $(grep -a 'clean_mIoU\|clean 등록수치' $LOGD/$N.log | tail -1 | cut -c1-160)"
clog "GPU2_E1SCREEN821_YEON_DONE"
