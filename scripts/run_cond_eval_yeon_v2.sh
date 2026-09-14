#!/bin/bash
# [DELIVER 조건별 평가 v2] yeon develop 새 체크아웃(TAPS 포함)에서 E1·E13 확정 시드1 을 다시 잰다.
#   v1(run_cond_eval_yeon.sh)은 옛 체크아웃(drone-MemorySAM-p38, TAPS 코드 없음)이라 E1s1 unexpected=64 ·
#   E13s1 unexpected=72 로 로드돼 무효였다(2026-09-15). 기준선 seed821 몫은 v1 결과를 쓴다.
# 인자: A|B.  A = GPU4: E1 확정 시드1(5조건) / B = GPU5: E13 확정 시드1(5조건)
# 첫 조건 로그의 로드 줄이 "missing=0 unexpected=0" 이 아니면 COND_LOAD_BAD 를 찍고 즉시 멈춘다.
set -u
LANE="$1"
N=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
C=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/_eval_ckpts/cond
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh; conda activate MMSS_SAM
cd $N || exit 1
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34; export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
D=/SSDb/jemo_maeng/dset/DELIVER; O=$N/logs/cond_eval
if [ "$LANE" = A ]; then
  TAG=E1s1; CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
  CK=$C/E1s1_epoch140_68.9_top1_checkpoint.pth; G=4
else
  TAG=E13s1; CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E13_confirm200.yaml
  CK=$C/E13s1_epoch100_67.84_top1_checkpoint.pth; G=5
fi
mkdir -p $O/$TAG
# 로드 검사: 첫 조건(cloud)만 먼저 돌려 로드 줄을 확인한 뒤 나머지를 돈다
echo "COND_START $(date '+%F %T') $TAG cloud (로드 검사)"
python tools/eval_per_domain.py --repo $N --dataset-root $D --split test --batch 1 --cfg $CFG --ckpt $TAG=$CK --conditions cloud --gpu $G --out-dir $O/$TAG
LD=$(grep -a 'ReliaDINO loaded' $O/$TAG/${TAG}__cloud.log | head -1)
echo "COND_LOAD $TAG $LD"
case "$LD" in *"missing=0 unexpected=0"*) ;; *) echo "COND_LOAD_BAD $TAG — 로드가 깨끗하지 않아 중단"; exit 1;; esac
echo "COND_DONE cloud $(grep -a '^mIoU:' $O/$TAG/${TAG}__cloud.log | tail -1)"
python tools/eval_per_domain.py --repo $N --dataset-root $D --split test --batch 1 --cfg $CFG --ckpt $TAG=$CK --conditions fog,night,rain,sun --gpu $G --out-dir $O/$TAG
echo "COND_DONE rc=$? $(date '+%F %T') $TAG"
for c in cloud fog night rain sun; do echo "COND_RESULT $TAG $c $(grep -a '^mIoU:' $O/$TAG/${TAG}__$c.log | tail -1)"; done
echo "COND_LANE_DONE $LANE $(date '+%F %T')"
