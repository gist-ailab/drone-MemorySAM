#!/bin/bash
# [DELIVER 조건별 평가] yeon GPU4·5 두 레인. 2026-09-15 생각정리 결정 — §0 스크린 규약의 "악조건 −0.5 미만 없음" 검사.
# 인자: A|B. A = GPU4: E1 확정 시드1(5조건) + 기준선 seed821(cloud,fog) / B = GPU5: E13 확정 시드1(5조건) + 기준선(night,rain,sun)
set -u
LANE="$1"; Y=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38; C=$Y/_eval_ckpts/cond; E=/SSDb/jemo_maeng/cond_eval_cfgs
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh; conda activate MMSS_SAM
cd $Y || exit 1
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34; export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
T=/SSDb/jemo_maeng/eval_per_domain.py; D=/SSDb/jemo_maeng/dset/DELIVER; O=$Y/logs/cond_eval
run() { echo "COND_START $(date '+%F %T') $1"; python $T --repo $Y --dataset-root $D --split test --batch 1 "${@:2}"; echo "COND_DONE rc=$? $(date '+%F %T') $1"; }
if [ "$LANE" = A ]; then
  run E1s1 --cfg $E/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml --ckpt E1s1=$C/E1s1_epoch140_68.9_top1_checkpoint.pth --conditions cloud,fog,night,rain,sun --gpu 4 --out-dir $O/E1s1
  run base821_a --cfg $E/yeon-deliver_rgbdel_P46_c3only_lam01_base_eval1024.yaml --ckpt base821=$C/base821_epoch90_67.3_top1_checkpoint.pth --conditions cloud,fog --gpu 4 --out-dir $O/base821
else
  run E13s1 --cfg $E/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E13_confirm200.yaml --ckpt E13s1=$C/E13s1_epoch100_67.84_top1_checkpoint.pth --conditions cloud,fog,night,rain,sun --gpu 5 --out-dir $O/E13s1
  run base821_b --cfg $E/yeon-deliver_rgbdel_P46_c3only_lam01_base_eval1024.yaml --ckpt base821=$C/base821_epoch90_67.3_top1_checkpoint.pth --conditions night,rain,sun --gpu 5 --out-dir $O/base821
fi
echo "COND_LANE_DONE $LANE $(date '+%F %T')"
