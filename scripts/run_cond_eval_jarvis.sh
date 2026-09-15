#!/bin/bash
# [DELIVER 조건별 평가 — 추가 시드, jarvis] 인자: <GPU> <TAG>...  (TAG = E1s2|E1s3|E13s2|E13s3)
# §5-27(E1·E13 확정 시드1) 의 조건별 검사를 시드2·3 으로 넓힌다. 절차는 scripts/run_cond_eval_yeon_v2.sh 와 같다:
# 첫 조건(cloud)으로 로드 줄을 확인하고 "missing=0 unexpected=0" 이 아니면 그 TAG 는 건너뛴다.
set -u
G="$1"; shift
N=/SSDb/jemo_maeng/src/drone-MemorySAM
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
cd $N || exit 1
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34; export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
D=/SSDb/jemo_maeng/dset/DELIVER; O=$N/logs/cond_eval; R=outputs/ReliaDINO
declare -A CFG=( [E1s2]=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260902_eval1024_E1_confirm200_s2.yaml
                 [E1s3]=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260903_eval1024_E1_confirm200_s3.yaml
                 [E13s2]=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260902_eval1024_E13_confirm200_s2.yaml
                 [E13s3]=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260903_eval1024_E13_confirm200_s3.yaml )
declare -A RUN=( [E1s2]=jarvis_deliver_rgbdel_P46_c3only_seed20260902_E1_confirm200_s2
                 [E1s3]=jarvis_deliver_rgbdel_P46_c3only_seed20260903_E1_confirm200_s3
                 [E13s2]=jarvis_deliver_rgbdel_P46_c3only_seed20260902_E13_confirm200_s2
                 [E13s3]=jarvis_deliver_rgbdel_P46_c3only_seed20260903_E13_confirm200_s3 )
for TAG in "$@"; do
  CK=$(ls $R/${RUN[$TAG]}/*/epoch*_top1_checkpoint.pth 2>/dev/null | head -1)
  C=${CFG[$TAG]}
  [ -f "$CK" ] && [ -f "$C" ] || { echo "COND_SKIP $TAG ckpt/cfg 없음 ck=$CK cfg=$C"; continue; }
  mkdir -p $O/$TAG
  echo "COND_START $(date '+%F %T') $TAG gpu=$G ckpt=$(basename $CK)"
  python tools/eval_per_domain.py --repo $N --dataset-root $D --split test --batch 1 --cfg $C --ckpt $TAG=$CK --conditions cloud --gpu $G --out-dir $O/$TAG
  LD=$(grep -a 'ReliaDINO loaded' $O/$TAG/${TAG}__cloud.log | head -1)
  echo "COND_LOAD $TAG $LD"
  case "$LD" in *"missing=0 unexpected=0"*) ;; *) echo "COND_LOAD_BAD $TAG — 건너뜀"; continue;; esac
  python tools/eval_per_domain.py --repo $N --dataset-root $D --split test --batch 1 --cfg $C --ckpt $TAG=$CK --conditions fog,night,rain,sun --gpu $G --out-dir $O/$TAG
  for c in cloud fog night rain sun; do echo "COND_RESULT $TAG $c $(grep -a '^mIoU:' $O/$TAG/${TAG}__$c.log | tail -1)"; done
  echo "COND_TAG_DONE $TAG $(date '+%F %T')"
done
echo "COND_LANE_END gpu=$G $(date '+%F %T')"
