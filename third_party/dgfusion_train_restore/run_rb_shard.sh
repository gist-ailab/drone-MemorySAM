#!/bin/bash
# 기준선 강건 평가기(robust_bench_eval.py) 한 샤드를 돌린다. DRN-260930-02.
#   WHICH=a|b   체크포인트 (a: DGFusion 발표재현 val-best 80k / b: 열화 커리큘럼 재학습 val-best 90k)
#   GPU=n       물리 GPU 번호(CUDA_VISIBLE_DEVICES)
#   TAG=이름    출력 폴더 이름(robust_out/<TAG>)
#   ONLY="id;id;..."  (선택) 이 케이스만. 케이스 id 에 | + 가 있어 ';' 로 구분한다.
#   LIMIT=N     (선택) 앞 N 장만(스모크)
# 출력: robust_out/<TAG>/shard.npz (케이스별 혼동행렬) + run.log
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
PY=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
[ -x "$PY" ] || PY=/home/jemo_maeng/anaconda3/envs/dgfusion/bin/python
export PYTHONPATH=$PWD:$PWD/OneFormer DETECTRON2_DATASETS=$PWD/datasets WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${GPU:?GPU 를 지정하라}
WHICH=${WHICH:?WHICH=a|b}
TAG=${TAG:?TAG 를 지정하라}
case "$WHICH" in
  a) CKPT=output/dgfusion_a_valbest80k/model_0079999.pth ;;
  b) CKPT=output/dgfusion_b_valbest90k/model_0089999.pth ;;
  *) echo "WHICH 는 a|b"; exit 2 ;;
esac
OUT=/SSDb/jemo_maeng/dgfusion_train/robust_out/$TAG
mkdir -p "$OUT"
ARGS=()
if [ -n "${ONLY:-}" ]; then
  IFS=';' read -r -a ONLY_ARR <<< "$ONLY"
  ARGS+=(--only "${ONLY_ARR[@]}")
fi
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
echo "$(date '+%F %T') [shard] $TAG 시작 WHICH=$WHICH GPU=$GPU ONLY=${ONLY:-전부}"
$PY tools/baseline_failure/robust_bench_eval.py \
  --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
  --weights "$CKPT" --split test --out "$OUT" \
  --dump_hists "$OUT/shard.npz" "${ARGS[@]}" \
  --opts MODEL.TEST.DEPTH_ON False > "$OUT/run.log" 2>&1
echo "$(date '+%F %T') [shard] $TAG 종료 exit=$?"
grep -E "CASE |총 .* 장 완료|Error|Traceback" "$OUT/run.log" | tail -70
