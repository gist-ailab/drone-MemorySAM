#!/bin/bash
# 우리 모델 D4 측정 — hpca100 판. jarvis 의 GPU 가 외부 사용자 학습으로 막혀 옮겨 왔다.
# 판정 세션이 정한 순서를 따른다: 부분 열화(RMM r=0.5) 두 건이 먼저고, 그다음 시드 확장,
# 마지막이 이미지별 기록이다. 3자 열화 비교가 먼저 필요하기 때문이다.
#
# 환경은 hpca100 기동 표준을 따른다(메모리 hpca100-cudnn-fix). pip install 금지 서버라
# 공유 venv 를 그대로 쓴다. DINOv3 백본은 이 venv 의 timm 1.0.24 가 만든다.
set -uo pipefail
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
WORK=/home/jovyan/SSDb/jemo_maeng/d4_ours
CFGDIR=$WORK/cfg
OUT=$WORK/out
BATCH=${BATCH:-16}          # A100 40GB — jarvis 4090 의 배치 8 보다 올려 잡는다
RATIO=${RATIO:-0.5}
SEED=${SEED:-0}
ALLOWED=${ALLOWED:-"1 2 3"} # GPU0 은 외부 프로세스가 34GiB 를 쓰고 있다
mkdir -p "$OUT"
cd "$REPO"

source /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/activate
export PYTHONPATH=$REPO/semseg/models/sam2:$REPO
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
_CUDNN=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)
[ -n "$_CUDNN" ] && export LD_LIBRARY_PATH="$_CUDNN:${LD_LIBRARY_PATH:-}"

C=$CFGDIR
K=$WORK/ckpts
# 이름 | 평가 config | 체크포인트
RUNS=(
"E1_s1|$C/hpca100-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml|$K/E1_s1_epoch140.pth"
"E1_s2|$C/hpca100-deliver_rgbdel_P46_c3only_seed20260902_eval1024_E1_confirm200_s2.yaml|$K/E1_s2_epoch134.pth"
"E1_s3|$C/hpca100-deliver_rgbdel_P46_c3only_seed20260903_eval1024_E1_confirm200_s3.yaml|$K/E1_s3_epoch70.pth"
"E13_s1|$C/hpca100-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E13_confirm200.yaml|$K/E13_s1_epoch100.pth"
)

free_gpu() {
  for g in $ALLOWED; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "${used:-99999}" -le 2000 ] && [ "${util:-100}" -le 10 ]; then echo "$g"; return; fi
  done
}

launch() {  # launch <tag> <cfg> <ckpt> <조건> [추가 인자...]
  local tag=$1 cfg=$2 ckpt=$3 cond=$4; shift 4
  [ -s "$OUT/$tag.json" ] && { echo "skip $tag" >> "$OUT/runner.log"; return 0; }
  [ -f "$ckpt" ] || { echo "$(date '+%F %T') 체크포인트 없음, 건너뜀: $tag ($ckpt)" >> "$OUT/runner.log"; return 0; }
  local g; g=$(free_gpu)
  while [ -z "$g" ]; do sleep 60; g=$(free_gpu); done
  echo "$(date '+%F %T') start $tag (GPU $g)" >> "$OUT/runner.log"
  CUDA_VISIBLE_DEVICES=$g python tools/baseline_failure/modality_zero_ablation.py \
    --cfg "$cfg" --model_path "$ckpt" --split test --batch "$BATCH" \
    --only "$cond" --out "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1 &
  sleep 120   # 이 프로세스의 메모리 점유가 nvidia-smi 에 반영된 뒤 다음 장을 고른다
}

echo "$(date '+%F %T') 시작 (batch=$BATCH, 허용 GPU: $ALLOWED)" > "$OUT/runner.log"

# ① 부분 열화 두 건 — 3자 비교에 먼저 필요하다. 시드 1 만 쓴다.
S1_CFG=$C/hpca100-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
S1_CKPT=$K/E1_s1_epoch140.pth
launch rmm_zero_depth "$S1_CFG" "$S1_CKPT" zero_depth --ratio "$RATIO" --ratio_seed "$SEED"
launch rmm_zero_img   "$S1_CFG" "$S1_CKPT" zero_img   --ratio "$RATIO" --ratio_seed "$SEED"
wait

# ② 시드 확장 — E1 시드2·3 과 E13 시드1 의 완전 제거 5조건.
for r in "${RUNS[@]:1}"; do
  name=${r%%|*}; rest=${r#*|}; cfg=${rest%%|*}; ckpt=${rest#*|}
  for c in base zero_img zero_depth zero_event zero_lidar; do
    launch "${name}_${c}" "$cfg" "$ckpt" "$c"
  done
done
wait

# ③ 시드1 이미지별 기록 — 조건별 Δ 산출용.
for c in base zero_depth zero_img; do
  launch "perimg_${c}" "$S1_CFG" "$S1_CKPT" "$c" --per_image_csv "$OUT/perimg_${c}.csv"
done
wait

echo "$(date '+%F %T') 전체 완료" >> "$OUT/runner.log"
cat "$OUT/runner.log"
