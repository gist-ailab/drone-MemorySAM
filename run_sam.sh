export cur_dir=`pwd`

# ── GPU 선택: 비어있는 GPU 자동 배정 ─────────────────────────────────────────
#   - 직접 지정: CUDA_VISIBLE_DEVICES=0,1 bash run_sam.sh   (그대로 존중)
#   - 자동 선택: NGPU=4 bash run_sam.sh                      (빈 GPU 4장 자동)
NGPU="${NGPU:-4}"
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh "$NGPU")" || {
  echo "[run_sam] 빈 GPU ${NGPU}장을 찾지 못했습니다. nvidia-smi로 확인 후 NGPU를 낮추거나 CUDA_VISIBLE_DEVICES를 직접 지정하세요." >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES
NPROC="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
echo "[run_sam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"

# ── 실행할 config 파일 설정 ────────────────────────────────────────────────
CFG="configs/bengio-multiaqua_rgbtl_P9_hardaug6.yaml"
# CFG="configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml"
# CFG="configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml"

# config 파일명에서 폴더명 자동 생성
cfg_name=$(basename ${CFG} .yaml)
folder_name="${cfg_name}"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)

python3 -u -m torch.distributed.launch --nproc_per_node=${NPROC} --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg ${CFG} \
2>&1 | tee "logs/${folder_name}/${cfg_name}_${timestamp}.log"

