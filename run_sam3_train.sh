export cur_dir=`pwd`

# ── 멀티 GPU (DDP) 예시 — 필요 시 주석 해제 ───────────────────────────────
# export CUDA_VISIBLE_DEVICES="6,7"
# CFG="configs/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml"
# cfg_name=$(basename ${CFG} .yaml); folder_name="${cfg_name}"
# mkdir -p logs/${folder_name}; timestamp=$(date +%Y%m%d_%H%M%S)
# export OMP_NUM_THREADS=1
# export HF_HUB_OFFLINE=1
# export PYTHONPATH="semseg/models/sam3:${PYTHONPATH}"   # import sam3
# torchrun --nproc_per_node=2 --master_port=21613 train_sam3_rbma.py \
#  --cfg ${CFG} \
# 2>&1 | tee "logs/${folder_name}/${cfg_name}_${timestamp}.log"

# ── GPU 선택: 비어있는 GPU 자동 배정 ─────────────────────────────────────────
#   - 직접 지정: CUDA_VISIBLE_DEVICES=3 bash run_sam3_train.sh   (그대로 존중)
#   - 자동 선택: NGPU=1 bash run_sam3_train.sh                   (빈 GPU 1장 자동)
NGPU="${NGPU:-1}"
CUDA_VISIBLE_DEVICES="$(bash scripts/pick_free_gpus.sh "$NGPU")" || {
  echo "[run_sam3] 빈 GPU ${NGPU}장을 찾지 못했습니다. nvidia-smi 확인 후 NGPU/CUDA_VISIBLE_DEVICES를 조정하세요." >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES
NPROC="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
echo "[run_sam3] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"

# OMP_NUM_THREADS 경고 방지 (권장)
export OMP_NUM_THREADS=1

# 공유 GPU 메모리 단편화 완화 (OOM "reserved but unallocated" 대응)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SAM3 전용 환경 (필수)
export HF_HUB_OFFLINE=1
export PYTHONPATH="semseg/models/sam3:${PYTHONPATH}"   # import sam3

# ── 실행할 config 파일 설정 ────────────────────────────────────────────────
CFG="configs/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml"

# config 파일명에서 폴더명 자동 생성
cfg_name=$(basename ${CFG} .yaml)
folder_name="${cfg_name}"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)

# torchrun. nproc는 자동 선택된 GPU 개수(${NPROC})를 따른다.
torchrun --nproc_per_node=${NPROC} --master_port=21613 train_sam3_rbma.py \
 --cfg ${CFG} \
2>&1 | tee "logs/${folder_name}/${cfg_name}_${timestamp}.log"
