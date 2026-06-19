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

export CUDA_VISIBLE_DEVICES="3"

# OMP_NUM_THREADS 경고 방지 (권장)
export OMP_NUM_THREADS=1

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

# torchrun (단일 GPU). DDP면 위 멀티 GPU 블록 참고.
torchrun --nproc_per_node=1 --master_port=21613 train_sam3_rbma.py \
 --cfg ${CFG} \
2>&1 | tee "logs/${folder_name}/${cfg_name}_${timestamp}.log"
