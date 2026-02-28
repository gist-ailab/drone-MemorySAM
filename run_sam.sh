export cur_dir=`pwd`
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ── 실행할 config 파일 설정 ────────────────────────────────────────────────
CFG="configs/levine-multiaqua_rgbtl_P18_hardaug5.yaml"
# CFG="configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml"
# CFG="configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml"

# config 파일명에서 폴더명 자동 생성
cfg_name=$(basename ${CFG} .yaml)
folder_name="${cfg_name}"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)

python3 -u -m torch.distributed.launch --nproc_per_node=8 --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg ${CFG} \
2>&1 | tee "logs/${folder_name}/${cfg_name}_${timestamp}.log"

