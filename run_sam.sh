export cur_dir=`pwd`
export save_exp_name="0124_DELIVER_RGBDEL_LORA_SAM2_P5_lora_lr00006"
export CUDA_VISIBLE_DEVICES="0,1"

folder_name="0124_DELIVER_RGBDEL_LORA_SAM2_P5_lora_lr00006"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)


python3 -u -m torch.distributed.launch --nproc_per_node=4 --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg configs/bai-deliver_rgbdel_sam.yaml \
2>&1 | tee "logs/${folder_name}/${save_exp_name}_${timestamp}.log"
