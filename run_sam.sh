export cur_dir=`pwd`
export save_exp_name="0202_DELIVER_RGBD_LORA_SAM2_P7_lora_lr000045"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

folder_name="0202_DELIVER_RGBD_LORA_SAM2_P7_lora_lr000045"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)


python3 -u -m torch.distributed.launch --nproc_per_node=4 --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg configs/lecun_deliver_rgbd_P7.yaml \
2>&1 | tee "logs/${folder_name}/${save_exp_name}_${timestamp}.log"

