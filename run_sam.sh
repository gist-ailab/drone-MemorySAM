export cur_dir=`pwd`
export save_exp_name="0116_DELIVER_RGBD_LORA_SAM2_P3_lora_lr00006"
export CUDA_VISIBLE_DEVICES="1,2,3,4"

folder_name="0116_DELIVER_RGBD_LORA_SAM2_P3_lora_lr00006"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)


python3 -u -m torch.distributed.launch --nproc_per_node=4 --master_port=21611 --use_env train_sam2_lora_paper.py \
 --cfg /SSDc/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM/configs/lecun_deliver_rgbd_sam.yaml \
2>&1 | tee "logs/${folder_name}/${save_exp_name}_${timestamp}.log"
