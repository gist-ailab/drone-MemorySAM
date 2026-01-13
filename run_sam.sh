export cur_dir=`pwd`
export save_exp_name="0108_DELIVER_RGBDL_SAM2_bplus_lora_lr00006"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

folder_name="deliver_rgbdl"
mkdir -p logs/${folder_name}
timestamp=$(date +%Y%m%d_%H%M%S)


python3 -u -m torch.distributed.launch --nproc_per_node=4 --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg /home/jovyan/SSDc/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM/configs/hpca-deliver_rgbdl_sam.yaml \
2>&1 | tee "logs/${folder_name}/${save_exp_name}_${timestamp}.log"
