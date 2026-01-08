export cur_dir=`pwd`
export save_exp_name="0106_DELIVER_RGBD_SAM2_bplus_lora_lr000045"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mkdir -p logs/deliver_rgbd
timestamp=$(date +%Y%m%d_%H%M%S)


python3 -u -m torch.distributed.launch --nproc_per_node=7 --master_port=21612 --use_env train_sam2_lora_paper.py \
 --cfg /SSDe/jemo_maeng/src/Project/Drone/drone-MemorySAM/configs/levine_deliver_rgbd_sam_recon.yaml \
2>&1 | tee "logs/deliver_rgbd/${save_exp_name}_${timestamp}.log"
