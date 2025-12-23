export cur_dir=`pwd`
export save_exp_name="1223_DELIVER_RGBDEL_SAM2_bplus_lora_lr00006"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=21612 --use_env train_sam2_lora.py \
 --cfg /SSDb/jemo_maeng/src/Project/Drone/detection/MemorySAM/configs/deliver_rgbdel_sam_recon.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"
