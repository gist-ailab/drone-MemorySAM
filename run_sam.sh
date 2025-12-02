export cur_dir=`pwd`
export save_exp_name="0904_DELIVER_RGBDepth_SAM2_bplus_lora_lr00008"
export CUDA_VISIBLE_DEVICES="0"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=21612 --use_env train_sam2_lora.py \
 --cfg /home/ailab-drone-demo/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM/configs/deliver_rgbdel_sam_custom.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"
