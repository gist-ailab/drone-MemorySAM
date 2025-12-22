export cur_dir=`pwd`
export save_exp_name="1207_DELIVERrgbdel_SAM2_bplus_lora_lr00003"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
python3 -m torch.distributed.launch --nproc_per_node=5 --master_port=21612 --use_env train_sam2_lora.py \
 --cfg configs/deliver_rgbdel_sam_lecun_original.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"
