export cur_dir=`pwd`
export save_exp_name="1227_DELIVER_RGBDL_SAM2_bplus_lora_lr00006"
export CUDA_VISIBLE_DEVICES="0,4,5,6,7"
python3 -m torch.distributed.launch --nproc_per_node=5 --master_port=21612 --use_env train_sam2_lora.py \
 --cfg configs/bengio_deliver_rgbdl_sam.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"
