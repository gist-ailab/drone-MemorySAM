export cur_dir=`pwd`
export save_exp_name="0917_DELIVER_RGBDepth_SAM2_bplus_lora_lr00008"
export CUDA_VISIBLE_DEVICES="0"
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=21612 --use_env val_mm_sam.py \
 --cfg /hpc2hdd/home/cliao127/MMSS-SAM-S1/configs/mcubes_rgbadn_sam.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"
