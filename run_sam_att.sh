#!/bin/bash
# LoRA-SAM with Learnable Modal Attention Fusion 학습 스크립트

export cur_dir=`pwd`
export save_exp_name="DELIVER_SAM2_LoRA_ATT"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"

python3 -m torch.distributed.launch --nproc_per_node=7 --master_port=21613 --use_env train_sam2_lora_att.py \
 --cfg configs/deliver_rgbdel_sam_att.yaml \
\
2>&1 \
| tee "sam/log/${save_exp_name}.`date`"

