CFG="configs/eval_config/levine-multiaqua_rgbtl_P15_hardaug5.yaml"
CKPT="outputs/MMSamP15/levine_multiaqua_rgbtl_P15_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/epoch46_93.92_top1_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)



python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
--mode test \
--macvi

python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
--mode test 