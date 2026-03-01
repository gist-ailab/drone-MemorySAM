CFG="configs/levine-multiaqua_rgbtl_P17_hardaug5.yaml"
CKPT="outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/periodic_epoch15_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)



python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
--mode test \
--macvi

python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
--mode test