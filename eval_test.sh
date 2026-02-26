CFG="configs/eval_config/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml"
CKPT="./outputs/MMSamBase/levine_multiaqua_rgbtl_LoRASam_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch24_93.54_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)

# python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
# --mode test \

python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
--mode test \
# --macvi