CFG="configs/eval_config/bengio-multiaqua_rgbtl_P14_hardaug5.yaml"
CKPT="outputs/MMSamP14/bengio_multiaqua_rgbtl_P14_hardaug5/MULTIAQUA_CMNeXt-B2_ilt/night_epoch47_90.75_top1_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)



python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
--mode test \
--macvi

python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
--mode test 