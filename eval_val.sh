CFG="configs/bengio-multiaqua_rgbtl_P9_hardaug6.yaml"
CKPT="outputs/MMSamP9/bengio_multiaqua_rgbtl_P9_hardaug6/MULTIAQUA_CMNeXt-B2_ilt/epoch43_94.08_top4_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)


python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
--mode val \
--macvi

python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
--mode val