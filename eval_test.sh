CFG="configs/eval_config/levine-multiaqua_rgbtl_P17_hardaug5.yaml"
CKPT_DIR="outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug5/MULTIAQUA_CMNeXt-B2_ilt"

# Night-Val top checkpoints (test = night images)
CKPTS=(
    "${CKPT_DIR}/night_epoch35_90.34_top1_checkpoint.pth"
    "${CKPT_DIR}/night_epoch36_89.62_top2_checkpoint.pth"
    "${CKPT_DIR}/night_epoch33_89.59_top3_checkpoint.pth"
)

for CKPT in "${CKPTS[@]}"; do
    echo "========================================"
    echo "Evaluating: $(basename ${CKPT})"
    echo "========================================"

    python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
        --mode test \
        --macvi

    python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
        --mode test
done
