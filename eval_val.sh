CFG="configs/eval_config/levine-multiaqua_rgbtl_P17_hardaug5.yaml"
CKPT_DIR="outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug5/MULTIAQUA_CMNeXt-B2_ilt"

# Day-Val top checkpoints
CKPTS=(
    "${CKPT_DIR}/epoch28_93.77_top1_checkpoint.pth"
    "${CKPT_DIR}/epoch22_93.73_top2_checkpoint.pth"
    "${CKPT_DIR}/epoch27_93.72_top3_checkpoint.pth"
)

for CKPT in "${CKPTS[@]}"; do
    echo "========================================"
    echo "Evaluating: $(basename ${CKPT})"
    echo "========================================"

    python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
        --mode val \
        --macvi

    python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
        --mode val
done
