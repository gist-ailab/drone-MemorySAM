CFG="configs/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml"
CKPT="outputs/MMSamBase/levine_multiaqua_rgbtl_LoRASam_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch24_93.54_checkpoint.pth"

cfg_name=$(basename ${CFG} .yaml)


# python3 -u val_multiaqua.py --cfg ${CFG} --model_path ${CKPT} \
#   --mode val --macvi 

python3 val_multiaqua_detailed.py --cfg ${CFG} --model_path ${CKPT} \
  --mode val 

# Analyze detailed_log.json
CKPT_DIR=$(dirname ${CKPT})
CKPT_PREFIX=$(basename ${CKPT} .pth | sed 's/_checkpoint//')
LORA_MODEL=$(grep 'LORA_MODEL' ${CFG} | head -1 | awk '{print $3}' | sed 's/LoRA_Sam_//')
DETAIL_DIR="${CKPT_DIR}/${CKPT_PREFIX}_val_pred_${LORA_MODEL}"
JSON_PATH="${DETAIL_DIR}/detailed_log.json"

if [ -f "${JSON_PATH}" ]; then
    echo ""
    echo "========== Analyzing detailed log =========="
    python3 analyze_detailed_log.py "${JSON_PATH}"
else
    echo "[WARN] detailed_log.json not found at: ${JSON_PATH}"
fi
