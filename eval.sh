#!/usr/bin/env bash
# Unified eval launcher — 구 eval_val.sh + eval_test.sh 통합 (mode만 상이했음)
#
# Usage:
#   bash eval.sh --mode val          # val 평가
#   bash eval.sh test                # 첫 인자로 mode 지정도 허용
#   CFG=<cfg.yaml> CKPT=<ckpt.pth> bash eval.sh --mode val   # config/ckpt 오버라이드
#   DRY_RUN=1 bash eval.sh --mode val                        # 커맨드만 출력(실행 안 함)
set -e

MODE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)   MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#--mode=}"; shift ;;
    val|test) MODE="$1"; shift ;;
    *) echo "[eval] unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ "$MODE" != "val" ] && [ "$MODE" != "test" ]; then
  echo "usage: bash eval.sh --mode val|test  (또는 bash eval.sh val|test)" >&2
  exit 1
fi

CFG="${CFG:-configs/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml}"
CKPT="${CKPT:-outputs/MMSamBase/levine_multiaqua_rgbtl_LoRASam_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch24_93.54_checkpoint.pth}"

CMD=(python3 val_multiaqua_detailed.py --cfg "${CFG}" --model_path "${CKPT}" --mode "${MODE}")

if [ -n "${DRY_RUN:-}" ]; then
  echo "[eval][dry-run] mode=${MODE}"
  echo "[eval][dry-run] ${CMD[*]}"
  exit 0
fi

"${CMD[@]}"

# Analyze detailed_log.json
CKPT_DIR=$(dirname "${CKPT}")
CKPT_PREFIX=$(basename "${CKPT}" .pth | sed 's/_checkpoint//')
LORA_MODEL=$(grep 'LORA_MODEL' "${CFG}" | head -1 | awk '{print $3}' | sed 's/LoRA_Sam_//')
DETAIL_DIR="${CKPT_DIR}/${CKPT_PREFIX}_${MODE}_pred_${LORA_MODEL}"
JSON_PATH="${DETAIL_DIR}/detailed_log.json"

if [ -f "${JSON_PATH}" ]; then
    echo ""
    echo "========== Analyzing detailed log =========="
    python3 analyze_detailed_log.py "${JSON_PATH}"
else
    echo "[WARN] detailed_log.json not found at: ${JSON_PATH}"
fi
