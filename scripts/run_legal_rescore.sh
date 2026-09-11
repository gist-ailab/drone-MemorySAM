#!/bin/bash
# 사용법: bash run_legal_rescore.sh <카드명> <GPU>
#   예: bash run_legal_rescore.sh E2s2 2
# val-best(top1, test_ 접두어 제외)를 자동으로 골라 test → val 순으로 재채점한다.
#
# 🔴 이 스크립트는 **hpca100 전용**이다. 레포 경로·venv·스크래치 경로가 하드코딩돼 있다.
#    서버 배치 위치는 /tmp/jemo_scratch/run_legal_rescore.sh 이고, 이 파일이 그 정본이다
#    (2026-09-11 에 서버에만 있던 것을 회수해 develop 에 올렸다 — CLAUDE.md §1.7).
#    고칠 일이 생기면 **여기를 고친 뒤 서버로 전송**하고, 서버에서 직접 고치지 마라.
#
# 재채점 config 이름 규약: /tmp/jemo_scratch/configs/<카드명>_eval1024.yaml
#    (<카드명>.yaml 은 학습 config 다. 둘을 헷갈리면 768/BS4 로 돌아 무효 수치가 나온다.)
set -u
CARD="$1"; GPU="${2:-2}"
declare -A DIRS=(
  [E2s2]=hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E2s2
  [E12]=hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E12
  [B0s2]=bengio_deliver_rgbdel_P46_c3only_seed20260902_screen40_B0s2
  [E3s2]=hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E3s2
  [E13s2]=hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E13s2
  [E13s3]=hpca100_deliver_rgbdel_P46_c3only_seed20260903_screen40_E13s3
  [E14]=hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E14
  [E15]=hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E15
)
if [ -z "${DIRS[$CARD]+x}" ]; then
  echo "${CARD}_LEGAL_ABORT: DIRS 매핑에 '$CARD' 가 없다. 이 스크립트 상단 DIRS 에 추가하라."
  echo "  등록된 카드: ${!DIRS[*]}"
  exit 1
fi
RUN="${DIRS[$CARD]}"
cd /home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/semseg/models/sam2:${PYTHONPATH:-}
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
D=/tmp/jemo_scratch/outputs/ReliaDINO/$RUN/DELIVER_ReliaDINO-ViTL16_idel
CKPT=$(ls -1 "$D"/epoch*_top1_checkpoint.pth 2>/dev/null | grep -v '/test_' | sort -t_ -k2 -V | tail -1)
[ -z "$CKPT" ] && { echo "${CARD}_LEGAL_ABORT: val-best 없음 ($D)"; exit 1; }
echo "${CARD}_LEGAL_CKPT=$CKPT"
CFG=/tmp/jemo_scratch/configs/${CARD}_eval1024.yaml
[ -f "$CFG" ] || { echo "${CARD}_LEGAL_ABORT: 재채점 config 가 없다 ($CFG). 로컬 레포 configs/eval/ 에서 전송하라."; exit 1; }

# 🔴 legal 규약 사전 검증 — 1024·BS1 이 아니면 돌기 전에 멈춘다.
#    BS4 로 돌면 진행 표시줄이 475/501 로 뜨고 수치가 무효가 된다(두 번 실증된 사고).
#    돌고 나서 로그를 보고 알아채면 GPU 시간을 통째로 버리므로 여기서 막는다.
EVAL_BS=$(awk '/^EVAL:/{f=1} f&&/^  BATCH_SIZE/{print $3; exit}' "$CFG")
EVAL_SZ=$(awk '/^EVAL:/{f=1} f&&/^  IMAGE_SIZE/{print $3 $4; exit}' "$CFG")
if [ "${EVAL_BS:-}" != "1" ]; then
  echo "${CARD}_LEGAL_ABORT: EVAL.BATCH_SIZE 가 '${EVAL_BS:-없음}' 이다. legal 재채점은 native-GT BS1 이어야 한다 ($CFG)"
  exit 1
fi
case "${EVAL_SZ:-}" in
  '[1024,1024]') : ;;
  *) echo "${CARD}_LEGAL_ABORT: EVAL.IMAGE_SIZE 가 '${EVAL_SZ:-없음}' 이다. 하네스 규약은 [1024, 1024] 다 ($CFG)"; exit 1 ;;
esac
echo "${CARD}_LEGAL_PRECHECK_OK: EVAL BS=$EVAL_BS SIZE=$EVAL_SZ"

TS=$(date +%Y%m%d_%H%M%S)
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python3.11
$PY val.py --cfg "$CFG" --mode test --model_path "$CKPT" 2>&1 | tee "/tmp/jemo_scratch/logs/${CARD}_legal_test_$TS.log"
echo "${CARD}_LEGAL_TEST_DONE"
$PY val.py --cfg "$CFG" --mode val  --model_path "$CKPT" 2>&1 | tee "/tmp/jemo_scratch/logs/${CARD}_legal_val_$TS.log"
echo "${CARD}_LEGAL_ALL_DONE"
