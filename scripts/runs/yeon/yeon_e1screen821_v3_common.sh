#!/bin/bash
# 서버 전용 실행 기록(yeon, 2026-10-08)
# E1스크린 s821(40ep, ckpt epoch35_67.25) 강건성 재기동 v3 (2026-09-29, 판정 세션 지시).
# 이전 2회(09-25, 09-26~27) 시도는 yeon /SSDb 100% 포화로 극심하게 느려지다 중단됨(209s/case,
# 4h46m에 34%) — 디스크 정리(09-29) 이후 재시도.
# 판정 세션 요구: 출력 경로 겹침 재발 방지 위해 <out>/<체인이름>/<케이스> 구조 + 잠금파일(.lock)로
# 스킵 판정(json 존재 여부 대신).
set -u
R=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-develop
CK=$R/ckpts_external/e1screen821_epoch35_67.25_top1_checkpoint.pth
C=$R/configs/eval/yeon-e1screen821_eval1024.yaml
CHAIN=yeon_e1screen821_v3
OUT=$R/robust_out/$CHAIN
LOGD=$R/logs
mkdir -p "$OUT" "$LOGD"
clog() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGD/${CHAIN}_${PROC_NAME:-main}_chain.log"; }

run_case() {
  # $1=케이스이름 $2..=mm_eval_v2.py 인자
  local case_name="$1"; shift
  local case_dir="$OUT/$case_name"
  local lock="$case_dir/.lock"
  mkdir -p "$case_dir"
  if [ -f "$lock" ]; then
    clog "$case_name SKIP(잠금 파일 존재 — 다른 체인이 선점)"
    return 0
  fi
  touch "$lock"
  clog "$case_name 시작"
  python tools/mm_eval_v2.py --model_path "$CK" --cfg "$C" --split test --batch 8 --seed 0 \
    --expected_clean_miou 55.94 --tol 0.1 --out "$case_dir" "$@" > "$LOGD/${CHAIN}_${case_name}.log" 2>&1
  local rc=$?
  clog "$case_name rc=$rc $(grep -a 'clean_mIoU\|clean 등록수치' "$LOGD/${CHAIN}_${case_name}.log" | tail -1 | cut -c1-160)"
  return $rc
}

[ -f "$CK" ] || { echo "ABORT ckpt 없음 $CK"; exit 1; }
[ -f "$C" ] || { echo "ABORT cfg 없음 $C"; exit 1; }
cd "$R" || exit 1
set +u
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM || exit 1
set -u
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$R:$R/semseg/models/sam2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
