#!/bin/bash
# 사용법: bash run_legal_rescore_jarvis_chain.sh <태그> <학습cfg식별문자열> <학습로그> <eval cfg> <ckpt 디렉터리> <test GPU> <val GPU>
#   예: bash run_legal_rescore_jarvis_chain.sh E13conf_s2 seed20260902_E13_confirm200_s2 \
#         logs/e13_confirm200_s2_launch_2gpu.log \
#         configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260902_eval1024_E13_confirm200_s2.yaml \
#         outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260902_E13_confirm200_s2/DELIVER_ReliaDINO-ViTL16_idel 1 7
#
# 🔴 jarvis 기본(레포 경로는 환경변수 REPO 로 바꿀 수 있다 — yeon 은
#    REPO=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38). pylibs_p34·anaconda3 MMSS_SAM 은
#    두 서버 공통(2026-09-14 yeon 기동 환경 실측). 200ep 런이 **완주한 뒤** val-best 로 legal 재채점을
#    test / val 두 GPU 에 병렬로 돌린다. 완주 전에는 기동하지 않는다(val-best 는 완주 후 확정 — 늦은 갱신 선례
#    E1 s1 ep138 · E13 s2 ep160 · E1 s2 ep112). 서버 배치 위치 = /SSDb/jemo_maeng/, 이 파일이 정본이다.
#
# 완주 판정 = 학습 로그에 'Total Training Time' 이 찍히고 AND 해당 cfg 의 train_reliadino.py 프로세스가 0 개.
#   pgrep 자기 매칭을 막으려고 'pgrep|bash -c' 줄을 뺀다(모니터 명세의 교훈).
# 산출: logs/<태그>_{test,val}_<ts>.log , 이 스크립트 콘솔에 CHAIN_* 마커.
set -u
TAG="$1"; KEY="$2"; TLOG="$3"; CFG="$4"; D="$5"; GT="$6"; GV="$7"
REPO="${REPO:-/SSDb/jemo_maeng/src/drone-MemorySAM}"
cd "$REPO" || { echo "CHAIN_ABORT: 레포 경로 없음 ($REPO)"; exit 1; }
nproc_of() { pgrep -af 'train_reliadino.py' | grep -- "$KEY" | grep -vc 'pgrep\|bash -c'; }

# 사전 검증 — 기동 전에 틀린 것을 잡는다
[ -f "$CFG" ]  || { echo "CHAIN_ABORT: eval cfg 없음 ($CFG)"; exit 1; }
[ -f "$TLOG" ] || { echo "CHAIN_ABORT: 학습 로그 없음 ($TLOG)"; exit 1; }
EVAL_BS=$(awk '/^EVAL:/{f=1} f&&/^  BATCH_SIZE/{print $3; exit}' "$CFG")
EVAL_SZ=$(awk '/^EVAL:/{f=1} f&&/^  IMAGE_SIZE/{print $3$4; exit}' "$CFG")
[ "$EVAL_BS" = "1" ] && [ "$EVAL_SZ" = "[1024,1024]" ] || { echo "CHAIN_ABORT: legal 규약 위반 BS=$EVAL_BS SIZE=$EVAL_SZ"; exit 1; }
N0=$(nproc_of)
if [ "$N0" = "0" ] && ! grep -aq 'Total Training Time' "$TLOG"; then
  echo "CHAIN_ABORT: 학습 프로세스가 안 보이는데 완주 표시도 없다 — 식별문자열($KEY) 확인 필요"; exit 1
fi
echo "CHAIN_WAIT $(date '+%F %T') 학습 프로세스 ${N0}개 — 완주 대기"

# 완주 대기
while true; do
  done_line=$(grep -a 'Total Training Time' "$TLOG" | tail -1)
  n=$(nproc_of)
  [ -n "$done_line" ] && [ "$n" = "0" ] && break
  if [ "$n" = "0" ] && [ -z "$done_line" ]; then
    sleep 180
    if [ "$(nproc_of)" = "0" ] && ! grep -aq 'Total Training Time' "$TLOG"; then
      echo "CHAIN_ABORT: 학습 프로세스가 완주 표시 없이 사라졌다(사망 의심). 재채점하지 않는다"; exit 1
    fi
  fi
  sleep 60
done
echo "CHAIN_TRAIN_DONE $(date '+%F %T') $done_line"

# val-best 선택 — test_ 접두어(test-best, 무효)는 glob 에서 이미 빠진다. 정확히 1 개여야 한다
mapfile -t TOPS < <(ls -1 "$D"/epoch*_top1_checkpoint.pth 2>/dev/null)
[ "${#TOPS[@]}" = "1" ] || { echo "CHAIN_ABORT: val-best top1 이 ${#TOPS[@]} 개 (${TOPS[*]:-없음})"; exit 1; }
CKPT="${TOPS[0]}"
echo "CHAIN_CKPT $CKPT"

# GPU 해제 대기(최대 10분): 두 장 모두 2000MiB 이하
for i in $(seq 1 20); do
  mt=$(nvidia-smi -i "$GT" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  mv=$(nvidia-smi -i "$GV" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$mt" -le 2000 ] && [ "$mv" -le 2000 ] && break
  [ "$i" = "20" ] && { echo "CHAIN_ABORT: GPU $GT/$GV 가 비지 않았다 (${mt}/${mv} MiB)"; exit 1; }
  sleep 30
done

# conda 활성화 스크립트는 미정의 변수를 참조하므로 set -u 를 잠시 끈다
set +u
source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM || { echo "CHAIN_ABORT: conda MMSS_SAM 활성화 실패"; exit 1; }
set -u
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34
TS=$(date +%Y%m%d_%H%M%S)
LT=logs/${TAG}_test_${TS}.log; LV=logs/${TAG}_val_${TS}.log
echo "CHAIN_LAUNCH $(date '+%F %T') test→GPU$GT ($LT) · val→GPU$GV ($LV)"
( CUDA_VISIBLE_DEVICES=$GT python val.py --cfg "$CFG" --mode test --model_path "$CKPT" > "$LT" 2>&1; echo "CONF_TEST_DONE rc=$?" ) &
( CUDA_VISIBLE_DEVICES=$GV python val.py --cfg "$CFG" --mode val  --model_path "$CKPT" > "$LV" 2>&1; echo "CONF_VAL_DONE rc=$?" ) &
wait
echo "CHAIN_RESULT test: $(grep -a '^mIoU: ' "$LT" | tail -1)"
echo "CHAIN_RESULT val : $(grep -a '^mIoU: ' "$LV" | tail -1)"
echo "RESCORE_CONFIRM_ALL_DONE $(date '+%F %T')"
