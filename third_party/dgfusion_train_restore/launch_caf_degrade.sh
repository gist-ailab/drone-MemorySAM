#!/bin/bash
# CAFuser (b) 재학습 기동 — DGFusion (b) 와 같은 이유·같은 두 변경(모달 드롭 0.2 + 열화
# 커리큘럼). DGFusion (b) 가 완주하고 val-best test 결과까지 나온 뒤 판정 세션이 승인했다
# (2026-09-23). depth 보조 헤드가 없는 CAFuser 는 D4 판독의 대조군 역할이다.
#
# 빈 GPU 4 장을 기다린다(누적 없음, 기존 결정대로). GPU7 은 감시 세션의 R2 재채점,
# GPU4 는 Q3 완주 뒤 Q2/Q3 재채점 예정이라 겹치면 그쪽과 직접 조율한다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/dgfusion_train
CFG=configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade.yaml
NEED=${NEED:-4}
TS=$(date +%Y%m%d_%H%M%S)
LOG=$REPO/logs/cafuser_deliver_degrade_${TS}.log
mkdir -p "$REPO/logs"
cd "$REPO"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$1 != 0 && $2 <= 2000 && $3 <= 10 {print $1}'
}

# 🔴 2026-09-23 재기동에서 겪은 실패의 교훈: 스냅샷 한 번만 보고 "비었다"고 판단하면,
# 다른 사용자의 간헐적 워크로드(추론 워커가 쉬다가 다시 GPU 를 채우는 패턴)를 빈 GPU로
# 오판해 기동 직후 OOM 이 난다(GPU3 에서 7 iteration 만에 실측). 그래서 후보 GPU 집합의
# 교집합이 STREAK 회 연속으로 유지될 때만 확보로 본다 — run_three_tables.sh 의 메모리
# 3연속 확인과 같은 발상이다.
STREAK=${STREAK:-3}
CANDIDATE=()
ok=0
echo "$(date '+%F %T') 빈 GPU ${NEED}장이 ${STREAK}회 연속 유지될 때까지 대기"
for i in $(seq 1 480); do
  CUR=($(free_gpus))
  if [ "${#CUR[@]}" -ge "$NEED" ]; then
    if [ "$ok" -eq 0 ]; then
      CANDIDATE=("${CUR[@]}")
    else
      INTER=($(comm -12 <(printf '%s\n' "${CANDIDATE[@]}" | sort) <(printf '%s\n' "${CUR[@]}" | sort)))
      CANDIDATE=("${INTER[@]}")
    fi
    if [ "${#CANDIDATE[@]}" -ge "$NEED" ]; then
      ok=$((ok + 1))
      echo "$(date '+%F %T') 후보 ${CANDIDATE[*]} 연속 ${ok}/${STREAK}"
    else
      ok=0
      CANDIDATE=("${CUR[@]}")
    fi
  else
    ok=0
    CANDIDATE=()
  fi
  [ "$ok" -ge "$STREAK" ] && break
  sleep 60
done
if [ "$ok" -lt "$STREAK" ]; then
  echo "$(date '+%F %T') ${NEED}장을 ${STREAK}회 연속 못 얻었다(현재 후보: ${CANDIDATE[*]:-없음}) — 기동하지 않는다"
  exit 1
fi
G=("${CANDIDATE[@]}")
SEL=$(IFS=,; echo "${G[*]:0:$NEED}")
echo "$(date '+%F %T') GPU $SEL 확보(${STREAK}회 연속 확인) — 기동"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer
export DETECTRON2_DATASETS=$PWD/datasets
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=$SEL nohup python train_net.py \
  --config-file "$CFG" --num-gpus "$NEED" \
  OUTPUT_DIR output/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade \
  > "$LOG" 2>&1 &
echo "$(date '+%F %T') 기동 pid=$! log=$LOG"
