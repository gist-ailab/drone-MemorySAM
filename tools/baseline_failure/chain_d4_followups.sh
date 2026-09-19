#!/bin/bash
# 앞선 연쇄(시드 확장 → 시드1 이미지별 → CAFuser 이미지별)가 끝나면 RMM r=0.5 측정 6회를
# 이어서 돌린다. GPU 를 겹쳐 잡지 않도록 순서를 준다.
#
# 종료 판정은 프로세스 이름이 아니라 각 러너가 로그에 남기는 "전체 완료" 표식으로 한다.
# pgrep 로 판정하면 원격에 보내는 명령줄 자체에 그 이름이 들어 있어 자기 자신을 세고,
# 영원히 끝나지 않는다(2026-09-19 실제로 4시간을 헛되이 기다렸다).
set -uo pipefail

wait_done() {  # wait_done <원격 로그 경로> <설명>
  local log=$1 what=$2 n=0
  while ! ssh jarvis "grep -q '전체 완료' '$log'" 2>/dev/null; do
    sleep 120
    n=$((n+1))
    if [ $n -gt 240 ]; then echo "$(date '+%F %T') 대기 한도 초과: $what"; return 1; fi
  done
  echo "$(date '+%F %T') 종료 확인: $what"
}

R=/SSDb/jemo_maeng/src/drone-MemorySAM
D=/SSDb/jemo_maeng/dgfusion_train

echo "$(date '+%F %T') 앞선 연쇄 종료 대기"
wait_done "$R/modal_zero_ours_seeds/runner.log" "시드 확장 15회" || exit 1

echo "$(date '+%F %T') 우리 모델 시드1 이미지별 측정 시작"
ssh jarvis "bash $R/tools/baseline_failure/run_ours_seed1_perimage.sh"

echo "$(date '+%F %T') CAFuser 이미지별 측정 시작"
ssh jarvis "bash $D/tools/baseline_failure/run_probe_cafuser_perimage.sh"

echo "$(date '+%F %T') 기준선 둘 RMM r=0.5 (4회)"
ssh jarvis "RATIO=0.5 SEED=0 BATCH=8 bash $D/tools/baseline_failure/run_baselines_rmm.sh"

echo "$(date '+%F %T') 우리 모델 RMM r=0.5 (2회)"
ssh jarvis "RATIO=0.5 SEED=0 BATCH=8 bash $R/tools/baseline_failure/run_ours_rmm.sh"
echo "$(date '+%F %T') 전체 연쇄 종료"
