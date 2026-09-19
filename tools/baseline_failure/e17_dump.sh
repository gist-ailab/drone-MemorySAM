#!/bin/bash
# E17(4탭 + 고해상도 세부 가지) 40ep 스크린 체크포인트의 test 예측 덤프.
# 세부 가지가 원거리·얇은 객체에 효과가 있었는지를 A6·셀 지도에 넣어 보려는 것이다.
# 체크포인트를 NAS -> 허브 -> hpca100 으로 옮기며 구간마다 md5 를 대조한다.
set -uo pipefail
NAS=/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/E17_detailbranch_screen40_seed20260821_20260918/epoch35_69.53_top1_checkpoint.pth
RELAY=/home/jemo/.claude/jobs/f92837ca/tmp/ckpt_relay/E17_s1_epoch35.pth
H=/home/jovyan/SSDb/jemo_maeng/d4_ours
REPO_H=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
LOCAL_REPO=/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/.claude/worktrees/eval-batch-vram-maximize

echo "$(date '+%F %T') NAS -> 허브"
rsync -t --no-perms --no-owner --no-group --partial "$NAS" "$RELAY" || exit 1
M_N=$(md5sum "$NAS" | awk '{print $1}')
M_L=$(md5sum "$RELAY" | awk '{print $1}')
[ "$M_N" = "$M_L" ] || { echo "NAS 대 허브 md5 불일치 — 중단"; exit 1; }

echo "$(date '+%F %T') 허브 -> hpca100"
ssh hpca100 "mkdir -p $H/ckpts" > /dev/null 2>&1
rsync -t --no-perms --no-owner --no-group --partial "$RELAY" "hpca100:$H/ckpts/E17_s1_epoch35.pth" || exit 1
M_H=$(ssh hpca100 "md5sum $H/ckpts/E17_s1_epoch35.pth" 2>/dev/null | awk '{print $1}')
[ "$M_L" = "$M_H" ] || { echo "허브 대 hpca100 md5 불일치 — 중단"; exit 1; }
echo "$(date '+%F %T') md5 일치 $M_L"

# 평가 config 와 덤프 도구를 올린다. E17 평가 config 는 이미 hpca100 경로를 쓴다.
scp -q "$LOCAL_REPO/configs/eval/hpca100-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E17.yaml" \
       "hpca100:$H/cfg/" || exit 1
scp -q "$LOCAL_REPO/tools/baseline_failure/dump_preds_ours.py" \
       "$LOCAL_REPO/tools/baseline_failure/common.py" \
       "hpca100:$REPO_H/tools/baseline_failure/" || exit 1

echo "$(date '+%F %T') 덤프 실행"
ssh hpca100 "bash $REPO_H/tools/baseline_failure/run_e17_dump.sh"
echo "$(date '+%F %T') 종료"
