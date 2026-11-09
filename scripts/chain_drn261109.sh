#!/usr/bin/env bash
# DRN-261109-01(RGB+LiDAR) / DRN-261109-02(RGB+Event) 체인 — user 2026-11-09 지시.
#   MUSES 2센서 교사(clean, 100ep) 완주 -> 교사 val-best top1 을 학생 config TRAIN.QAF.TEACHER_CKPT 에 채움
#   -> 같은 GPU 로 C2D2 학생(100ep) -> 학생 val-best top1 로 MUSES val 250장 전체 예측 덤프(tools/qualbank_dump.py).
# 사용: bash scripts/chain_drn261109.sh <rgbl|rgbe> <gpus 예: 0,1,2,3> [start_stage teacher|student|dump]
#   레포 루트(yeon 체크아웃)에서 tmux 창 안에서 띄운다. GPU 수는 16 의 약수(1·2·4·8)로 둘 것(eff-batch 16 = BS1 x world x accum).
#   각 학습은 last_checkpoint.pth 의 epoch 이 EPOCHS 에 닿을 때까지 AUTO_RESUME 으로 최대 2회 재기동한다(LAB STANDARD §6 재시도 상한).
set -u
TAG=${1:?rgbl|rgbe}; GPUS=${2:?gpus}; STAGE=${3:-teacher}
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1
case "$TAG" in
  rgbl) MI=il; ID=DRN-261109-01 ;;
  rgbe) MI=ie; ID=DRN-261109-02 ;;
  *) echo "unknown tag $TAG" >&2; exit 2 ;;
esac
TNAME=yeon-muses_${TAG}_P39_1_physaugoff_taps_full100_E1M
SNAME=yeon-muses_${TAG}_P39_1_physaugoff_taps_full100_A2_s2_768
TCFG=configs/${TNAME}.yaml
SCFG=configs/${SNAME}.yaml
TDIR=outputs/ReliaDINO/yeon_muses_${TAG}_P39_1_physaugoff_taps_full100_E1M/MUSES_ReliaDINO-ViTL16_${MI}
SDIR=outputs/ReliaDINO/yeon_muses_${TAG}_P39_1_physaugoff_taps_full100_A2_s2_768/MUSES_ReliaDINO-ViTL16_${MI}
QDIR=outputs/qualbank/${SNAME}_val
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
case "$NPROC" in 1|2|4|8) ;; *) echo "GPU 수 $NPROC 는 16 의 약수가 아니라 eff-batch 16 이 깨진다" >&2; exit 2 ;; esac
PORT=$((29740 + RANDOM % 200))
mkdir -p logs
CHAIN_LOG="logs/chain_${ID}_${TAG}_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[chain $ID $(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

source /home/jemo_maeng/anaconda3/etc/profile.d/conda.sh && conda activate MMSS_SAM || { echo "conda activate 실패" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1

ckpt_epoch() {  # last_checkpoint.pth 의 epoch (없으면 0)
  python - "$1" <<'EOF'
import sys, os, torch
p = sys.argv[1]
if not os.path.isfile(p):
    print(0); sys.exit()
try:
    ck = torch.load(p, map_location='cpu', mmap=True, weights_only=False)
except TypeError:
    ck = torch.load(p, map_location='cpu')
print(int(ck.get('epoch', 0)))
EOF
}
cfg_epochs() { python -c "import yaml,sys;print(int(yaml.safe_load(open(sys.argv[1]))['TRAIN']['EPOCHS']))" "$1"; }
val_top1() { ls "$1"/epoch*_top1_checkpoint.pth 2>/dev/null | head -1; }

train_until_done() {  # $1=cfg $2=save_dir(exp)
  local cfg=$1 dir=$2 name; name=$(basename "$cfg" .yaml)
  local want; want=$(cfg_epochs "$cfg")
  local attempt=0 e
  while :; do
    e=$(ckpt_epoch "$dir/last_checkpoint.pth")
    if [ "$e" -ge "$want" ]; then log "$name 완주 (epoch $e/$want)"; return 0; fi
    if [ "$attempt" -ge 3 ]; then log "🔴 $name 재기동 상한(2회) 초과, epoch $e/$want 에서 중단"; return 1; fi
    attempt=$((attempt + 1))
    mkdir -p "logs/$name"
    local L="logs/$name/${name}_$(date +%Y%m%d_%H%M%S).log"
    log "$name 기동 시도 $attempt (epoch $e/$want) GPUS=$GPUS NPROC=$NPROC PORT=$PORT LOG=$L"
    torchrun --nproc_per_node="$NPROC" --master_port="$PORT" train_reliadino.py --cfg "$cfg" 2>&1 | tee "$L"
    log "$name torchrun 종료 rc=${PIPESTATUS[0]}"
    PORT=$((PORT + 1))
    sleep 30
  done
}

if [ "$STAGE" = teacher ]; then
  log "1/3 교사 $TCFG"
  train_until_done "$TCFG" "$TDIR" || exit 1
  STAGE=student
fi

if [ "$STAGE" = student ]; then
  TBEST=$(val_top1 "$TDIR")
  [ -n "$TBEST" ] && [ -f "$TBEST" ] || { log "🔴 교사 val-best top1 없음 ($TDIR) — 학생 기동 안 함"; exit 1; }
  TBEST_ABS="$REPO/$TBEST"
  log "2/3 교사 val-best top1 = $TBEST_ABS (md5 $(md5sum "$TBEST" | cut -c1-32))"
  cp -n "$SCFG" "$SCFG.pre_teacher_fill" 2>/dev/null || true
  python - "$SCFG" "$TBEST_ABS" <<'EOF'
import re, sys, yaml
p, ck = sys.argv[1], sys.argv[2]
t = open(p, encoding='utf-8').read()
t2, n = re.subn(r"^    TEACHER_CKPT      : .*$",
                lambda m: f"    TEACHER_CKPT      : '{ck}'   # 교사 val-best top1, chain_drn261109.sh 가 교사 완주 후 채움",
                t, count=1, flags=re.M)
assert n == 1, 'TEACHER_CKPT 줄을 못 찾음'
open(p, 'w', encoding='utf-8').write(t2)
assert yaml.safe_load(open(p))['TRAIN']['QAF']['TEACHER_CKPT'] == ck
print('TEACHER_CKPT filled:', ck)
EOF
  [ $? -eq 0 ] || { log "🔴 TEACHER_CKPT 채우기 실패"; exit 1; }
  log "학생 $SCFG 기동 (로그에서 '[QAF-T] teacher ... missing=0 unexpected=0' 확인 필요)"
  train_until_done "$SCFG" "$SDIR" || exit 1
  STAGE=dump
fi

if [ "$STAGE" = dump ]; then
  SBEST=$(val_top1 "$SDIR")
  [ -n "$SBEST" ] || { log "🔴 학생 val-best top1 없음 ($SDIR)"; exit 1; }
  MIOU=$(basename "$SBEST" | sed -E 's/^epoch[0-9]+_([0-9.]+)_top1_.*/\1/')
  log "3/3 qualbank 덤프 $SBEST (학습기 val mIoU $MIOU, DDP 분할 패딩 차이 허용 tol 0.3) -> $QDIR"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python tools/qualbank_dump.py --cfg "$SCFG" --model_path "$SBEST" \
    --split val --out "$QDIR" --expected-miou "$MIOU" --tol 0.3 2>&1 | tee -a "$CHAIN_LOG"
  log "qualbank 종료 rc=${PIPESTATUS[0]}"
fi
log "체인 끝"
