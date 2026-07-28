#!/usr/bin/env bash
# reproduce_eval.sh — 대표 체크포인트의 정량 지표를 명령 한 줄로 재현한다.
#
#   bash scripts/reproduce_eval.sh <bench>
#   <bench> ∈ { deliver | muses | muses-official | multiaqua | det }
#
# 하는 일:
#   1) python/config/ckpt/데이터 경로가 실제로 있는지 먼저 검사하고, 없으면 한국어 에러로 즉시 종료
#   2) GPU 미지정 시 scripts/pick_free_gpus.sh 로 빈 GPU 1장을 자동 선택
#   3) 기존 평가 진입점(tools/eval_reliadino_ckpt.py · tools/eval_muses_official.py ·
#      val.py · val_det.py)을 호출 — 이 스크립트는 평가 로직을 새로 구현하지 않는다
#   4) 결과 지표(mIoU / AP50)를 stdout에 표로 요약
#
# 환경변수 오버라이드:
#   PY=<python>        기본 /home/jemo/anaconda3/envs/MMSS_SAM/bin/python
#   CFG=<yaml>         평가 config (벤치별 기본값은 아래 case 블록)
#   CKPT=<path>        체크포인트
#   DATA_ROOT=<dir>    데이터셋 루트 (config의 DATASET.ROOT를 덮어씀)
#   ANN_VAL=<json>     det 전용 — COCO val annotation
#   GPU=<idx>          예: GPU=3 (미지정 시 자동 선택)
#   OUT_DIR=<dir>      로그/산출물 위치 (기본 ./outputs/reproduce/<bench>_<timestamp>, .gitignore 대상)
#   SPLIT=val|test|both  seg 전용 (기본: deliver=both, muses=val)
#   SCORE_THRESH=<float> det 전용 (기본 0.0 = 학습 중 eval과 동일 조건)
#   DRY_RUN=1          실제 실행 없이 검사 + 실행할 명령만 출력
#
# 수치의 출처와 기대값은 REPRODUCE.md 참조.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_ROOT="${NAS_ROOT:-/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM}"

die() { printf '\n\033[31m[에러]\033[0m %s\n' "$1" >&2; shift; for l in "$@"; do printf '        %s\n' "$l" >&2; done; exit 1; }
info() { printf '\033[36m[재현]\033[0m %s\n' "$1"; }

usage() {
  cat <<'EOF'
사용법: bash scripts/reproduce_eval.sh <bench>

  <bench>
    deliver         DELIVER 4모달(img/depth/event/lidar) seg — P34-ReliaDINO val/test mIoU
    muses           MUSES 3모달(img/lidar/event) seg — P39.1-rank(seed2) val mIoU (학습 내부 프로토콜)
    muses-official  MUSES val을 공식 native 해상도(1080x1920) 프로토콜로 재채점
    multiaqua       MULTIAQUA(RGB+LiDAR+Thermal) seg — P9 val mIoU
    det             poongsan indoor 멀티모달 검출 — D1-recovered COCO AP/AP50

환경변수: PY CFG CKPT DATA_ROOT ANN_VAL GPU OUT_DIR SPLIT DRY_RUN
자세한 설명·기대 수치·데이터 준비는 REPRODUCE.md 를 보라.
EOF
}

[ $# -ge 1 ] || { usage; exit 1; }
case "${1:-}" in -h|--help|help) usage; exit 0 ;; esac

BENCH="$1"
PY="${PY:-/home/jemo/anaconda3/envs/MMSS_SAM/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

# ── 벤치별 기본값 ────────────────────────────────────────────────────────────
# 기본 ckpt 경로는 정규 웨이트 루트($NAS_ROOT/ckpts, ckpts_backup) 기준.
case "$BENCH" in
  deliver)
    CFG="${CFG:-$REPO_ROOT/configs/b200-deliver_rgbdel_P34_reliadino.yaml}"
    CKPT="${CKPT:-$NAS_ROOT/ckpts/P34_final_20260713/epoch120_68.19_top1_checkpoint.pth}"
    DATA_ROOT="${DATA_ROOT:-/ailab_mat2/dataset/DELIVER}"
    SPLIT="${SPLIT:-both}"
    ;;
  muses|muses-official)
    CFG="${CFG:-$REPO_ROOT/configs/jarvis-muses_rgbel_P39_1_rank.yaml}"
    CKPT="${CKPT:-$NAS_ROOT/ckpts_backup/jarvis/ReliaDINO/jarvis_muses_rgbel_P39_1_rank_seed2/MUSES_ReliaDINO-ViTL16_ile/epoch208_82.62_top1_checkpoint.pth}"
    DATA_ROOT="${DATA_ROOT:-/ailab_mat2/dataset/MUSES}"
    SPLIT="${SPLIT:-val}"   # MUSES test는 GT 비공개 → Codabench 제출로만 채점
    ;;
  multiaqua)
    CFG="${CFG:-$REPO_ROOT/configs/eval/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml}"
    CKPT="${CKPT:-}"
    DATA_ROOT="${DATA_ROOT:-/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2}"
    SPLIT="${SPLIT:-val}"
    ;;
  det)
    CFG="${CFG:-$REPO_ROOT/configs/det/det_D1_recovered_yeon.yaml}"
    CKPT="${CKPT:-$NAS_ROOT/ckpts/det_D1_recovered_20260723/best_checkpoint.pth}"
    DATA_ROOT="${DATA_ROOT:-/ailab_mat2/Projects/Drone/DATA/260618_poongsan}"
    ANN_VAL="${ANN_VAL:-$DATA_ROOT/final/annotations/instances_test_common.json}"
    SPLIT="${SPLIT:-val}"
    ;;
  *)
    usage
    die "알 수 없는 벤치마크: '$BENCH'" "지원: deliver | muses | muses-official | multiaqua | det"
    ;;
esac

# ── 사전 검사 (여기서 막아야 GPU를 잡고 죽는 일이 없다) ──────────────────────
[ -x "$PY" ] || die "python 실행 파일이 없다: $PY" \
  "conda env 'MMSS_SAM'을 만들거나, PY=<python 경로> 로 지정하라." \
  "예: PY=\$(which python) bash scripts/reproduce_eval.sh $BENCH"

"$PY" -c 'import torch, yaml, numpy' 2>/dev/null || die \
  "python 환경에 torch/pyyaml/numpy가 없다: $PY" \
  "REPRODUCE.md '전제조건' 절의 패키지 표를 보고 환경을 갖춰라."

[ -f "$CFG" ] || die "config 파일이 없다: $CFG" "CFG=<yaml> 로 다른 config를 지정할 수 있다."

if [ -z "${CKPT:-}" ]; then
  die "이 벤치($BENCH)의 기본 체크포인트가 정의돼 있지 않다." \
      "CKPT=<체크포인트 경로> 를 직접 지정하라." \
      "예: CKPT=/path/to/epoch131_94.41_top1_checkpoint.pth bash scripts/reproduce_eval.sh $BENCH"
fi
[ -f "$CKPT" ] || die "체크포인트가 없다: $CKPT" \
  "정규 웨이트 루트: $NAS_ROOT/ckpts (+ ckpts_backup)" \
  "연구실 NAS가 마운트돼 있지 않으면 접근할 수 없다 — REPRODUCE.md '체크포인트' 절 참조." \
  "CKPT=<경로> 로 로컬 사본을 지정할 수 있다."

[ -d "$DATA_ROOT" ] || die "데이터셋 루트가 없다: $DATA_ROOT" \
  "DATA_ROOT=<데이터셋 경로> 로 지정하라 (레이아웃은 REPRODUCE.md '데이터셋' 절)."

if [ "$BENCH" = "det" ]; then
  [ -f "$ANN_VAL" ] || die "det COCO annotation이 없다: $ANN_VAL" "ANN_VAL=<json 경로> 로 지정하라."
fi

# ── GPU 선택 (빈 GPU만 — 기존 헬퍼 재사용) ──────────────────────────────────
if [ -z "${GPU:-}" ]; then
  GPU="$(bash "$REPO_ROOT/scripts/pick_free_gpus.sh" 1 2>/dev/null || true)"
  [ -n "$GPU" ] || die "빈 GPU가 없다 (기준: memory.used<=2000MiB & util<=10%)." \
    "다른 실험이 끝나길 기다리거나, GPU=<idx> 로 직접 지정하라." \
    "확인: nvidia-smi"
  info "빈 GPU 자동 선택: $GPU"
else
  info "GPU 지정됨: $GPU"
fi
# 헬퍼가 nvidia-smi 에러 문자열을 stdout으로 흘리는 경우가 있다(NVML driver/library
# version mismatch 등). 그대로 CUDA_VISIBLE_DEVICES 에 넣으면 GPU0 에 붙어 남의 학습을
# 밟을 수 있으므로, 숫자 목록이 아니면 여기서 막는다.
case "$GPU" in
  *[!0-9,]*|""|,*|*,)
    die "GPU 지정값이 유효한 인덱스 목록이 아니다: '$GPU'" \
        "nvidia-smi 가 정상 동작하는지 먼저 확인하라 (NVML driver/library mismatch면 관리자 리부트 필요)." \
        "임시 회피: GPU=<idx> 로 직접 지정." ;;
esac

STAMP="$("$PY" -c 'import time;print(time.strftime("%Y%m%d_%H%M%S"))')"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/reproduce/${BENCH}_${STAMP}}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/eval.log"

info "bench     : $BENCH"
info "config    : $CFG"
info "checkpoint: $CKPT"
info "data root : $DATA_ROOT"
info "out dir   : $OUT_DIR"

# config의 데이터 경로만 로컬 값으로 덮어쓴 임시 config 생성 (원본 configs/ 는 건드리지 않음).
# val.py / val_det.py 는 --dataset-root 인자가 없어서 이 방식이 필요하다.
make_local_cfg() {  # $1 = 출력 경로
  OUT="$1" SRC="$CFG" ROOT="$DATA_ROOT" ANN="${ANN_VAL:-}" BENCH="$BENCH" "$PY" - <<'PYEOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ['SRC']))
ds = cfg.setdefault('DATASET', {})
ds['ROOT'] = os.environ['ROOT']
if isinstance(ds.get('PHYSAUG'), dict):
    ds['PHYSAUG']['ENABLE'] = False          # 학습용 증강은 평가에서 끈다
if os.environ.get('ANN'):
    ds['ANNOTATION_VAL'] = os.environ['ANN']
mdl = cfg.setdefault('MODEL', {})
mdl['RESUME_ENABLE'] = False
mdl['RESUME_PATH'] = ''
mdl['AUTO_RESUME'] = False
cfg.setdefault('TEST', {})['FILE'] = os.environ['ROOT']
yaml.safe_dump(cfg, open(os.environ['OUT'], 'w'), sort_keys=False, allow_unicode=True)
print(f"[재현] 임시 config 생성: {os.environ['OUT']}")
PYEOF
}

run() {  # 명령을 로그에 남기며 실행
  printf '\033[36m[재현]\033[0m 실행: %s\n' "$*"
  if [ "$DRY_RUN" = "1" ]; then
    printf '\033[33m[DRY_RUN]\033[0m 실제 실행은 건너뛴다.\n'
    return 0
  fi
  ( cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" \
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$@" ) 2>&1 | tee "$LOG"
}

# ── 벤치별 실행 ──────────────────────────────────────────────────────────────
case "$BENCH" in
  deliver|muses)
    # ReliaDINO 계열(P34~P44): 학습 시 eval 경로를 그대로 재사용하는 정규 평가기.
    run "$PY" tools/eval_reliadino_ckpt.py \
        --cfg "$CFG" --ckpt "$CKPT" --split "$SPLIT" \
        --gpu 0 --dataset-root "$DATA_ROOT"
    ;;
  muses-official)
    # MUSES 리더보드와 같은 native 1080x1920 해상도로 재채점.
    run "$PY" tools/eval_muses_official.py \
        --cfg "$CFG" --ckpt "$CKPT" --gpu 0 \
        --dataset-root "$DATA_ROOT" --out "$OUT_DIR/muses_official.json"
    ;;
  multiaqua)
    LOCAL_CFG="$OUT_DIR/cfg_local.yaml"; make_local_cfg "$LOCAL_CFG"
    run "$PY" val.py --cfg "$LOCAL_CFG" --mode "$SPLIT" \
        --model_path "$CKPT" --save_dir "$OUT_DIR/pred"
    ;;
  det)
    LOCAL_CFG="$OUT_DIR/cfg_local.yaml"; make_local_cfg "$LOCAL_CFG"
    # score_thresh 0.0 = 학습 중 eval(train_det.py evaluate())과 동일 조건.
    # val_det.py 기본값 0.3은 낮은-score 박스를 버려 COCO AP를 낮춘다 → 기록 수치와 어긋난다.
    run "$PY" val_det.py --cfg "$LOCAL_CFG" --det_checkpoint "$CKPT" \
        --mode "$SPLIT" --score_thresh "${SCORE_THRESH:-0.0}" --save_dir "$OUT_DIR/det"
    ;;
esac

if [ "$DRY_RUN" = "1" ]; then
  printf '\n\033[33m[DRY_RUN]\033[0m 사전 검사 통과. 실제로 돌리려면 DRY_RUN 없이 다시 실행하라.\n'
  exit 0
fi

# ── 결과 요약 ────────────────────────────────────────────────────────────────
echo
echo "==================== 재현 결과 요약 ($BENCH) ===================="
if [ "$BENCH" = "det" ]; then
  MET="$OUT_DIR/det/metrics.json"
  if [ -f "$MET" ]; then
    MET="$MET" "$PY" - <<'PYEOF'
import json, os
m = json.load(open(os.environ['MET']))
rows = [(k, m[k]) for k in ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large') if k in m]
w = max(len(k) for k, _ in rows)
print(f"{'metric'.ljust(w)} | value")
print('-' * (w + 9))
for k, v in rows:
    print(f"{k.ljust(w)} | {v:.4f}")
PYEOF
  else
    echo "⚠️ metrics.json이 없다 ($MET) — 로그를 확인하라: $LOG"
  fi
else
  # eval_reliadino_ckpt: "[G0a][val] n=2005  mIoU: 68.1900 ..."  /  val.py: "mIoU: 93.29  mAcc: ..."
  if grep -qE 'mIoU:' "$LOG"; then
    printf '%-10s | %s\n' "split" "mIoU"
    printf -- '-----------+---------\n'
    grep -E 'mIoU:' "$LOG" | while IFS= read -r line; do
      split=$(printf '%s' "$line" | grep -oE '\[(val|test)\]' | tr -d '[]' | head -1)
      miou=$(printf '%s' "$line" | sed -E 's/.*mIoU:[[:space:]]*([0-9.]+).*/\1/')
      printf '%-10s | %s\n' "${split:-$SPLIT}" "$miou"
    done
  elif [ -f "$OUT_DIR/muses_official.json" ]; then
    OUTJSON="$OUT_DIR/muses_official.json" "$PY" - <<'PYEOF'
import json, os
d = json.load(open(os.environ['OUTJSON']))
for k, v in d.items():
    if isinstance(v, (int, float)):
        print(f"{k:24s} | {v:.4f}")
PYEOF
  else
    echo "⚠️ 지표를 찾지 못했다 — 전체 로그를 확인하라: $LOG"
  fi
fi
echo "================================================================"
echo "로그: $LOG"
echo "기대 수치와 그 출처는 REPRODUCE.md '재현 명령' 표를 보라."
