#!/usr/bin/env bash
# run_eval_autobatch.sh — 기준선(detectron2 --eval-only) 평가를 빈 GPU 1장에서
# 가능한 큰 배치로 돌린다(A10, user 상시 지시 "모든 실험은 VRAM 을 최대한 채워서 쓴다").
#
# 사전 조건: 기준선 저장소에 d2_eval_batch.patch 가 적용돼 있어야 한다
# (build_test_loader 가 BF_EVAL_BATCH 를 build_detection_test_loader 의 batch_size 로 넘김).
# conda env(dgfusion)·PYTHONPATH 등 실행 환경은 호출자가 미리 활성화한다.
#
# 사용법:
#   run_eval_autobatch.sh --repo <기준선 저장소 경로> --cfg <config yaml> \
#       --weights <ckpt.pth> --out <출력 디렉터리> --log <로그 경로> \
#       [--expect_miou <값>] [--dry_run] [KEY VALUE ...]
#     - KEY VALUE 쌍은 detectron2 명령 끝에 그대로 전달되는 오버라이다
#       (예: MODEL.TEST.DEPTH_ON False DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)").
#     - --expect_miou: 실행 후 로그의 mIoU 와 ±0.01 이내인지 검사(밖이면 종료 코드 3).
#     - --dry_run: GPU·모델 없이 후보 목록과 최종 명령만 출력하고 종료.
#
# 배치 결정: GPU 총 메모리(nvidia-smi --query-gpu=memory.total)로 급을 나눠 큰 값부터
# 시도한다. 24GB 급(≥22528MiB)이면 "16 12 8 6 4 2 1". 실행 중 CUDA out of memory 면
# 다음 후보로 내려 재시도하고, 성공하면 그 배치(eval_batch)와 실행 중 관측한 최대
# 사용량(peak_mem_MiB)을 로그와 요약(<out>/autobatch_summary.json)에 남긴다.
# OOM 이외의 에러는 재시도 없이 그대로 실패한다(원문은 로그에 유지).
#
# 종료 코드: 0 성공 · 1 사용법/환경 에러 · 2 전 후보 OOM · 3 --expect_miou ±0.01 밖 ·
#            그 외 = 평가 프로세스의 원본 종료 코드.
#
# 테스트 주입: BF_AUTOBATCH_TOTAL_MIB 를 주면 nvidia-smi 대신 그 값(MiB)으로 후보를
# 계산한다(스모크 --dry_run 검증용). 빈 GPU 판정 한계는 GPU_MAXMEM/GPU_MAXUTIL 환경변수로.
set -uo pipefail

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

REPO="" CFG="" WEIGHTS="" OUT="" LOG="" EXPECT_MIOU="" DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo)        REPO="$2"; shift 2 ;;
        --cfg)         CFG="$2"; shift 2 ;;
        --weights)     WEIGHTS="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        --log)         LOG="$2"; shift 2 ;;
        --expect_miou) EXPECT_MIOU="$2"; shift 2 ;;
        --dry_run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) break ;;   # 이후는 detectron2 KEY VALUE 오버라이드
    esac
done
OVERRIDES=("$@")

die_usage() { echo "ERROR: $*" >&2; usage >&2; exit 1; }

[ -n "$REPO" ]     || die_usage "--repo 가 필요하다"
[ -n "$CFG" ]      || die_usage "--cfg 가 필요하다"
[ -n "$WEIGHTS" ]  || die_usage "--weights 가 필요하다"
[ -n "$OUT" ]      || die_usage "--out 이 필요하다"
[ -n "$LOG" ]      || die_usage "--log 이 필요하다"
if [ $(( ${#OVERRIDES[@]} % 2 )) -ne 0 ]; then
    die_usage "오버라이드는 KEY VALUE 쌍이어야 한다(홀수 개가 들어왔다): ${OVERRIDES[*]}"
fi
if [ -n "$EXPECT_MIOU" ]; then
    awk -v v="$EXPECT_MIOU" 'BEGIN{ if (v !~ /^-?[0-9]+(\.[0-9]+)?$/) exit 1 }' \
        || die_usage "--expect_miou 는 숫자여야 한다: $EXPECT_MIOU"
fi

GPU_MAXMEM="${GPU_MAXMEM:-2000}"   # 빈 GPU 판정: memory.used ≤ 이 값(MiB)
GPU_MAXUTIL="${GPU_MAXUTIL:-10}"   #                  && util ≤ 이 값(%)

# GPU 총 메모리(MiB) 급 → 배치 후보(큰 값부터).
candidates_for_total_mib() {
    local t="$1"
    if   [ "$t" -ge 73728 ]; then echo "64 48 32 24 16 12 8 6 4 2 1"   # 80GB 급(A100 등)
    elif [ "$t" -ge 45056 ]; then echo "32 24 16 12 8 6 4 2 1"         # 48GB 급(A6000 등)
    elif [ "$t" -ge 22528 ]; then echo "16 12 8 6 4 2 1"               # 24GB 급(3090/4090 등)
    elif [ "$t" -ge 10240 ]; then echo "8 6 4 2 1"                     # 11GB 급
    else                          echo "4 2 1"                         # 8GB 이하
    fi
}

# 총 메모리 확보: 주입값 > nvidia-smi. 없으면 에러(추측 금지).
TOTAL_MIB="${BF_AUTOBATCH_TOTAL_MIB:-}"
TOTAL_SRC="BF_AUTOBATCH_TOTAL_MIB(주입)"
if [ -z "$TOTAL_MIB" ]; then
    TOTAL_SRC="nvidia-smi"
fi

CMD=(python train_net.py --config-file "$CFG" --eval-only
     MODEL.WEIGHTS "$WEIGHTS" OUTPUT_DIR "$OUT")
if [ "${#OVERRIDES[@]}" -gt 0 ]; then
    CMD+=("${OVERRIDES[@]}")
fi
CMD_STR="BF_EVAL_BATCH=<B> CUDA_VISIBLE_DEVICES=<GPU> ${CMD[*]}"

if [ "$DRY_RUN" -eq 1 ]; then
    # GPU·모델 없이 후보 목록과 최종 명령만 출력.
    if [ -z "$TOTAL_MIB" ]; then
        TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                    | head -1 | tr -d ' ') || TOTAL_MIB=""
        if [ -z "$TOTAL_MIB" ]; then
            echo "ERROR: GPU 총 메모리를 확인할 수 없다(nvidia-smi 실패). 스모크 등 GPU 없는 환경에서는 BF_AUTOBATCH_TOTAL_MIB=<MiB> 를 주라." >&2
            exit 1
        fi
    fi
    CANDS=$(candidates_for_total_mib "$TOTAL_MIB")
    FIRST_B=$(echo "$CANDS" | cut -d' ' -f1)
    GPU_SHOW="${CUDA_VISIBLE_DEVICES:-<자동선택>}"
    echo "[dry-run] total_mem_MiB=$TOTAL_MIB (source=$TOTAL_SRC)"
    echo "[dry-run] candidates: $CANDS"
    echo "[dry-run] final command:"
    echo "cd $REPO && BF_EVAL_BATCH=$FIRST_B CUDA_VISIBLE_DEVICES=$GPU_SHOW ${CMD[*]}"
    echo "[dry-run] OOM 시 candidates 순서로 내려가며 재시도한다."
    exit 0
fi

# ---- 실행 모드: 빈 GPU 1장 확보 ----
GPU="" GPU_SRC=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    case "$CUDA_VISIBLE_DEVICES" in
        *,*) echo "ERROR: 이 실행기는 빈 GPU 1장 기준이다 — CUDA_VISIBLE_DEVICES 가 2개 이상이다: $CUDA_VISIBLE_DEVICES" >&2; exit 1 ;;
    esac
    GPU="$CUDA_VISIBLE_DEVICES"
    GPU_SRC="CUDA_VISIBLE_DEVICES(사용자 지정 존중)"
else
    GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
          | awk -F', *' -v mm="$GPU_MAXMEM" -v mu="$GPU_MAXUTIL" \
                '$2+0<=mm && $3+0<=mu {print $1; exit}') || GPU=""
    if [ -z "$GPU" ]; then
        echo "ERROR: 빈 GPU(memory.used≤${GPU_MAXMEM}MiB && util≤${GPU_MAXUTIL}%)가 없다." >&2
        exit 1
    fi
    GPU_SRC="자동 선택(빈 GPU 규칙)"
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

if [ -z "$TOTAL_MIB" ]; then
    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [ -z "$TOTAL_MIB" ]; then
        echo "ERROR: GPU $GPU 의 총 메모리를 읽지 못했다(nvidia-smi 실패)." >&2
        exit 1
    fi
fi
[ "$TOTAL_MIB" -ge 1 ] 2>/dev/null || { echo "ERROR: total_mem_MiB 가 정수가 아니다: $TOTAL_MIB" >&2; exit 1; }

CANDS=$(candidates_for_total_mib "$TOTAL_MIB")
mkdir -p "$OUT"
mkdir -p "$(dirname "$LOG")"
: >> "$LOG"

echo "[autobatch] 시작 $(date -Iseconds) gpu=$GPU($GPU_SRC) total_mem_MiB=$TOTAL_MIB($TOTAL_SRC) 후보: $CANDS" >> "$LOG"
echo "[autobatch] 명령 템플릿: cd $REPO && $CMD_STR" >> "$LOG"

# ---- 후보를 큰 값부터 시도 ----
OOMED=""
for B in $CANDS; do
    ATTEMPT=$(mktemp /tmp/autobatch_attempt.XXXXXX.log)
    PEAK_FILE=$(mktemp /tmp/autobatch_peak.XXXXXX)
    echo 0 > "$PEAK_FILE"
    echo "[autobatch] $(date -Iseconds) 시도 eval_batch=$B" >> "$LOG"
    ( cd "$REPO" && BF_EVAL_BATCH="$B" CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" ) \
        > "$ATTEMPT" 2>&1 &
    RUN_PID=$!
    # 실행 중 최대 사용량 관측(2초 간격 폴링, 실패 시 무시).
    ( while kill -0 "$RUN_PID" 2>/dev/null; do
          m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
          cur=$(cat "$PEAK_FILE" 2>/dev/null || echo 0)
          if [ -n "$m" ] && [ "$m" -gt "$cur" ] 2>/dev/null; then echo "$m" > "$PEAK_FILE"; fi
          sleep 2
      done ) &
    MON_PID=$!
    RC=0
    wait "$RUN_PID" || RC=$?
    wait "$MON_PID" 2>/dev/null
    cat "$ATTEMPT" >> "$LOG"
    PEAK=$(cat "$PEAK_FILE")

    if [ "$RC" -eq 0 ]; then
        echo "[autobatch] 성공 eval_batch=$B peak_mem_MiB=$PEAK (총 ${TOTAL_MIB}MiB 중 $(awk -v p="$PEAK" -v t="$TOTAL_MIB" 'BEGIN{printf "%.0f", p/t*100}')%)" >> "$LOG"
        MIOU_LINE=""
        DELTA_NOTE=""
        if [ -n "$EXPECT_MIOU" ]; then
            # 로그의 mIoU: 마지막 등장값(최종 요약). 못 찾으면 검증 불가 에러.
            MIOU=$(grep -oiE "mIoU[^0-9A-Za-z]*[0-9]+(\.[0-9]+)?" "$ATTEMPT" \
                   | grep -oE "[0-9]+(\.[0-9]+)?" | tail -1)
            if [ -z "$MIOU" ]; then
                echo "ERROR: --expect_miou 를 줬지만 로그에서 mIoU 를 찾지 못했다 — 로그 포맷을 확인하고 수동 대조하라: $ATTEMPT 원문은 $LOG 에 있다." >&2
                rm -f "$ATTEMPT" "$PEAK_FILE"
                exit 1
            fi
            if awk -v e="$EXPECT_MIOU" -v m="$MIOU" 'BEGIN{d=e-m; if(d<0)d=-d; exit (d>0.01)}'; then
                MIOU_LINE=", \"expect_miou\": $EXPECT_MIOU, \"log_miou\": $MIOU, \"miou_delta\": $(awk -v e="$EXPECT_MIOU" -v m="$MIOU" 'BEGIN{printf "%.4f", (e-m)}')"
                DELTA_NOTE=" mIoU=$MIOU (기대 $EXPECT_MIOU, ±0.01 이내)"
            else
                echo "[autobatch] 실패 mIoU=$MIOU 가 기대 $EXPECT_MIOU 과 ±0.01 밖이다 — 종료 코드 3" >> "$LOG"
                echo "ERROR: mIoU=$MIOU 가 --expect_miou $EXPECT_MIOU 과 ±0.01 밖이다(배치 확대 등가성 위반 — 배치 1 로 되돌려 재검하라)." >&2
                printf '{\n  "timestamp": "%s",\n  "eval_batch": %s,\n  "peak_mem_MiB": %s,\n  "total_mem_MiB": %s,\n  "gpu": "%s",\n  "gpu_source": "%s",\n  "candidates": "%s",\n  "oom_batches": "%s",\n  "command": "cd %s && %s",\n  "exit_code": 3%s\n}\n' \
                    "$(date -Iseconds)" "$B" "$PEAK" "$TOTAL_MIB" "$GPU" "$GPU_SRC" "$CANDS" "$OOMED" \
                    "$REPO" "BF_EVAL_BATCH=$B CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}" \
                    ", \"expect_miou\": $EXPECT_MIOU, \"log_miou\": $MIOU" > "$OUT/autobatch_summary.json"
                rm -f "$ATTEMPT" "$PEAK_FILE"
                exit 3
            fi
        fi
        printf '{\n  "timestamp": "%s",\n  "eval_batch": %s,\n  "peak_mem_MiB": %s,\n  "total_mem_MiB": %s,\n  "gpu": "%s",\n  "gpu_source": "%s",\n  "candidates": "%s",\n  "oom_batches": "%s",\n  "command": "cd %s && %s",\n  "exit_code": 0%s\n}\n' \
            "$(date -Iseconds)" "$B" "$PEAK" "$TOTAL_MIB" "$GPU" "$GPU_SRC" "$CANDS" "$OOMED" \
            "$REPO" "BF_EVAL_BATCH=$B CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}" \
            "$MIOU_LINE" > "$OUT/autobatch_summary.json"
        echo "[autobatch] 요약 저장: $OUT/autobatch_summary.json${DELTA_NOTE}"
        rm -f "$ATTEMPT" "$PEAK_FILE"
        exit 0
    fi

    if grep -qiE "CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY" "$ATTEMPT"; then
        OOMED="$OOMED $B"
        echo "[autobatch] eval_batch=$B 에서 CUDA out of memory — 다음 후보로 내려간다" >> "$LOG"
        rm -f "$ATTEMPT" "$PEAK_FILE"
        continue
    fi
    # OOM 이외의 에러 — 재시도 없이 실패(원문은 이미 $LOG 에 있다).
    echo "[autobatch] OOM 이외의 실패 rc=$RC — 재시도 없이 종료한다(원문은 위 로그)" >> "$LOG"
    rm -f "$ATTEMPT" "$PEAK_FILE"
    exit "$RC"
done

echo "[autobatch] 모든 후보($CANDS)에서 OOM — 배치 1 로도 안 되는 환경 문제일 수 있다" >> "$LOG"
echo "ERROR: 모든 후보에서 CUDA out of memory — 로그를 확인하라: $LOG" >&2
exit 2
