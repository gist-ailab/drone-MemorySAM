#!/bin/bash
# 서버 전용 실행 기록(jarvis, 2026-10-08)
# 우리 모델(E1 확정 시드1, P46 C3-only) 모달 제거 측정 — 기준선과 같은 축(정규화 후 0).
# 우리 파이프라인은 데이터셋 변환에서 정규화를 마친 뒤 텐서를 넘기므로, 입력 텐서를 0 으로
# 두는 것이 곧 "정규화 후 0" 이다(기준선의 normalized 개입과 같은 축).
# 빈 GPU 1 장을 기다렸다가, 짧은 표본으로 들어가는 최대 배치를 찾고 전량을 돌린다.
set -uo pipefail
REPO=/SSDb/jemo_maeng/src/drone-MemorySAM
CFG=configs/eval/jarvis-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1_confirm200.yaml
CKPT=$REPO/outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200/DELIVER_ReliaDINO-ViTL16_idel/epoch140_68.9_top1_checkpoint.pth
OUT=$REPO/modal_zero_ours
mkdir -p "$OUT"
cd "$REPO"

source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate MMSS_SAM
# jarvis 에서 ReliaDINO 백본(DINOv3)은 별도 경로의 timm 1.0.24 가 있어야 만들어진다
# (기본 환경의 timm 0.4.12 는 vit_large_patch16_dinov3 를 모른다). servers.conf jarvis 행 참조.
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$REPO/semseg/models/sam2:$REPO
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

free_gpu() {  # GPU0 은 사용자 예약이라 제외
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$1 != 0 && $2 <= 2000 && $3 <= 10 {print $1; exit}'
}

echo "$(date '+%F %T') 빈 GPU 대기" | tee -a "$OUT/runner.log"
G=$(free_gpu)
while [ -z "$G" ]; do sleep 60; G=$(free_gpu); done
echo "$(date '+%F %T') GPU $G 확보" | tee -a "$OUT/runner.log"

# 들어가는 최대 배치 탐색 — 표본 16 장으로 시도한다.
BATCH=1
for B in 8 6 4 2 1; do
  if CUDA_VISIBLE_DEVICES=$G python tools/baseline_failure/modality_zero_ablation.py \
       --cfg "$CFG" --model_path "$CKPT" --split test --batch "$B" --limit 16 \
       --out "$OUT/probe_batch_${B}.json" > "$OUT/probe_batch_${B}.log" 2>&1; then
    BATCH=$B
    echo "$(date '+%F %T') 배치 $B 통과 — 이 값으로 전량 실행" | tee -a "$OUT/runner.log"
    break
  fi
  echo "$(date '+%F %T') 배치 $B 실패 — 마지막 오류: $(grep -E 'Error|error:' "$OUT/probe_batch_${B}.log" | tail -1)" | tee -a "$OUT/runner.log"
done

echo "$(date '+%F %T') 전량 실행 시작 (batch=$BATCH, GPU $G)" | tee -a "$OUT/runner.log"
CUDA_VISIBLE_DEVICES=$G python tools/baseline_failure/modality_zero_ablation.py \
  --cfg "$CFG" --model_path "$CKPT" --split test --batch "$BATCH" \
  --out "$OUT/modal_zero_ours_E1_seed1.json" > "$OUT/full_run.log" 2>&1
echo "$(date '+%F %T') 종료 exit=$?" | tee -a "$OUT/runner.log"
tail -25 "$OUT/full_run.log"
