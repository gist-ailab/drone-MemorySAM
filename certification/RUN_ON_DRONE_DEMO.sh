#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#   공인인증 정량평가 — drone-demo 실행 스크립트 (원커맨드)
#
#     bash certification/RUN_ON_DRONE_DEMO.sh
#
#   경로·환경이 이미 박혀 있어 인자 없이 실행하면 된다.
#   바꿔야 할 때만 인자로 준다:
#     bash certification/RUN_ON_DRONE_DEMO.sh <DATA_ROOT> <GPU>
#
#   측정 항목 / 반복 횟수
#     ① 표적위치 정확도            mAP50           5회
#     ② 영상 열화시 표적인식 성공률  mAP50(야간)      5회
#     ③ 피아식별 정확도            분류 accuracy    2회 (가중치 고정, 평가만)
#
#   산출물: runs/cert_all_<타임스탬프>/
#     summary.txt        한눈 요약표 (여기만 봐도 됨)
#     console.log        전 과정 콘솔 로그
#     det_trial1..5/     회차별 cert_report.json · inference_log.txt · viz/*.png
#     iff/               iff_report.json · viz/(정답 파랑 / 오답 빨강)
#
#   소요: 약 20분 (검출 3.5분 x 5회 + 피아식별 수초)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$HOME/drone-MemorySAM-cert"
DATA_ROOT="${1:-$HOME/poongsan_v2}"
GPU="${2:-0}"
CKPT="$REPO/weights/det_D1_vitsp_20260723/best_checkpoint.pth"

# 인증용 conda 환경 (torch cu128, RTX 5090)
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate drone_cert
export PYTHON="$HOME/anaconda3/envs/drone_cert/bin/python"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$REPO"

# ---- 실행 전 점검: 없으면 여기서 멈추는 게 낫다 ----
[ -f "$CKPT" ]                || { echo "✗ 검출 체크포인트 없음: $CKPT"; exit 1; }
[ -d "$DATA_ROOT/_final_ann" ] || { echo "✗ 데이터 루트가 아님(_final_ann 없음): $DATA_ROOT"; exit 1; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.used --format=csv,noheader \
  || { echo "✗ GPU $GPU 를 못 찾음"; exit 1; }

echo "────────────────────────────────────────────────"
echo " 공인인증 정량평가 시작"
echo "   모델      : D1 ViT-S+ (ReliaDINO 3-modal + RF-DETR)"
echo "   데이터    : $DATA_ROOT"
echo "   GPU       : $GPU"
echo "   반복      : 표적인식 5회 / 영상열화 5회 / 피아식별 2회"
echo "────────────────────────────────────────────────"

bash certification/run_cert_all.sh "$CKPT" "$DATA_ROOT" "$GPU"
