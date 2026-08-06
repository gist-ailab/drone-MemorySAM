#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  공인인증 정량목표 통합 러너 — 3개 항목을 반복 횟수대로 실행하고 한눈에 요약
#
#   ① 표적위치 정확도            mAP50            반복 5회   [D1 ViT-S+ 검출]
#   ② 영상 열화시 표적인식 성공률  mAP50 (야간)      반복 5회   [①과 동일 실행에서 분해]
#   ③ 피아식별 정확도            분류 accuracy     반복 2회   [MobileNetV3-small 크롭분류, 가중치 고정]
#
#   bash certification/run_cert_all.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]
#
# 전 과정 콘솔 출력 + 로그 저장 + 추론 중 시각화(GT|Pred 오버레이 PNG) 포함.
# 산출물: runs/cert_all_<TS>/{summary.txt, console.log, det_trial<N>/, iff/}
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CKPT="${1:?usage: run_cert_all.sh <best_checkpoint.pth> [DATA_ROOT] [GPU]}"
DATA_ROOT="${2:-$HOME/poongsan_v2}"
GPU="${3:-0}"
DET_TRIALS="${DET_TRIALS:-5}"        # ①② 표적위치·영상열화 = 5회
IFF_TRIALS="${IFF_TRIALS:-2}"        # ③ 피아식별 = 2회
IFF_CROPS="${IFF_CROPS:-$HOME/dset/poongsan_iff_crops}"
PY="${PYTHON:-python}"

cd "$REPO"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="runs/cert_all_${TS}"; mkdir -p "$OUT"
CONSOLE="$OUT/console.log"

# 전체 출력을 콘솔과 로그 파일에 동시 기록
exec > >(tee "$CONSOLE") 2>&1

B=$'\033[1m'; A=$'\033[33m'; G=$'\033[32m'; R=$'\033[0m'
echo "${B}╔════════════════════════════════════════════════════════════════╗${R}"
echo "${B}║        공인인증 정량목표 평가 — 통합 실행                        ║${R}"
echo "${B}╚════════════════════════════════════════════════════════════════╝${R}"
echo "  checkpoint : $(basename "$CKPT")"
echo "  data root  : $DATA_ROOT"
echo "  output     : $OUT   (콘솔로그·추론로그·시각화 PNG 저장)"
echo "  GPU        : $GPU"
echo ""
echo "  ${B}측정 항목${R}"
echo "   ① 표적위치 정확도           mAP50           반복 ${DET_TRIALS}회"
echo "   ② 영상 열화시 표적인식 성공률 mAP50(야간)      반복 ${DET_TRIALS}회"
echo "   ③ 피아식별 정확도           분류 accuracy    반복 ${IFF_TRIALS}회"
echo ""
echo "  ${B}사용 모델${R}"
echo "   ①② 검출  : ReliaDINO ViT-S+ (frozen DINOv3 ViT-S+/16 + per-modal LoRA"
echo "               + reliability-gated fusion) + RF-DETR NMS-free head"
echo "               3-modal RGB+LiDAR+Thermal, 768x768"
echo "   ③  분류  : MobileNetV3-small (ImageNet init) 128x128 크롭, Allies vs Enemies"
echo ""

# ────────────────────────── ①② 검출: mAP 반복 ──────────────────────────
echo "${A}▶ [1/2] 표적위치 정확도 + 영상 열화시 표적인식 성공률 — 검출 ${DET_TRIALS}회${R}"
for i in $(seq 1 "$DET_TRIALS"); do
  echo ""
  echo "${B}──── 검출 시행 $i / $DET_TRIALS ────${R}"
  # 서브셸 환경변수로만 넘긴다 ($OUT 자체를 덮어쓰면 경로가 중첩된다)
  OUT="$OUT/det_trial$i" bash "$HERE/run_cert.sh" "$CKPT" "$DATA_ROOT" "$GPU" --auto || \
    echo "  (검출 시행 $i 실패 — rc=$?)"
done

# ────────────────────────── ③ 피아식별: 분류 반복 ──────────────────────────
echo ""
echo "${A}▶ [2/2] 피아식별 정확도 — 크롭 분류 ${IFF_TRIALS}회${R}"
if [ ! -d "$IFF_CROPS/test" ]; then
  echo "  크롭 데이터셋이 없어 생성합니다 -> $IFF_CROPS"
  "$PY" "$HERE/build_iff_crops.py" \
      --train-root "$HOME/dset/poongsan_v2_yolo_rgb/images/train" --train-flat \
      --train-ann "$HOME/poongsan_v2_train3modal/_final_ann/instances_train_egofill.json" \
      --test-root "$DATA_ROOT" --out "$IFF_CROPS"
fi
IFF_CKPT="${IFF_CKPT:-$REPO/weights/iff_mobilenetv3.pt}"
if [ ! -f "$IFF_CKPT" ]; then
  echo "  피아식별 분류기 체크포인트가 없어 1회 학습합니다 -> $IFF_CKPT"
  "$PY" "$HERE/iff_eval.py" --mode train --data "$IFF_CROPS" \
      --epochs "${IFF_EPOCHS:-12}" --ckpt "$IFF_CKPT" --gpu "$GPU"
fi
# 인증 시험은 '가중치 고정 + 평가만' 반복 (①② 와 동일한 성격)
"$PY" "$HERE/iff_eval.py" --mode eval --data "$IFF_CROPS" --trials "$IFF_TRIALS" \
    --ckpt "$IFF_CKPT" --out "$OUT/iff" --gpu "$GPU"

# ────────────────────────── 통합 요약 ──────────────────────────
"$PY" - "$OUT" "$DET_TRIALS" "$IFF_TRIALS" <<'PY' | tee "$OUT/summary.txt"
import json, os, statistics, sys
out, dn, inn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
B, A, G, R = '\033[1m', '\033[33m', '\033[32m', '\033[0m'

det = []
for i in range(1, dn + 1):
    p = f'{out}/det_trial{i}/cert_report.json'
    if os.path.exists(p):
        det.append(json.load(open(p)))
iff = json.load(open(f'{out}/iff/iff_report.json')) if os.path.exists(f'{out}/iff/iff_report.json') else None

def ms(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

print(f'\n{B}╔══════════════════════════════════════════════════════════════════════╗{R}')
print(f'{B}║                   공인인증 정량목표 — 최종 결과                        ║{R}')
print(f'{B}╚══════════════════════════════════════════════════════════════════════╝{R}')
print(f'  {"항목":34s}{"지표":14s}{"반복":>4s}   {"결과 (평균 ± 표준편차)":>24s}')
print('  ' + '─' * 70)

if det:
    mu, sd = ms([d['overall']['AP50'] for d in det])
    print(f'  {"① 표적위치 정확도":32s}{"mAP50":14s}{len(det):>3d}회   {A}{mu:.4f} ± {sd:.4f}{R}')
    mu_n, sd_n = ms([d['night']['AP50'] for d in det])
    print(f'  {"② 영상 열화시 표적인식 성공률":28s}{"mAP50(야간)":12s}{len(det):>3d}회   {A}{mu_n:.4f} ± {sd_n:.4f}{R}')
    mu_d, sd_d = ms([d['normal']['AP50'] for d in det])
    mu_f, sd_f = ms([d['fps'] for d in det])
    print(f'  {"   └ 참고: 주간 mAP50":32s}{"":14s}{"":>4s}   {mu_d:.4f} ± {sd_d:.4f}')
    print(f'  {"   └ 참고: 속도":32s}{"FPS":14s}{"":>4s}   {mu_f:.2f} ± {sd_f:.2f}')
else:
    print('  ①② 검출 결과 없음 (cert_report.json 미생성)')

if iff:
    print(f'  {"③ 피아식별 정확도":32s}{"accuracy":14s}{iff["n_trials"]:>3d}회   '
          f'{A}{iff["mean"]["acc"]:.4f} ± {iff["std"]["acc"]:.4f}{R}')
    print(f'  {"   └ 야간 / 주간":32s}{"":14s}{"":>4s}   '
          f'{iff["mean"]["acc_night"]:.4f} / {iff["mean"]["acc_day"]:.4f}')
    print(f'  {"   └ Allies / Enemies recall":32s}{"":14s}{"":>4s}   '
          f'{iff["mean"]["recall_allies"]:.4f} / {iff["mean"]["recall_enemies"]:.4f}')
else:
    print('  ③ 피아식별 결과 없음')

print('  ' + '─' * 70)
print(f'  {B}사용 모델{R}')
print('    ①② ReliaDINO ViT-S+ (DINOv3 ViT-S+/16 frozen + per-modal LoRA + reliability-gated')
print('        fusion + SimpleFPN) + RF-DETR NMS-free head · 3-modal · 768x768')
if iff:
    print(f'    ③  {iff.get("model", "MobileNetV3-small")}')
print(f'\n  {B}산출물{R}  {out}/')
print( '    summary.txt · console.log · det_trial*/{console_*.log, inference_log.txt,')
print( '    cert_report.json, viz/*.png(GT|Pred 오버레이)} · iff/iff_report.json')
PY

echo ""
echo "${G}✔ 전체 완료 — 요약: $OUT/summary.txt${R}"
