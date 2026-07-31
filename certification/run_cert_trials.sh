#!/usr/bin/env bash
# 인증 반복 시험 러너 — 항목 (A) 표적위치 정확도 / (B) 영상 열화시 표적인식 성공률.
# 둘 다 지표가 mAP 이고 반복 2회이므로 cert_eval 을 N회(기본 2) 돌려 평균±표준편차를 낸다.
# (기존 run_cert.sh 는 1회만 돌고 cert_report.json 을 덮어써서 반복 집계가 불가능했다.)
#
#   bash run_cert_trials.sh <best_checkpoint.pth> [DATA_ROOT] [GPU] [TRIALS]
#
# 각 회차 산출물은 runs/cert_trials/trial<N>/ 에 분리 저장되고,
# 마지막에 all/night/normal mAP50 의 평균±std 를 출력 + trials_summary.json 저장.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CKPT="${1:?usage: run_cert_trials.sh <ckpt> [DATA_ROOT] [GPU] [TRIALS]}"
DATA_ROOT="${2:-$HOME/poongsan_v2}"
GPU="${3:-0}"
TRIALS="${4:-2}"

cd "$REPO"
OUT="runs/cert_trials"; mkdir -p "$OUT"
echo "[trials] 반복 시험 ${TRIALS}회 — ckpt=$(basename "$CKPT") data=$DATA_ROOT gpu=$GPU"

for i in $(seq 1 "$TRIALS"); do
  echo "======== trial $i / $TRIALS ========"
  OUT="$OUT/trial$i" bash "$HERE/run_cert.sh" "$CKPT" "$DATA_ROOT" "$GPU" --auto
done

# ---- 집계: 각 trial 의 cert_report.json 을 읽어 평균±std ----
"${PYTHON:-python}" - "$OUT" "$TRIALS" <<'PY'
import json, statistics, sys, os
out, n = sys.argv[1], int(sys.argv[2])
rows = []
for i in range(1, n + 1):
    p = f'{out}/trial{i}/cert_report.json'
    if os.path.exists(p):
        rows.append(json.load(open(p)))
if not rows:
    print('  (no cert_report.json found — did the trials run?)'); raise SystemExit(1)

def ms(get):
    v = [get(r) for r in rows]
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

print(f"\n╔═══ 반복 시험 집계 ({len(rows)}회) ═══╗")
for lab, get in (('all    mAP50', lambda r: r['overall']['AP50']),
                 ('night  mAP50', lambda r: r['night']['AP50']),
                 ('normal mAP50', lambda r: r['normal']['AP50']),
                 ('all    mAP',   lambda r: r['overall']['AP']),
                 ('FPS',          lambda r: r['fps'])):
    mu, sd = ms(get)
    print(f'  {lab:14s} {mu:.4f} ± {sd:.4f}')
mu, sd = ms(lambda r: r['overall']['AP50'])
print(f'\n  ►►  mAP50 = {mu:.4f} ± {sd:.4f}  ({len(rows)}회 평균)  ◄◄')
print('╚════════════════════════════╝')
json.dump({'n_trials': len(rows),
           'mean': {'AP50': ms(lambda r: r['overall']['AP50'])[0],
                    'AP50_night': ms(lambda r: r['night']['AP50'])[0],
                    'AP50_normal': ms(lambda r: r['normal']['AP50'])[0],
                    'AP': ms(lambda r: r['overall']['AP'])[0],
                    'fps': ms(lambda r: r['fps'])[0]},
           'std': {'AP50': ms(lambda r: r['overall']['AP50'])[1],
                   'AP50_night': ms(lambda r: r['night']['AP50'])[1],
                   'AP50_normal': ms(lambda r: r['normal']['AP50'])[1],
                   'AP': ms(lambda r: r['overall']['AP'])[1],
                   'fps': ms(lambda r: r['fps'])[1]},
           'trials': [{'AP50': r['overall']['AP50'], 'AP50_night': r['night']['AP50'],
                       'AP50_normal': r['normal']['AP50'], 'fps': r['fps']} for r in rows]},
          open(f'{out}/trials_summary.json', 'w'), indent=2)
print(f'  summary: {out}/trials_summary.json\n')
PY
