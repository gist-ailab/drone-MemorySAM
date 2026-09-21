#!/usr/bin/env bash
# run_cycle.sh — 자동 배치 1사이클: 배치(place) -> 240초 대기 -> 검증(verify).
# 사람이 읽는 요약을 마지막에 출력한다.
#
# 사용법:
#   bash scripts/autoplace/run_cycle.sh          # 배치 + 대기 + 검증
#   bash scripts/autoplace/run_cycle.sh --dry    # place.py 계획만 출력하고 종료(기동 없음)
#
# 종료코드: place 또는 verify 가 실패하면 0 이 아니다(정기 점검 크론에서 활용).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== [autoplace] 1단계: 배치 (place.py) =="

if [ "${1:-}" = "--dry" ]; then
  python3 "$SCRIPT_DIR/place.py"
  rc=$?
  echo "== [autoplace] --dry 모드: 계획만 출력하고 종료 (기동·검증 없음) =="
  exit "$rc"
fi

python3 "$SCRIPT_DIR/place.py" --launch
rc=$?
if [ $rc -ne 0 ]; then
  echo "== [autoplace] 요약: place.py 실패(rc=$rc) — 사이클 중단. 위 로그 확인 ==" >&2
  exit "$rc"
fi

echo
echo "== [autoplace] 2단계: 240초 대기 (기동 안정화 — 파라미터 덤프·첫 iteration 대기) =="
sleep 240

echo
echo "== [autoplace] 3단계: 검증 (verify.py) =="
python3 "$SCRIPT_DIR/verify.py"
rc=$?
if [ $rc -ne 0 ]; then
  echo "== [autoplace] 요약: 검증 FAIL 있음 — 해당 세션 로그와 GPU 상태를 직접 확인하라 ==" >&2
  exit "$rc"
fi

echo "== [autoplace] 요약: 배치 후 검증 통과 =="
exit 0
