#!/usr/bin/env bash
# probe_free_gpus.sh — 서버 한 대의 빈 GPU 인덱스를 조사해 공백 구분 한 줄로 출력한다.
#
# 사용법: bash scripts/autoplace/probe_free_gpus.sh <서버alias>
#
# 빈 GPU 판정: memory.used <= ${GPU_MAXMEM:-2000} MiB 그리고 util <= ${GPU_MAXUTIL:-10} %
# servers.conf(scripts/servers.conf, '|' 구분 6열)의 policy:
#   off      -> 아무것도 출력하지 않고 종료코드 0 (ssh 자체를 하지 않는다)
#   ban:1,2  -> 해당 인덱스는 결과에서 제외
#   없음/'-'  -> 제한 없음
# ssh 실패 시 stderr 에 한 줄 남기고 빈 출력, 종료코드 1.
# 출력 순서는 메모리 사용량 오름차순(scripts/pick_free_gpus.sh 관례와 동일).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${AUTOPLACE_SERVERS_CONF:-$SCRIPT_DIR/../servers.conf}"

alias_name="${1:-}"
if [ -z "$alias_name" ]; then
  echo "probe_free_gpus: 서버 alias 가 필요함 (usage: bash scripts/autoplace/probe_free_gpus.sh <서버alias>)" >&2
  exit 1
fi
if [ ! -f "$CONF" ]; then
  echo "probe_free_gpus: servers.conf 를 찾을 수 없음: $CONF" >&2
  exit 1
fi

# lookup: 해당 alias 행의 policy(6번째 필드, 없으면 빈 값)를 꺼낸다. 매칭 행이 없으면 빈 출력.
row="$(awk -F'|' -v a="$alias_name" -v S=$'\037' '
  /^[[:space:]]*#/ {next}  /^[[:space:]]*$/ {next}
  { t=$1; gsub(/^[ \t]+|[ \t]+$/,"",t); gsub(/\r/,"",t)
    if (t==a) { p=$6; gsub(/^[ \t]+|[ \t]+$/,"",p); gsub(/\r/,"",p); print "1" S p; exit } }' "$CONF")"
if [ -z "$row" ]; then
  echo "probe_free_gpus: 서버 '$alias_name' 이(가) $CONF 에 없음" >&2
  exit 1
fi
policy="${row#*$'\037'}"
policy_trim="$(printf '%s' "$policy" | tr -d '[:space:]')"

case "$policy_trim" in
  off) exit 0 ;;   # 정책상 배치 금지 서버: 조사하지 않고 조용히 끝낸다
esac
banned=""
case "$policy_trim" in
  ban:*) banned="${policy_trim#ban:}" ;;
  ""|-|none) ;;
  *) echo "probe_free_gpus: 알 수 없는 policy '$policy' — 제한 없음으로 취급" >&2 ;;
esac

errfile="$(mktemp)"
trap 'rm -f "$errfile"' EXIT
if ! out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$alias_name" \
      "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>"$errfile")"; then
  echo "probe_free_gpus: ssh/nvidia-smi 실패 ($alias_name): $(tr '\n' ' ' < "$errfile" | cut -c1-200)" >&2
  exit 1
fi

# CSV 파싱: 빈 GPU 만 남기고 (메모리, 인덱스) 쌍으로 메모리 오름차순 정렬 후 인덱스만 출력.
free="$(printf '%s\n' "$out" | awk -F',' -v mm="${GPU_MAXMEM:-2000}" -v mu="${GPU_MAXUTIL:-10}" '
  { gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3)
    if ($2+0 <= mm && $3+0 <= mu) print ($2+0)"\t"$1 }' | sort -n | awk -F'\t' '{print $2}')"

if [ -n "$banned" ]; then
  free="$(printf '%s\n' "$free" | awk -v ban="$banned" '
    BEGIN{n=split(ban,b,","); for(i=1;i<=n;i++) skip[b[i]]=1}
    NF && !($1 in skip) {print $1}')"
fi

printf '%s\n' "$free" | sed '/^$/d' | paste -sd' ' -
