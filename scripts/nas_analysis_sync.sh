#!/bin/bash
# NAS 분석 산출물 공유·연동 (user 지시 2026-09-18: 산출물은 /ailab_mat2 에 두고 서버 간 연동)
#
# 공유 루트 = $NAS_ANALYSIS_ROOT (기본값 아래) — /ailab_mat2 는 hub·yeon·lecun 에 마운트돼 있고
# hpca100 에는 마운트가 없으므로(2026-09-18 확인) hpca100 은 hub 를 거쳐야 한다.
#
#   bash scripts/nas_analysis_sync.sh init     <run_id>
#   bash scripts/nas_analysis_sync.sh push     <run_id> <src_path> <rel_dest> [server]
#   bash scripts/nas_analysis_sync.sh pull     <run_id> <rel_src> <dst_path>  [server]
#   bash scripts/nas_analysis_sync.sh verify   <run_id> <rel> <local_path>    [server]
#   bash scripts/nas_analysis_sync.sh manifest <run_id>
#   bash scripts/nas_analysis_sync.sh ls       <run_id> [rel]
#   bash scripts/nas_analysis_sync.sh mounts
#
# server 를 주면 그 서버에서 rsync 를 실행한다(그 서버의 /ailab_mat2 마운트 사용). 생략하면 hub.
# NFS 가 그룹 변경을 막으므로 속성 보존 옵션을 끈다(-rt --no-perms --no-owner --no-group).
set -uo pipefail

NAS_ANALYSIS_ROOT=${NAS_ANALYSIS_ROOT:-/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/analysis}
RSYNC_OPTS="-rt --no-perms --no-owner --no-group --human-readable"
MOUNTED_SERVERS="yeon lecun"   # /ailab_mat2 마운트 확인된 원격 서버 (hpca100 은 없음)

die() { echo "ERROR: $*" >&2; exit 1; }

run_at() {  # run_at <server|hub> <command>
  local where=$1; shift
  if [ "$where" = "hub" ] || [ -z "$where" ]; then
    bash -c "$*"
  else
    ssh "$where" "$*"
  fi
}

cmd=${1:-}; [ -n "$cmd" ] || die "부명령이 필요하다 (init|push|pull|verify|manifest|ls|mounts)"
shift || true

case "$cmd" in
  mounts)
    echo "hub: $(df -h /ailab_mat2 2>/dev/null | tail -1 || echo NOT_MOUNTED)"
    for s in $MOUNTED_SERVERS; do
      echo "$s: $(ssh -o ConnectTimeout=10 "$s" 'df -h /ailab_mat2 2>/dev/null | tail -1 || echo NOT_MOUNTED' 2>/dev/null)"
    done
    ;;

  init)
    run_id=${1:?run_id}; base="$NAS_ANALYSIS_ROOT/$run_id"
    mkdir -p "$base"/{code,raw,preds,metrics,mining,reports}
    [ -f "$base/README.md" ] || cat > "$base/README.md" <<EOF
# $run_id — 분석 산출물 공유 루트

- 생성: $(date '+%F %T') (scripts/nas_analysis_sync.sh init)
- 규약: raw=입력 원본 사본(체크포인트·예측 JSON), preds=복원 예측 PNG, metrics=이미지별 지표,
  mining=실패 채굴 결과, reports=판정 문서, code=실행 시점 도구 사본.
- 대장: MANIFEST.tsv (scripts/nas_analysis_sync.sh manifest $run_id 로 갱신).
- 서버 간 연동: /ailab_mat2 는 hub·yeon·lecun 에 마운트. hpca100 은 hub 경유.
EOF
    echo "초기화 완료: $base"
    ;;

  push)
    run_id=${1:?run_id}; src=${2:?src_path}; rel=${3:?rel_dest}; server=${4:-hub}
    dst="$NAS_ANALYSIS_ROOT/$run_id/$rel"
    run_at "$server" "mkdir -p \"\$(dirname '$dst')\" && rsync $RSYNC_OPTS '$src' '$dst'" || die "push 실패"
    echo "push 완료: [$server] $src -> $dst"
    ;;

  pull)
    run_id=${1:?run_id}; rel=${2:?rel_src}; dst=${3:?dst_path}; server=${4:-hub}
    src="$NAS_ANALYSIS_ROOT/$run_id/$rel"
    run_at "$server" "mkdir -p \"\$(dirname '$dst')\" && rsync $RSYNC_OPTS '$src' '$dst'" || die "pull 실패"
    echo "pull 완료: [$server] $src -> $dst"
    ;;

  verify)  # rsync 체크섬 비교로 차이 유무만 확인(전송하지 않는다)
    run_id=${1:?run_id}; rel=${2:?rel}; local_path=${3:?local_path}; server=${4:-hub}
    nas="$NAS_ANALYSIS_ROOT/$run_id/$rel"
    out=$(run_at "$server" "rsync -rtn --checksum --no-perms --no-owner --no-group --itemize-changes '$local_path' '$nas'" 2>&1)
    diffs=$(echo "$out" | grep -c '^[<>ch]' || true)
    if [ "$diffs" -eq 0 ]; then echo "verify 통과: 차이 없음 ($local_path == $nas)"; else
      echo "verify 실패: 차이 $diffs 건"; echo "$out" | head -20; exit 1; fi
    ;;

  manifest)  # md5 를 전부 계산하므로 /ailab_mat2 가 빠른 서버(yeon·lecun)에서 돌리는 편이 낫다.
    run_id=${1:?run_id}; server=${2:-hub}; base="$NAS_ANALYSIS_ROOT/$run_id"
    script_on_nas="$NAS_ANALYSIS_ROOT/$run_id/code/nas_analysis_sync.sh"
    if [ "$server" != "hub" ]; then
      [ -f "$script_on_nas" ] || die "원격 실행에는 NAS 사본이 필요하다: push $run_id <이 스크립트> code/nas_analysis_sync.sh"
      ssh "$server" "bash '$script_on_nas' manifest '$run_id'" || die "원격 manifest 실패"
      exit 0
    fi
    [ -d "$base" ] || die "$base 없음"
    out="$base/MANIFEST.tsv"
    { echo -e "rel_path\tbytes\tmtime\tmd5"
      find "$base" -type f ! -name MANIFEST.tsv -printf '%P\t%s\t%TY-%Tm-%Td %TH:%TM\n' \
        | sort | while IFS=$'\t' read -r p s m; do
            echo -e "$p\t$s\t$m\t$(md5sum "$base/$p" | cut -d' ' -f1)"
          done
    } > "$out"
    echo "대장 갱신: $out ($(($(wc -l < "$out") - 1)) 파일)"
    ;;

  ls)
    run_id=${1:?run_id}; rel=${2:-}
    du -sh "$NAS_ANALYSIS_ROOT/$run_id/$rel" 2>/dev/null
    find "$NAS_ANALYSIS_ROOT/$run_id/$rel" -maxdepth 2 -mindepth 1 -printf '%y %10s %P\n' 2>/dev/null | head -60
    ;;

  *) die "알 수 없는 부명령: $cmd" ;;
esac
