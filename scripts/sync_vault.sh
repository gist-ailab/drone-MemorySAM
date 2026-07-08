#!/usr/bin/env bash
# research_vault 동기화: NAS 옵시디언 볼트(canonical) → repo 사본(.claude_logs/research_vault)
#
# 방향: NAS → repo 단방향 pull. NAS가 단일 정본(canonical)이다.
#   - 새 실험 리포트는 NAS 볼트(실험폴더 P<N>_<이름>/)에 먼저 쓰고, 이 스크립트로 repo에 반영한다.
#   - repo 쪽에서만 만들어진 문서(예: Cowork 산출물)는 NAS로 수동 복사 후 pull.
# 제외: README의 기존 규칙(무거운 원천 데이터·미검증 소스맵) + 옵시디언 내부 폴더 + PDF.
#
# 사용: bash scripts/sync_vault.sh [--dry-run]

set -euo pipefail

NAS=/nas_jm/Research/26_MultimodalSeg
REPO_VAULT="$(cd "$(dirname "$0")/.." && pwd)/.claude_logs/research_vault"

if [ ! -d "$NAS" ]; then
  echo "ERROR: NAS vault not mounted at $NAS (sshfs 확인)" >&2
  exit 1
fi

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

rsync -av $DRY \
  --exclude '.obsidian/' \
  --exclude '.trash/' \
  --exclude 'sources/db/' \
  --exclude 'sources/pdfs/' \
  --exclude 'sources/raw/' \
  --exclude 'sources/archive/' \
  --exclude 'sources/01_*' \
  --exclude 'sources/02_*' \
  --exclude 'sources/05_*' \
  --exclude 'sources/06_*' \
  --exclude '*.pdf' \
  "$NAS/" "$REPO_VAULT/"

echo "동기화 완료: $NAS → $REPO_VAULT $DRY"
echo "다음: git diff로 확인 후 커밋 (.claude_logs/research_vault)"
