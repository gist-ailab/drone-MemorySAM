#!/bin/bash
# NAS Obsidian 볼트 → repo 사본 동기화
# 사본(.claude_logs/research/vault/)은 손편집 금지 — 항상 이 스크립트로 재생성한다.
# 제외 정책은 vault README.md의 "포함/제외" 규칙을 따른다:
#   OpenAlex DB(노이즈) · source map 노트 · PDF 원문 · 트렌드 워치 스캐폴드 · .obsidian/.trash
set -euo pipefail

SRC=/nas_jm/Research/26_MultimodalSeg
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$REPO_ROOT/.claude_logs/research/vault"

[ -d "$SRC" ] || { echo "ERROR: NAS 볼트 미마운트: $SRC" >&2; exit 1; }

rsync -av --delete \
  --exclude '.obsidian/' \
  --exclude '.trash/' \
  --exclude '*.pdf' \
  --exclude 'sources/pdfs/' \
  --exclude 'sources/raw/' \
  --exclude 'sources/db/' \
  --exclude '*.json' --exclude '*.jsonl' --exclude '*.csv' --exclude '*.sqlite' \
  --exclude 'sources/01_source_index*' \
  --exclude 'sources/02_openalex*' \
  --exclude 'sources/02_source_map*' \
  --exclude 'sources/02_top_venue*' \
  --exclude 'sources/05_*' \
  --exclude 'sources/06_*' \
  --exclude 'README.md' \
  "$SRC/" "$DST/"

# 동기화 일자를 README 헤더에 기록
sed -i "s/^> \*\*마지막 동기화\*\*: .*/> **마지막 동기화**: $(date +%F) (scripts\/sync_research_vault.sh)/" "$DST/README.md" 2>/dev/null || true
echo "synced: $SRC -> $DST ($(date +%F))"
