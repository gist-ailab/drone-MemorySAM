#!/bin/bash
# Build the IEEE Sensors Journal manuscript.
#   bash build.sh            -> build/root.pdf
#   bash build.sh snapshot   -> also copy to drafts/IEEE-Sensors-draft_<YYYY-MM-DD>.pdf (commit that file)
# The build runs in build/ so that LaTeX by-products never touch the versioned sources.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build"
rsync -a --delete --exclude 'root.pdf' "$HERE/latex/" "$HERE/build/src/"
cd "$HERE/build/src"
latexmk -pdf -interaction=nonstopmode -quiet root.tex > /dev/null 2>&1 || true
errs=$(grep -c '^!' root.log || true)
cp root.pdf "$HERE/build/root.pdf"
echo "errors: $errs  pages: $(pdfinfo root.pdf | awk '/Pages/{print $2}')  -> build/root.pdf"
if [ "$1" = "snapshot" ]; then
  out="$HERE/drafts/IEEE-Sensors-draft_$(date +%F).pdf"
  cp root.pdf "$out"
  echo "snapshot -> ${out#$HERE/}"
fi
[ "$errs" = "0" ]
