#!/bin/bash
# Build the IEEE Access manuscript.
#   bash build.sh            -> build/root.pdf
#   bash build.sh snapshot   -> also copy to drafts/IEEE-Access-draft_<YYYY-MM-DD>.pdf (commit that file)
# The build runs in build/ so that LaTeX by-products never touch the versioned sources.
# Fig. 1 is TikZ; ieeeaccess.cls conflicts with pgf, so it is compiled first as a standalone PDF
# (figures/src/fig_arch_standalone.tex -> latex/figures/fig_arch.pdf, versioned).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build/fig"
# 1) Fig. 1 standalone
rsync -a "$HERE/latex/figures/" "$HERE/build/fig/figures/"
cp "$HERE/figures/src/fig_arch_standalone.tex" "$HERE/build/fig/"
( cd "$HERE/build/fig" && pdflatex -interaction=nonstopmode fig_arch_standalone.tex > /dev/null 2>&1 ) || true
if [ -s "$HERE/build/fig/fig_arch_standalone.pdf" ]; then
  cp "$HERE/build/fig/fig_arch_standalone.pdf" "$HERE/latex/figures/fig_arch.pdf"
else
  echo "Fig. 1 standalone build failed: see build/fig/fig_arch_standalone.log"; exit 1
fi
# 2) manuscript
rsync -a --delete --exclude 'root.pdf' "$HERE/latex/" "$HERE/build/src/"
cd "$HERE/build/src"
latexmk -pdf -interaction=nonstopmode -quiet root.tex > /dev/null 2>&1 || true
errs=$(grep -c '^!' root.log || true)
cp root.pdf "$HERE/build/root.pdf"
echo "errors: $errs  pages: $(pdfinfo root.pdf | awk '/Pages/{print $2}')  -> build/root.pdf"
if [ "$1" = "snapshot" ]; then
  out="$HERE/drafts/IEEE-Access-draft_$(date +%F).pdf"
  cp root.pdf "$out"
  echo "snapshot -> ${out#$HERE/}"
fi
[ "$errs" = "0" ]
