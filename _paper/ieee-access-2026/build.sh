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
# 2) manuscript (best value per column bold, second underlined: figures/src/mark_best.py, idempotent)
python3 "$HERE/figures/src/mark_best.py" > /dev/null
rsync -a --delete --exclude 'root.pdf' "$HERE/latex/" "$HERE/build/src/"
cd "$HERE/build/src"
latexmk -pdf -interaction=nonstopmode -quiet root.tex > /dev/null 2>&1 || true
errs=$(grep -c '^!' root.log || true)
cp root.pdf "$HERE/build/root.pdf"
echo "errors: $errs  pages: $(pdfinfo root.pdf | awk '/Pages/{print $2}')  -> build/root.pdf"
# When built inside a git worktree (.claude/worktrees/<name>/...), mirror the PDF to the same path in the
# main checkout so the user always opens one file (build/ is untracked, so this never touches git).
# Override with PAPER_PDF_MIRROR=<path>, or PAPER_PDF_MIRROR=none to skip.
case "$HERE" in
  */.claude/worktrees/*)
    main="${HERE%%/.claude/worktrees/*}/${HERE#*/.claude/worktrees/*/}"
    mirror="${PAPER_PDF_MIRROR:-$main/build/root.pdf}"
    if [ "$mirror" != "none" ] && [ -d "$(dirname "$(dirname "$mirror")")" ]; then
      mkdir -p "$(dirname "$mirror")" && cp root.pdf "$mirror" && echo "mirror -> $mirror"
    fi ;;
esac
if [ "$1" = "snapshot" ]; then
  out="$HERE/drafts/IEEE-Access-draft_$(date +%F).pdf"
  cp root.pdf "$out"
  echo "snapshot -> ${out#$HERE/}"
fi
# NAS copy for reading from Windows/phone (user 2026-11-10): \\nas_jm\jemo\Paper = /nas_jm/Paper on this hub,
# laid out as Paper/<project>/<paper folder>/ (project folder 26_MemorySAM+ already holds this project's papers).
# /nas_jm is sshfs, so every copy has a timeout and a failure never fails the build.
# Override with PAPER_PDF_NAS=<dir>, or PAPER_PDF_NAS=none to skip.
nas="${PAPER_PDF_NAS:-/nas_jm/Paper/26_MemorySAM+/$(basename "$HERE")}"
if [ "$nas" != "none" ] && timeout 10 test -d "$(dirname "$nas")"; then
  if timeout 20 mkdir -p "$nas" && timeout 60 cp root.pdf "$nas/root.pdf"; then
    echo "nas -> $nas/root.pdf"
    if [ "$1" = "snapshot" ]; then
      timeout 20 mkdir -p "$nas/drafts" && timeout 60 cp root.pdf "$nas/drafts/IEEE-Access-draft_$(date +%F).pdf" && echo "nas snapshot -> $nas/drafts/"
    fi
  else
    echo "nas copy skipped (timeout or error): $nas"
  fi
fi
[ "$errs" = "0" ]
