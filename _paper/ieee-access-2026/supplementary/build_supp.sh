#!/bin/bash
# Build the revision-reserve draft (supp_reserve.tex; the submitted supplementary is built by ../build.sh). Regenerates the tables first.
set -e
cd "$(dirname "$0")"
python3 make_supp_tables.py
cp ../latex/IEEEtran.bst . 2>/dev/null || true
pdflatex -interaction=nonstopmode supp_reserve.tex > /dev/null || true
bibtex supp_reserve > /dev/null || true
pdflatex -interaction=nonstopmode supp_reserve.tex > /dev/null || true
pdflatex -interaction=nonstopmode supp_reserve.tex > /dev/null || true
grep -c "^!" supp_reserve.log | xargs -I{} echo "errors: {}"
pdfinfo supp_reserve.pdf 2>/dev/null | grep Pages
