#!/bin/bash
# Build the supplementary draft (revision reserve). Regenerates the tables first.
set -e
cd "$(dirname "$0")"
python3 make_supp_tables.py
cp ../latex/IEEEtran.bst . 2>/dev/null || true
pdflatex -interaction=nonstopmode supp_main.tex > /dev/null || true
bibtex supp_main > /dev/null || true
pdflatex -interaction=nonstopmode supp_main.tex > /dev/null || true
pdflatex -interaction=nonstopmode supp_main.tex > /dev/null || true
grep -c "^!" supp_main.log | xargs -I{} echo "errors: {}"
pdfinfo supp_main.pdf 2>/dev/null | grep Pages
