#!/bin/bash
# Qualitative-figure pipeline (run from anywhere).
#   bash figures/qualbank/make.sh status          what is in the bank
#   bash figures/qualbank/make.sh bank <dataset>  ingest every ready model not yet in the bank, score, candidates
#   bash figures/qualbank/make.sh figs            render figures from selection.yaml and rebuild the PDF
# The bank and scores are slow (NAS); figs only reads them, so a scene change costs one `figs` run.
set -e
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
case "$1" in
  status) python3 figures/qualbank/qualbank.py status ;;
  bank)   python3 figures/qualbank/qualbank.py all "$2" ;;
  figs)   python3 figures/src/make_qual_figs.py && bash build.sh ;;
  *) echo "usage: make.sh status | bank <dataset> | figs"; exit 1 ;;
esac
