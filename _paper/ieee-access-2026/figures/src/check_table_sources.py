"""Check that every number printed in a manuscript table appears in notes/table_sources.md.

Reads latex/sections/*.tex, extracts the tabular environments (comment lines skipped), collects
decimal numbers and 3-4 digit integers, and lists those missing from notes/table_sources.md.
Exit code 1 if anything is missing. Run from the paper folder: python figures/src/check_table_sources.py
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
src = (PAPER / "notes" / "table_sources.md").read_text(encoding="utf-8")
missing = {}
for tex in sorted((PAPER / "latex" / "sections").glob("*.tex")):
    body = tex.read_text(encoding="utf-8")
    for tab in re.findall(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", body, re.S):
        for line in tab.splitlines():
            if line.strip().startswith("%"):
                continue
            for n in re.findall(r"(?<![\w.])\d+\.\d+(?!\w)|(?<![\w.])\d{3,4}(?![\w.])", line):
                if n not in src:
                    missing.setdefault(tex.name, set()).add(n)
if missing:
    for f, ns in missing.items():
        print(f"{f}: not in notes/table_sources.md -> {sorted(ns)}")
    sys.exit(1)
print("all table numbers have a source entry")
