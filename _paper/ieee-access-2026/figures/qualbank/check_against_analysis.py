"""Cross-check: the bank's candidates must reproduce the scenes analysis selected by hand (inbox 20261030-175233).

Compares figures/source_data/qual_scenes_deliver.csv (analysis, 24 scenes: top 1-3 and reverse 1 per condition)
with figures/source_data/qualbank/deliver_candidates.csv (this system) on image id and delta.
Usage (from _paper/ieee-access-2026): python figures/qualbank/check_against_analysis.py
"""
import csv
from collections import Counter
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
rd = lambda p: [r for r in csv.DictReader(l for l in open(p, encoding="utf-8") if not l.startswith("#"))]
old = rd(PAPER / "figures/source_data/qual_scenes_deliver.csv")
new = rd(PAPER / "figures/source_data/qualbank/deliver_candidates.csv")
nm = {(r["kind"], r["group"], r["rank"]): r for r in new}
bad = 0
for r in old:
    n = nm.get((r["kind"], r["group"], r["rank"]))
    same = n is not None and n["image_id"] == r["scene_id"] and abs(float(n["delta"]) - float(r["delta"])) < 1e-3
    if not same:
        bad += 1
        print("DIFF", r["kind"], r["group"], r["rank"], r["scene_id"], r["delta"], "|",
              n and n["image_id"], n and n["delta"])
print(f"compared {len(old)} scenes, differing {bad}")
print("candidate rows by kind:", dict(Counter(r["kind"] for r in new)))
