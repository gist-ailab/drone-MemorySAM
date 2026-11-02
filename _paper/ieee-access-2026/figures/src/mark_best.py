"""Mark the best value of every result column in bold and the second best with an underline.

Rule (user 2026-11-02): in each listed table, per numeric column, the highest value is \\best{} (bold) and the
second highest is \\second{} (underline). Ties share the mark. A column (or block) with fewer than two numeric
values gets no mark. Cells that are not plain numbers (todo, --, check marks) are ignored.
Scope: a table is ranked as a whole unless it is listed in BLOCKED, where each block that starts with a
"\\multicolumn{..}{..}{\\emph{...}}" header line is ranked on its own (the blocks there use different columns).
The script is idempotent: it first removes \\best{}/\\second{}/\\textbf{} around plain numbers, then re-marks.
build.sh runs it before LaTeX, so a changed number re-ranks automatically.

Usage (from _paper/ieee-access-2026):  python figures/src/mark_best.py
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
FILES = [PAPER / "latex" / "sections" / "5_experiments.tex"]
TABLES = ["tab:deliver", "tab:muses", "tab:mcubes", "tab:robust", "tab:condition",
          "tab:ablation", "tab:ablation_loo", "tab:abl_robust"]
BLOCKED = {"tab:robust", "tab:abl_robust"}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
WRAP = re.compile(r"\\(?:best|second|textbf)\{(-?\d+(?:\.\d+)?)\}")


def split_cells(row):
    body = row.rstrip()
    assert body.endswith("\\\\"), row
    return body[:-2].split("&")


def rank_rows(lines, idx):
    """idx = indices of data rows in one ranking scope; edits lines in place."""
    cells = {i: split_cells(lines[i]) for i in idx}
    ncol = max(len(c) for c in cells.values())
    for col in range(ncol):
        vals = {}
        for i, c in cells.items():
            if col < len(c) and NUM.match(c[col].strip()):
                vals[i] = float(c[col].strip())
        uniq = sorted(set(vals.values()), reverse=True)
        if len(vals) < 2 or len(uniq) < 1:
            continue
        first = uniq[0]
        second = uniq[1] if len(uniq) > 1 else None
        for i, v in vals.items():
            raw = cells[i][col]
            txt = raw.strip()
            mark = "best" if v == first else ("second" if v == second else None)
            if mark:
                cells[i][col] = raw.replace(txt, f"\\{mark}{{{txt}}}", 1)
    for i, c in cells.items():
        lines[i] = "&".join(c) + "\\\\\n"


def process(text, label):
    start = text.index(f"\\label{{{label}}}")
    b = text.index("\\begin{tabular", start)
    e = text.index("\\end{tabular", b)
    seg = text[b:e]
    seg = WRAP.sub(r"\1", seg)
    lines = seg.splitlines(keepends=True)
    scopes, cur, header_seen = [], [], False
    for i, l in enumerate(lines):
        s = l.strip()
        if label in BLOCKED and s.startswith("\\multicolumn") and "\\emph" in s:
            if cur:
                scopes.append(cur)
            cur = []
            continue
        if "&" in s and s.endswith("\\\\") and not s.startswith("\\multicolumn"):
            cells = [c.strip() for c in s[:-2].split("&")]
            if any(NUM.match(c) for c in cells):
                cur.append(i)
    if cur:
        scopes.append(cur)
    for sc in scopes:
        rank_rows(lines, sc)
    return text[:b] + "".join(lines) + text[e:]


def main():
    for f in FILES:
        t = f.read_text(encoding="utf-8")
        for lab in TABLES:
            if f"\\label{{{lab}}}" in t:
                t = process(t, lab)
        f.write_text(t, encoding="utf-8")
    print("mark_best: done")


if __name__ == "__main__":
    main()
