"""Verify every BibTeX entry against CrossRef, arXiv and DBLP (title / first author / year / arXiv id).

Usage: python verify_bib.py <references.bib> <report.md>
Status per entry:
  verified  - a database record with title similarity >= 0.92 and matching first-author surname
  suspect   - best record similarity in [0.75, 0.92) or author/year/id mismatch
  not_found - nothing above 0.75 in any database
Network only; no packages beyond the standard library.
"""
import difflib
import json
import unicodedata
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "bib-verify/1.0 (mailto:maengjemo98@gmail.com)"}


def get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\"[^\"]*\"|\d+)", body):
            v = f.group(2).strip()
            if v[0] in "{\"":
                v = v[1:-1]
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", v)
        entries.append((kind, key, fields))
    return entries


def norm(t):
    t = re.sub(r'\\[\"\'`^~=.]\{?([a-zA-Z])\}?', r"\1", t or "")   # LaTeX accents -> base letter
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"\\[a-zA-Z]+|[{}\\\"']", "", t)
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def first_surname(authors):
    if not authors:
        return ""
    a = re.split(r"\s+and\s+", authors)[0]
    a = re.sub(r"[{}\\\"']", "", a)
    s = a.split(",")[0] if "," in a else a.split()[-1]
    return norm(s)[0] if norm(s) else ""


def crossref(title):
    q = urllib.parse.quote(title)
    d = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"))
    out = []
    for it in d["message"]["items"]:
        out.append(dict(src="crossref", title=(it.get("title") or [""])[0],
                        first=(it.get("author") or [{}])[0].get("family", ""),
                        year=str((it.get("issued", {}).get("date-parts") or [[None]])[0][0]),
                        id=it.get("DOI", ""), venue=(it.get("container-title") or [""])[0]))
    return out


def arxiv(title):
    words = [w for w in norm(title) if len(w) > 2][:10]
    q = urllib.parse.quote(" AND ".join(f"ti:{w}" for w in words))
    x = get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=3")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        au = e.find("a:author/a:name", ns)
        out.append(dict(src="arxiv", title=e.findtext("a:title", "", ns), first=(au.text.split()[-1] if au is not None else ""),
                        year=e.findtext("a:published", "", ns)[:4], id=e.findtext("a:id", "", ns).split("/abs/")[-1], venue="arXiv"))
    return out


def arxiv_id(aid):
    x = get(f"http://export.arxiv.org/api/query?id_list={aid}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        au = e.find("a:author/a:name", ns)
        out.append(dict(src="arxiv-id", title=e.findtext("a:title", "", ns), first=(au.text.split()[-1] if au is not None else ""),
                        year=e.findtext("a:published", "", ns)[:4], id=e.findtext("a:id", "", ns).split("/abs/")[-1], venue="arXiv"))
    return out


def arxiv_phrase(title):
    t = " ".join(norm(title))
    q = urllib.parse.quote(f'ti:"{t}"')
    x = get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=3")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(x).findall("a:entry", ns):
        au = e.find("a:author/a:name", ns)
        out.append(dict(src="arxiv", title=e.findtext("a:title", "", ns), first=(au.text.split()[-1] if au is not None else ""),
                        year=e.findtext("a:published", "", ns)[:4], id=e.findtext("a:id", "", ns).split("/abs/")[-1], venue="arXiv"))
    return out


def dblp(title):
    q = urllib.parse.quote(title)
    d = json.loads(get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=3"))
    out = []
    for h in d["result"]["hits"].get("hit", []):
        i = h["info"]
        au = i.get("authors", {}).get("author", [])
        au = au if isinstance(au, list) else [au]
        first = au[0]["text"].split()[-1] if au else ""
        out.append(dict(src="dblp", title=i.get("title", ""), first=first, year=i.get("year", ""),
                        id=i.get("doi", i.get("ee", "")), venue=i.get("venue", "")))
    return out


def main(bib, report):
    entries = parse_bib(open(bib).read())
    rows = []
    for kind, key, f in entries:
        title = f.get("title", "")
        cands, errs = [], []
        axm = re.search(r"(\d{4}\.\d{4,5})", " ".join(f.get(k, "") for k in ("journal", "eprint", "note", "url")))
        calls = [crossref, arxiv_phrase, dblp]
        if axm:
            calls.insert(0, lambda t, a=axm.group(1): arxiv_id(a))
        for fn in calls:
            try:
                cands += fn(title)
            except Exception as ex:  # network hiccup: record and continue
                errs.append(f"{getattr(fn, '__name__', 'arxiv_id')}: {type(ex).__name__}")
            time.sleep(3.1)
        best = max(cands, key=lambda c: sim(title, c["title"]), default=None)
        s = sim(title, best["title"]) if best else 0.0
        fa = first_surname(f.get("author", ""))
        notes = []
        status = "not_found"
        if best and s >= 0.75:
            au_ok = (not fa) or fa in norm(best["first"]) or any(fa in norm(c["first"]) for c in cands if sim(title, c["title"]) >= 0.9)
            yr = f.get("year", "")
            yr_ok = (not yr) or yr == best["year"] or any(yr == c["year"] for c in cands if sim(title, c["title"]) >= 0.9)
            ax = re.search(r"(\d{4}\.\d{4,5})", f.get("journal", "") + f.get("eprint", "") + f.get("note", ""))
            ax_ok = True
            if ax:
                ax_ok = any(ax.group(1) in c["id"] for c in cands)
                if not ax_ok:
                    notes.append(f"arXiv id {ax.group(1)} not among matches")
            if not au_ok:
                notes.append(f"first author '{fa}' vs '{best['first']}'")
            if not yr_ok:
                notes.append(f"year {yr} vs {best['year']}")
            status = "verified" if (s >= 0.92 and au_ok and yr_ok and ax_ok) else "suspect"
        rows.append(dict(key=key, status=status, sim=round(s, 2), bib_title=title,
                         match=(best or {}).get("title", ""), src=(best or {}).get("src", ""),
                         year=(best or {}).get("year", ""), id=(best or {}).get("id", ""),
                         notes="; ".join(notes + errs)))
        print(key, status, round(s, 2), flush=True)
    with open(report, "w") as w:
        w.write("# Reference check\n\nTool: figures/src/verify_bib.py (CrossRef + arXiv + DBLP, title/first-author/year/arXiv-id match).\n\n")
        cnt = {k: sum(r["status"] == k for r in rows) for k in ("verified", "suspect", "not_found")}
        w.write(f"Summary: {len(rows)} entries — verified {cnt['verified']}, suspect {cnt['suspect']}, not_found {cnt['not_found']}\n\n")
        w.write("| key | status | sim | best match (source, year, id) | notes |\n|---|---|---|---|---|\n")
        for r in rows:
            m = f"{r['match'][:90]} ({r['src']}, {r['year']}, {r['id']})" if r["match"] else "-"
            w.write(f"| {r['key']} | {r['status']} | {r['sim']} | {m} | {r['notes']} |\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
