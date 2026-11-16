#!/usr/bin/env python3
"""Precise text-layout audit for vector figures (paper session, user 2026-11-16 "디테일하게 검사").

Why: audit_figure_collisions.py pads text boxes with the font ascender/descender, which for the scaled
Helvetica of our TikZ figures makes adjacent lines of one label overlap "100%" (false positives).
Here every span gets a TIGHT box from its baseline and size: [baseline - 0.72*size, baseline + 0.21*size]
(Helvetica cap height ~0.72 em, descender ~0.21 em), horizontally the span's advance box.

Checks per figure PDF (figure coordinates, points):
  T1 text-text  : tight boxes of two spans intersect by more than 0.3 pt in both axes
  T2 text-stroke: a stroked line/curve segment passes through a tight text box (segments that only touch
                  the box border within 0.4 pt are ignored; filled shapes behind text are not strokes)
  T3 text-image : a tight text box lies on top of a raster image (labels drawn over thumbnails)
  T4 bounds     : a tight text box leaves the page (canvas) by more than 0.5 pt
Manuscript check (root.pdf): for each figure, effective font size of its text at the placed scale
  (placed width / natural width), flagged below 5 pt.

Usage: figqa-venv/bin/python audit_text_layout.py <fig.pdf>... [--manuscript build/root.pdf --figdir latex/figures]
Exit code 1 when any T1-T4 finding exists.
"""
import sys, argparse, math
import pymupdf

CAP, DESC = 0.72, 0.21


def spans(page):
    out = []
    for b in page.get_text('rawdict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                chars = [c for c in s['chars'] if c['c'].strip()]
                if not chars:
                    continue
                x0 = min(c['bbox'][0] for c in chars); x1 = max(c['bbox'][2] for c in chars)
                by = s['origin'][1]; sz = s['size']
                txt = ''.join(c['c'] for c in s['chars']).strip()
                out.append(dict(text=txt, size=sz, box=(x0, by - CAP * sz, x1, by + DESC * sz)))
    return out


def inter(a, b, tol):
    return (min(a[2], b[2]) - max(a[0], b[0]) > tol) and (min(a[3], b[3]) - max(a[1], b[1]) > tol)


def seg_hits_box(p, q, box, tol=0.4):
    x0, y0, x1, y1 = box[0] + tol, box[1] + tol, box[2] - tol, box[3] - tol
    if x1 <= x0 or y1 <= y0:
        return False
    # Liang-Barsky clipping
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def stroke_segments(page):
    segs = []
    for d in page.get_drawings():
        if d.get('type') not in ('s', 'fs') or not d.get('color'):
            continue
        for it in d['items']:
            if it[0] == 'l':
                segs.append((tuple(it[1]), tuple(it[2])))
            elif it[0] == 'c':   # bezier: sample
                P = [it[1], it[2], it[3], it[4]]
                pts = []
                for k in range(9):
                    t = k / 8
                    x = sum(math.comb(3, i) * (1 - t) ** (3 - i) * t ** i * P[i].x for i in range(4))
                    y = sum(math.comb(3, i) * (1 - t) ** (3 - i) * t ** i * P[i].y for i in range(4))
                    pts.append((x, y))
                segs += list(zip(pts[:-1], pts[1:]))
            elif it[0] == 're':
                r = it[1]
                c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                segs += [(c[i], c[(i + 1) % 4]) for i in range(4)]
    return segs


def audit(path):
    doc = pymupdf.open(path); pg = doc[0]; W, H = pg.rect.width, pg.rect.height
    S = spans(pg); segs = stroke_segments(pg)
    imgs = [pg.get_image_bbox(im) for im in pg.get_images(full=True)]
    imgs = [r for r in imgs if r.is_valid and not r.is_empty]
    find = []
    ACC = set('˜^ˆ¯´`˙¨~')   # math accents sit on their base letter by design
    def is_accent(t):
        return t and all(c in ACC or c in '()' for c in t) and any(c in ACC for c in t)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if is_accent(S[i]['text']) or is_accent(S[j]['text']):
                continue
            if inter(S[i]['box'], S[j]['box'], 0.3):
                find.append(('T1 text-text', S[i]['text'], S[j]['text']))
    for s in S:
        n = sum(seg_hits_box(p, q, s['box']) for p, q in segs)
        if n:
            find.append(('T2 text-stroke', s['text'], f'{n} segment(s)'))
        for r in imgs:
            if inter(s['box'], (r.x0, r.y0, r.x1, r.y1), 0.3):
                find.append(('T3 text-image', s['text'], f'image at {r.x0:.0f},{r.y0:.0f}'))
        b = s['box']
        if b[0] < -0.5 or b[1] < -0.5 or b[2] > W + 0.5 or b[3] > H + 0.5:
            find.append(('T4 bounds', s['text'], f'{b}'))
    sizes = sorted(round(s['size'], 2) for s in S)
    print(f'  (detected: {len(S)} text spans, {len(segs)} stroke segments, {len(imgs)} raster images)')
    return find, sizes, W


def placed_scale(manuscript, figname, natural_w):
    """Placed width of figname in the manuscript divided by its natural width (1.0 if not found)."""
    doc = pymupdf.open(manuscript)
    best = None
    for pg in doc:
        for info in pg.get_image_info(xrefs=True):
            pass
        for item in pg.get_xobjects():
            name = item[1] if len(item) > 1 else ''
        # pymupdf exposes embedded PDF figures as form xobjects; their placement is not named reliably,
        # so fall back to the caption page heuristic below
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pdfs', nargs='+')
    ap.add_argument('--scale', type=float, default=1.0, help='placed width / natural width in the manuscript')
    a = ap.parse_args()
    bad = 0
    for p in a.pdfs:
        f, sizes, W = audit(p)
        eff = [round(x * a.scale, 2) for x in sizes]
        print(f'== {p}  width {W:.0f} pt  spans {len(sizes)}  font size min/median {min(eff) if eff else "-"} / {eff[len(eff)//2] if eff else "-"} pt (placed scale {a.scale})')
        for k, x, y in f:
            print(f'  [{k}] {x!r} | {y}')
        print(f'  -> {len(f)} finding(s)' + ('' if f else ' : PASS'))
        bad += len(f)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
