#!/usr/bin/env python3
"""
Analyze per-domain val.py logs -> per-domain x per-class IoU matrix + failure
classification. Reusable for any model (P28/P29/P30/...): it only reads the
per-condition logs produced by tools/eval_per_domain.py (or any val.py --detailed
/--macvi test log that prints the per-class grid table).

Classifies each class into:
  - domain-invariant dead : max IoU across domains < --dead-thresh  (structural ceiling)
  - domain-sensitive      : spread(max-min) > --spread-thresh       (real per-domain failure)
  - robust                : otherwise

Also reports per-domain mIoU and its spread (small spread => val/test gap is NOT
weather/condition shift, but per-class transfer failure).

Example
-------
python tools/analyze_per_domain.py --logs-dir ~/eval_P28_out \
  --label ep178=ep178_test55.27 --label ep100=ep100_val63.4 \
  --out analysis_P28_per_domain.md
"""
import argparse, re
from pathlib import Path

ROW = re.compile(r"\|\s*([A-Za-z][A-Za-z ]*?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")

def parse_log(logf):
    cls_iou, mean = {}, None
    if not Path(logf).exists():
        return cls_iou, mean
    for ln in open(logf, encoding='utf-8', errors='ignore'):
        m = ROW.match(ln)
        if not m:
            continue
        cls, iou = m.group(1).strip(), float(m.group(2))
        if cls == 'Class':
            continue
        if cls == 'Mean':
            mean = iou
        else:
            cls_iou[cls] = iou
    return cls_iou, mean

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs-dir', required=True)
    ap.add_argument('--label', action='append', required=True, metavar='NAME=PREFIX',
                    help="label=log-filename-prefix (file = <prefix>__<condition>.log); repeatable")
    ap.add_argument('--conditions', default='cloud,fog,night,rain,sun')
    ap.add_argument('--dead-thresh', type=float, default=10.0)
    ap.add_argument('--spread-thresh', type=float, default=12.0)
    ap.add_argument('--gap-thresh', type=float, default=6.0,
                    help="per-domain mIoU spread below this => 'not domain shift'")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
    labels = [(s.split('=', 1)[0], s.split('=', 1)[1]) for s in args.label]
    d = Path(args.logs_dir).expanduser()
    L = []
    def emit(s=''):
        L.append(s)

    for name, prefix in labels:
        data, means = {}, {}
        for c in conds:
            cls_iou, mean = parse_log(d / f"{prefix}__{c}.log")
            data[c], means[c] = cls_iou, mean
        classes = list(next((v for v in data.values() if v), {}).keys())
        emit(f"### {name} — per-domain × per-class IoU")
        emit()
        emit("| class | " + " | ".join(conds) + " | min | spread |")
        emit("|" + "---|" * (len(conds) + 3))
        dead, sensitive, robust = [], [], []
        for cls in classes:
            vals = [data[c].get(cls) for c in conds]
            vv = [v for v in vals if v is not None]
            if not vv:
                continue
            mn, mx = min(vv), max(vv); sp = mx - mn
            emit(f"| {cls} | " + " | ".join(f"{v:.1f}" if v is not None else "—" for v in vals)
                 + f" | {mn:.1f} | {sp:.1f} |")
            if mx < args.dead_thresh:
                dead.append((cls, mx))
            elif sp > args.spread_thresh:
                worst = conds[vals.index(mn)]
                sensitive.append((cls, sp, worst, mn))
            else:
                robust.append((cls, mn))
        mvals = [means[c] for c in conds]
        emit("| **mIoU** | " + " | ".join(f"{m:.2f}" if m is not None else "—" for m in mvals) + " | | |")
        emit()
        mm = [m for m in mvals if m is not None]
        gap = (max(mm) - min(mm)) if mm else float('nan')
        verdict = "갭은 도메인시프트 아님 → per-class transfer 문제" if gap < args.gap_thresh else "도메인시프트가 갭에 기여"
        emit(f"- **per-domain mIoU spread = {gap:.2f}** ({verdict})")
        emit(f"- **도메인-불변 사망** (max IoU < {args.dead_thresh:g}): "
             + (", ".join(f"{c}({m:.1f})" for c, m in dead) or "없음"))
        emit(f"- **도메인-민감** (spread > {args.spread_thresh:g}): "
             + (", ".join(f"{c}(±{sp:.0f}, 최약 {w} {mn:.0f})" for c, sp, w, mn in sensitive) or "없음"))
        emit(f"- **강건**: " + (", ".join(c for c, _ in robust) or "없음"))
        emit()

    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding='utf-8')
        print(f"[written] {args.out}")

if __name__ == '__main__':
    main()
