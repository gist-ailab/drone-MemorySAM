#!/usr/bin/env python3
"""Supplementary draft tables (revision reserve, user 2026-11-13). Writes supplementary/tables_generated.tex.

Sources (all read-only):
  S1 EMM per subset : NAS analysis_logs/mmssb_table5_20261109/*/eval_txt/{s821,s902,s903,cmnext}/eval_*.txt
                      (public benchmark code, DELIVER val 2005 images; the file whose table header is 'Missing Modalities')
  S2/S3 MUSES test  : figures/source_data/muses_test_server/muses_test_server.csv (server outputs pasted by the user)
                      MM SAM-adapter per condition: arXiv 2509.10408 Table 6 (values typed below)
  S4 MCubeS classes : NAS analysis_logs/mcubes_rescore_20261104/class_table.md
  S5 depth removal  : figures/source_data/fig_dependence.csv (test split)
"""
import csv, glob, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAS = Path('/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs')
out = []


def bold_best(vals, fmt='{:.2f}'):
    """Return formatted strings with the best (max) in bold and the second underlined."""
    order = sorted(set(v for v in vals if v is not None), reverse=True)
    res = []
    for v in vals:
        if v is None:
            res.append('--')
        elif v == order[0]:
            res.append(r'\textbf{' + fmt.format(v) + '}')
        elif len(vals) > 2 and len(order) > 1 and v == order[1]:
            res.append(r'\underline{' + fmt.format(v) + '}')
        else:
            res.append(fmt.format(v))
    return res


# ---------------- S1: EMM per subset ----------------
def emm_table(path):
    t = Path(path).read_text()
    if '\nMissing Modalities' not in t:
        return None
    d = {}
    for m in re.finditer(r"^(\(\)|\('[^)]*\))\s+([\d.]+)", t, flags=re.M):
        d[m.group(1)] = float(m.group(2))
    for key, lab in (('Mean', 'Mean'), ('Expectation_p0.2', 'E(.2)'), ('Expectation_p0.1', 'E(.1)'), ('Expectation_p0.05', 'E(.05)')):
        m = re.search(rf'^{re.escape(key)}\s+([\d.]+)', t, flags=re.M)
        d[lab] = float(m.group(1))
    return d


emm = {}
for who in ('s821', 's902', 's903', 'cmnext'):
    for f in sorted(glob.glob(str(NAS / 'mmssb_table5_20261109' / '*' / 'eval_txt' / who / 'eval_*.txt'))):
        r = emm_table(f)
        if r:
            emm[who] = r
            break
assert set(emm) == {'s821', 's902', 's903', 'cmnext'}, emm.keys()
keys = list(emm['s821'].keys())
name = {"()": "none (clean)"}
out.append(r"""\begin{table}[t]
\caption{Entire-missing evaluation on the DELIVER validation set per removed-sensor subset (mIoU, \%), public benchmark code~\cite{mmssbench2025}. R: RGB, D: depth, E: event, L: LiDAR.}
\label{tab:s_emm}
\centering\small
\begin{tabular}{lcc}
\toprule
Removed sensors & CMNeXt (released) & Ours (mean of 3 runs) \\
\midrule""")
for k in keys:
    ours = sum(emm[s][k] for s in ('s821', 's902', 's903')) / 3
    lab = name.get(k, k if k.startswith('E(') else k.strip("()").replace("'", '').replace(',', ' +').replace('RGB', 'R').strip(' +'))
    if k == 'Mean':
        out.append(r'\midrule')
        lab = 'Average over 15 subsets'
    a, b = bold_best([emm['cmnext'][k], ours])
    out.append(f'{lab} & {a} & {b} \\\\')
out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S2: MUSES test per condition ----------------
rows = list(csv.DictReader(open(ROOT / 'figures/source_data/muses_test_server/muses_test_server.csv')))
M = {}
for r in rows:
    M.setdefault(r['model'], {})[r['condition']] = r
mods = [('A2rgbl_s2_1024_20261110', 'Ours, RGB-L'), ('A2rgbe_s2_1536_20261111', 'Ours, RGB-E'), ('A2m4_s2_1536_20261109', 'Ours, RGB-L-R-E')]
# MM SAM-adapter, arXiv 2509.10408 Table 6 (All, Day, Night, Clear, Fog, Rain, Snow)
mmsam = {'RGB-L': [81.07, 83.34, 74.97, 80.82, 74.12, 80.00, 80.85], 'RGB-E': [79.92, 83.39, 72.38, 81.52, 68.97, 78.92, 78.72]}
conds = ['all', 'day', 'night', 'clear', 'fog', 'rain', 'snow']
out.append(r"""\begin{table*}[t]
\caption{MUSES test set per condition (mIoU, \%, official evaluation server). MM SAM-adapter values are from~\cite{mmsamadapter2025} (Table~6); ours are one submission each.}
\label{tab:s_muses_cond}
\centering\small
\begin{tabular}{lccccccc}
\toprule
Model & All & Day & Night & Clear & Fog & Rain & Snow \\
\midrule""")
table = [('MM SAM-adapter, RGB-L', mmsam['RGB-L']), ('MM SAM-adapter, RGB-E', mmsam['RGB-E'])]
for m, lab in mods:
    table.append((lab, [float(M[m][c]['mIoU']) for c in conds]))
cols = list(zip(*[v for _, v in table]))
fmt_cols = [bold_best(list(c)) for c in cols]
for i, (lab, _) in enumerate(table):
    out.append(lab + ' & ' + ' & '.join(fmt_cols[j][i] for j in range(len(conds))) + r' \\')
    if i == 1:
        out.append(r'\midrule')
out.append(r"""\bottomrule
\end{tabular}
\vspace{4pt}

\begin{tabular}{lcccccccc}
\toprule
Model & Clear day & Clear night & Fog day & Fog night & Rain day & Rain night & Snow day & Snow night \\
\midrule""")
combo = ['clear_day', 'clear_night', 'fog_day', 'fog_night', 'rain_day', 'rain_night', 'snow_day', 'snow_night']
vals = [[float(M[m][c]['mIoU']) for c in combo] for m, _ in mods]
fc = [bold_best(list(c)) for c in zip(*vals)]
for i, (m, lab) in enumerate(mods):
    out.append(lab + ' & ' + ' & '.join(fc[j][i] for j in range(len(combo))) + r' \\')
out.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------- S3: MUSES test per class ----------------
CL = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
out.append(r"""\begin{table}[t]
\caption{MUSES test set per class (IoU, \%, official evaluation server, all conditions).}
\label{tab:s_muses_cls}
\centering\small
\begin{tabular}{lccc}
\toprule
Class & RGB-L & RGB-E & RGB-L-R-E \\
\midrule""")
for c in CL + ['mIoU']:
    key = c if c != 'mIoU' else 'mIoU'
    v = [float(M[m]['all'][key]) for m, _ in mods]
    if c == 'mIoU':
        out.append(r'\midrule')
    f = bold_best(v)
    out.append(f'{c.capitalize() if c != "mIoU" else "mIoU"} & ' + ' & '.join(f) + r' \\')
out.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S4: MCubeS per class ----------------
md = (NAS / 'mcubes_rescore_20261104' / 'class_table.md').read_text().splitlines()
cls_rows = [l for l in md if l.startswith('| ') and not l.startswith('| 모델') and not l.startswith('| 클래스') and l.count('|') > 8]
out.append(r"""\begin{table}[t]
\caption{MCubeS test set per class (IoU, \%, mean of two runs): the clean-trained model without the high-resolution path and our full model.}
\label{tab:s_mcubes_cls}
\centering\small
\begin{tabular}{lcc}
\toprule
Class & Clean-trained, no high-res. & Ours \\
\midrule""")
for l in cls_rows:
    p = [x.strip() for x in l.strip('|').split('|')]
    c, ours, e1 = p[0], float(p[3]), float(p[5])
    f = bold_best([e1, ours])
    out.append(f"{c.replace('_', ' ').capitalize()} & {f[0]} & {f[1]} \\\\")
out.append(r"""\midrule
mIoU & 58.78 & \textbf{59.78} \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S5: depth removal per condition ----------------
dep = list(csv.DictReader(open(ROOT / 'figures/source_data/fig_dependence.csv')))
order = ['Sun', 'Cloud', 'Rain', 'Fog', 'Night', 'Low exp.']
models = []
for r in dep:
    if r['model'] not in models:
        models.append(r['model'])
D = {(r['model'], r['condition']): float(r['drop_without_depth']) for r in dep}
out.append(r"""\begin{table*}[t]
\caption{Drop in mIoU (points) on the DELIVER test set when depth is removed (zero-filled), per condition; lower is better. Data of Fig.~7 of the main paper.}
\label{tab:s_depth}
\centering\small
\begin{tabular}{lcccccc}
\toprule
Model & """ + ' & '.join(order) + r""" \\
\midrule""")
lab = {'DGFusion (reproduced)': 'DGFusion (our reproduction)', 'Ours w/o C2D2': 'Ours, clean training, no high-res. path (2 runs)', 'Ours': 'Ours (3 runs)'}
for m in models:
    out.append(lab.get(m, m) + ' & ' + ' & '.join(f"{D[(m, c)]:.1f}" for c in order) + r' \\')
out.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")

(HERE / 'tables_generated.tex').write_text('% generated by make_supp_tables.py — do not edit by hand\n' + '\n'.join(out))
print('ok', len(out))
