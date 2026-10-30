"""Data figures of the IEEE Access manuscript (Python / matplotlib).

What this script reads (no number is typed into this file):
  figures/source_data/fig_dependence.csv    -> Fig. dependence (drop in mIoU when depth is removed, per condition)
  figures/source_data/fig_robust_noise.csv  -> Fig. robust (a)  (mIoU vs salt-and-pepper density)
  figures/source_data/fig_robust_single.csv -> Fig. robust (b)  (mIoU when only one sensor is left)
  <repo>/.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv
                                     -> Fig. thin (per-class gain of the high-resolution path, paired by seed)
Every row of the three figures/source_data CSVs carries source_file and source_locator columns that point to the
canonical record (judgment-ledger row, analysis document line). An empty value means "not measured yet";
it is drawn as a red dashed "pending" bar and never interpolated.

What it writes (default out dir = latex/figures):
  fig_dependence.pdf, fig_robust.pdf, fig_thin.pdf and one <name>.json sidecar per figure that records the
  input files with their md5, the design source, and this script.

How each figure is drawn:
  fig_dependence: one line per model over the six DELIVER conditions; y = drop in mIoU (points). Missing
                  conditions are left as gaps. Legend placed in the empty lower band.
  fig_robust:     (a) one line per model over S&P density 0/.05/.10/.20, y = mIoU.
                  (b) grouped bars, one group per remaining sensor, one bar per model.
  fig_thin:       one bar per class = mean of the three seed differences (d821, d902, d903), black dots =
                  the three single-seed differences; classes sorted by the mean; thin classes in bold.

Colours: carbon@11.84.0 (~/labdesign/build/py/carbon_palette.py). Fonts: TeX Gyre Heros (Helvetica clone).
Width: IEEE Access single column 3.36 in (242.7 pt, measured in the class on 2026-10-29).

Usage (from the paper folder _paper/ieee-access-2026 in the repo):
  python figures/src/make_data_figs.py [out_dir]
"""
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[2]            # _paper/ieee-access-2026
DATA = PAPER / "figures" / "source_data"
REPO = PAPER.parents[1]                                 # repo root
PERCLASS = REPO / ".claude_logs" / "experiments" / "analysis" / "DRN-261018-01" / "a2_vs_q2_perclass_3seed.csv"

COL_W = 3.36  # inch, IEEE Access single column (242.7 pt, measured 2026-10-29)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.titlesize": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 4.5, "ytick.major.pad": 3,
    "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 3.5,
})

# carbon@11.84.0 roles / sequential (blue) ramp
FG, MUTED, BORDER = "#161616", "#525252", "#c6c6c6"
SEQ = ["#ffffff", "#e4eeff", "#cadcff", "#afcbff", "#94b9ff",
       "#7aa8fe", "#5f96fe", "#4485fe", "#2a73fe", "#0f62fe"]
DANGER = "#da1e28"
# model families: baselines = neutral greys (line style separates them); ours = blue ramp, final model darkest
STYLE = {
    "DGFusion (official)":   dict(color=MUTED, ls=(0, (1.2, 1.2)), marker="o", mfc="white"),
    "DGFusion (reproduced)": dict(color=MUTED, ls=(0, (1.2, 1.2)), marker="s", mfc="white"),
    "DGFusion (retrained)":  dict(color=MUTED, ls="-", marker="o", mfc=MUTED),
    "CAFuser (retrained)":   dict(color=FG, ls=(0, (3, 1.5)), marker="s", mfc="white"),
    "Ours w/o C2D2":         dict(color=SEQ[4], ls="-", marker="^", mfc=SEQ[4]),
    "Ours w/o high-res":     dict(color=SEQ[7], ls="-", marker="D", mfc=SEQ[7]),
    "Ours":                  dict(color=SEQ[9], ls="-", marker="o", mfc=SEQ[9], lw=1.8),
}
CONDITIONS = ["Sun", "Cloud", "Rain", "Fog", "Night", "Low exp."]
SENSORS = ["RGB", "Depth", "Event", "LiDAR"]
THIN = {"Pole", "TrafficLight", "Pedestrian", "Static"}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_long(path, key, col, value):
    """Read a long-format CSV into {model: {col_value: float or None}} keeping file order."""
    out = OrderedDict()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        v = r[value].strip()
        out.setdefault(r[key], OrderedDict())[r[col]] = float(v) if v else None
    return out


def label_axes(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def save(fig, stem, inputs):
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    meta = {"design_source": "carbon@11.84.0", "script": "figures/src/make_data_figs.py",
            "inputs": {str(Path(p).relative_to(REPO)): md5(p) for p in inputs}}
    json.dump(meta, open(f"{stem}.json", "w"), indent=1, ensure_ascii=False)
    plt.close(fig)


def fig_dependence(out):
    src = DATA / "fig_dependence.csv"
    dep = read_long(src, "model", "condition", "drop_without_depth")
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(CONDITIONS))
    for name, by_cond in dep.items():
        st = dict(STYLE[name]); lw = st.pop("lw", 1.2)
        v = np.array([np.nan if by_cond.get(c) is None else by_cond[c] for c in CONDITIONS], float)
        if np.all(np.isnan(v)):
            continue
        ax.plot(x, v, lw=lw, markeredgecolor=st["color"], label=name, **st, zorder=3)
    ax.set_xticks(x, CONDITIONS)
    ax.set_ylabel("mIoU drop without depth (points)")
    ax.set_ylim(0, 42)
    h, l = ax.get_legend_handles_labels()
    l = [s.replace("DGFusion (reproduced)", "DGFusion (reproduced), test")
          .replace("Ours w/o high-res", "Ours, C2D2 (w/o high-res)") for s in l]
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.2, 0.04), handlelength=2.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, out / "fig_dependence", [src])


def fig_robust(out):
    src_a = DATA / "fig_robust_noise.csv"
    src_b = DATA / "fig_robust_single.csv"
    noise = read_long(src_a, "model", "sp_density", "miou")
    single = read_long(src_b, "model", "sensor_left", "miou")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for name, by_d in noise.items():
        st = dict(STYLE[name]); lw = st.pop("lw", 1.2)
        xs = [float(k) for k in by_d]
        a.plot(xs, list(by_d.values()), lw=lw, markeredgecolor=st["color"], **st, zorder=3)
    ticks = sorted({float(k) for d in noise.values() for k in d})
    a.set_xticks(ticks, ["0" if t == 0 else f"{t:.2f}".lstrip("0") for t in ticks])
    a.set_xlabel("Salt-and-pepper density")
    a.set_ylabel("mIoU (%)")
    a.set_ylim(0, 62)
    a.set_xlim(-0.012, 0.212)
    label_axes(a, "a")

    names = list(single.keys())
    w = 0.8 / len(names)
    xs = np.arange(len(SENSORS))
    for i, name in enumerate(names):
        st = STYLE[name]
        for j, s in enumerate(SENSORS):
            xx = xs[j] + (i - (len(names) - 1) / 2) * w
            v = single[name].get(s)
            if v is None:   # not measured yet: red dashed stub, never a value
                b.bar(xx, 4, w * 0.9, color="white", edgecolor=DANGER, lw=0.6, ls=(0, (1.5, 1)), zorder=2)
                continue
            face = "white" if st["mfc"] == "white" else st["color"]
            b.bar(xx, v, w * 0.9, color=face, edgecolor=st["color"], lw=0.6, zorder=2)
    b.set_xticks(xs, [f"{s}\nonly" for s in SENSORS])
    b.set_ylim(0, 62)
    b.set_yticklabels([])
    b.tick_params(axis="y", left=False)
    label_axes(b, "b")
    handles = [mpl.lines.Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"], marker=STYLE[n]["marker"],
                                mfc=STYLE[n]["mfc"], mec=STYLE[n]["color"], lw=STYLE[n].get("lw", 1.2))
               for n in names]
    labels = list(names)
    if any(v is None for d in single.values() for v in d.values()):
        handles.append(mpl.patches.Patch(facecolor="white", edgecolor=DANGER, lw=0.6, ls=(0, (1.5, 1))))
        labels.append("Ours, pending")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
    save(fig, out / "fig_robust", [src_a, src_b])


def fig_thin(out):
    rows = list(csv.DictReader(open(PERCLASS, encoding="utf-8")))
    data = [(r["class"], [float(r["d821"]), float(r["d902"]), float(r["d903"])]) for r in rows]
    data.sort(key=lambda t: np.mean(t[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = np.arange(len(data))
    for i, (c, d) in enumerate(data):
        m = np.mean(d)
        col = SEQ[9] if c in THIN else SEQ[5]
        ax.bar(i, m, 0.7, color=col if m >= 0 else "white", edgecolor=col, lw=0.6, zorder=2)
        ax.scatter([i] * 3, d, s=4, color=FG, zorder=3, lw=0)
    ax.axhline(0, color=FG, lw=0.6)
    ax.set_xticks(x, [c for c, _ in data], rotation=90, fontsize=5.8)
    ax.set_xlim(-0.7, len(data) - 0.3)
    for t, (c, _) in zip(ax.get_xticklabels(), data):
        if c in THIN:
            t.set_fontweight("bold")
    ax.set_ylabel("Gain from high-res path\n(mIoU points)")
    ax.set_ylim(-21, 21)
    ax.text(0.99, 0.99, "bar: mean of 3 seeds, dot: one seed\nbold: thin classes",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED, linespacing=1.3)
    fig.tight_layout(pad=0.3)
    save(fig, out / "fig_thin", [PERCLASS])


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / "latex" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dependence(out)
    fig_robust(out)
    fig_thin(out)
