"""Data figures for the IEEE Sensors Journal manuscript (Python / matplotlib).

Every number below is copied from a canonical record; the source is written next to it.
Values not yet measured are None and are drawn as an explicit "pending" mark, never interpolated.
Colours: carbon@11.84.0 (labdesign build/py/carbon_palette.py). Fonts: TeX Gyre Heros (Helvetica
metric clone, matches the TikZ figures). Single-column width 88.9 mm (3.5 in).

Usage: python make_data_figs.py <out_dir> <perclass_csv>
  perclass_csv = repo .claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv
Outputs: fig_dependence.pdf, fig_robust.pdf, fig_thin.pdf (+ .svg, + .json sidecar).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COL_W = 3.5  # inch, IEEE single column
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
# model families: baselines = neutral greys (line style separates them); ours = blue ramp, hero darkest
STYLE = {
    "DGFusion (official)":   dict(color=MUTED, ls=(0, (1.2, 1.2)), marker="o", mfc="white"),
    "DGFusion (retrained)":  dict(color=MUTED, ls="-", marker="o", mfc=MUTED),
    "CAFuser (retrained)":   dict(color=FG, ls=(0, (3, 1.5)), marker="s", mfc="white"),
    "Ours, clean training":  dict(color=SEQ[4], ls="-", marker="^", mfc=SEQ[4]),
    "Ours w/o high-res":     dict(color=SEQ[7], ls="-", marker="D", mfc=SEQ[7]),
    "Ours":                  dict(color=SEQ[9], ls="-", marker="o", mfc=SEQ[9], lw=1.8),
}


def label_axes(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def pending(ax, x, y, text="pending"):
    ax.text(x, y, text, fontsize=5.5, color=DANGER, ha="center", va="center", style="italic")


# ---------------------------------------------------------------------------
# Fig. 3 (manuscript numbering): sensor dependence by condition — drop when depth is removed.
# Source: experiments/judgment-ledger.md 2026-09-30 row (oracle_cond probe).
#   Ours, clean training = E1 confirmed run, val.  Ours w/o high-res = Q2 seed 821, val
#   (only sun/night/low exposure listed).  DGFusion (official) = same row, TEST split (mismatch).
# ---------------------------------------------------------------------------
COND = ["Sun", "Cloud", "Rain", "Fog", "Night", "Low exp."]
DEP = {
    "DGFusion (official)":  [23.1, 24.9, 27.6, 25.9, 32.6, 37.1],
    "Ours, clean training": [23.5, 23.9, 23.7, 26.8, 28.7, 38.7],
    "Ours w/o high-res":    [3.9, None, None, None, 10.1, 13.4],
    "Ours":                 [None] * 6,
}


def fig_dependence(out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(COND))
    for name, vals in DEP.items():
        st = dict(STYLE[name])
        lw = st.pop("lw", 1.2)
        v = np.array([np.nan if a is None else a for a in vals], float)
        if np.all(np.isnan(v)):
            continue
        ax.plot(x, v, lw=lw, markeredgecolor=st["color"], label=name, **st, zorder=3)
    ax.set_xticks(x, COND)
    ax.set_ylabel("mIoU drop without depth (points)")
    ax.set_ylim(0, 42)
    ax.yaxis.grid(True, color=BORDER, lw=0.4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    h, l = ax.get_legend_handles_labels()
    l = [x.replace("DGFusion (official)", "DGFusion (official)$^{\\dagger}$")
          .replace("Ours w/o high-res", "Ours, two-pass training") for x in l]
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.2, 0.04), handlelength=2.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, out / "fig_dependence")


# ---------------------------------------------------------------------------
# Fig. robust: (a) salt-and-pepper noise on all sensors, test, same protocol;
# (b) one sensor left (EMM single-sensor subsets), test.
# Sources: (a) clean/S&P — judgment-ledger 2026-10-01 row (DGFusion official / retrained),
#   tasks/DRN-261019-03/analysis.md (Ours w/o high-res = Q2, Ours = A2; 3-seed means).
#   (b) judgment-ledger 2026-10-01 row: DGFusion img/depth/event/LiDAR only 12.1/43.1/1.2/3.2
#   (official) and 49.0/44.5/25.0/25.0 (retrained); Q2 3 seeds 53.9/46.0/31.0/22.9.
#   Ours (A2) single-sensor values not in a canonical record yet -> pending.
# ---------------------------------------------------------------------------
NOISE = [0.0, 0.05, 0.10, 0.20]
NM = {
    "DGFusion (official)":  [55.68, 37.97, 16.71, 5.52],
    "DGFusion (retrained)": [54.77, 48.89, 40.51, 23.34],
    "Ours w/o high-res":    [56.59, 53.25, 51.64, 48.79],
    "Ours":                 [57.74, 53.11, 51.36, 47.85],
}
SINGLE = ["RGB", "Depth", "Event", "LiDAR"]
ONE = {
    "DGFusion (official)":  [12.1, 43.1, 1.2, 3.2],
    "DGFusion (retrained)": [49.0, 44.5, 25.0, 25.0],
    "Ours w/o high-res":    [53.9, 46.0, 31.0, 22.9],
    "Ours":                 [None, None, None, None],
}


def fig_robust(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for name, vals in NM.items():
        st = dict(STYLE[name]); lw = st.pop("lw", 1.2)
        a.plot(NOISE, vals, lw=lw, markeredgecolor=st["color"], **st, zorder=3)
    a.set_xticks(NOISE, ["0", ".05", ".10", ".20"])
    a.set_xlabel("Salt-and-pepper density")
    a.set_ylabel("mIoU (%)")
    a.set_ylim(0, 62)
    a.set_xlim(-0.012, 0.212)
    label_axes(a, "a")

    names = list(ONE.keys())
    w = 0.2
    xs = np.arange(len(SINGLE))
    for i, name in enumerate(names):
        st = STYLE[name]
        vals = ONE[name]
        for j, v in enumerate(vals):
            xx = xs[j] + (i - 1.5) * w
            if v is None:
                b.bar(xx, 4, w * 0.9, color="white", edgecolor=DANGER, lw=0.6, ls=(0, (1.5, 1)), zorder=2)
                continue
            face = "white" if st["mfc"] == "white" else st["color"]
            b.bar(xx, v, w * 0.9, color=face, edgecolor=st["color"], lw=0.6, zorder=2)
    b.set_xticks(xs, [f"{s}\nonly" for s in SINGLE])
    b.set_ylim(0, 62)
    b.set_yticklabels([])
    b.tick_params(axis="y", left=False)
    label_axes(b, "b")
    # shared legend under both panels
    handles = [mpl.lines.Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"], marker=STYLE[n]["marker"],
                                mfc=STYLE[n]["mfc"], mec=STYLE[n]["color"], lw=STYLE[n].get("lw", 1.2))
               for n in names]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor=DANGER, lw=0.6, ls=(0, (1.5, 1))))
    fig.legend(handles, ["DGFusion (official)", "DGFusion (retrained)", "Ours w/o high-res", "Ours",
                         "Ours, pending"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
    save(fig, out / "fig_robust")


# ---------------------------------------------------------------------------
# Fig. thin: per-class gain of the high-resolution path, paired by seed (A2 - Q2, test).
# Source: experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv (3 seeds, legal v2 test 1024).
# ---------------------------------------------------------------------------
THIN = {"Pole", "TrafficLight", "Pedestrian", "Static"}


def fig_thin(out, csv_path):
    rows = list(csv.DictReader(open(csv_path)))
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
    save(fig, out / "fig_thin")


def save(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    json.dump({"design_source": "carbon@11.84.0", "script": "figures/src/make_data_figs.py"},
              open(f"{stem}.json", "w"))
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fig_dependence(out)
    fig_robust(out)
    fig_thin(out, sys.argv[2])
