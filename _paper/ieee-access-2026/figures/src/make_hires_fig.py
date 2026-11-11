#!/usr/bin/env python3
"""Fig. hires (6_analysis): per-pixel feature magnitude of the intermediate-feature taps vs the high-resolution path at stride 4.

Panels come from the analysis run DRN-261104-02 (method verdict adopt 2026-11-11, inbox 20261111-132936):
NAS analysis_logs/hires_viz_a2s821_20261104/panels/<image>/{rgb,gt,c_norm,d_norm}.png
(A2 s821 epoch40_68.93_top1, md5 f05b54d4..., tools/viz_hires_pyramid.py; (c) = taps resampled to s4, (d) = high-resolution path s4).
The SimpleFPN panel (b) is left out: its levels end in LayerNorm2d, so its per-pixel norm is nearly constant (verdict note).
Usage: python3 figures/src/make_hires_fig.py   -> latex/figures/fig_hires.pdf (+ .json provenance)
"""
import hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parents[2]
SRC = Path("/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/hires_viz_a2s821_20261104/panels")
SCENES = [("Night", "img/night/test/MAP_10_point5/001200_rgb_front"),
          ("Cloud", "img/cloud/test/MAP_10_point12/013150_rgb_front")]
COLS = [("rgb", "RGB input"), ("gt", "Ground truth"), ("c_norm", "Taps (stride 16)"), ("d_norm", "High-res. path")]
TEXT_W = 7.0
FG, EDGE = "#3C3838", "#8F8B8B"   # carbon@11.84.0 warm-gray-80 / warm-gray-50 (design-source: carbon@11.84.0)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42})


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    n = len(SCENES) * len(COLS)
    gap, lab_h, grp_gap = 0.03, 0.30, 0.10
    tile = (TEXT_W - gap * (n - 1) - grp_gap) / n
    fig_h = tile + lab_h
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    used = {}
    x = 0.0
    for si, (sname, sid) in enumerate(SCENES):
        for ci, (key, title) in enumerate(COLS):
            p = SRC / sid / f"{key}.png"
            ax = fig.add_axes([x / TEXT_W, 0, tile / TEXT_W, tile / fig_h])
            ax.imshow(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS), interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color(EDGE)
            ax.set_title(title, fontsize=6.5, color=FG, pad=2)
            used[str(p)] = md5(p)
            x += tile + gap
        fig.text((x - gap - 2 * tile - gap / 2) / TEXT_W, (tile + lab_h - 0.06) / fig_h, sname, ha="center", va="top",
                 fontsize=7, color=FG, fontweight="bold")
        x += grp_gap
    out = HERE / "latex/figures/fig_hires.pdf"
    fig.savefig(out)
    json.dump({"source": str(SRC), "scenes": SCENES, "inputs_md5": used, "design_source": "carbon@11.84.0"},
              open(out.with_suffix(".json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
