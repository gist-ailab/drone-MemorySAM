"""Qualitative figures of the IEEE Access manuscript (Python / matplotlib).

What this script reads (no number is typed into this file):
  figures/source_data/qual_scenes_deliver.csv
      scene table from analysis (inbox 20261030-175233, md5 efe580b4...). One row per selected scene with the
      per-image mIoU of every model and the selection rule in its header comments
      (delta = image mIoU(ours) - max(baselines); top 3 per condition + 1 reverse, no repeated sequence).
  <dump>/  = NAS analysis_logs/paper_qual_deliver_20261030/  (default; override with --dump)
      <model>/<scene>.png   raw train-ID prediction, 1042x1042, ignore = 255
      gt/<scene>.png        ground truth, same axis (legal v2, native 1042)
      input/<scene>__img.png RGB input;  manifest.csv = checkpoint path + md5 of every model
  <repo>/semseg/datasets/deliver.py   the DELIVER colour palette (parsed, not copied)

What it writes (default out dir = latex/figures):
  fig_adverse_deliver.pdf  + fig_adverse_deliver.json sidecar (input md5s, chosen scenes, design source).

How the figure is drawn:
  rows    = conditions night, fog, rain, low exposure; in each, the rank-1 "top" scene of the table.
  columns = RGB input, ground truth, DGFusion (official), CAFuser (retrained), Ours w/o high-res, Ours.
  Labels are coloured with the dataset palette; ignore pixels are black. With --errors, a prediction tile
  keeps its colour only where it disagrees with the ground truth (GT != 255) and is washed to 80 % white
  where it is correct, so the error regions stand out; output = fig_adverse_deliver_err.pdf. The number under each prediction
  is that model's image mIoU for the scene, read from the CSV (x100, one decimal). Label maps are resized
  with nearest neighbour, the RGB with Lanczos, to 420 px per tile before embedding.

Usage (from _paper/ieee-access-2026):  python figures/src/make_qual_figs.py [--dump DIR] [--out DIR]
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
REPO = PAPER.parents[1]
SCENES = PAPER / "figures" / "source_data" / "qual_scenes_deliver.csv"
DUMP = Path("/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/"
            "paper_qual_deliver_20261030")
DELIVER_PY = REPO / "semseg" / "datasets" / "deliver.py"

TEXT_W = 6.99  # inch, IEEE Access two-column text width (measured 2026-10-29)
TILE_PX = 420
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 7,
})
FG, MUTED = "#161616", "#525252"          # carbon@11.84.0 text roles
ROWS = [("night", "Night"), ("fog", "Fog"), ("rain", "Rain"), ("low_exposure", "Low exposure")]
COLS = [("rgb", "RGB input"), ("gt", "Ground truth"),
        ("DGFofficial", "DGFusion (official)"), ("CAFb", "CAFuser (retrained)"),
        ("Q2s821", "Ours w/o high-res"), ("A2s821", "Ours")]


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def palette():
    """Parse the 25 RGB triplets of DELIVER.PALETTE from deliver.py (order = train ID)."""
    src = DELIVER_PY.read_text(encoding="utf-8")
    block = src[src.index("PALETTE = torch.tensor(["):]
    block = block[:block.index("])") + 2]
    rgb = [tuple(int(v) for v in m) for m in re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+)\]", block)]
    assert len(rgb) == 25, len(rgb)
    lut = np.zeros((256, 3), np.uint8)       # 255 (ignore) and unused IDs stay black
    lut[:25] = np.array(rgb, np.uint8)
    return lut


def scene_file(scene_id):
    return scene_id.replace("/", "__") + ".png"


def tile_label(path, lut):
    a = np.array(Image.open(path))
    return np.array(Image.fromarray(lut[a]).resize((TILE_PX, TILE_PX), Image.NEAREST))


def tile_error(path, gt_path, lut):
    a = np.array(Image.open(path))
    g = np.array(Image.open(gt_path))
    col = lut[a].astype(np.float32)
    ok = (a == g) | (g == 255)
    col[ok] = 0.2 * col[ok] + 0.8 * 255.0
    return np.array(Image.fromarray(col.astype(np.uint8)).resize((TILE_PX, TILE_PX), Image.NEAREST))


def tile_rgb(path):
    return np.array(Image.open(path).convert("RGB").resize((TILE_PX, TILE_PX), Image.LANCZOS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--out", type=Path, default=PAPER / "latex" / "figures")
    ap.add_argument("--errors", action="store_true", help="wash out correct pixels (see docstring)")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(l for l in open(SCENES, encoding="utf-8") if not l.startswith("#"))]
    pick = {g: next(r for r in rows if r["group"] == g and r["kind"] == "top" and r["rank"] == "1")
            for g, _ in ROWS}
    lut = palette()

    nr, nc = len(ROWS), len(COLS)
    gap_w, gap_h, lab_w, head_h, foot_h = 0.035, 0.15, 0.17, 0.16, 0.0
    tile = (TEXT_W - lab_w - gap_w * (nc - 1)) / nc
    fig_h = head_h + nr * (tile + gap_h)
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    used = {}
    for i, (g, gname) in enumerate(ROWS):
        r = pick[g]
        f = scene_file(r["scene_id"])
        y0 = fig_h - head_h - i * (tile + gap_h) - tile
        for j, (key, cname) in enumerate(COLS):
            x0 = lab_w + j * (tile + gap_w)
            ax = fig.add_axes([x0 / TEXT_W, y0 / fig_h, tile / TEXT_W, tile / fig_h])
            if key == "rgb":
                p = a.dump / "input" / f.replace(".png", "__img.png")
                ax.imshow(tile_rgb(p), interpolation="none")
            else:
                p = a.dump / key / f
                if a.errors and key != "gt":
                    ax.imshow(tile_error(p, a.dump / "gt" / f, lut), interpolation="none")
                else:
                    ax.imshow(tile_label(p, lut), interpolation="none")
            used[str(p)] = md5(p)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color(MUTED)
            if i == 0:
                ax.set_title(cname, fontsize=7, color=FG, pad=3,
                             fontweight="bold" if key == "A2s821" else "normal")
            if key not in ("rgb", "gt"):
                v = 100 * float(r[f"mIoU_{key}"])
                ax.text(0.5, -0.025, f"{v:.1f}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color=FG, fontweight="bold" if key == "A2s821" else "normal")
        fig.text(0.02 / TEXT_W, (y0 + tile / 2) / fig_h, gname, rotation=90, ha="left", va="center",
                 fontsize=7, color=FG)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / ("fig_adverse_deliver_err" if a.errors else "fig_adverse_deliver")
    fig.savefig(f"{stem}.pdf", dpi=300, pad_inches=0.01)
    plt.close(fig)
    meta = {"design_source": "carbon@11.84.0 (text colours); DELIVER dataset palette for labels",
            "script": "figures/src/make_qual_figs.py",
            "scene_table": {str(SCENES.relative_to(REPO)): md5(SCENES)},
            "palette_source": {str(DELIVER_PY.relative_to(REPO)): md5(DELIVER_PY)},
            "manifest": {str(a.dump / "manifest.csv"): md5(a.dump / "manifest.csv")},
            "scenes": {g: pick[g]["scene_id"] for g, _ in ROWS},
            "tiles": used}
    json.dump(meta, open(f"{stem}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
