"""Qualitative figures of the IEEE Access manuscript (Python / matplotlib).

What this script reads (no number and no scene is typed into this file):
  figures/qualbank/selection.yaml     which scenes go into which figure (the file to edit for a change)
  figures/qualbank/registry.yaml      datasets, models, bank location, input paths
  figures/source_data/qualbank/<ds>_candidates.csv, <ds>_per_image.csv
                                      ranked candidates and every image's mIoU per model (qualbank.py candidates)
  <bank>/<ds>_<split>/<model>/<image_id>.png, gt/<image_id>.png
                                      whole-split predictions at GT resolution (qualbank.py ingest)
  <input_root>/<image_id>.png         the RGB tile;  <repo>/<classes_from> the class names and colours (parsed)

What it writes (default out dir = latex/figures):
  <figure>.pdf + <figure>.json for every figure in selection.yaml (the json lists each tile's file and md5,
  the chosen image ids with their delta, and the md5 of selection/registry/candidates), and
  fig_class_legend_<ds>.pdf for every dataset used.

How a figure is drawn: one row per selected scene, one column per selection `columns` entry. Labels use the
dataset palette, ignore pixels are black. With errors: true a prediction tile keeps its colour only where it
disagrees with the ground truth (GT != 255) and is washed to 80 % white where it is correct. The number under a
prediction tile is that model's image mIoU (x100) from <ds>_per_image.csv. Label maps are resized with nearest
neighbour and the RGB with Lanczos, to TILE_PX per tile.

Usage (from _paper/ieee-access-2026):  python figures/src/make_qual_figs.py [--only NAME ...] [--out DIR]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
REPO = PAPER.parents[1]
QB = PAPER / "figures" / "qualbank"
SRC = PAPER / "figures" / "source_data" / "qualbank"
sys.path.insert(0, str(QB))
import qualbank  # noqa: E402

TEXT_W = 6.99  # inch, IEEE Access two-column text width (measured 2026-10-29)
TILE_PX = 420
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 7,
})
FG, MUTED = "#161616", "#525252"          # carbon@11.84.0 text roles


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    return [r for r in csv.DictReader(l for l in open(p, encoding="utf-8") if not l.startswith("#"))]


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
    im = Image.open(path)
    if im.mode in ("I", "I;16", "F", "L"):         # depth / LiDAR maps: stretch 1-99 percentile to grey
        a = np.array(im).astype(np.float32)
        v = a[a > 0]
        lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0.0, 1.0)
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * (a > 0)
        im = Image.fromarray((a * 255).astype(np.uint8))
    return np.array(im.convert("RGB").resize((TILE_PX, TILE_PX), Image.LANCZOS))


def resolve(row, cand):
    if "image" in row:
        return row["image"]
    p = row["pick"]
    hit = [c for c in cand if c["group"] == p["group"] and c["kind"] == p["kind"] and int(c["rank"]) == int(p["rank"])]
    assert hit, f"no candidate {p}"
    return hit[0]["image_id"]


def input_path(ds, image_id, key="rgb"):
    rel = image_id
    for old, new in ds["inputs"][key]:
        rel = rel.replace(old, new)
    return Path(ds["input_root"]) / f"{rel}.png"


def draw(name, spec, reg, out):
    dname = spec["dataset"]
    ds = reg["datasets"][dname]
    bank = qualbank.bank_dir(reg, dname)
    lut, _ = qualbank.palette_of(ds)
    cand = read_csv(SRC / f"{dname}_candidates.csv")
    per = {r["image_id"]: r for r in read_csv(SRC / f"{dname}_per_image.csv")}
    cols = spec["columns"]
    sel = [(resolve(r, cand), r["label"]) for r in spec["rows"]]
    nr, nc = len(sel), len(cols)
    gap_w, gap_h, lab_w, head_h = 0.035, 0.15, 0.17, 0.16
    tile = (TEXT_W - lab_w - gap_w * (nc - 1)) / nc
    fig_h = head_h + nr * (tile + gap_h)
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    used = {}
    for i, (iid, rname) in enumerate(sel):
        y0 = fig_h - head_h - i * (tile + gap_h) - tile
        for j, key in enumerate(cols):
            x0 = lab_w + j * (tile + gap_w)
            ax = fig.add_axes([x0 / TEXT_W, y0 / fig_h, tile / TEXT_W, tile / fig_h])
            gt = bank / "gt" / f"{iid}.png"
            if key in ("rgb", "input"):                # "input" = the row's `input` sensor (default rgb)
                p = input_path(ds, iid, "rgb" if key == "rgb" else spec["rows"][i].get("input", "rgb"))
                ax.imshow(tile_rgb(p), interpolation="none")
            elif key == "gt":
                p = gt
                ax.imshow(tile_label(p, lut), interpolation="none")
            else:
                p = bank / key / f"{iid}.png"
                ax.imshow(tile_error(p, gt, lut) if spec["errors"] else tile_label(p, lut), interpolation="none")
            used[str(p)] = md5(p)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color(MUTED)
            ours = key == ds["ours"]
            if i == 0:
                title = ({"rgb": "RGB input", "input": "Affected input", "gt": "Ground truth"}.get(key)
                         or ds["models"][key]["label"])
                ax.set_title(title, fontsize=7, color=FG, pad=3, fontweight="bold" if ours else "normal")
            if key not in ("rgb", "input", "gt"):
                v = 100 * float(per[iid][f"mIoU_{key}"])
                ax.text(0.5, -0.025, f"{v:.1f}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color=FG, fontweight="bold" if ours else "normal")
        if "input" in cols and spec["rows"][i].get("input", "rgb") != "rgb":
            rname = f"{rname} ({spec['rows'][i]['input']})"
        fig.text(0.02 / TEXT_W, (y0 + tile / 2) / fig_h, rname, rotation=90, ha="left", va="center",
                 fontsize=7, color=FG)
    stem = out / name
    fig.savefig(f"{stem}.pdf", dpi=300, pad_inches=0.01)
    plt.close(fig)
    json.dump({"design_source": "carbon@11.84.0 (text colours); dataset palette for labels",
               "script": "figures/src/make_qual_figs.py", "errors_mode": spec["errors"], "columns": cols,
               "inputs_md5": {"selection.yaml": md5(QB / "selection.yaml"), "registry.yaml": md5(QB / "registry.yaml"),
                              f"{dname}_candidates.csv": md5(SRC / f"{dname}_candidates.csv"),
                              f"{dname}_per_image.csv": md5(SRC / f"{dname}_per_image.csv")},
               "rows": [[rname, iid, per[iid]["delta"], per[iid]["best_compare"]] for iid, rname in sel],
               "tiles": used}, open(f"{stem}.json", "w"), indent=1, ensure_ascii=False)
    print(f"[fig] {name}: {nr} rows")


def draw_legend(dname, reg, out):
    ds = reg["datasets"][dname]
    names = qualbank.classes_of(ds)
    lut, n = qualbank.palette_of(ds)
    assert len(names) == n, (len(names), n)
    ncol = 9
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(TEXT_W, 0.14 * nrow + 0.04))
    cw = TEXT_W / ncol
    for k, nm in enumerate(names):
        rr, cc = divmod(k, ncol)
        y = 1 - (rr + 0.5) / nrow
        fig.patches.append(mpl.patches.Rectangle(((cc * cw + 0.05) / TEXT_W, y - 0.25 / nrow), 0.12 / TEXT_W,
                                                 0.5 / nrow, transform=fig.transFigure,
                                                 facecolor=lut[k] / 255.0, edgecolor=MUTED, lw=0.3))
        fig.text((cc * cw + 0.21) / TEXT_W, y, nm, va="center", ha="left", fontsize=6.3, color=FG)
    fig.savefig(f"{out / f'fig_class_legend_{dname}'}.pdf", pad_inches=0.01)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PAPER / "latex" / "figures")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    reg = qualbank.load_reg()
    sel = yaml.safe_load(open(QB / "selection.yaml", encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    used = set()
    for name, spec in sel["figures"].items():
        if a.only and name not in a.only:
            continue
        spec = {**sel["defaults"], **spec}
        draw(name, spec, reg, a.out)
        used.add(spec["dataset"])
    for d in used:
        draw_legend(d, reg, a.out)


if __name__ == "__main__":
    main()
