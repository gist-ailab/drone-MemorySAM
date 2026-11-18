#!/usr/bin/env python3
"""Why thin objects vanish in stride-16 ViT tokens, and what the high-resolution path changes (user 2026-11-18).

For a DELIVER test scene and one thin class, crop the window where the model WITH the high-resolution path (A2s821)
gets right the most pixels of the class (misses and false alarms) that the same model WITHOUT it (Q2s821, same seed, same training) gets wrong, and show:
  (a) RGB crop with the 16-pixel token grid of the ViT (each cell = one token)
  (b) token view: the crop averaged per token cell, i.e. the finest detail one token can address
  (c) stride-16 backbone taps (feature magnitude, from hires_viz panels c_norm)
  (d) high-resolution path (feature magnitude, panels d_norm)
  (e) prediction without the path, (f) prediction with the path, (g) ground truth
Inputs: NAS hires_viz_a2s821_20261104/panels (rgb, c_norm, d_norm, gt rendered at 1042 px) and the qualbank label PNGs.
Usage: python make_smallobj_viz.py --out <png> [--scene ID:CLASS ...]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER / "figures" / "qualbank"))
import qualbank  # noqa: E402

NAS = Path("/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs")
PANEL_ROOTS = [NAS / "hires_viz_a2s821_20261118" / "panels", NAS / "hires_viz_a2s821_20261104" / "panels"]
MPL_PAPER = {"font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
             "pdf.fonttype": 42, "font.size": 7}
TEXT_W = 6.99  # inch, IEEE Access two-column text width
BANK = NAS / "paper_qualbank" / "deliver_test"
CLS = {"TrafficSign": 11, "Pole": 4, "Pedestrian": 3, "Static": 18, "TrafficLight": 17}
DEFAULT = ["img/fog/test/MAP_7_point84/011250_rgb_front:Pole",
           "img/cloud/test/MAP_10_point62/042100_rgb_front:TrafficLight",
           "img/cloud/test/MAP_10_point12/013150_rgb_front:Pedestrian"]
W_CROP = 176          # crop side in native pixels (~11 tokens)
TOKEN = 1042 / 64     # model input 1024 -> 64x64 tokens; panels are at native 1042 px


def lab(model, sid):
    return np.array(Image.open(BANK / model / f"{sid}.png"))


def best_window(gain, w):
    """Top-left of the w x w window containing the most 'gain' pixels (integral image)."""
    ii = np.pad(gain.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(y), int(x), int(s[y, x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", nargs="*", default=DEFAULT)
    ap.add_argument("--paper", action="store_true", help="two-column paper size, 7 pt text, PDF")
    a = ap.parse_args()
    reg = qualbank.load_reg()
    lut, _ = qualbank.palette_of(reg["datasets"]["deliver"])

    def col(m):
        out = lut[np.clip(m, 0, len(lut) - 1)].copy()
        out[m == 255] = 0
        return out

    if a.paper:
        plt.rcParams.update(MPL_PAPER)
        titles = ["RGB, 16-px\ntoken grid", "Token\naverage", "Backbone features\n(stride 16)",
                  "High-res. path\n(stride 4)", "Without\nhigh-res. path", "With\nhigh-res. path", "Ground\ntruth"]
        fs_t, fs_l, fs_x = 7, 7, 6.5
        fig, axs = plt.subplots(len(a.scene), 7, figsize=(TEXT_W, 1.08 * len(a.scene) + 0.32), squeeze=False)
    else:
        titles = ["(a) RGB + 16-px token grid", "(b) what one token sees", "(c) ViT features, stride 16",
                  "(d) high-res path, stride 4", "(e) w/o high-res path", "(f) with high-res path", "(g) ground truth"]
        fs_t, fs_l, fs_x = 8, 8, 7
        fig, axs = plt.subplots(len(a.scene), 7, figsize=(14, 2.15 * len(a.scene)), squeeze=False)
    for r, spec in enumerate(a.scene):
        sid, cname = spec.split(":"); k = CLS[cname]
        P = next(r_ / sid for r_ in PANEL_ROOTS if (r_ / sid / "d_norm.png").exists())
        rgb = np.array(Image.open(P / "rgb.png").convert("RGB"))
        cn = np.array(Image.open(P / "c_norm.png").convert("RGB"))
        dn = np.array(Image.open(P / "d_norm.png").convert("RGB"))
        gt, a2, q2 = lab("gt", sid), lab("A2s821", sid), lab("Q2s821", sid)
        # pixels of the class that only the model with the path gets right (misses and false alarms alike)
        gain = ((a2 == k) == (gt == k)) & ((q2 == k) != (gt == k)) & (gt != 255)
        y, x, n = best_window(gain, W_CROP)
        sl = (slice(y, y + W_CROP), slice(x, x + W_CROP))
        # token view: average RGB over each token cell of the full image, then crop
        ty = (np.arange(rgb.shape[0]) / TOKEN).astype(int); tx = (np.arange(rgb.shape[1]) / TOKEN).astype(int)
        tok = np.zeros_like(rgb, dtype=float)
        for c in range(3):
            s = np.zeros((64, 64)); cnt = np.zeros((64, 64))
            np.add.at(s, (ty[:, None], tx[None, :]), rgb[..., c]); np.add.at(cnt, (ty[:, None], tx[None, :]), 1)
            tok[..., c] = (s / np.maximum(cnt, 1))[ty[:, None], tx[None, :]]
        tiles = [rgb[sl], tok.astype(np.uint8)[sl], cn[sl], dn[sl], col(q2)[sl], col(a2)[sl], col(gt)[sl]]
        iou = lambda p: ((p == k) & (gt == k))[sl].sum() / max(((p == k) | (gt == k))[sl].sum(), 1)
        for j, t in enumerate(tiles):
            ax = axs[r, j]; ax.imshow(t, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:   # token grid
                for g in np.arange(np.ceil(x / TOKEN) * TOKEN - x, W_CROP, TOKEN):
                    ax.axvline(g - 0.5, color="w", lw=0.5, alpha=0.7)
                for g in np.arange(np.ceil(y / TOKEN) * TOKEN - y, W_CROP, TOKEN):
                    ax.axhline(g - 0.5, color="w", lw=0.5, alpha=0.7)
                ax.set_ylabel(f"{sid.split('/')[1].capitalize()}\n{cname}", fontsize=fs_l)
            if j in (4, 5):
                v = iou(q2 if j == 4 else a2)
                ax.set_xlabel(f"IoU {100 * v:.0f}" if a.paper else f"{cname} IoU in crop {v:.2f}", fontsize=fs_x, labelpad=1.5)
            if r == 0:
                ax.set_title(titles[j], fontsize=fs_t, pad=3)
        print(f"{sid} {cname}: crop y{y} x{x}, {n} recovered px")
    fig.tight_layout(pad=0.2, w_pad=0.25, h_pad=0.3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300 if a.paper else 200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
