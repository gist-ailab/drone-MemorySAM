"""Sensor thumbnails for Fig. 1 (architecture overview).

Input: one DELIVER test scene (RGB, depth, event, LiDAR, semantic GT), fetched from
jarvis:/SSDb/jemo_maeng/dset/DELIVER (scene sun/test/MAP_10_point21/002500).
Output: figures/fig1_thumbs/{rgb,depth,event,lidar,gt}.png, 512 px, used by fig_arch.tex.

Colours: depth/LiDAR use the Carbon sequential ramp (design-source: carbon@11.84.0);
GT uses the DELIVER palette from semseg/datasets/deliver.py (dataset convention).
Usage: python make_arch_thumbs.py <scene_root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SCENE = ("sun", "MAP_10_point21", "002500")
SIZE = 512
# carbon@11.84.0 SEQUENTIAL (labdesign build/py/carbon_palette.py)
SEQ = ["#ffffff", "#e4eeff", "#cadcff", "#afcbff", "#94b9ff",
       "#7aa8fe", "#5f96fe", "#4485fe", "#2a73fe", "#0f62fe"]
# DELIVER palette, index = GT id - 1 (semseg/datasets/deliver.py:28-52)
PALETTE = np.array([
    [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
    [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
    [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
    [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160], [170, 120, 50],
    [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100], [0, 0, 70]], np.uint8)


def ramp(values):
    """Map values in [0, 1] onto the sequential ramp (linear interpolation)."""
    stops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEQ], float)
    x = np.clip(values, 0, 1) * (len(stops) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    t = (x - lo)[..., None]
    return (stops[lo] * (1 - t) + stops[hi] * t).astype(np.uint8)


def path(root, modal, suffix):
    cond, mp, idx = SCENE
    return Path(root) / modal / cond / "test" / mp / f"{idx}_{suffix}_front.png"


def main(root, out, pred=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rs = lambda im: im.resize((SIZE, SIZE), Image.LANCZOS)

    rgb = rs(Image.open(path(root, "img", "rgb")).convert("RGB"))
    rgb.save(out / "rgb.png")

    # illustrative degraded RGB for the training panel (blur + low contrast + Gaussian noise);
    # mimics the kinds of synthetic degradation, not the exact training sampler
    a = np.asarray(rgb.filter(ImageFilter.GaussianBlur(2.0)), float)
    a = 0.45 * a + 0.55 * a.mean()
    a += np.random.default_rng(0).normal(0, 22, a.shape)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out / "rgb_degraded.png")

    # depth: near = dark blue, far = white (sky saturates to white)
    d = np.asarray(Image.open(path(root, "depth", "depth")), float) / 255.0
    rs(Image.fromarray(ramp(1.0 - d ** 0.5))).save(out / "depth.png")

    # event: keep polarity colours, put them on white instead of black
    e = np.asarray(Image.open(path(root, "event", "event")).convert("RGB")).astype(int)
    on = e.sum(-1) > 60
    ev = np.full(e.shape, 255, np.uint8)
    ev[on & (e[..., 0] > e[..., 2])] = (218, 30, 40)   # carbon red-60
    ev[on & (e[..., 2] >= e[..., 0])] = (15, 98, 254)  # carbon blue-60
    rs(Image.fromarray(ev)).save(out / "event.png")

    # LiDAR: sparse returns, dilate so points survive the down-sampling
    l = np.asarray(Image.open(path(root, "lidar", "lidar")), float)
    hit = l > 0
    v = np.zeros_like(l)
    v[hit] = 1.0 - (l[hit] - l[hit].min()) / max(np.ptp(l[hit]), 1e-6)
    li = ramp(0.25 + 0.75 * v)
    li[~hit] = 255
    img = Image.fromarray(li).filter(ImageFilter.MinFilter(3))
    rs(img).save(out / "lidar.png")

    # GT: R channel holds class id + 1 (0 = unlabeled -> white)
    g = np.asarray(Image.open(path(root, "semantic", "semantic")))[..., 0].astype(int) - 1
    gt = np.full(g.shape + (3,), 255, np.uint8)
    ok = (g >= 0) & (g < len(PALETTE))
    gt[ok] = PALETTE[g[ok]]
    rs(Image.fromarray(gt)).save(out / "gt.png")

    # prediction of the full model (A2 seed 821, legal v2 dump; 0-based train ids, 255 = ignore)
    if pred is not None:
        p = np.asarray(Image.open(pred)).astype(int)
        pr = np.full(p.shape + (3,), 255, np.uint8)
        ok = p < len(PALETTE)
        pr[ok] = PALETTE[p[ok]]
        Image.fromarray(pr).resize((SIZE, SIZE), Image.NEAREST).save(out / "pred.png")


if __name__ == "__main__":
    # pred: NAS analysis_logs/a2_test1024_dump_20261016/s821/test/pred/img/sun/test/MAP_10_point21/002500_rgb_front.png
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
