#!/usr/bin/env python
"""Render a polished MULTIAQUA (MaCVi challenge) segmentation comparison video.

Composes 50 validation (daytime) + 50 test (night) frames into a single MP4:
per frame it shows the three input modalities (RGB / LiDAR / Thermal) on the top
row and three prediction overlays (baseline / ours / ground-truth) on the bottom
row, with official challenge-server per-frame mIoU badges. All predictions are
pre-computed PNG masks; no model inference happens here.

Usage
-----
    python tools/render_multiaqua_compare_video.py --out /path/to/compare.mp4

    # smoke test: dump 4 composites + the intro card instead of encoding a video
    python tools/render_multiaqua_compare_video.py \
        --out /tmp/preview/video.mp4 --preview /tmp/preview

The default prediction/score directories are resolved relative to the repo root
(the parent of tools/). Override any of them on the command line when running
from a worktree where outputs/ is absent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
W, H, FPS = 1920, 1080, 30
FFMPEG = "/usr/bin/ffmpeg"

NOTO = "/usr/share/fonts/opentype/noto"
DEJAVU_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_FILES = {
    "bold": (f"{NOTO}/NotoSansCJK-Bold.ttc", 0),
    "med": (f"{NOTO}/NotoSansCJK-Medium.ttc", 0),
    "reg": (f"{NOTO}/NotoSansCJK-Regular.ttc", 0),
    "mono": (DEJAVU_MONO, 0),
}


def hx(s):
    """'#RRGGBB' -> (r, g, b)."""
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


PRIMARY = hx("#E6EAF2")
SECONDARY = hx("#8B95AD")
TEAL = hx("#14B8A6")
VIOLET = hx("#8B5CF6")
ACCENT = hx("#34D399")
BORDER = hx("#2A3350")
RED = hx("#F87171")
BG_TOP = hx("#0B1020")
BG_BOT = hx("#121A2E")
PANEL_BG = hx("#0E1424")
CHIP_BG = (8, 12, 24, 184)  # rgba(8,12,24,0.72)

# raw label id -> (name, color); ignore ids (0, 255) get no color.
CLASS_RGB = {1: hx("#F59E0B"), 2: hx("#F43F5E"), 3: hx("#38BDF8"), 4: hx("#A78BFA")}
CLASS_NAMES = [("Static obstacle", 1), ("Dynamic obstacle", 2),
               ("Water", 3), ("Sky", 4)]

# Layout geometry — horizontal
MARGIN, GAP, PW, PH, RADIUS = 40, 20, 600, 338, 14
COLX = [MARGIN, MARGIN + PW + GAP, MARGIN + 2 * (PW + GAP)]  # [40, 660, 1280]

# Layout geometry — vertical. Derived from the fixed block heights so the free
# space is shared out instead of pooling at the bottom: the empty bands above
# the header, between the two panel rows, and below the footer come out roughly
# equal (~45 px), with a moderate breathing gap between header/row-1 and
# row-2/footer (~37 px). Panels stay width-bound at 600x338.
HEADER_TOP = 45          # title y; an equal empty band sits above it
HEADER_H = 100           # title + subtitle + right-side pill/counter block
LABEL_OFFSET = 26        # row caption sits this far above its panel row
GAP_ROWS = 45            # empty band between row-1 panels and row-2 caption
GAP_EDGE = 37            # header->row-1 and row-2->footer breathing room

ROW1_Y = HEADER_TOP + HEADER_H + GAP_EDGE + LABEL_OFFSET   # 208
ROW2_Y = ROW1_Y + PH + GAP_ROWS + LABEL_OFFSET             # 617
FOOTER_Y = ROW2_Y + PH + GAP_EDGE + 6                      # 998 (legend line)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def natkey(stem):
    """Natural sort key: 'lj4_1_009220' -> ['lj', 4, '_', 1, '_', 9220]."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", stem)]


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def draw_tracked(d, pos, text, font, fill, tracking=2):
    """Draw letter-spaced text (for tracked caps labels)."""
    x, y = pos
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking
    return x


def make_background():
    t = np.linspace(0.0, 1.0, H)[:, None]
    col = (1 - t) * np.array(BG_TOP, np.float32) + t * np.array(BG_BOT, np.float32)
    g = np.repeat(col[:, None, :], W, axis=1)
    return Image.fromarray(g.astype(np.uint8))


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


def boundary_mask(lab):
    """True where a 4-neighbour label differs (crisp class edges)."""
    b = np.zeros(lab.shape, bool)
    dh = lab[:, 1:] != lab[:, :-1]
    b[:, 1:] |= dh
    b[:, :-1] |= dh
    dv = lab[1:, :] != lab[:-1, :]
    b[1:, :] |= dv
    b[:-1, :] |= dv
    return b


# -----------------------------------------------------------------------------
# Data loading / frame selection
# -----------------------------------------------------------------------------
def read_stems(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_csv_miou(path):
    """{stem: mIoU_float} from a frames_*.csv (mIoU may be NaN)."""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem = (row.get("image") or "").strip()
            if not stem:
                continue
            try:
                d[stem] = float(row.get("mIoU", ""))
            except (TypeError, ValueError):
                d[stem] = float("nan")
    return d


def load_scores(scores_dir):
    """-> ({'val': {stem: miou}, 'test': {...}}, summary_dict)."""
    frames = {
        "val": load_csv_miou(os.path.join(scores_dir, "frames_val.csv")),
        "test": load_csv_miou(os.path.join(scores_dir, "frames_test.csv")),
    }
    summ = {}
    sp = os.path.join(scores_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summ = json.load(f)
    return frames, summ


def select_frames(split, stems, ours_m, base_m, n, mode):
    """Pick n frames for a split, then order them by natural sort of the stem."""
    if mode == "uniform":
        ordered = sorted(stems, key=natkey)
        if n >= len(ordered):
            chosen = ordered
        else:
            idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
            chosen = [ordered[i] for i in sorted(set(idx.tolist()))]
    else:  # gap
        common = [s for s in stems if s in ours_m and s in base_m
                  and not isnan(ours_m[s]) and not isnan(base_m[s])]
        common.sort(key=lambda s: (ours_m[s] - base_m[s]), reverse=True)
        chosen = common[:n]

    chosen.sort(key=natkey)
    out = []
    for i, s in enumerate(chosen):
        om = ours_m.get(s, float("nan"))
        bm = base_m.get(s, float("nan"))
        gap = (om - bm) if not (isnan(om) or isnan(bm)) else float("nan")
        out.append({"split": split, "stem": s, "ours_mIoU": om,
                    "base_mIoU": bm, "gap": gap, "idx": i + 1})
    for fi in out:
        fi["total"] = len(out)
    return out


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------
class Renderer:
    def __init__(self, args):
        self.a = args
        self.R = os.path.join(args.data_root, "MULTIAQUA_night")
        self._fonts = {}
        self._rgb_cache = {}
        self.bg = make_background()

        self.ours_frames, self.ours_summary = load_scores(args.ours_scores)
        self.base_frames, self.base_summary = load_scores(args.base_scores)

        val_stems = read_stems(os.path.join(args.data_root, "val.txt"))
        test_stems = read_stems(os.path.join(args.data_root, "test.txt"))
        self.val_frames = select_frames(
            "val", val_stems, self.ours_frames["val"], self.base_frames["val"],
            args.n_val, args.select)
        self.test_frames = select_frames(
            "test", test_stems, self.ours_frames["test"], self.base_frames["test"],
            args.n_test, args.select)

    # -- fonts --------------------------------------------------------------
    def font(self, kind, size):
        key = (kind, size)
        if key not in self._fonts:
            path, idx = FONT_FILES[kind]
            self._fonts[key] = ImageFont.truetype(path, size, index=idx)
        return self._fonts[key]

    # -- source paths -------------------------------------------------------
    def p_rgb(self, stem):
        return os.path.join(self.R, "data", "zed", f"{stem}.png")

    def p_lidar(self, stem):
        return os.path.join(self.R, "data", "lidar_processed2", f"{stem}_lidar.png")

    def p_thermal_raw(self, stem):
        return os.path.join(self.R, "data", "thermal_camera", f"{stem}.png")

    def p_thermal_fb(self, stem):
        return os.path.join(self.R, "data", "thermal_processed", f"{stem}_thermal.png")

    def p_gt(self, stem):
        return os.path.join(self.R, "annotations", f"{stem}.png")

    def p_pred(self, base_dir, stem):
        return os.path.join(base_dir, f"{stem}.png")

    # -- modality panels ----------------------------------------------------
    def _placeholder(self, text):
        img = Image.new("RGB", (PW, PH), PANEL_BG)
        d = ImageDraw.Draw(img)
        f = self.font("reg", 18)
        tw = d.textlength(text, font=f)
        d.text(((PW - tw) / 2, PH / 2 - 12), text, font=f, fill=SECONDARY)
        return np.asarray(img)

    def _rgb_downscaled(self, stem):
        """Downscaled RGB panel (PW, PH, 3) uint8, or None if the file is
        missing. Cached per stem: the RGB PNGs live on a slow sshfs NAS and are
        needed up to four times while composing a single frame (row-1 RGB panel,
        baseline overlay, ours overlay, and the ground-truth panel), so each
        file is read only once. The cache is bounded to keep memory flat."""
        if stem in self._rgb_cache:
            return self._rgb_cache[stem]
        im = cv2.imread(self.p_rgb(stem))
        down = (cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (PW, PH),
                           interpolation=cv2.INTER_AREA)
                if im is not None else None)
        if len(self._rgb_cache) >= 8:
            self._rgb_cache.clear()
        self._rgb_cache[stem] = down
        return down

    def render_rgb(self, stem):
        im = self._rgb_downscaled(stem)
        if im is None:
            return self._placeholder("not available")
        return im

    def render_lidar(self, stem):
        im = cv2.imread(self.p_lidar(stem), cv2.IMREAD_UNCHANGED)
        if im is None:
            return self._placeholder("not available")
        if im.ndim == 3:
            im = im[..., 0]
        val = im > 0
        if not val.any():
            return self._placeholder("no returns")
        nz = im[val].astype(np.float32)
        lo, hi = np.percentile(nz, 2), np.percentile(nz, 98)
        if hi <= lo:
            hi = lo + 1.0
        norm8 = (np.clip((im.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        # Dilate at full resolution so sparse points survive downscaling.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (self.a.lidar_dilate, self.a.lidar_dilate))
        val_d = cv2.dilate((val.astype(np.uint8) * 255), k)
        norm_d = cv2.dilate(norm8, k)
        color = cv2.applyColorMap(norm_d, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = (val_d > 0).astype(np.float32)
        # Premultiplied-alpha downscale so dots keep their color.
        down_ca = cv2.resize(color * alpha[..., None], (PW, PH), interpolation=cv2.INTER_AREA)
        down_a = cv2.resize(alpha, (PW, PH), interpolation=cv2.INTER_AREA)
        col = np.zeros((PH, PW, 3), np.float32)
        m = down_a > 1e-4
        col[m] = down_ca[m] / down_a[m, None]
        a3 = np.clip(down_a, 0, 1)[..., None]
        bg = np.array(PANEL_BG, np.float32)
        out = bg[None, None] * (1 - a3) + col * a3
        return out.astype(np.uint8)

    def render_thermal(self, stem):
        im = cv2.imread(self.p_thermal_raw(stem), cv2.IMREAD_UNCHANGED)
        if im is None:
            im = cv2.imread(self.p_thermal_fb(stem), cv2.IMREAD_UNCHANGED)
        if im is None:
            return self._placeholder("not available")
        if im.ndim == 3:
            im = im[..., 0]
        val = im > 0
        if not val.any():
            return self._placeholder("no signal")
        v = im[val].astype(np.float32)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        st = (np.clip((im.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        st[~val] = 0
        st = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(st)
        color = cv2.applyColorMap(st, cv2.COLORMAP_INFERNO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color[~val] = PANEL_BG
        # The sensor only covers a quadrilateral (~36%) in the middle of the
        # frame, so crop to the valid mask's bounding box (with ~2% padding)
        # and fit that crop into the panel to make the thermal FOV fill it.
        ys, xs = np.where(val)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pady = max(1, round((y1 - y0) * 0.02))
        padx = max(1, round((x1 - x0) * 0.02))
        y0, y1 = max(0, y0 - pady), min(color.shape[0], y1 + pady)
        x0, x1 = max(0, x0 - padx), min(color.shape[1], x1 + padx)
        return self._fit_panel(color[y0:y1, x0:x1])

    def _fit_panel(self, img):
        """Letterbox img into a (PW, PH) panel preserving aspect ratio: scale to
        fit, center, and pad the margins with PANEL_BG."""
        h, w = img.shape[:2]
        scale = min(PW / w, PH / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((PH, PW, 3), PANEL_BG, np.uint8)
        ox, oy = (PW - nw) // 2, (PH - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    # -- overlays -----------------------------------------------------------
    def _rgb_panel(self, stem):
        im = self._rgb_downscaled(stem)
        if im is None:
            return np.full((PH, PW, 3), PANEL_BG, np.uint8)
        return im

    def render_overlay(self, stem, label_path):
        base = self._rgb_panel(stem).astype(np.float32)
        lab = (cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
               if label_path and os.path.exists(label_path) else None)
        if lab is None:
            return base.astype(np.uint8)
        lab = cv2.resize(lab, (PW, PH), interpolation=cv2.INTER_NEAREST)
        out = base
        for cls, col in CLASS_RGB.items():
            m = lab == cls
            if m.any():
                out[m] = out[m] * 0.45 + np.array(col, np.float32) * 0.55
        b = boundary_mask(lab)
        out[b] = out[b] * 0.4 + np.array((255, 255, 255), np.float32) * 0.6
        return out.astype(np.uint8)

    def render_gt_withheld(self, stem):
        im = self._rgb_downscaled(stem)
        if im is None:
            base = np.full((PH, PW, 3), PANEL_BG, np.uint8)
        else:
            base = cv2.GaussianBlur(im, (0, 0), 2.0)
            base = (base.astype(np.float32) * 0.35).astype(np.uint8)
        img = Image.fromarray(base)
        d = ImageDraw.Draw(img)
        f = self.font("med", 20)
        for i, line in enumerate(["Ground truth withheld",
                                  "scored on the MaCVi challenge server"]):
            tw = d.textlength(line, font=f)
            d.text(((PW - tw) / 2, PH / 2 - 26 + i * 30), line, font=f, fill=SECONDARY)
        return np.asarray(img)

    # -- compositing primitives --------------------------------------------
    def paste_panel(self, base, arr, x, y, accent=False):
        base.paste(Image.fromarray(np.ascontiguousarray(arr)), (x, y),
                   rounded_mask(PW, PH, RADIUS))
        d = ImageDraw.Draw(base)
        d.rounded_rectangle([x, y, x + PW - 1, y + PH - 1], radius=RADIUS,
                            outline=BORDER, width=1)
        if accent:
            d.rounded_rectangle([x + 1, y + 1, x + PW - 2, y + PH - 2],
                                radius=RADIUS, outline=ACCENT, width=3)

    def chip(self, od, x, y, text, font, text_fill=(255, 255, 255),
             bg=CHIP_BG, anchor="lt"):
        tw = od.textlength(text, font=font)
        asc, desc = font.getmetrics()
        padx, pady = 8, 4
        w, h = tw + 2 * padx, asc + desc + 2 * pady
        if anchor == "rt":
            x = x - w
        od.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=bg)
        od.text((x + padx, y + pady), text, font=font, fill=text_fill)
        return w, h

    # -- full frame ---------------------------------------------------------
    def compose_frame(self, fi):
        stem = fi["stem"]
        val = fi["split"] == "val"
        base = self.bg.copy()
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(base)
        od = ImageDraw.Draw(overlay)

        # Row 1: input modalities.
        panels1 = [("RGB", self.render_rgb(stem)),
                   ("LiDAR", self.render_lidar(stem)),
                   ("Thermal · sensor FOV", self.render_thermal(stem))]
        # Row 2: prediction overlays.
        base_ov = self.render_overlay(stem, self.p_pred(self.a.base_pred, stem))
        ours_ov = self.render_overlay(stem, self.p_pred(self.a.ours_pred, stem))
        gt_ov = (self.render_overlay(stem, self.p_gt(stem)) if val
                 else self.render_gt_withheld(stem))
        panels2 = [(self.a.base_label, base_ov), (self.a.ours_label, ours_ov),
                   ("Ground Truth", gt_ov)]

        chip_font = self.font("med", 16)
        for col, (cap, arr) in enumerate(panels1):
            x = COLX[col]
            self.paste_panel(base, arr, x, ROW1_Y)
            self.chip(od, x + 8, ROW1_Y + 8, cap, chip_font)

        for col, (cap, arr) in enumerate(panels2):
            x = COLX[col]
            accent = col == 1  # ours
            self.paste_panel(base, arr, x, ROW2_Y, accent=accent)
            self.chip(od, x + 8, ROW2_Y + 8, cap, chip_font)

        # Score badges (baseline @ col0, ours @ col1).
        self._score_badge(od, COLX[0], ROW2_Y, fi["base_mIoU"])
        gap_h = self._score_badge(od, COLX[1], ROW2_Y, fi["ours_mIoU"])
        if not isnan(fi["gap"]):
            g = fi["gap"]
            col = ACCENT if g >= 0 else RED
            self.chip(od, COLX[1] + PW - 8, ROW2_Y + 8 + gap_h + 6,
                      f"{g:+.1f}", self.font("bold", 15),
                      text_fill=col, anchor="rt")

        self._header(d, fi)
        draw_tracked(d, (COLX[0], ROW1_Y - LABEL_OFFSET), "INPUT MODALITIES",
                     self.font("med", 16), SECONDARY, 2)
        draw_tracked(d, (COLX[0], ROW2_Y - LABEL_OFFSET), "PREDICTION OVERLAY",
                     self.font("med", 16), SECONDARY, 2)
        self._footer(d)

        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        return np.asarray(base)

    def _score_badge(self, od, x, y, miou):
        if isnan(miou):
            return 0
        _, h = self.chip(od, x + PW - 8, y + 8, f"mIoU {miou:.1f}",
                         self.font("bold", 15), anchor="rt")
        return h

    def _header(self, d, fi):
        d.text((MARGIN, HEADER_TOP), self.a.title,
               font=self.font("bold", 40), fill=PRIMARY)
        d.text((MARGIN, HEADER_TOP + 48), self.a.subtitle,
               font=self.font("reg", 20), fill=SECONDARY)

        val = fi["split"] == "val"
        pill_txt = "VALIDATION · DAY" if val else "TEST · NIGHT"
        pill_col = TEAL if val else VIOLET
        pf = self.font("bold", 18)
        tw = d.textlength(pill_txt, font=pf)
        w, h = tw + 28, 34
        x1, y0 = W - MARGIN, HEADER_TOP + 2
        x0 = x1 - w
        d.rounded_rectangle([x0, y0, x1, y0 + h], radius=h // 2, fill=pill_col)
        asc, desc = pf.getmetrics()
        d.text((x0 + 14, y0 + (h - asc - desc) / 2), pill_txt, font=pf, fill=(11, 16, 32))

        cf = self.font("med", 20)
        ct = f"{fi['idx']} / {fi['total']}"
        d.text((x1 - d.textlength(ct, font=cf), y0 + h + 8), ct, font=cf, fill=PRIMARY)
        mf = self.font("mono", 15)
        d.text((x1 - d.textlength(fi["stem"], font=mf), y0 + h + 38),
               fi["stem"], font=mf, fill=SECONDARY)

    def _footer(self, d):
        fy = FOOTER_Y
        x = MARGIN
        lf = self.font("med", 15)
        for name, cls in CLASS_NAMES:
            d.rounded_rectangle([x, fy, x + 16, fy + 16], radius=5, fill=CLASS_RGB[cls])
            x += 24
            d.text((x, fy), name, font=lf, fill=PRIMARY)
            x += d.textlength(name, font=lf) + 28

        rf = self.font("reg", 15)
        l1 = "Per-frame mIoU = official challenge-server scores"
        d.text((W - MARGIN - d.textlength(l1, font=rf), fy - 6), l1, font=rf, fill=SECONDARY)
        if self.a.select == "gap":
            n = (f"{self.a.n_val}" if self.a.n_val == self.a.n_test
                 else f"{self.a.n_val} val / {self.a.n_test} test")
            l2 = f"Frames: {n} per split with the largest Ours − Baseline gap"
        else:
            l2 = "Frames: uniform sampling per split"
        d.text((W - MARGIN - d.textlength(l2, font=rf), fy + 16), l2, font=rf, fill=SECONDARY)

    # -- cards --------------------------------------------------------------
    def _summ_vals(self, summ):
        return (summ.get("val_mIoU", float("nan")),
                summ.get("test_mIoU", float("nan")),
                summ.get("M", float("nan")))

    def _summ_vals_dict(self, summ):
        return {"val_mIoU": summ.get("val_mIoU"),
                "test_mIoU": summ.get("test_mIoU"), "M": summ.get("M")}

    def render_table_card(self):
        base = self.bg.copy()
        d = ImageDraw.Draw(base)
        tf = self.font("bold", 64)
        tw = d.textlength(self.a.title, font=tf)
        d.text(((W - tw) / 2, 250), self.a.title, font=tf, fill=PRIMARY)
        sf = self.font("reg", 24)
        sw = d.textlength(self.a.subtitle, font=sf)
        d.text(((W - sw) / 2, 340), self.a.subtitle, font=sf, fill=SECONDARY)

        # Results table.
        cols = ["Val mIoU", "Test mIoU", "M-score"]
        col_x = [990, 1250, 1510]
        label_x = 420
        hf = self.font("med", 22)
        d.text((label_x, 470), "Model", font=hf, fill=SECONDARY)
        for cx, name in zip(col_x, cols):
            d.text((cx - d.textlength(name, font=hf) / 2, 470), name, font=hf, fill=SECONDARY)

        rows = [(self.a.base_label, self._summ_vals(self.base_summary), False),
                (self.a.ours_label, self._summ_vals(self.ours_summary), True)]
        ry = 540
        rf = self.font("med", 24)
        for label, (v, t, m), hi in rows:
            if hi:
                d.rounded_rectangle([label_x - 24, ry - 8, 1560, ry + 46],
                                    radius=10, fill=(0x1B, 0x2E, 0x2A))
            fill = ACCENT if hi else PRIMARY
            d.text((label_x, ry), label, font=rf, fill=fill)
            for cx, val in zip(col_x, (v, t, m)):
                s = "—" if isnan(val) else f"{val:.2f}"
                d.text((cx - d.textlength(s, font=rf) / 2, ry), s, font=rf, fill=fill)
            ry += 78

        ff = self.font("reg", 20)
        foot = "M = 0.75·Val + 0.25·Test (MaCVi)"
        d.text(((W - d.textlength(foot, font=ff)) / 2, ry + 6), foot, font=ff, fill=SECONDARY)
        return np.asarray(base)

    def render_section_card(self, split):
        base = self.bg.copy()
        d = ImageDraw.Draw(base)
        val = split == "val"
        name = "Validation · Day" if val else "Test · Night"
        line = ("Ground truth available · %d frames" % self.a.n_val if val
                else "Night-only, ground truth withheld · %d frames" % self.a.n_test)
        col = TEAL if val else VIOLET
        nf = self.font("bold", 72)
        d.text(((W - d.textlength(name, font=nf)) / 2, 420), name, font=nf, fill=col)
        lf = self.font("reg", 28)
        d.text(((W - d.textlength(line, font=lf)) / 2, 530), line, font=lf, fill=SECONDARY)
        return np.asarray(base)

    # -- scene sequencing ---------------------------------------------------
    def scene_specs(self):
        specs = [("intro", None), ("section", "val")]
        specs += [("frame", fi) for fi in self.val_frames]
        specs += [("section", "test")]
        specs += [("frame", fi) for fi in self.test_frames]
        specs += [("outro", None)]
        return specs

    def render_spec(self, spec):
        kind, data = spec
        if kind == "intro":
            return self.render_table_card(), 3.5
        if kind == "outro":
            return self.render_table_card(), 3.0
        if kind == "section":
            return self.render_section_card(data), 2.2
        return self.compose_frame(data), self.a.hold

    # -- encode -------------------------------------------------------------
    def encode(self, out):
        if not os.path.exists(FFMPEG):
            raise RuntimeError(f"ffmpeg not found at {FFMPEG}")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        # Route stderr to a temp log file (not a PIPE) so a chatty/failing
        # ffmpeg can't deadlock us on an undrained pipe; we surface its tail
        # only if the encode fails.
        log = tempfile.NamedTemporaryFile(mode="w+", suffix=".log",
                                          prefix="ffmpeg_", delete=False)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=log)

        black = np.zeros((H, W, 3), np.uint8)
        fade_n = max(1, round(self.a.fade * FPS))
        total = [0]
        broken = [False]  # set once ffmpeg's stdin closes early (it died)

        def emit(a):
            if broken[0]:
                return
            try:
                proc.stdin.write(np.ascontiguousarray(a, np.uint8).tobytes())
            except BrokenPipeError:
                broken[0] = True
                return
            total[0] += 1

        def crossfade(a, b):
            af, bf = a.astype(np.float32), b.astype(np.float32)
            for t in range(1, fade_n + 1):
                al = t / fade_n
                emit((af * (1 - al) + bf * al).astype(np.uint8))

        specs = self.scene_specs()
        prev = black
        done = 0
        for spec in specs:
            comp, hold = self.render_spec(spec)
            crossfade(prev, comp)
            for _ in range(max(1, round(hold * FPS))):
                emit(comp)
            prev = comp
            if spec[0] == "frame":
                done += 1
                if done % 10 == 0:
                    print(f"  encoded {done} frames...", flush=True)
        crossfade(prev, black)

        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        log.flush()
        log.close()
        if proc.returncode != 0:
            with open(log.name) as f:
                tail = "".join(f.readlines()[-30:])
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}; last log lines:\n"
                f"{tail}")
        os.unlink(log.name)
        return total[0] / FPS

    # -- preview ------------------------------------------------------------
    def preview(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        picks = []
        if self.val_frames:
            picks.append(("val_first", self.val_frames[0]))
            picks.append(("val_mid", self.val_frames[len(self.val_frames) // 2]))
        if self.test_frames:
            picks.append(("test_first", self.test_frames[0]))
            picks.append(("test_last", self.test_frames[-1]))
        saved = []
        for name, fi in picks:
            arr = self.compose_frame(fi)
            p = os.path.join(outdir, f"preview_{name}.png")
            cv2.imwrite(p, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            saved.append(p)
        intro, _ = self.render_spec(("intro", None))
        p = os.path.join(outdir, "preview_intro.png")
        cv2.imwrite(p, cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))
        saved.append(p)
        return saved


# -----------------------------------------------------------------------------
# Sidecar outputs
# -----------------------------------------------------------------------------
def write_manifest(path, frames):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["order", "split", "stem", "base_mIoU", "ours_mIoU", "gap"])
        order = 1
        for fi in frames:
            def fmt(x):
                return "" if isnan(x) else f"{x:.4f}"
            wr.writerow([order, fi["split"], fi["stem"],
                         fmt(fi["base_mIoU"]), fmt(fi["ours_mIoU"]), fmt(fi["gap"])])
            order += 1


def write_sidecar(path, r):
    a = r.a
    caption = (f"{a.title}: baseline vs ours on {len(r.val_frames)} val (day) + "
               f"{len(r.test_frames)} test (night) MULTIAQUA frames, "
               f"RGB+LiDAR+Thermal with official challenge-server mIoU.")
    data = {
        "title": a.title,
        "subtitle": a.subtitle,
        "splits": {
            "val": {"count": len(r.val_frames), "condition": "day",
                    "ground_truth": True},
            "test": {"count": len(r.test_frames), "condition": "night",
                     "ground_truth": False},
        },
        "selection": a.select,
        "labels": {"baseline": a.base_label, "ours": a.ours_label},
        "prediction_dirs": {"baseline": a.base_pred, "ours": a.ours_pred},
        "score_dirs": {"baseline": a.base_scores, "ours": a.ours_scores},
        "summaries": {
            "baseline": r._summ_vals_dict(r.base_summary),
            "ours": r._summ_vals_dict(r.ours_summary),
        },
        "submission_ids": {"baseline": a.base_sub, "ours": a.ours_sub},
        "fps": FPS, "hold": a.hold, "fade": a.fade,
        "caption": caption,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--repo_root", default=repo_root,
                   help="root used to resolve relative default dirs")
    p.add_argument("--data_root",
                   default="/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2")
    p.add_argument("--ours_pred",
                   default="outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/"
                           "MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_eval_macvi")
    p.add_argument("--base_pred",
                   default="outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/"
                           "MULTIAQUA_CMNeXt-B2_ilt_beforeAug/eval_macvi")
    p.add_argument("--ours_scores",
                   default="outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/"
                           "MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_16710_results")
    p.add_argument("--base_scores",
                   default="outputs/MMSamP8/lecun_multiaqua_rgbtl_P8/"
                           "MULTIAQUA_CMNeXt-B2_ilt_beforeAug/15509_results")
    p.add_argument("--base_label", default="Baseline · P8 (no night-aug)")
    p.add_argument("--ours_label", default="Ours · GatedMemorySAM")
    p.add_argument("--title", default="GatedMemorySAM")
    p.add_argument("--subtitle",
                   default="MaCVi · MULTIAQUA Challenge — "
                           "RGB + LiDAR + Thermal semantic segmentation")
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--select", choices=["gap", "uniform"], default="gap")
    p.add_argument("--hold", type=float, default=1.6)
    p.add_argument("--fade", type=float, default=0.35)
    p.add_argument("--lidar_dilate", type=int, default=9)
    p.add_argument("--base_sub", type=int, default=15509)
    p.add_argument("--ours_sub", type=int, default=16710)
    p.add_argument("--preview", default=None,
                   help="dump 4 composites + intro card to DIR and exit")
    return p


def resolve_dirs(a):
    """Make relative default dirs absolute against repo_root."""
    for k in ("ours_pred", "base_pred", "ours_scores", "base_scores"):
        v = getattr(a, k)
        if not os.path.isabs(v):
            setattr(a, k, os.path.join(a.repo_root, v))


def main():
    a = build_parser().parse_args()
    resolve_dirs(a)
    r = Renderer(a)

    print(f"Selection ({a.select}): {len(r.val_frames)} val + "
          f"{len(r.test_frames)} test frames")
    for tag, frames in (("val", r.val_frames), ("test", r.test_frames)):
        if not frames:
            print(f"  {tag}: (none)")
            continue
        gaps = [f["gap"] for f in frames if not isnan(f["gap"])]
        gtxt = (f"gap {min(gaps):+.1f}..{max(gaps):+.1f}" if gaps else "gap n/a")
        print(f"  {tag}: {frames[0]['stem']} .. {frames[-1]['stem']}  ({gtxt})")

    out_stem = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    manifest = out_stem + "_frames.csv"
    write_manifest(manifest, r.val_frames + r.test_frames)
    print(f"Wrote manifest: {manifest}")

    if a.preview:
        saved = r.preview(a.preview)
        print("Preview PNGs:")
        for p in saved:
            print(f"  {p}")
        return

    sidecar = out_stem + ".json"
    write_sidecar(sidecar, r)
    print(f"Wrote sidecar: {sidecar}")

    print("Encoding...")
    dur = r.encode(a.out)
    print(f"Done: {a.out}  ({dur:.1f} s, {dur / 60:.2f} min)")


if __name__ == "__main__":
    main()
