"""Low-light (dark-tail) augmentation for classic YOLOv5 — TRAIN-ONLY.

Why this exists
---------------
The certification test set carries a *deep low-light tail* that the train set
lacks. Measured per-clip RGB luma (0-255) on poongsan_v2:

    train 5 clips : mean luma 117.2   (p10 78-99,   dark-pixel% 8-14)
    test  3 clips : mean luma 108.6   (p10 48.6,     dark-pixel% up to 22)

So the gap is NOT a global brightness offset (only -8.6) — it is the presence
of very dark frames (mean luma ~50, 22% near-black pixels) in test that are
under-represented in train. This augmentation manufactures that tail: it
*darkens* a fraction of train frames toward luma ~40-70 and adds low-light
sensor noise, teaching illumination invariance without touching the network.

i.MX safety
-----------
Purely photometric, applied at train time only. Bounding boxes are unchanged
(no geometric op), the inference graph is identical, so the exported model
still compiles for the i.MX NPU.

Calibration note (albumentations RandomGamma convention)
--------------------------------------------------------
RandomGamma applies ``img ** (gamma/100)``; gamma/100 > 1 DARKENS. To take a
train frame from luma ~117 (~0.46 norm) down to ~50 (~0.20) needs gamma ~= 206:
    0.46 ** g = 0.20  ->  g = ln(0.20)/ln(0.46) ~= 2.06
Hence gamma_limit reaches ~210. (A naive (60,110) would have *brightened* — the
opposite of what the data needs; this was caught by measuring first.)
"""
from __future__ import annotations


def build_night_transforms(strength: str = "calibrated"):
    """Return a list of albumentations photometric transforms (dark-tail).

    strength: 'calibrated' (data-matched), 'mild', or 'strong'.
    Version-tolerant: skips a transform if the installed albumentations lacks it
    or renamed its args, rather than crashing the whole pipeline.
    """
    import albumentations as A

    # albumentations 2.x renamed several args and only *warns* on the old ones
    # (it does not raise), so version-detect explicitly rather than try/except.
    major = int(A.__version__.split(".")[0])

    # gamma_limit / brightness ranges per strength; all DARKENING-biased.
    cfg = {
        "mild":       dict(gamma=(90, 160),  bright=(-0.30, 0.05), pg=0.5, pb=0.5),
        "calibrated": dict(gamma=(90, 210),  bright=(-0.40, 0.10), pg=0.7, pb=0.6),
        "strong":     dict(gamma=(110, 240), bright=(-0.55, 0.00), pg=0.8, pb=0.7),
    }[strength]

    T = []

    def _try(fn):
        try:
            T.append(fn())
        except (TypeError, ValueError):
            pass  # transform not supported in this albumentations version

    # 1) dark-tail: gamma darkening (the primary lever)
    _try(lambda: A.RandomGamma(gamma_limit=cfg["gamma"], p=cfg["pg"]))
    # 2) low exposure + reduced contrast
    _try(lambda: A.RandomBrightnessContrast(
        brightness_limit=cfg["bright"], contrast_limit=(-0.30, 0.15), p=cfg["pb"]))
    # 3) low-light sensor noise (shot+read); ISONoise expects uint8 RGB
    _try(lambda: A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.6), p=0.4))
    # 4) generic additive noise — GaussNoise arg name changed at 2.x
    if major >= 2:
        _try(lambda: A.GaussNoise(std_range=(0.02, 0.14), p=0.3))    # >=2.x
    else:
        _try(lambda: A.GaussNoise(var_limit=(5.0, 50.0), p=0.3))     # <=1.x
    # 5) mild compression artefacts (dark scenes compress poorly)
    if major >= 2:
        _try(lambda: A.ImageCompression(quality_range=(45, 90), p=0.2))          # >=2.x
    else:
        _try(lambda: A.ImageCompression(quality_lower=45, quality_upper=90, p=0.2))  # <=1.x

    return T


# ---------------------------------------------------------------------------
# Self-test: darken a real image and report the luma shift, so we can confirm
# the calibration and albumentations compatibility ON THE TARGET SERVER before
# launching any training (repo policy: measure the loader, don't assume).
#   python night_aug.py <image.png> [strength]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import numpy as np
    import albumentations as A

    import glob
    import os
    import cv2

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    strength = sys.argv[2] if len(sys.argv) > 2 else "calibrated"

    T = build_night_transforms(strength)
    print(f"[night_aug] strength={strength}  built {len(T)} transforms:")
    for t in T:
        print(f"    - {t.__class__.__name__}  (p={getattr(t, 'p', '?')})")

    # resolve a REPRESENTATIVE image: if given a directory, sample and pick the
    # median-luma frame (clip-start frames can be near-black and would make the
    # darkening test meaningless).
    img_path = arg
    if arg and os.path.isdir(arg):
        import numpy as _np
        files = sorted(glob.glob(os.path.join(arg, "*.png")) + glob.glob(os.path.join(arg, "*.jpg")))
        idx = _np.linspace(0, len(files) - 1, min(80, len(files))).astype(int)
        scored = [(cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2GRAY).mean(), files[i]) for i in idx]
        scored.sort()
        img_path = scored[len(scored) // 2][1]  # median luma
        print(f"[night_aug] picked median-luma frame from dir: {os.path.basename(img_path)}")

    if img_path:
        im = cv2.imread(img_path)
        assert im is not None, f"cannot read {img_path}"
        luma0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()
        comp = A.Compose(T)
        lumas = []
        for _ in range(200):
            out = comp(image=im)["image"]
            lumas.append(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
        lumas = np.array(lumas)
        print(f"\n[night_aug] source luma            : {luma0:6.1f}")
        print(f"[night_aug] augmented luma (200x)  : mean {lumas.mean():6.1f}  "
              f"p10 {np.percentile(lumas,10):5.1f}  p50 {np.percentile(lumas,50):5.1f}  "
              f"p90 {np.percentile(lumas,90):5.1f}")
        print(f"[night_aug] target test tail       :  ~50 (p10) — "
              f"{'REACHES dark tail OK' if np.percentile(lumas,10) <= 70 else 'TOO BRIGHT — raise gamma'}")
