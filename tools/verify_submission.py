#!/usr/bin/env python3
"""verify_submission.py — hard checks on the MUSES Codabench submission PNGs.

Spec being checked (source: Codabench comp. 14005 "Submission Instructions" page,
fetched via https://www.codabench.org/api/competitions/14005/):
  - zip contains a `labelTrainIds/` directory holding the PNGs; size limit 200 MB
  - filename pattern {sequence}_frame_{frame:0>6}*.png, exactly one per test image
  - dimensions must match the input RGB: 1920x1080
  - labels encoded as trainIDs (road == 0), i.e. 0..18
The 255-is-unsafe call is inference, not a quoted spec: the reference implementation
(CAFuser muses_sem_evaluator.py) argmaxes over 19 channels and thus never emits 255,
and a stock detectron2-style scorer would crash on a 255 prediction. We assert 0..18.
"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
from PIL import Image

PRED_DIR = sys.argv[1]
TEST_RGB = sys.argv[2]

fails = []
def check(cond, msg):
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        fails.append(msg)

pngs = sorted(glob.glob(os.path.join(PRED_DIR, '*.png')))
rgbs = sorted(glob.glob(os.path.join(TEST_RGB, '*', '*', '*.png')))

print("=== 1. file count ===")
check(len(pngs) == 750, f"750 PNGs present (got {len(pngs)})")
check(len(rgbs) == 750, f"750 test RGB images (got {len(rgbs)})")

print("=== 2. filename <-> test-stem 1:1 ===")
expect = {os.path.basename(r)[:-len('_frame_camera.png')] for r in rgbs}
got = {os.path.basename(p)[:-len('.png')] for p in pngs}
check(expect == got, f"stems match exactly (missing={sorted(expect - got)[:3]}, "
                     f"extra={sorted(got - expect)[:3]})")
pat = re.compile(r'^REC\d{4}_frame_\d{6}$')
bad = [g for g in got if not pat.match(g)]
check(not bad, f"all names match {{sequence}}_frame_{{frame:0>6}} (bad={bad[:3]})")
check(len(got) == 750, f"750 unique names (got {len(got)})")

print("=== 3. per-file format (all 750) ===")
bad_mode, bad_size, bad_dtype, bad_val, pal = [], [], [], [], []
nuniq_hist = Counter()
px = np.zeros(256, dtype=np.int64)
for p in pngs:
    im = Image.open(p)
    if im.mode != 'L':
        bad_mode.append((os.path.basename(p), im.mode))
    if im.size != (1920, 1080):            # PIL: (W, H)
        bad_size.append((os.path.basename(p), im.size))
    if im.palette is not None:
        pal.append(os.path.basename(p))
    a = np.array(im)
    if a.dtype != np.uint8:
        bad_dtype.append((os.path.basename(p), str(a.dtype)))
    if a.shape != (1080, 1920):
        bad_size.append((os.path.basename(p), a.shape))
    mn, mx = int(a.min()), int(a.max())
    if mn < 0 or mx > 18:
        bad_val.append((os.path.basename(p), mn, mx))
    c = np.bincount(a.reshape(-1), minlength=256)
    px += c
    nuniq_hist[int((c > 0).sum())] += 1

check(not bad_mode, f"all mode 'L' single-channel grayscale (bad={bad_mode[:3]})")
check(not pal, f"no palette (bad={pal[:3]})")
check(not bad_size, f"all 1920x1080 / array (1080,1920) (bad={bad_size[:3]})")
check(not bad_dtype, f"all uint8 (bad={bad_dtype[:3]})")
check(not bad_val, f"all values within trainID 0-18, no 255 (bad={bad_val[:3]})")
check(px[255] == 0, f"zero pixels equal 255/ignore (got {int(px[255])})")
check(px[19:].sum() == 0, f"zero pixels outside 0-18 (got {int(px[19:].sum())})")

print("=== 4. degenerate / collapsed predictions ===")
n_le2 = sum(v for k, v in nuniq_hist.items() if k <= 2)
check(n_le2 == 0, f"no image with <=2 distinct classes (got {n_le2})")
check(min(nuniq_hist) >= 3, f"min distinct classes per image = {min(nuniq_hist)}")
print(f"  distinct-classes-per-image histogram: "
      f"{dict(sorted(nuniq_hist.items()))}")
present = [i for i in range(19) if px[i] > 0]
check(len(present) == 19, f"all 19 trainIDs appear somewhere (got {len(present)})")

CLS = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
       "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
       "truck", "bus", "train", "motorcycle", "bicycle"]
tot = px[:19].sum()
print("=== 5. global class pixel share (%) ===")
for i, c in enumerate(CLS):
    print(f"  {i:>2} {c:<15} {100.0 * px[i] / tot:7.4f}")

print()
if fails:
    print(f"RESULT: {len(fails)} CHECK(S) FAILED")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
