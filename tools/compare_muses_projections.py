#!/usr/bin/env python
"""
Verify + quantitatively compare two MUSES projection sets (baseline vs DGFusion-param).

Checks per modality: file count, no failures, shape (1080,1920,3), dtype
(lidar/radar uint16, event uint8); then coverage % (fraction of pixels carrying a real
measurement) and per-channel stats, baseline vs new.

Coverage definition (important):
  lidar/radar PNGs store (raw + 100) * 150 as uint16, so background raw 0 -> 15000.
  A pixel is "covered" iff any channel != 15000. Using !=0 would report 100% and is wrong.
  event PNGs are raw uint8 counts, background 0 -> covered iff any channel != 0.

Channel stats are reported in RAW units (lidar/radar: png/150 - 100), over covered
pixels only, so they are comparable to DGFusion's DATASETS.PIXEL_MEAN.
"""
import argparse
import json
import os
import numpy as np
import cv2

MODS = {
    'lidar':        dict(sub='lidar',        suffix='_lidar.png',        dtype=np.uint16, bg=15000,
                         chans=['range_m', 'intensity', 'height_m']),
    'event_camera': dict(sub='event_camera', suffix='_event_camera.png', dtype=np.uint8,  bg=0,
                         chans=['pos_count', 'neg_count', 'zero']),
    'radar':        dict(sub='radar',        suffix='_radar.png',        dtype=np.uint16, bg=15000,
                         chans=['range_m', 'intensity', 'zero']),
}


def raw(img, modality):
    if MODS[modality]['bg'] == 15000:
        return img.astype(np.float64) / 150.0 - 100.0
    return img.astype(np.float64)


def stats_for(path, modality):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, 'unreadable'
    err = None
    if im.shape != (1080, 1920, 3):
        err = f'shape {im.shape}'
    if im.dtype != MODS[modality]['dtype']:
        err = (err or '') + f' dtype {im.dtype}'
    bg = MODS[modality]['bg']
    cov = (im != bg).any(axis=2)
    r = raw(im, modality)
    out = dict(cov=float(cov.mean()))
    for c in range(3):
        v = r[:, :, c][cov]
        out[f'c{c}'] = dict(mean=float(v.mean()) if v.size else 0.0,
                            min=float(v.min()) if v.size else 0.0,
                            max=float(v.max()) if v.size else 0.0,
                            p50=float(np.percentile(v, 50)) if v.size else 0.0,
                            p99=float(np.percentile(v, 99)) if v.size else 0.0)
    # global (all-pixel) mean, directly comparable to DGFusion DATASETS.PIXEL_MEAN
    out['global_mean'] = [float(r[:, :, c].mean()) for c in range(3)]
    return out, err


def walk(root, sub, suffix):
    hits = []
    for dp, _, fns in os.walk(os.path.join(root, sub)):
        for fn in fns:
            if fn.endswith(suffix):
                hits.append(os.path.join(dp, fn))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--muses_root', default='/ailab_mat2/dataset/MUSES')
    ap.add_argument('--baseline', default='projected_to_rgb')
    ap.add_argument('--new', default='projected_to_rgb_dgf')
    ap.add_argument('--sample', type=int, default=60, help='files per modality for stats')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    report = {'baseline': a.baseline, 'new': a.new, 'modalities': {}}
    for m, spec in MODS.items():
        base_files = walk(os.path.join(a.muses_root, a.baseline), spec['sub'], spec['suffix'])
        new_files = walk(os.path.join(a.muses_root, a.new), spec['sub'], spec['suffix'])
        rel_b = {os.path.relpath(f, os.path.join(a.muses_root, a.baseline)) for f in base_files}
        rel_n = {os.path.relpath(f, os.path.join(a.muses_root, a.new)) for f in new_files}
        entry = dict(n_baseline=len(base_files), n_new=len(new_files),
                     missing_in_new=sorted(rel_b - rel_n)[:10],
                     extra_in_new=sorted(rel_n - rel_b)[:10],
                     n_missing=len(rel_b - rel_n))
        # deterministic sample present in both
        common = sorted(rel_b & rel_n)
        idx = np.linspace(0, len(common) - 1, min(a.sample, len(common))).astype(int)
        sample = [common[i] for i in idx]
        errs, agg = [], {'baseline': [], 'new': []}
        for rel in sample:
            for tag, folder in (('baseline', a.baseline), ('new', a.new)):
                s, e = stats_for(os.path.join(a.muses_root, folder, rel), m)
                if e:
                    errs.append(f'{tag}:{rel}:{e}')
                if s:
                    agg[tag].append(s)
        for tag in ('baseline', 'new'):
            if not agg[tag]:
                continue
            entry[tag] = dict(
                n_sampled=len(agg[tag]),
                coverage_pct=float(np.mean([x['cov'] for x in agg[tag]]) * 100),
                coverage_pct_std=float(np.std([x['cov'] for x in agg[tag]]) * 100),
                global_mean=[float(np.mean([x['global_mean'][c] for x in agg[tag]])) for c in range(3)],
                channels={spec['chans'][c]: {
                    k: float(np.mean([x[f'c{c}'][k] for x in agg[tag]]))
                    for k in ('mean', 'min', 'max', 'p50', 'p99')} for c in range(3)},
            )
        if 'baseline' in entry and 'new' in entry:
            b, n = entry['baseline']['coverage_pct'], entry['new']['coverage_pct']
            entry['coverage_ratio_new_over_baseline'] = float(n / b) if b else None
        entry['errors'] = errs
        report['modalities'][m] = entry

    txt = json.dumps(report, indent=2)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w') as f:
            f.write(txt)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
