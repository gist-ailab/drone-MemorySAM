#!/usr/bin/env python
"""
Independent oracle check: DGFusion publishes DATASETS.PIXEL_MEAN/STD for LIDAR and RADAR
in its MUSES config. Those are global per-channel statistics over the TRAIN split in RAW
sensor units (background pixels included as 0).

If our re-projection truly reproduces DGFusion's pipeline, the stats we measure on our
regenerated train-split PNGs must land on their published numbers. This validates the
parameters end-to-end without needing their code to run.

  lidar/radar PNG -> raw = png/150 - 100  (background 15000 -> 0)
  event PNG is already raw uint8 counts, but DGFusion's event mean is in [0,1] units
  (it divides by 255), so we scale accordingly.
"""
import argparse
import os
import numpy as np
import cv2

DGF_PUBLISHED = {
    'lidar': dict(mean=[7.6631646, 9.861262, 0.23872735],
                  std=[16.53752009, 23.13748461, 1.71006056]),
    'radar': dict(mean=[3.5392952, 8.275344, 0.0],
                  std=[13.59349095, 18.27715183, 1.0]),
    'event_camera': dict(mean=[0.12577528, 0.12728328, 0.0],
                         std=[0.54420582, 0.47625199, 1.0]),
}
CAFUSER_PUBLISHED = {
    'lidar': dict(mean=[4.91737, 6.149373, 0.27025607],
                  std=[13.34715295, 17.63751258, 1.23208644]),
    'radar': dict(mean=[4.003564, 9.342789, 0.0],
                  std=[14.60685343, 19.29406236, 1.0]),
}
SUB = {'lidar': ('lidar', '_lidar.png'), 'radar': ('radar', '_radar.png'),
       'event_camera': ('event_camera', '_event_camera.png')}


def raw(im, m):
    # Event PIXEL_MEAN in the DGFusion config (~0.126) is in RAW EVENT-COUNT units,
    # not [0,1]: measured raw-count means land on it directly. No /255 rescale.
    if m == 'event_camera':
        return im.astype(np.float64)
    return im.astype(np.float64) / 150.0 - 100.0


def collect(root, folder, m, n):
    sub, suf = SUB[m]
    files = []
    for dp, _, fns in os.walk(os.path.join(root, folder, sub, 'train')):
        files += [os.path.join(dp, f) for f in fns if f.endswith(suf)]
    files.sort()
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)
    files = [files[i] for i in idx]
    # streaming mean/var over all pixels
    s = np.zeros(3); ss = np.zeros(3); cnt = 0
    for f in files:
        r = raw(cv2.imread(f, cv2.IMREAD_UNCHANGED), m)
        s += r.reshape(-1, 3).sum(0); ss += (r.reshape(-1, 3) ** 2).sum(0)
        cnt += r.shape[0] * r.shape[1]
    mean = s / cnt
    std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 0))
    return mean, std, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--muses_root', default='/ailab_mat2/dataset/MUSES')
    ap.add_argument('--folders', default='projected_to_rgb,projected_to_rgb_dgf')
    ap.add_argument('--n', type=int, default=120)
    a = ap.parse_args()
    folders = a.folders.split(',')
    for m in ['lidar', 'radar', 'event_camera']:
        print(f'\n===== {m}  (train split, {a.n} files sampled, RAW units, all pixels)')
        pub = DGF_PUBLISHED.get(m)
        if pub:
            print(f'  DGFusion published mean = {np.round(pub["mean"], 4)}')
            print(f'  DGFusion published std  = {np.round(pub["std"], 4)}')
        caf = CAFUSER_PUBLISHED.get(m)
        if caf:
            print(f'  CAFuser  published mean = {np.round(caf["mean"], 4)}')
        for folder in folders:
            p = os.path.join(a.muses_root, folder, SUB[m][0], 'train')
            if not os.path.isdir(p):
                print(f'  {folder:26s} MISSING')
                continue
            mean, std, n = collect(a.muses_root, folder, m, a.n)
            line = f'  {folder:26s} mean = {np.round(mean, 4)}  std = {np.round(std, 4)}  (n={n})'
            if pub:
                dm = np.abs(mean - np.array(pub['mean']))
                rel = dm / np.maximum(np.abs(pub['mean']), 1e-6)
                line += f'\n  {"":26s} |Δ| vs DGFusion mean = {np.round(dm, 4)}  rel = {np.round(rel * 100, 1)}%'
            print(line)


if __name__ == '__main__':
    main()
