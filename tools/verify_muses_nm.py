"""Verify our NM (noisy-modality) implementations against the public benchmark code, bit for bit.

Nothing about the benchmark is re-typed here: the input space and the noise come from the
byte-identical upstream copies in third_party/mmss_robust_bench/
  - upstream/val_mm_NM.py (benchmark repo @53c7a6e): add_salt_and_pepper_noise,
    add_gaussian_noise, called in the upstream order of val_mm_NM.py:104-113
    (S&P on every sensor, then Gaussian on every sensor except event)
  - bench_upstream/augmentations_mm.py (DELIVER @b3a76c3): Normalize, which defines the space the
    noise is added in (RGB /255 + ImageNet normalization, other sensors /255).
Checks
  1. space: upstream Normalize == our semseg Normalize (NORM_ALL_MODALS false) on raw uint8 inputs
  2. DELIVER  tools/missing_modality_eval.bench_noise   (img, depth, event, lidar)
  3. MUSES    tools/muses_robust_eval.bench_nm          (img, lidar, event, radar; radar like LiDAR)
  4. d2 hook  third_party/dgfusion_train_restore/muses_robust/muses_robust_hook (raw 0-255 inputs)
     for NM, and EMM fill (= 0 in the upstream space)
Each noise check runs upstream Normalize on the same raw tensors, applies the upstream functions
with torch.manual_seed(seed), and compares with ours using torch.Generator().manual_seed(seed).

Usage: python tools/verify_muses_nm.py [--out result.json]
"""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BENCH = REPO / 'third_party/mmss_robust_bench'
FILES = {'upstream/val_mm_NM.py': 'b677d2dfbaa4b5d998597f0d2d2d650d',
         'bench_upstream/augmentations_mm.py': '35ee0b7e9b99c68a9d275af95cb0350e'}
LEVELS = {'low': (0.05, 0.1), 'mid': (0.1, 0.2), 'high': (0.2, 0.5)}
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_upstream():
    for rel, want in FILES.items():
        got = md5(BENCH / rel)
        if got != want:
            raise RuntimeError(f'{rel} changed: {got}')
    src = (BENCH / 'upstream/val_mm_NM.py').read_text()
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
            and n.name in ('add_salt_and_pepper_noise', 'add_gaussian_noise')]
    ns = {'np': np, 'torch': torch, 'cfg': {'DEVICE': 'cpu'}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'upstream/val_mm_NM.py', 'exec'), ns)
    spec = importlib.util.spec_from_file_location('bench_aug', BENCH / 'bench_upstream/augmentations_mm.py')
    aug = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aug)
    return ns['add_salt_and_pepper_noise'], ns['add_gaussian_noise'], aug.Normalize()


def raw_sample(names, seed, shape=(3, 40, 56)):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randint(0, 256, shape, generator=g, dtype=torch.uint8) for n in names}


def upstream_space(norm, raw):
    """Upstream Normalize (key 'img' = RGB) on a copy of the raw sample."""
    return norm({k: v.clone() for k, v in raw.items()})


def upstream_noise(space, names, level, seed, sp, gauss):
    d, s = LEVELS[level]
    torch.manual_seed(seed)
    x = [space[n].unsqueeze(0).clone() for n in names]   # batch 1, as our runners
    x = [sp(t, density=d) for t in x]
    for i, n in enumerate(names):
        if n != 'event':
            x[i] = gauss(x[i], mean=0.0, std=s)
    return x


def maxdiff(a, b):
    return float((a - b).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args()
    sp, gauss, up_norm = load_upstream()
    res = {'upstream_md5': {k: md5(BENCH / k) for k in FILES}}

    # 1. space
    from semseg.augmentations_mm import Normalize as OurNormalize
    names_d = ['img', 'depth', 'event', 'lidar']
    raw = raw_sample(names_d, 1)
    up = upstream_space(up_norm, raw)
    ours = OurNormalize(norm_all_modals=False)({k: v.clone() for k, v in raw.items()})
    res['space_max_abs_diff'] = {n: maxdiff(up[n], ours[n]) for n in names_d}
    res['space_nonrgb_range'] = [float(up['depth'].min()), float(up['depth'].max())]

    # 2. DELIVER bench_noise
    from tools.missing_modality_eval import bench_noise
    res['deliver'] = {}
    for level, (d, s) in LEVELS.items():
        ref = upstream_noise(up, names_d, level, a.seed, sp, gauss)
        got = bench_noise([up[n].unsqueeze(0) for n in names_d], names_d, d, s,
                          torch.Generator().manual_seed(a.seed))
        res['deliver'][level] = {n: maxdiff(r, g) for n, r, g in zip(names_d, ref, got)}

    # 3. MUSES bench_nm (radar handled like LiDAR: upstream functions applied to it too)
    from tools.muses_robust_eval import bench_nm
    names_m = ['img', 'lidar', 'event', 'radar']
    raw_m = raw_sample(names_m, 2)
    up_m = upstream_space(up_norm, raw_m)
    res['muses'] = {}
    for level in LEVELS:
        ref = upstream_noise(up_m, names_m, level, a.seed, sp, gauss)
        got = bench_nm([up_m[n].unsqueeze(0) for n in names_m], level,
                       torch.Generator().manual_seed(a.seed), 'all')
        res['muses'][level] = {n: maxdiff(r, g) for n, r, g in zip(names_m, ref, got)}

    # 4. detectron2 hook: raw 0-255 inputs, camera pixel_mean/std = ImageNet * 255
    hook = importlib.import_module('third_party.dgfusion_train_restore.muses_robust.muses_robust_hook')
    d2 = ('CAMERA', 'LIDAR', 'EVENT_CAMERA', 'RADAR')
    mean = torch.tensor([123.675, 116.28, 103.53, 20., 21., 22., 64., 64., 64., 5., 6., 7.])
    std = torch.tensor([58.395, 57.12, 57.375, 30., 31., 32., 50., 50., 50., 15., 16., 17.])

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = list(d2)
            self.register_buffer('pixel_mean', mean.view(-1, 1, 1))
            self.register_buffer('pixel_std', std.view(-1, 1, 1))
            self.got = None

        def forward(self, batch):
            self.got = batch
            return batch

    def run_hook(env):
        os.environ.update(env)
        m = Fake()
        hook.install(m)
        m([{n: raw_m[k].float() for n, k in zip(d2, names_m)}])
        return m.got[0]

    def to_up_space(x, key):
        x = x.float() / 255.0
        return (x - IMNET_MEAN) / IMNET_STD if key == 'img' else x

    res['d2_hook'] = {'nm': {}}
    for level in LEVELS:
        out = run_hook(dict(MUSES_ROBUST_PROTOCOL='nm', MUSES_ROBUST_LEVEL=level,
                            MUSES_ROBUST_SEED=str(a.seed), MUSES_ROBUST_NM_SCOPE='all'))
        ref = upstream_noise(up_m, names_m, level, a.seed, sp, gauss)
        res['d2_hook']['nm'][level] = {n: maxdiff(to_up_space(out[n], k), ref[i][0])
                                       for i, (n, k) in enumerate(zip(d2, names_m))}
    out = run_hook(dict(MUSES_ROBUST_PROTOCOL='emm', MUSES_ROBUST_SUBSET='img'))
    res['d2_hook']['emm_img_only'] = {
        'camera_unchanged': bool(torch.equal(out['CAMERA'], raw_m['img'].float())),
        'others_zero_in_bench_space': all(bool((out[n] == 0).all()) for n in d2[1:])}

    worst = {'space': max(res['space_max_abs_diff'].values()),
             'deliver': max(v for lv in res['deliver'].values() for v in lv.values()),
             'muses': max(v for lv in res['muses'].values() for v in lv.values()),
             'd2_hook': max(v for lv in res['d2_hook']['nm'].values() for v in lv.values())}
    res['worst'] = worst
    emm = res['d2_hook']['emm_img_only']
    res['pass'] = (worst['space'] <= 1e-6 and worst['deliver'] <= 1e-6 and worst['muses'] <= 1e-6
                   and worst['d2_hook'] <= 1e-4 and emm['camera_unchanged']
                   and emm['others_zero_in_bench_space'])
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
