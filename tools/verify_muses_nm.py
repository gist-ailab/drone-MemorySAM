"""Verify tools/muses_robust_eval.bench_nm against the public NM benchmark code, bit for bit.

The two noise functions are taken verbatim from the byte-identical upstream copy
third_party/mmss_robust_bench/upstream/val_mm_NM.py (Chenfei-Liao/Multi-Modal-Semantic-
Segmentation-Robustness-Benchmark @53c7a6e, md5 b677d2dfbaa4b5d998597f0d2d2d650d) and
called in the upstream order (val_mm_NM.py:104-113: S&P on every sensor, then Gaussian on
every sensor except event) on tensors in the upstream input space (RGB ImageNet-normalized,
other sensors raw 0-255). Radar is appended to the S&P and Gaussian lists like LiDAR.

Usage:
  python tools/verify_muses_nm.py                       # synthetic tensors only
  python tools/verify_muses_nm.py --cfg <muses cfg> --n 3   # also real MUSES val samples
Writes a JSON summary with --out.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.muses_robust_eval import LEVELS, MODALS, bench_nm  # noqa: E402

UPSTREAM = REPO / 'third_party/mmss_robust_bench/upstream/val_mm_NM.py'
UPSTREAM_MD5 = 'b677d2dfbaa4b5d998597f0d2d2d650d'


def load_upstream():
    src = UPSTREAM.read_bytes()
    md5 = hashlib.md5(src).hexdigest()
    if md5 != UPSTREAM_MD5:
        raise RuntimeError(f'upstream copy changed: {md5}')
    tree = ast.parse(src.decode())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ('add_salt_and_pepper_noise', 'add_gaussian_noise')]
    ns = {'np': np, 'torch': torch, 'cfg': {'DEVICE': 'cpu'}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(UPSTREAM), 'exec'), ns)
    return ns['add_salt_and_pepper_noise'], ns['add_gaussian_noise'], md5


def upstream_nm(images, level, seed, sp, gauss):
    density, std = LEVELS[level]
    torch.manual_seed(seed)
    up = [x.clone() if m == 'img' else x * 255.0 for x, m in zip(images, MODALS)]
    for i in range(len(up)):
        up[i] = sp(up[i], density=density)
    for i, m in enumerate(MODALS):
        if m != 'event':
            up[i] = gauss(up[i], mean=0.0, std=std)
    return [x if m == 'img' else x / 255.0 for x, m in zip(up, MODALS)]


def compare(images, level, seed, sp, gauss):
    ref = upstream_nm(images, level, seed, sp, gauss)
    ours = bench_nm(images, level, torch.Generator().manual_seed(seed), 'all')
    rec = {}
    for m, x, a, b in zip(MODALS, images, ref, ours):
        scale = 1.0 if m == 'img' else 255.0  # report in bench units
        d = (b - x) * scale
        rec[m] = {'max_abs_diff_vs_upstream_bench_units': float(((a - b) * scale).abs().max()),
                  'changed_fraction': float((d != 0).float().mean()),
                  'noise_std_bench_units': float(d.std())}
    return rec


def check_d2_hook(sp, gauss, seed):
    """Same comparison for the detectron2 input hook used for DGFusion/CAFuser (raw 0-255 inputs)."""
    import importlib, os
    hook_mod = importlib.import_module('third_party.dgfusion_train_restore.muses_robust.muses_robust_hook')
    names = ('CAMERA', 'LIDAR', 'EVENT_CAMERA', 'RADAR')
    mean = torch.tensor([123.675, 116.28, 103.53, 20., 21., 22., 64., 64., 64., 5., 6., 7.])
    std = torch.tensor([58.395, 57.12, 57.375, 30., 31., 32., 50., 50., 50., 15., 16., 17.])

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = list(names)
            self.register_buffer('pixel_mean', mean.view(-1, 1, 1))
            self.register_buffer('pixel_std', std.view(-1, 1, 1))
            self.got = None

        def forward(self, batch):
            self.got = batch
            return batch

    g = torch.Generator().manual_seed(7)
    raw = {n: torch.randint(0, 256, (3, 48, 64), generator=g).float() for n in names}
    out = {'nm': {}, 'emm': {}}
    for level in LEVELS:
        os.environ.update(MUSES_ROBUST_PROTOCOL='nm', MUSES_ROBUST_LEVEL=level, MUSES_ROBUST_SEED=str(seed),
                          MUSES_ROBUST_NM_SCOPE='all')
        m = Fake()
        hook_mod.install(m)
        m([{n: raw[n].clone() for n in names}])
        mu, sd = mean[:3].view(3, 1, 1), std[:3].view(3, 1, 1)
        ours_space = [((raw['CAMERA'] - mu) / sd).unsqueeze(0)] + [(raw[n] / 255.0).unsqueeze(0) for n in names[1:]]
        ref = upstream_nm(ours_space, level, seed, sp, gauss)
        diffs = {}
        for i, n in enumerate(names):
            got = m.got[0][n]
            got_b = (got - mu) / sd if n == 'CAMERA' else got
            ref_b = ref[i][0] if n == 'CAMERA' else ref[i][0] * 255.0
            diffs[n] = float((got_b - ref_b).abs().max())
        out['nm'][level] = diffs
    os.environ.update(MUSES_ROBUST_PROTOCOL='emm', MUSES_ROBUST_SUBSET='img')
    m = Fake()
    hook_mod.install(m)
    m([{n: raw[n].clone() for n in names}])
    out['emm']['img_only'] = {
        'camera_unchanged': bool(torch.equal(m.got[0]['CAMERA'], raw['CAMERA'])),
        'others_raw_zero': all(bool((m.got[0][n] == 0).all()) for n in names[1:])}
    return out


def check_deliver_bench_noise(sp, gauss, seed):
    """tools/missing_modality_eval.bench_noise (DELIVER, sensors img/depth/event/lidar) vs upstream order
    val_mm_NM.py:104-113 (S&P RGB, D, E, L; Gaussian RGB, D, L) in the upstream input space."""
    from tools.missing_modality_eval import bench_noise
    names = ['img', 'depth', 'event', 'lidar']
    g = torch.Generator().manual_seed(11)
    base = [torch.randn(2, 3, 40, 56, generator=g)] + [torch.rand(2, 3, 40, 56, generator=g) for _ in names[1:]]
    out = {}
    for level, (d, s_) in LEVELS.items():
        torch.manual_seed(seed)
        up = [base[0].clone()] + [x * 255.0 for x in base[1:]]
        up = [sp(x, density=d) for x in up]
        for i, n in enumerate(names):
            if n != 'event':
                up[i] = gauss(up[i], mean=0.0, std=s_)
        ref = [up[0]] + [x / 255.0 for x in up[1:]]
        ours = bench_noise(base, names, d, s_, torch.Generator().manual_seed(seed))
        out[level] = {n: float(((a - b) * (1.0 if n == 'img' else 255.0)).abs().max())
                      for n, a, b in zip(names, ref, ours)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args()
    sp, gauss, md5 = load_upstream()
    result = {'upstream': str(UPSTREAM.relative_to(REPO)), 'upstream_md5': md5, 'synthetic': {}, 'muses': []}
    g = torch.Generator().manual_seed(123)
    synth = [torch.randn(1, 3, 64, 96, generator=g)] + \
            [torch.rand(1, 3, 64, 96, generator=g) * 0.4 for _ in MODALS[1:]]
    for level in LEVELS:
        result['synthetic'][level] = compare(synth, level, a.seed, sp, gauss)
    if a.cfg:
        import yaml
        from tools.eval_muses_official import MUSES, get_val_augmentation
        cfg = yaml.safe_load(Path(a.cfg).read_text())
        d = cfg['DATASET']
        ds = MUSES(d['ROOT'], 'val', get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'], dataset_cfg=d),
                   d['MODALS'], return_meta=True, proj_dir=d.get('PROJ_DIR', 'projected_to_rgb'))
        for k in range(a.n):
            sample = ds[k]
            images = [x.unsqueeze(0) for x in sample[0]]
            entry = {'index': k, 'input_range': {m: [float(x.min()), float(x.max())]
                                                 for m, x in zip(MODALS, images)}}
            for level in LEVELS:
                entry[level] = compare(images, level, a.seed + k, sp, gauss)
            result['muses'].append(entry)
    result['d2_hook'] = check_d2_hook(sp, gauss, a.seed)
    result['deliver_bench_noise'] = check_deliver_bench_noise(sp, gauss, a.seed)
    deliver_worst = max(v for lv in result['deliver_bench_noise'].values() for v in lv.values())
    result['deliver_bench_noise_worst_max_abs_diff'] = deliver_worst
    hook_worst = max(v for lv in result['d2_hook']['nm'].values() for v in lv.values())
    worst = max(v['max_abs_diff_vs_upstream_bench_units']
                for block in [result['synthetic']] + result['muses']
                for lv, mods in block.items() if lv in LEVELS for v in mods.values())
    result['worst_max_abs_diff'] = worst
    result['d2_hook_worst_max_abs_diff'] = hook_worst
    emm = result['d2_hook']['emm']['img_only']
    result['pass'] = worst <= 1e-4 and deliver_worst <= 1e-3 and hook_worst <= 1e-3 and emm['camera_unchanged'] and emm['others_raw_zero']
    text = json.dumps(result, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    sys.exit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
