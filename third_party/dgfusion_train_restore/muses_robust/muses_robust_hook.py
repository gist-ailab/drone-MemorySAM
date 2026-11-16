"""MUSES four-sensor detectron2 input hook; install after checkpoint loading.

The model normalizes each raw modality as (raw - pixel_mean slice) / pixel_std
in dgfusion.py/cafuser.py. This hook operates just before that normalization.
"""
import os

import torch


MODALS = ('CAMERA', 'LIDAR', 'EVENT_CAMERA', 'RADAR')
LEVELS = {'low': (0.05, 0.1), 'mid': (0.1, 0.2), 'high': (0.2, 0.5)}
ALIASES = {'img': 'CAMERA', 'camera': 'CAMERA', 'lidar': 'LIDAR',
           'event': 'EVENT_CAMERA', 'event_camera': 'EVENT_CAMERA', 'radar': 'RADAR'}


def parse_subset(value):
    parts = value.lower().split(',')
    subset = {ALIASES[p.strip()] for p in parts}
    if not subset or len(subset) != len(parts):
        raise ValueError('Expected a nonempty present subset of img,lidar,event,radar')
    return subset


def bench_sp_noise(x, density, generator):
    """Exact coordinate sampling of DELIVER val_mm_NM.py:64-77."""
    import math
    count = math.ceil(density * x.numel() * 0.5)
    out = x.clone()
    salt = tuple(torch.randint(0, size, (count,), generator=generator).to(x.device) for size in x.shape)
    out[salt] = x.max()
    pepper = tuple(torch.randint(0, size, (count,), generator=generator).to(x.device) for size in x.shape)
    out[pepper] = x.min()
    return out


def install(model):
    protocol = os.environ.get('MUSES_ROBUST_PROTOCOL', 'clean').lower()
    if protocol not in ('clean', 'emm', 'rmm', 'nm'):
        raise ValueError(f'Unknown protocol {protocol}')
    if protocol == 'clean':
        return
    if set(model.modalities) != set(MODALS) or len(model.modalities) != 4:
        raise RuntimeError(f'Unexpected MUSES model modality order: {model.modalities}')
    mean = model.pixel_mean.detach().flatten().cpu()
    std = model.pixel_std.detach().flatten().cpu()
    if mean.numel() != 12 or std.numel() != 12 or (std <= 0).any():
        raise RuntimeError('Expected four three-channel pixel_mean/std slices')
    stats = {name: (mean[3*i:3*i+3, None, None], std[3*i:3*i+3, None, None])
             for i, name in enumerate(model.modalities)}
    seed = int(os.environ.get('MUSES_ROBUST_SEED', '0'))
    generator = torch.Generator().manual_seed(seed)
    if protocol in ('emm', 'rmm'):
        present = parse_subset(os.environ['MUSES_ROBUST_SUBSET'])
        missing = set(MODALS) - present
        ratio = float(os.environ.get('MUSES_ROBUST_RATIO', '0.25'))
        if protocol == 'rmm' and ratio not in (0.25, 0.5, 0.75):
            raise ValueError('RMM ratio must be 0.25, 0.5 or 0.75')
    else:
        level = os.environ['MUSES_ROBUST_LEVEL'].lower()
        density, sigma = LEVELS[level]
        nm_scope = os.environ.get('MUSES_ROBUST_NM_SCOPE', 'all').lower()  # all = public bench, rgb = RGB only
        if nm_scope not in ('all', 'rgb'):
            raise ValueError('MUSES_ROBUST_NM_SCOPE must be all or rgb')

    # Public bench input space (DELIVER augmentations_mm.Normalize @b3a76c3): RGB is
    # ImageNet-normalized, other sensors are raw 0-255 floats. "Missing" = 0 in that space
    # (RGB -> channel mean, others -> raw 0), NM noise is drawn in that space and in the
    # upstream order (S&P on every target sensor, then Gaussian except event).
    def to_bench(name, raw, mu, sd):
        return (raw.float() - mu) / sd if name == 'CAMERA' else raw.float()

    def from_bench(name, x, mu, sd):
        return x * sd + mu if name == 'CAMERA' else x

    def hook(_module, inputs):
        for sample in inputs[0]:
            names = [n for n in model.modalities
                     if not (protocol == 'nm' and nm_scope == 'rgb' and n != 'CAMERA')
                     and not (protocol in ('emm', 'rmm') and n not in missing)]
            bench = {}
            for name in names:
                raw = sample[name]
                if raw.ndim != 3 or raw.shape[0] != 3:
                    raise RuntimeError(f'{name}: expected 3,H,W raw tensor, got {tuple(raw.shape)}')
                mu, sd = (v.to(device=raw.device, dtype=torch.float32) for v in stats[name])
                bench[name] = (to_bench(name, raw, mu, sd).unsqueeze(0), mu, sd)  # batch 1, as upstream
            if protocol == 'emm':
                for name in names:
                    x, mu, sd = bench[name]
                    bench[name] = (torch.zeros_like(x), mu, sd)
            elif protocol == 'rmm':
                for name in names:
                    x, mu, sd = bench[name]
                    keep = (torch.rand(x.shape, generator=generator) >= ratio).to(x.device)
                    bench[name] = (torch.where(keep, x, torch.zeros_like(x)), mu, sd)
            else:
                for name in names:
                    x, mu, sd = bench[name]
                    bench[name] = (bench_sp_noise(x, density, generator), mu, sd)
                for name in names:
                    if name != 'EVENT_CAMERA':  # bench: no Gaussian on event
                        x, mu, sd = bench[name]
                        bench[name] = (x + (torch.randn(x.size(), generator=generator) * sigma + 0.0).to(x.device), mu, sd)
            for name in names:
                x, mu, sd = bench[name]
                result = from_bench(name, x[0], mu, sd)
                sample[name] = result
                if name == 'CAMERA':
                    sample['image'] = result
        return inputs

    model.register_forward_pre_hook(hook)
    print(f'[muses_robust] protocol={protocol} modalities={model.modalities} seed={seed} '
          f"nm_scope={os.environ.get('MUSES_ROBUST_NM_SCOPE', 'all')} level={os.environ.get('MUSES_ROBUST_LEVEL')} "
          f"subset={os.environ.get('MUSES_ROBUST_SUBSET')} ratio={os.environ.get('MUSES_ROBUST_RATIO')} "
          'space=bench(RGB normalized, others raw 0-255)', flush=True)
