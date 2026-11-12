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
    salt = tuple(torch.randint(0, size, (count,), generator=generator) for size in x.shape)
    out[salt] = x.max()
    pepper = tuple(torch.randint(0, size, (count,), generator=generator) for size in x.shape)
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

    def hook(_module, inputs):
        for sample in inputs[0]:
            for name in model.modalities:
                if protocol == 'nm' and name != 'CAMERA':
                    continue
                if protocol in ('emm', 'rmm') and name not in missing:
                    continue
                raw = sample[name]
                if raw.ndim != 3 or raw.shape[0] != 3:
                    raise RuntimeError(f'{name}: expected 3,H,W raw tensor, got {tuple(raw.shape)}')
                mu, sd = (v.to(device=raw.device, dtype=torch.float32) for v in stats[name])
                if protocol == 'emm':
                    result = mu.expand_as(raw).clone()
                elif protocol == 'rmm':
                    keep = (torch.rand(raw.shape, generator=generator) >= ratio).to(raw.device)
                    result = torch.where(keep, raw, mu.expand_as(raw))
                else:
                    normalized = (raw.float() - mu) / sd
                    noisy = bench_sp_noise(normalized.unsqueeze(0), density, generator)[0]
                    noisy = noisy + torch.randn(noisy.shape, generator=generator).to(noisy.device) * sigma
                    result = noisy * sd + mu
                sample[name] = result
                if name == 'CAMERA':
                    sample['image'] = result
        return inputs

    model.register_forward_pre_hook(hook)
    print(f'[muses_robust] protocol={protocol} modalities={model.modalities} seed={seed}', flush=True)
