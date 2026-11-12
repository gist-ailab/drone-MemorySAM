#!/usr/bin/env python3
"""Native-resolution MUSES robustness evaluation for four-sensor ReliaDINO."""
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if '--gpu' in sys.argv:
    index = sys.argv.index('--gpu')
    if index + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[index + 1]

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.eval_muses_official import (MUSES, build_reliadino, get_val_augmentation,
                                       iou_from_hist, letterbox_valid_box,
                                       sanity_check_geometry)
from tools.missing_modality_eval import (bench_sp_noise, enumerate_present_subsets,
                                         missing_of, rmm_degrade, zero_fill)

MODALS = ('img', 'lidar', 'event', 'radar')
LEVELS = {'low': (0.05, 0.1), 'mid': (0.1, 0.2), 'high': (0.2, 0.5)}


def rgb_noise(images, level, generator):
    """DELIVER Table 4 NM, restricted to RGB; S&P then Gaussian in normalized space."""
    density, std = LEVELS[level]
    out = list(images)
    out[0] = bench_sp_noise(out[0], density, generator)
    out[0] = out[0] + torch.randn(out[0].shape, generator=generator).to(out[0].device) * std
    return out


def cases(args):
    if args.protocol == 'emm':
        subsets = enumerate_present_subsets(4)
        if args.subset:
            requested = tuple(i for i, name in enumerate(MODALS) if name in args.subset.split(','))
            if requested not in subsets or len(requested) != len(args.subset.split(',')):
                raise ValueError('Invalid --subset; use a nonempty comma-separated subset of img,lidar,event,radar')
            subsets = [requested]
        return [('+'.join(MODALS[i] for i in s), lambda base, miss=missing_of(s, 4): zero_fill(base, miss))
                for s in subsets]
    if args.protocol == 'rmm':
        ratios = [args.ratio] if args.ratio is not None else [0.25, 0.5, 0.75]
        if any(r not in (0.25, 0.5, 0.75) for r in ratios):
            raise ValueError('--ratio must be 0.25, 0.5 or 0.75')
        result = []
        for r in ratios:
            for s in enumerate_present_subsets(4):
                miss = missing_of(s, 4)
                gen = torch.Generator().manual_seed(args.seed)
                result.append((f'r{r}_' + '+'.join(MODALS[i] for i in s),
                               lambda base, m=miss, ratio=r, g=gen: rmm_degrade(base, m, ratio, g)))
        return result
    if args.protocol == 'nm':
        levels = [args.level] if args.level else list(LEVELS)
        return [(level, lambda base, lev=level, g=torch.Generator().manual_seed(args.seed):
                 rgb_noise(base, lev, g)) for level in levels]
    return [('clean', lambda base: list(base))]


def accumulate(hist, gt, pred, n):
    valid = (gt != 255) & (gt < n) & (pred < n)
    flat = gt[valid].long() * n + pred[valid].long()
    hist += torch.bincount(flat, minlength=n * n).reshape(n, n).double()


def evaluate_case(name, apply, model, loader, n, class_names, device, args, geometry, side, out, epoch):
    hist_full = torch.zeros(n, n, dtype=torch.double, device=device)
    hist_small = torch.zeros_like(hist_full)
    hist_cond = OrderedDict()
    counts = OrderedDict()
    n_done = 0
    for images, label, meta in tqdm(loader, desc=name):
        base = [x.to(device, non_blocking=True) for x in images]
        # MUSES val augmentation has already normalized each tensor here.
        with torch.no_grad():
            logits, _ = model(apply(base), True)
            accumulate(hist_small, label[0].to(device), logits.argmax(1)[0], n)
            h, w = int(meta['orig_h'][0]), int(meta['orig_w'][0])
            t0, t1, l0, l1 = letterbox_valid_box(h, w, logits.shape[-1])
            pred = F.interpolate(logits[:, :, t0:t1, l0:l1], size=(h, w),
                                 mode='bilinear', align_corners=False).argmax(1)[0]
            gt = meta['orig_label'][0].to(device)
            accumulate(hist_full, gt, pred, n)
            img_path = Path(meta['paths']['img'][0])
            cond = '/'.join(img_path.parts[-3:-1])
            if cond not in hist_cond:
                hist_cond[cond] = torch.zeros_like(hist_full)
                counts[cond] = 0
            accumulate(hist_cond[cond], gt, pred, n)
            counts[cond] += 1
            if args.dump:
                target = out / 'pred' / cond / img_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.byte().cpu().numpy()).save(target)
        n_done += 1
        if args.limit and n_done >= args.limit:
            break
    out.mkdir(parents=True, exist_ok=True)
    full_iou, full_p, full_all, present = iou_from_hist(hist_full)
    small_iou, small_p, small_all, _ = iou_from_hist(hist_small)
    report = {'ckpt': str(args.ckpt), 'epoch': epoch, 'n_images': n_done, 'protocol': args.protocol,
              'setting': name, 'geometry': geometry,
              'official_native_1080x1920': {
                  'mIoU_present_classes': round(full_p, 4), 'mIoU_all19_nan_as_0': round(full_all, 4),
                  'per_class': {c: None if np.isnan(v) else round(float(v), 4)
                                for c, v in zip(class_names, full_iou)},
                  'classes_absent_in_gt': [c for c, p in zip(class_names, present) if not p]},
              f'internal_letterbox_{side}': {'mIoU_present_classes': round(small_p, 4),
                                          'mIoU_all19_nan_as_0': round(small_all, 4),
                                          'per_class': {c: None if np.isnan(v) else round(float(v), 4)
                                                        for c, v in zip(class_names, small_iou)}},
              'per_condition_official': {}}
    for cond, hist in hist_cond.items():
        iou, miou, _, present = iou_from_hist(hist)
        report['per_condition_official'][cond] = {
            'n_images': counts[cond], 'mIoU_present_classes': round(miou, 4),
            'n_present_classes': int(present.sum()),
            'per_class': {c: None if np.isnan(v) else round(float(v), 4)
                          for c, v in zip(class_names, iou)}}
        np.save(out / f"hist_{cond.replace('/', '_')}.npy", hist.cpu().numpy())
    np.save(out / 'hist_full.npy', hist_full.cpu().numpy())
    np.save(out / f'hist_{side}.npy', hist_small.cpu().numpy())
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('cfg', 'ckpt', 'out'):
        ap.add_argument('--' + flag, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dataset-root')
    ap.add_argument('--side', type=int)
    ap.add_argument('--protocol', choices=('clean', 'emm', 'rmm', 'nm'), default='clean')
    ap.add_argument('--subset')
    ap.add_argument('--ratio', type=float)
    ap.add_argument('--level', choices=tuple(LEVELS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    dcfg = cfg['DATASET']
    if dcfg['NAME'].upper() != 'MUSES' or tuple(dcfg['MODALS']) != MODALS:
        raise ValueError('Expected four-sensor MUSES config in img,lidar,event,radar order')
    if args.dataset_root:
        dcfg['ROOT'] = args.dataset_root
    if args.side:
        cfg['EVAL']['IMAGE_SIZE'] = [args.side, args.side]
    side = cfg['EVAL']['IMAGE_SIZE'][0]
    geometry = sanity_check_geometry(side=side)
    geometry = {k: list(v) if isinstance(v, tuple) else v for k, v in geometry.items()}
    ds = MUSES(dcfg['ROOT'], 'val', get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'], dataset_cfg=dcfg),
               dcfg['MODALS'], return_meta=True, proj_dir=dcfg.get('PROJ_DIR', 'projected_to_rgb'))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda')
    model = build_reliadino(cfg, ds.n_classes)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Checkpoint mismatch: {result}')
    model = model.to(device).eval()
    rows = []
    evaluations = cases(args)
    for name, apply in evaluations:
        out = Path(args.out) if len(evaluations) == 1 else Path(args.out) / name
        report = evaluate_case(name, apply, model, loader, ds.n_classes, ds.CLASSES,
                               device, args, geometry, side, out, ck.get('epoch'))
        rows.append({'subset': name, 'mIoU': report['official_native_1080x1920']['mIoU_present_classes'],
                     **{k: v['mIoU_present_classes'] for k, v in report['per_condition_official'].items()}})
    if args.protocol in ('emm', 'rmm') and len(rows) > 1:
        groups = {'emm': rows} if args.protocol == 'emm' else {
            f'r{r}': [row for row in rows if row['subset'].startswith(f'r{r}_')]
            for r in ([args.ratio] if args.ratio is not None else [0.25, 0.5, 0.75])}
        for group, members in groups.items():
            keys = set.intersection(*(set(row) for row in members)) - {'subset'}
            rows.append({'subset': f'{group}_mean15',
                         **{key: round(sum(float(row[key]) for row in members) / len(members), 4)
                            for key in keys}})
    if len(rows) > 1:
        cols = ['subset', 'mIoU'] + sorted(set().union(*(row.keys() for row in rows)) - {'subset', 'mIoU'})
        with (Path(args.out) / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
