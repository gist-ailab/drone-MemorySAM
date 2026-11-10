#!/usr/bin/env python3
"""Dump a trained ReliaDINO checkpoint's predictions on a WHOLE evaluation split, in the paper's
qualbank format (_paper/ieee-access-2026/figures/qualbank/), for any dataset class in semseg.datasets.

Why: qualitative figures pick scenes from every image of the split (user 2026-11-02: "인퍼런스는 피규어화
할 수 있게 항상 준비"). tools/baseline_failure/dump_preds_ours.py is DELIVER-only (image-id rule and image
counts are fixed there); this tool reuses the trainer's own evaluation path instead:
  - dataset and transform built exactly as train_reliadino.py (get_val_augmentation(EVAL.IMAGE_SIZE),
    eval(DATASET.NAME)(ROOT, split, ...)), batch 1, no shuffle;
  - model = build_reliadino(cfg, n_classes) + checkpoint 'model_state_dict' (missing/unexpected printed);
  - prediction = argmax of model(images, True)[0], the same tensor train_reliadino.evaluate() scores.
Output (<out>/):
  pred/<image_id>.png   uint8 train-ID at label resolution
  gt/<image_id>.png     uint8 train-ID, ignore = 255
  summary.json          dataset mIoU recomputed with semseg.metrics.Metrics (must match the trainer's log)
image_id = dataset.files[i] made relative to DATASET.ROOT without extension (MCubeS: the list-file name).
Usage:
  python tools/qualbank_dump.py --cfg <train cfg> --model_path <ckpt> --split val --out <dir> [--expected-miou X]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from semseg.augmentations_mm import get_val_augmentation  # noqa: E402
from semseg.datasets import *  # noqa: E402,F401,F403  (eval(DATASET.NAME) as in train_reliadino.py)
from semseg.metrics import Metrics  # noqa: E402
from semseg.models.reliadino import build_reliadino  # noqa: E402


def image_id(ds, i, root):
    f = str(ds.files[i])
    r = str(root).rstrip('/')
    if f.startswith(r + '/'):
        f = f[len(r) + 1:]
    return os.path.splitext(f)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-miou', type=float, default=None)
    ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--root', default=None, help='override DATASET.ROOT (e.g. the same data on another machine)')
    ap.add_argument('--arch', choices=['reliadino', 'cmnext'], default='reliadino',
                    help='cmnext = released CMNeXt weights (semseg.models.cmnext, MODEL.BACKBONE e.g. CMNeXt-B2), '
                         'loaded strictly; used for public-baseline qualitative dumps (DRN-261027-01)')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding='utf-8'))
    dcfg, ecfg = cfg['DATASET'], cfg['EVAL']
    if a.root:
        dcfg['ROOT'] = a.root
    tf = get_val_augmentation(ecfg['IMAGE_SIZE'], dataset_cfg=dcfg)
    ds = eval(dcfg['NAME'])(dcfg['ROOT'], a.split, tf, dcfg['MODALS'])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.model_path, map_location='cpu')
    if a.arch == 'cmnext':
        from semseg.models.cmnext import CMNeXt
        model = CMNeXt(cfg['MODEL']['BACKBONE'], ds.n_classes, dcfg['MODALS'])
        sd = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
        res = model.load_state_dict(sd, strict=True)
        ck = {}
    else:
        model = build_reliadino(cfg, ds.n_classes)
        res = model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} epoch={ck.get("epoch")}')
    model.to(dev).eval()
    out = Path(a.out)
    metrics = Metrics(ds.n_classes, ds.ignore_label, dev)
    # CPU fused SDPA returns NaN on heavily -inf-masked rows of the query head's masked attention
    # (found 2026-11-02, torch 2.3 CPU); the MATH backend computes the same softmax attention exactly.
    import contextlib
    from torch.nn.attention import sdpa_kernel, SDPBackend
    ctx = sdpa_kernel(SDPBackend.MATH) if dev.type == 'cpu' else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        for i, (images, labels) in enumerate(tqdm(dl, desc='dump')):
            images = [x.to(dev) for x in images]
            labels = labels.to(dev)
            logits = model(images) if a.arch == 'cmnext' else model(images, True)[0]
            metrics.update(logits.softmax(dim=1), labels)
            pred = logits.argmax(1)[0].byte().cpu().numpy()
            gt = labels[0].cpu().numpy().astype(np.int64)
            gt = np.where((gt < 0) | (gt >= 255), 255, gt).astype(np.uint8)
            iid = image_id(ds, i, dcfg['ROOT'])
            for sub, arr in (('pred', pred), ('gt', gt)):
                p = out / sub / f'{iid}.png'
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(p)
    ious, miou = metrics.compute_iou()
    n = len(list((out / 'pred').rglob('*.png')))
    summ = {'cfg': a.cfg, 'model_path': a.model_path, 'epoch': ck.get('epoch'), 'split': a.split,
            'n_images': n, 'dataset_len': len(ds), 'miou': miou, 'ious': ious,
            'missing_keys': len(res.missing_keys), 'unexpected_keys': len(res.unexpected_keys),
            'arch': a.arch, 'script': 'tools/qualbank_dump.py'}
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(f'[dump] {n}/{len(ds)} images, mIoU {miou:.2f} -> {out}')
    assert n == len(ds), 'image count mismatch (flattened ids?)'
    if a.expected_miou is not None:
        assert abs(miou - a.expected_miou) <= a.tol, f'mIoU {miou:.2f} != expected {a.expected_miou}'


if __name__ == '__main__':
    main()
