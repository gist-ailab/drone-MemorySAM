"""M0-a harness — per-condition, per-class eval of a PLAIN (non-LoRA) CMNeXt
checkpoint on DELIVER, e.g. the official jamycheung/DELIVER weights.

Purpose (P34 fallback branch): if the feasibility probe is rejected, test
whether public SOTA checkpoints ALSO fail on the DELIVER-test dead classes
(Wall / Bridge / Water / TrafficLight), i.e. whether the failure is
data-inherent rather than model-specific.

Why a new script: val.py is hard-wired to SAM2+LoRA (build_sam2 + LoRA_Sam*),
and tools/eval_per_domain.py shells out to val.py, so neither can drive a bare
CMNeXt. val_mm.py CAN (eval(MODEL.NAME) + raw state_dict) but is hardcoded to
split='val' and cases=[None]. This script = val_mm.py's model path +
eval_per_domain.py's per-condition loop, model built ONCE.

Official ckpt (google drive folder 1OWteEOrjfrC3VNg3sxJFZPHz9urkZ3lm):
  cmnext_b2_deliver_rgbdel.pth  (raw state_dict, backbone.*/decode_head.*,
  224.5MB, official mIoU 66.30, md5 e3ba84b58cf38b22117a82adf17c61c6)

Usage:
  python tools/eval_public_cmnext.py \
      --cfg configs/b200-deliver_rgbdel_P34_reliadino.yaml \
      --ckpt /path/to/cmnext_b2_deliver_rgbdel.pth \
      --split test --gpu 0 --out-dir outputs/public_cmnext_eval
  # --cfg is only used for DATASET.{NAME,ROOT,MODALS} + EVAL.IMAGE_SIZE;
  # any DELIVER rgbdel config works. --conditions all -> 10 corner cases too.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.augmentations_mm import get_val_augmentation      # noqa: E402
from semseg.datasets import *                                 # noqa: E402,F401,F403
from semseg.metrics import Metrics                            # noqa: E402
from semseg.models import *                                   # noqa: E402,F401,F403

BASE_CONDITIONS = ['cloud', 'fog', 'night', 'rain', 'sun']
CORNER_CONDITIONS = ['motionblur', 'overexposure', 'underexposure',
                     'lidarjitter', 'eventlowres']
DEAD_CLASSES = ['Wall', 'Bridge', 'Water', 'TrafficLight']


@torch.no_grad()
def evaluate(model, dataloader, device):
    """val_mm.evaluate equivalent (single-scale, no flip)."""
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    for images, labels in tqdm(dataloader, desc='eval', leave=False):
        images = [x.to(device, non_blocking=True) for x in images]
        labels = labels.to(device, non_blocking=True)
        preds = model(images).softmax(dim=1)
        metrics.update(preds, labels)
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    return acc, macc, f1, mf1, ious, miou


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cfg', required=True,
                    help='any DELIVER yaml (DATASET.* + EVAL.IMAGE_SIZE reused)')
    ap.add_argument('--ckpt', required=True,
                    help='plain CMNeXt raw state_dict .pth (official weights)')
    ap.add_argument('--backbone', default='CMNeXt-B2')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--conditions', default='base',
                    help="'base' (cloud,fog,night,rain,sun), 'all' (+5 corner "
                         "cases), 'none' (merged, case=None), or CSV list")
    ap.add_argument('--gpu', default='0', help='CUDA device id, or "cpu"')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', default=None,
                    help='default: <ckpt_dir>/public_cmnext_<split>_<ts>')
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    ds_cfg = cfg['DATASET']
    image_size = cfg.get('EVAL', {}).get('IMAGE_SIZE', [1024, 1024])
    modals = ds_cfg['MODALS']

    if args.conditions == 'base':
        cases = list(BASE_CONDITIONS)
    elif args.conditions == 'all':
        cases = BASE_CONDITIONS + CORNER_CONDITIONS
    elif args.conditions == 'none':
        cases = [None]
    else:
        cases = [c.strip() for c in args.conditions.split(',') if c.strip()]

    device = torch.device('cpu' if args.gpu == 'cpu' else f'cuda:{args.gpu}')
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f'checkpoint not found: {ckpt_path}'
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir or ckpt_path.parent / f'public_cmnext_{args.split}_{ts}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── plain CMNeXt, built ONCE (val_mm.py:145-146 pattern) ────────────────
    transform = get_val_augmentation(image_size)
    probe_ds = eval(ds_cfg['NAME'])(ds_cfg['ROOT'], args.split, transform, modals, cases[0])
    n_classes = probe_ds.n_classes
    class_names = list(probe_ds.CLASSES)
    model = CMNeXt(args.backbone, n_classes, modals)              # noqa: F405
    state = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:   # tolerate wrapped
        state = state['model_state_dict']
    msg = model.load_state_dict(state, strict=True)
    print(f'[load] {ckpt_path.name}: {msg}')
    model = model.to(device).eval()

    dead_idx = [class_names.index(c) for c in DEAD_CLASSES if c in class_names]
    results = {}
    for case in cases:
        ds = probe_ds if case == cases[0] else \
            eval(ds_cfg['NAME'])(ds_cfg['ROOT'], args.split, transform, modals, case)
        if len(ds) == 0:
            print(f'[skip] {case}: 0 images')
            continue
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True)
        t0 = time.time()
        acc, macc, f1, mf1, ious, miou = evaluate(model, dl, device)
        label = case or 'ALL'
        results[label] = {'n_images': len(ds), 'mIoU': float(miou),
                          'mAcc': float(macc), 'mF1': float(mf1),
                          'per_class_iou': {c: float(v) for c, v in
                                            zip(class_names, ious)}}
        table = {'Class': class_names + ['Mean'], 'IoU': list(ious) + [miou],
                 'F1': list(f1) + [mf1], 'Acc': list(acc) + [macc]}
        rpt = (f"\n===== {ckpt_path.name} | split={args.split} | case={label} "
               f"| {len(ds)} imgs | {time.time()-t0:.0f}s =====\n"
               + tabulate(table, headers='keys', floatfmt='.2f') + '\n')
        print(rpt)
        with open(out_dir / 'report.txt', 'a') as f:
            f.write(rpt)

    # ── dead-class verdict table (the M0-a question) ─────────────────────────
    if results and dead_idx:
        rows = [[lbl, r['n_images'], f"{r['mIoU']:.2f}"]
                + [f"{r['per_class_iou'][c]:.2f}" for c in DEAD_CLASSES]
                for lbl, r in results.items()]
        summary = ('\n===== DEAD-CLASS SUMMARY (public CMNeXt) =====\n'
                   + tabulate(rows, headers=['case', 'imgs', 'mIoU'] + DEAD_CLASSES)
                   + '\n')
        print(summary)
        with open(out_dir / 'report.txt', 'a') as f:
            f.write(summary)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'ckpt': str(ckpt_path), 'split': args.split,
                   'image_size': image_size, 'modals': modals,
                   'results': results}, f, indent=2)
    print(f'[done] report: {out_dir}/report.txt  json: {out_dir}/results.json')


if __name__ == '__main__':
    main()
