#!/usr/bin/env python3
"""MUSES 공식 val 평가와 같은 경로로 추론·채점하고 원본 해상도 예측·GT PNG와 조건별 혼동행렬을 저장한다.

eval_muses_official.py 와 같은 경로, 동결 파일 미수정.

사용 예:
  python tools/muses_dump_preds.py --cfg <cfg> --ckpt <ckpt> --gpu 0 --out <dir> --save-pred-dir <dir>/preds
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --gpu must be honored before torch initializes CUDA (torch is imported below),
# same early-argv-scan convention as tools/module_diagnostics.py / eval_reliadino_ckpt.py.
if '--gpu' in sys.argv:
    _gi = sys.argv.index('--gpu')
    if _gi + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_gi + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
import torchvision.transforms.functional as TF                      # noqa: E402
import yaml                                                         # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402
from tqdm import tqdm                                               # noqa: E402

from semseg.augmentations_mm import get_val_augmentation            # noqa: E402
from semseg.datasets import *                                       # noqa: F401,F403,E402
from semseg.datasets.muses import MUSES                             # noqa: E402
from semseg.models.reliadino.model import build_reliadino           # noqa: E402
from PIL import Image                                                # noqa: E402
from tools.eval_muses_official import letterbox_valid_box, sanity_check_geometry, iou_from_hist  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True, help='training config used for the ckpt (protocol source of truth)')
    ap.add_argument('--ckpt', required=True, help='*_checkpoint.pth or a raw state_dict')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dataset-root', default=None, help='override DATASET.ROOT only (server-local path)')
    ap.add_argument('--out', required=True, help='output dir for report.json + hist_*.npy')
    ap.add_argument('--save-pred-dir', required=True, help='output dir for per-image pred/ and gt/ PNG files')
    ap.add_argument('--limit', type=int, default=None, help='stop after N images (smoke test)')
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    save_pred_dir = Path(args.save_pred_dir)
    saved_pred_paths = set()

    cfg = yaml.safe_load(open(args.cfg))
    dcfg, ecfg = cfg['DATASET'], cfg['EVAL']
    if str(dcfg.get('NAME', '')).strip().upper() != 'MUSES':
        raise SystemExit(f"--cfg must be a MUSES config (DATASET.NAME={dcfg.get('NAME')!r}); "
                         "this tool's letterbox geometry is MUSES-specific.")
    if args.dataset_root:
        dcfg['ROOT'] = args.dataset_root
    device = torch.device(cfg.get('DEVICE', 'cuda'))

    # build_reliadino sizes the ViT pos-embed from TRAIN.IMAGE_SIZE (model.py:1163)
    # while the val transform resizes to EVAL.IMAGE_SIZE — a mismatch silently
    # evaluates on a geometry the backbone was not interpolated for.
    side = ecfg['IMAGE_SIZE'][0]
    train_side = cfg.get('TRAIN', {}).get('IMAGE_SIZE', [side])[0]
    if train_side != side:
        print(f"[warn] TRAIN.IMAGE_SIZE={train_side} != EVAL.IMAGE_SIZE={side} — "
              f"backbone pos-embed is built for {train_side}.", flush=True)

    geo = sanity_check_geometry(side=side)
    print(f"[sanity] letterbox inverse verified @side={side}: {geo}", flush=True)

    valtransform = get_val_augmentation(ecfg['IMAGE_SIZE'], dataset_cfg=dcfg)
    ds = MUSES(dcfg['ROOT'], 'val', valtransform, dcfg['MODALS'], return_meta=True,
               proj_dir=dcfg.get('PROJ_DIR', 'projected_to_rgb'))
    n_classes, class_names = ds.n_classes, ds.CLASSES
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = build_reliadino(cfg, n_classes)
    # weights_only=False: our ckpts carry optimizer/scaler state, not just tensors
    # (torch>=2.6 flips this default and would refuse to load them).
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)
    msg = model.load_state_dict(state, strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, \
        f"state_dict mismatch: missing={msg.missing_keys[:3]} unexpected={msg.unexpected_keys[:3]}"
    print(f"[ckpt] {Path(args.ckpt).name} epoch={ck.get('epoch', '?')} loaded clean", flush=True)
    model = model.to(device).eval()

    hist_full = torch.zeros(n_classes, n_classes, dtype=torch.double, device=device)
    hist_1024 = torch.zeros(n_classes, n_classes, dtype=torch.double, device=device)
    hist_cond = OrderedDict()
    n_cond = OrderedDict()          # images actually accumulated per condition

    def accum(h, gt, pred):
        keep = (gt != 255) & (gt < n_classes) & (pred < n_classes)
        if keep.sum() == 0:
            return
        flat = gt[keep].long() * n_classes + pred[keep].long()
        h += torch.bincount(flat, minlength=n_classes ** 2)[:n_classes ** 2] \
            .view(n_classes, n_classes).double()

    n_done = 0
    with torch.no_grad():
        for images, label, meta in tqdm(loader, desc='official-eval'):
            images = [x.to(device, non_blocking=True) for x in images]
            label = label.to(device, non_blocking=True)

            logits, _ = model(images, True)                       # (1,19,1024,1024), fp32

            # --- (A) trainer-internal protocol: score at the letterboxed 1024^2
            accum(hist_1024, label.squeeze(0), logits.argmax(1).squeeze(0))

            # --- (B) official protocol: un-letterbox -> native res -> score vs orig GT
            H, W = int(meta['orig_h'][0]), int(meta['orig_w'][0])
            t0, t1, l0, l1 = letterbox_valid_box(H, W, logits.shape[-1])
            crop = logits[:, :, t0:t1, l0:l1]
            up = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
            pred_full = up.argmax(1).squeeze(0)
            gt_full = meta['orig_label'][0].to(device)
            assert gt_full.shape == pred_full.shape, f"{gt_full.shape} vs {pred_full.shape}"

            accum(hist_full, gt_full, pred_full)

            cond = '/'.join(Path(meta['paths']['img'][0]).parts[-3:-1])   # weather/tod
            if cond not in hist_cond:
                hist_cond[cond] = torch.zeros(n_classes, n_classes, dtype=torch.double,
                                              device=device)
                n_cond[cond] = 0
            accum(hist_cond[cond], gt_full, pred_full)
            n_cond[cond] += 1

            stem = Path(meta['paths']['img'][0]).stem
            pred_path = save_pred_dir / 'pred' / cond / f'{stem}.png'
            gt_path = save_pred_dir / 'gt' / cond / f'{stem}.png'
            assert pred_path not in saved_pred_paths, f"duplicate prediction path: {pred_path}"
            saved_pred_paths.add(pred_path)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_full.cpu().numpy().astype(np.uint8)).save(pred_path)
            Image.fromarray(gt_full.cpu().numpy().astype(np.uint8)).save(gt_path)

            n_done += 1
            if args.limit and n_done >= args.limit:
                break

    iou_f, miou_f_present, miou_f_all, present_f = iou_from_hist(hist_full)
    iou_s, miou_s_present, miou_s_all, _ = iou_from_hist(hist_1024)

    report = {
        'ckpt': str(args.ckpt),
        'epoch': ck.get('epoch', None),
        'n_images': n_done,
        'pred_dump_dir': str(save_pred_dir),
        'n_pred_saved': len(saved_pred_paths),
        'geometry': {k: list(v) if isinstance(v, tuple) else v for k, v in geo.items()},
        'official_native_1080x1920': {
            'mIoU_present_classes': round(miou_f_present, 4),
            'mIoU_all19_nan_as_0': round(miou_f_all, 4),
            'per_class': {c: (None if np.isnan(v) else round(float(v), 4))
                          for c, v in zip(class_names, iou_f)},
            'classes_absent_in_gt': [c for c, p in zip(class_names, present_f) if not p],
        },
        'internal_letterbox_1024': {
            'mIoU_present_classes': round(miou_s_present, 4),
            'mIoU_all19_nan_as_0': round(miou_s_all, 4),
            'per_class': {c: (None if np.isnan(v) else round(float(v), 4))
                          for c, v in zip(class_names, iou_s)},
        },
        'per_condition_official': {},
    }
    for cond, h in hist_cond.items():
        iou_c, miou_c_p, miou_c_all, present_c = iou_from_hist(h)
        report['per_condition_official'][cond] = {
            # images ACTUALLY accumulated (the old count scanned the whole val list,
            # which disagreed with the histogram whenever --limit was used).
            'n_images': int(n_cond[cond]),
            'mIoU_present_classes': round(miou_c_p, 4),
            'n_present_classes': int(present_c.sum()),
            'per_class': {c: (None if np.isnan(v) else round(float(v), 4))
                          for c, v in zip(class_names, iou_c)},
        }

    hist_cond_dir = outdir / 'hist_cond'
    hist_cond_dir.mkdir(parents=True, exist_ok=True)
    for cond, h in hist_cond.items():
        np.save(hist_cond_dir / f"{cond.replace('/', '_')}.npy", h.cpu().numpy())

    np.save(outdir / 'hist_full.npy', hist_full.cpu().numpy())
    np.save(outdir / 'hist_1024.npy', hist_1024.cpu().numpy())
    with open(outdir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n===== OFFICIAL (native 1080x1920, cumulative confusion over val) =====")
    print(f"  mIoU (present classes) = {miou_f_present:.2f}")
    print(f"  mIoU (all 19, nan->0)  = {miou_f_all:.2f}")
    print("\n===== INTERNAL (letterbox 1024^2, trainer protocol) =====")
    print(f"  mIoU (all 19, nan->0)  = {miou_s_all:.2f}   <- should reproduce the 81.02-style number")
    print("\n--- per-class IoU (official) ---")
    for c, v, s in zip(class_names, iou_f, iou_s):
        print(f"  {c:<15} official {v:6.2f}   internal1024 {s:6.2f}   delta {v - s:+6.2f}")
    print("\n--- per-condition (official) ---")
    for cond, d in report['per_condition_official'].items():
        print(f"  {cond:<14} n={d['n_images']:>3}  mIoU={d['mIoU_present_classes']:.2f}  "
              f"(present {d['n_present_classes']}/19)")
    print(f"\n[saved] {outdir}/report.json")


if __name__ == '__main__':
    main()
