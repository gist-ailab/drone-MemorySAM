#!/usr/bin/env python3
"""
tools/eval_muses_official.py — MUSES val을 **공식(native) 1080x1920 해상도**로 재평가
(레터박스 1024^2 내부 지표와 동일 forward에서 동시 산출해 사과-대-사과 비교).

Why: train_reliadino.py's evaluate() accumulates the confusion matrix at the
letterboxed 1024x1024 working resolution (the label is padded to square with
ignore=255 and then downsampled 1920->1024). That number (81.02 @ ep276) is an
*internal* metric and is NOT comparable to the MUSES/Cityscapes leaderboard,
which scores predictions against the native 1080x1920 *_gt_labelTrainIds.png.

This script runs the SAME forward pass (same transform, same fp32, no MSF/TTA --
exactly as evaluate() does), then:
  1. crops the letterbox padding out of the logits,
  2. bilinearly upsamples the cropped logits to the native 1080x1920,
  3. argmaxes and accumulates the confusion matrix against meta['orig_label'].

It accumulates BOTH histograms in one pass (hist_1024 = the trainer's internal
protocol, hist_full = official) so the two are strictly apples-to-apples: same
checkpoint, same weights, same forward, differing ONLY in the scoring geometry.

Per-condition (weather x time-of-day) histograms are accumulated too.

Native GT comes from the dataset itself: semseg/datasets/muses.py returns
meta['orig_label'] / ['orig_h'] / ['orig_w'] when constructed with return_meta=True,
so this tool never re-reads *_gt_labelTrainIds.png on its own.

Example:
  python tools/eval_muses_official.py \
    --cfg configs/hpca100-muses_rgbelr_P34_reliadino.yaml \
    --ckpt outputs/ReliaDINO/.../epoch276_81.02_top1_checkpoint.pth \
    --gpu 0 --out ~/muses_official_eval_P34
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


# ---------------------------------------------------------------- letterbox geometry
def letterbox_valid_box(orig_h: int, orig_w: int, side: int):
    """Region of the (side x side) network output that holds the real image.

    Mirrors MUSES._pad_to_square (short side padded symmetrically, (S-h)//2 on
    top/left) followed by augmentations_mm.Resize (square S -> side x side).
    Returns (t0, t1, l0, l1) in output-pixel coords.
    """
    S = max(orig_h, orig_w)
    top = (S - orig_h) // 2
    left = (S - orig_w) // 2
    s = side / S
    t0, t1 = int(round(top * s)), int(round((top + orig_h) * s))
    l0, l1 = int(round(left * s)), int(round((left + orig_w) * s))
    return t0, t1, l0, l1


def sanity_check_geometry(orig_h=1080, orig_w=1920, side=1024, ignore=255):
    """Round-trip proof that the inverse-letterbox indices are correct.

    (a) full-res: pad a random label to square, crop back with the geometry ->
        must be BIT-IDENTICAL to the original.
    (b) network-res: the padded rows of the letterboxed+downsampled label must be
        exactly the ignore value, and the kept box must contain no ignore-padding
        rows -- i.e. the crop box lines up with the real-image band.
    """
    rng = np.random.RandomState(0)
    lbl = torch.from_numpy(rng.randint(0, 19, (1, orig_h, orig_w)).astype(np.uint8))

    padded = MUSES._pad_to_square(lbl, fill=ignore)
    S = max(orig_h, orig_w)
    assert padded.shape[1:] == (S, S), f"pad shape {padded.shape}"

    t0f, t1f, l0f, l1f = letterbox_valid_box(orig_h, orig_w, S)   # side == S -> full res
    back = padded[:, t0f:t1f, l0f:l1f]
    assert back.shape == lbl.shape, f"roundtrip shape {back.shape} vs {lbl.shape}"
    assert torch.equal(back, lbl), "FULL-RES ROUNDTRIP MISMATCH"

    small = TF.resize(padded, (side, side), TF.InterpolationMode.NEAREST)
    t0, t1, l0, l1 = letterbox_valid_box(orig_h, orig_w, side)
    inside = small[:, t0:t1, l0:l1]
    assert (inside != ignore).all(), "crop box still contains ignore padding"
    n_ign_out = (small == ignore).sum().item() - (inside == ignore).sum().item()
    n_ign_tot = (small == ignore).sum().item()
    assert n_ign_out == n_ign_tot, "some padding leaked outside the crop box"
    # keys are side-agnostic: `side` is now taken from EVAL.IMAGE_SIZE, not hardcoded 1024.
    return dict(net_side=int(side), box_net=(t0, t1, l0, l1), box_full=(t0f, t1f, l0f, l1f),
                pad_rows_net=int(side - (t1 - t0)))


# ---------------------------------------------------------------- metrics
def iou_from_hist(hist: torch.Tensor):
    """Per-class IoU from a confusion matrix (rows = GT, cols = pred)."""
    hist = hist.double()
    inter = hist.diag()
    union = hist.sum(0) + hist.sum(1) - inter
    present = hist.sum(1) > 0                      # class has GT pixels
    iou = torch.where(union > 0, inter / union, torch.full_like(union, float('nan')))
    miou_present = iou[present].mean().item() * 100 if present.any() else float('nan')
    iou_z = torch.nan_to_num(iou, nan=0.0)
    miou_all = iou_z.mean().item() * 100            # trainer convention (nan -> 0)
    return (iou * 100).cpu().numpy(), miou_present, miou_all, present.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True, help='training config used for the ckpt (protocol source of truth)')
    ap.add_argument('--ckpt', required=True, help='*_checkpoint.pth or a raw state_dict')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dataset-root', default=None, help='override DATASET.ROOT only (server-local path)')
    ap.add_argument('--out', required=True, help='output dir for report.json + hist_*.npy')
    ap.add_argument('--limit', type=int, default=None, help='stop after N images (smoke test)')
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

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

            n_done += 1
            if args.limit and n_done >= args.limit:
                break

    iou_f, miou_f_present, miou_f_all, present_f = iou_from_hist(hist_full)
    iou_s, miou_s_present, miou_s_all, _ = iou_from_hist(hist_1024)

    report = {
        'ckpt': str(args.ckpt),
        'epoch': ck.get('epoch', None),
        'n_images': n_done,
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
