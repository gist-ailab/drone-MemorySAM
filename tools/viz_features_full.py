#!/usr/bin/env python3
"""
tools/viz_features_full.py — PER-IMAGE (whole test set) RBMA feature/fusion viewer + numeric CSV.

Extends tools/viz_features.py with two things the base tool lacks:
  (1) Runs over the ENTIRE split (default) and writes a per-image metrics CSV (incremental).
  (2) Visualizes the PROPOSED MODULE's effect: for corroboration-capable models (LoRA_Sam_P32)
      it runs a 2nd forward with `core.corroboration_bias = False` (== P31/P28 self-entropy path,
      byte-identical) and shows fused-feat ON vs OFF and the |ON-OFF| change map — i.e. "what the
      proposed module changed in the fused feature". If the model has no such attribute it silently
      skips the OFF pass (change columns = NaN, no diff row).

Per panel (rows):
  R1 inputs (per modality) | GT | Pred | Error
  R2 per-modal ENCODER featPCA | FUSED featPCA (ON = proposed module active)
  R3 per-modal RELIABILITY (1-H)
  R4 per-modal DECODER argmax pred  ("which modality carries which class")
  R5 per-modal UAMM fusion weight
  R6 [proposed-module diff] FUSED featPCA ON | FUSED featPCA OFF | |ON-OFF| L2 change map

Per-image CSV columns:
  idx, stem, condition, miou, miou_off, dmiou (on-off),
  iou_<Class> for all classes (NaN if absent in GT),
  smiou_<modal> (single-modality mIoU from that modal's own decoder),
  rel_<modal> (mean reliability), uamm_<modal> (mean fusion weight),
  best_modal (argmax smiou), top_uamm_modal (argmax mean uamm), misalloc (best_modal != top_uamm),
  corrb_change_mean, corrb_change_rel  (||fused_on-fused_off|| stats)

Example (full test set on B200 GPU7):
  python tools/viz_features_full.py --cfg configs/b200-deliver_rgbdel_P32_physaug.yaml \
    --model_path outputs/MMSamP32/.../test_epoch74_53.62_top1_checkpoint.pth \
    --split test --gpu 7 --out-dir ~/viz_P32_full --csv ~/viz_P32_full/per_image.csv
"""
import argparse, os, sys, math, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V  # reuse load_model / create_dataset / get_val_augmentation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(t):
    a = t.detach().float().cpu().numpy()
    a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 3:
        a = a * IMAGENET_STD + IMAGENET_MEAN
    else:
        a = a[..., :1]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return np.clip(a, 0, 1)


def pca_rgb(feat):
    C, H, W = feat.shape
    x = feat.detach().float().reshape(C, -1).t()
    x = x - x.mean(0, keepdim=True)
    try:
        _, _, Vt = torch.pca_lowrank(x, q=3)
        y = x @ Vt[:, :3]
    except Exception:
        y = x[:, :3]
    y = (y - y.min(0).values) / (y.max(0).values - y.min(0).values + 1e-8)
    return y.reshape(H, W, 3).cpu().numpy()


def reliability_map(logits):
    p = F.softmax(logits.detach().float(), dim=0)
    H = -(p * (p + 1e-8).log()).sum(0)
    return (1.0 - H / math.log(p.shape[0])).cpu().numpy()


def to_hw(arr):
    a = np.asarray(arr)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def per_image_iou(pred, gt, num_classes, ignore):
    """Return (miou_over_present, per_class_iou[list len num_classes with nan if absent])."""
    valid = gt != ignore
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    ious = [float('nan')] * num_classes
    present = []
    for c in range(num_classes):
        pc = p == c
        gc = g == c
        inter = np.logical_and(pc, gc).sum()
        union = np.logical_or(pc, gc).sum()
        if union > 0:
            iou = 100.0 * inter / union
            ious[c] = iou
            if gc.sum() > 0:            # present in GT -> counts toward mIoU
                present.append(iou)
    miou = float(np.mean(present)) if present else float('nan')
    return miou, ious


def resize_to(pred_hw, gh, gw):
    ph, pw = pred_hw.shape
    if (ph, pw) == (gh, gw):
        return pred_hw
    return np.asarray(torch.nn.functional.interpolate(
        torch.tensor(pred_hw)[None, None].float(), size=(gh, gw), mode='nearest')[0, 0]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--case', default=None, help="restrict to one condition (default: all)")
    ap.add_argument('--num', type=int, default=-1, help="max images (-1 = all)")
    ap.add_argument('--start', type=int, default=0, help="start index (for chunked/parallel runs)")
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--csv', default=None, help="per-image CSV path (default <out>/per_image.csv)")
    ap.add_argument('--panel-every', type=int, default=1, help="render a panel every N images (metrics always computed)")
    ap.add_argument('--no-panels', action='store_true', help="metrics/CSV only, skip PNGs")
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    out = Path(args.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / 'panels').mkdir(exist_ok=True)
    csv_path = Path(args.csv).expanduser() if args.csv else out / 'per_image.csv'

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root: ds_cfg['ROOT'] = args.dataset_root
    if args.case is not None: ds_cfg['CASE'] = args.case
    if isinstance(ds_cfg.get('PHYSAUG'), dict): ds_cfg['PHYSAUG']['ENABLE'] = False

    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    image_size = cfg['EVAL']['IMAGE_SIZE'] if args.split == 'val' else cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(image_size, dataset_cfg=ds_cfg)
    split = 'val' if args.split == 'val' else 'test'
    dataset, has_gt = V.create_dataset(ds_cfg, split, transform, args.split, macvi=False, eval_day=False)
    classes = list(dataset.CLASSES); palette = dataset.PALETTE
    ds_cls = V._get_dataset_class(dataset)
    modals = ds_cfg.get('MODALS')
    ignore = getattr(dataset, 'ignore_label', 255)
    ncls = len(classes)

    # DELIVER path layout is <root>/img/<condition>/<split>/<scene>/<frame>.png -> condition per idx.
    ds_files = getattr(dataset, 'files', None)

    def cond_of(idx):
        if args.case is not None:
            return args.case
        if ds_files is not None and 0 <= idx < len(ds_files):
            parts = str(ds_files[idx]).replace('\\', '/').split('/')
            if 'img' in parts:
                j = parts.index('img')
                if j + 1 < len(parts):
                    return parts[j + 1]
        return split

    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model
    has_corrb = hasattr(core, 'corroboration_bias')
    print(f"[viz] split={split} n={len(dataset)} classes={ncls} modals={modals} has_corrb={has_corrb}", flush=True)

    n_total = len(dataset)
    end = n_total if args.num < 0 else min(args.start + args.num, n_total)
    indices = range(args.start, end)

    # CSV header
    fieldnames = (['idx', 'stem', 'condition', 'miou', 'miou_off', 'dmiou']
                  + [f'iou_{c}' for c in classes]
                  + [f'smiou_{m}' for m in modals]
                  + [f'rel_{m}' for m in modals]
                  + [f'uamm_{m}' for m in modals]
                  + ['best_modal', 'top_uamm_modal', 'misalloc',
                     'corrb_dlogit_mean', 'corrb_dlogit_rel', 'corrb_frac_flipped'])
    write_header = not csv_path.exists() or args.start == 0
    fcsv = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for n, idx in enumerate(indices):
        images, label, meta = dataset[idx]
        stem = meta.get('stem', f"idx{idx}") if isinstance(meta, dict) else f"idx{idx}"
        cond = cond_of(idx)
        imgs = [im.unsqueeze(0).to(device) for im in images]
        m = len(imgs)
        gt = np.asarray(label).astype(np.int64)
        gh, gw = gt.shape

        # --- forward WITH proposed module (ON) ---
        if has_corrb: core.corroboration_bias = True
        with torch.no_grad():
            m_output, m_feat = model(imgs, multimask_output=True)
        # capture diagnostics from THIS forward before a 2nd forward overwrites them
        per_feat = getattr(core, '_last_per_modal_feats', None)
        per_out = getattr(core, '_last_per_modal_outputs', None)
        uamm_sp = getattr(core, '_last_uamm_spatial', None)
        m_feat_on = m_feat[0].detach().float().cpu()
        final_pred = m_output[0].argmax(0).cpu().numpy().astype(np.int64)
        final_pred_r = resize_to(final_pred, gh, gw)
        miou, ious = per_image_iou(final_pred_r, gt, ncls, ignore)

        # per-modal single-modality mIoU + reliability means (from ON forward's per-modal decoders)
        smiou = [float('nan')] * m
        rel_mean = [float('nan')] * m
        if per_out is not None:
            for i in range(m):
                pm = per_out[i][0].argmax(0).cpu().numpy().astype(np.int64)
                pm_r = resize_to(pm, gh, gw)
                smiou[i], _ = per_image_iou(pm_r, gt, ncls, ignore)
                rel_mean[i] = float(np.mean(reliability_map(per_out[i][0])))
        uamm_mean = [float('nan')] * m
        if uamm_sp is not None:
            for i in range(m):
                uamm_mean[i] = float(np.mean(to_hw(uamm_sp[i][0])))

        # --- forward WITHOUT proposed module (OFF) for the change map/metric ---
        # NOTE: the returned fused feature `m_feat` is identical ON/OFF (corroboration acts on the
        # memory-attention decode path, not on the returned fused tensor). The observable effect of
        # the proposed module is therefore on the OUTPUT logits / prediction — that is what we diff.
        miou_off = float('nan'); change_mean = float('nan'); change_rel = float('nan')
        frac_flipped = float('nan'); logit_change_hw = None; final_pred_off_r = None
        if has_corrb and m >= 2:
            core.corroboration_bias = False
            with torch.no_grad():
                m_output_off, _ = model(imgs, multimask_output=True)
            core.corroboration_bias = True
            lo_on = m_output[0].detach().float().cpu()            # (C,h,w) logits, proposed ON
            lo_off = m_output_off[0].detach().float().cpu()       # (C,h,w) logits, proposed OFF
            fp_off = lo_off.argmax(0).numpy().astype(np.int64)
            final_pred_off_r = resize_to(fp_off, gh, gw)
            miou_off, _ = per_image_iou(final_pred_off_r, gt, ncls, ignore)
            l2 = (lo_on - lo_off).pow(2).sum(0).sqrt()            # (h,w) per-pixel logit change
            logit_change_hw = l2.numpy()
            change_mean = float(l2.mean())
            change_rel = float(l2.mean() / (lo_on.pow(2).sum(0).sqrt().mean() + 1e-8))
            frac_flipped = float((lo_on.argmax(0) != lo_off.argmax(0)).float().mean())

        dmiou = (miou - miou_off) if (miou == miou and miou_off == miou_off) else float('nan')
        best_modal = modals[int(np.nanargmax(smiou))] if any(s == s for s in smiou) else ''
        top_uamm = modals[int(np.nanargmax(uamm_mean))] if any(u == u for u in uamm_mean) else ''
        misalloc = int(bool(best_modal) and bool(top_uamm) and best_modal != top_uamm)

        row = {'idx': idx, 'stem': stem, 'condition': cond,
               'miou': round(miou, 3) if miou == miou else '',
               'miou_off': round(miou_off, 3) if miou_off == miou_off else '',
               'dmiou': round(dmiou, 3) if dmiou == dmiou else ''}
        for c, v in zip(classes, ious):
            row[f'iou_{c}'] = round(v, 2) if v == v else ''
        for mm, v in zip(modals, smiou): row[f'smiou_{mm}'] = round(v, 2) if v == v else ''
        for mm, v in zip(modals, rel_mean): row[f'rel_{mm}'] = round(v, 4) if v == v else ''
        for mm, v in zip(modals, uamm_mean): row[f'uamm_{mm}'] = round(v, 4) if v == v else ''
        row['best_modal'] = best_modal; row['top_uamm_modal'] = top_uamm; row['misalloc'] = misalloc
        row['corrb_dlogit_mean'] = round(change_mean, 5) if change_mean == change_mean else ''
        row['corrb_dlogit_rel'] = round(change_rel, 5) if change_rel == change_rel else ''
        row['corrb_frac_flipped'] = round(frac_flipped, 5) if frac_flipped == frac_flipped else ''
        writer.writerow(row); fcsv.flush()

        # --- panel ---
        if not args.no_panels and (n % args.panel_every == 0):
            err = ((final_pred_r != gt) & (gt != ignore)).astype(np.float32)
            ncol = m + 3
            nrow = 6 if (logit_change_hw is not None) else 5
            fig, ax = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
            for r in range(nrow):
                for c in range(ncol):
                    ax[r, c].axis('off')

            def show(r, c, img, title, cmap=None):
                ax[r, c].imshow(img, cmap=cmap); ax[r, c].set_title(title, fontsize=8); ax[r, c].axis('off')

            for i in range(m):
                di = denorm(images[i]).squeeze()
                show(0, i, di, f"in:{modals[i]}", cmap='gray' if di.ndim == 2 else None)
            show(0, m,   ds_cls.decode_segmap(gt.astype(np.uint8), palette), "GT")
            show(0, m + 1, ds_cls.decode_segmap(final_pred_r.astype(np.uint8), palette), f"Pred ({miou:.1f})" if miou == miou else "Pred")
            show(0, m + 2, err, "Error", cmap='Reds')
            if per_feat is not None:
                for i in range(m): show(1, i, pca_rgb(per_feat[i][0]), f"featPCA:{modals[i]}")
            show(1, m, pca_rgb(m_feat[0]), "FUSED featPCA")
            if per_out is not None:
                for i in range(m): show(2, i, reliability_map(per_out[i][0]), f"reliab:{modals[i]}", cmap='viridis')
            if per_out is not None:
                for i in range(m):
                    pm = per_out[i][0].argmax(0).cpu().numpy().astype(np.uint8)
                    show(3, i, ds_cls.decode_segmap(pm, palette), f"pred:{modals[i]} ({smiou[i]:.1f})" if smiou[i] == smiou[i] else f"pred:{modals[i]}")
            if uamm_sp is not None:
                for i in range(m): show(4, i, to_hw(uamm_sp[i][0]), f"UAMM w:{modals[i]}", cmap='magma')
            if logit_change_hw is not None:
                # proposed-module effect lives in the OUTPUT (m_feat is identical ON/OFF)
                show(5, 0, ds_cls.decode_segmap(final_pred_r.astype(np.uint8), palette),
                     f"Pred ON ({miou:.1f})" if miou == miou else "Pred ON")
                show(5, 1, ds_cls.decode_segmap(final_pred_off_r.astype(np.uint8), palette),
                     f"Pred OFF ({miou_off:.1f})" if miou_off == miou_off else "Pred OFF")
                show(5, 2, logit_change_hw, f"|dlogit| ON-OFF (flip {frac_flipped*100:.1f}%)", cmap='inferno')
                flip = (final_pred_r != final_pred_off_r).astype(np.float32)
                show(5, 3, flip, "argmax flip (proposed effect)", cmap='Reds')
            fig.suptitle(f"{Path(args.model_path).stem} | {cond} | {stem} | mIoU {miou:.1f} (d{dmiou:+.1f})"
                         if miou == miou else f"{stem}", fontsize=10)
            fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.06, hspace=0.12)
            fp = out / 'panels' / f"panel_{idx:05d}_{cond}_{stem}.png"
            fig.savefig(fp, dpi=90); plt.close(fig)

        if n % 25 == 0:
            print(f"[viz] {n+1}/{len(indices)} idx={idx} {cond}/{stem} mIoU={miou:.1f} d={dmiou:+.2f} misalloc={misalloc}", flush=True)

    fcsv.close()
    print(f"[viz] DONE -> csv={csv_path} panels={out/'panels'}", flush=True)


if __name__ == '__main__':
    main()
