"""Detection breakdown: overall triplet + per-class AP + night/normal split.

Answers two of the three standard detection questions for ANY model in this repo
(the third — did the module actually do anything — is det_module_ablation.py):

  1) per-class performance   -> AP / AP50 / GT count per category
  2) low-light performance   -> the same metrics on the night clips vs the rest

Model-agnostic: driven by the training config + det checkpoint only.

  python tools/det_eval_breakdown.py --cfg configs/det/det_P37a_cefr_yeon.yaml \
      --ckpt outputs/.../best_checkpoint.pth --out analysis/P37a_breakdown
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _det_common import (DEFAULT_LOWLIGHT_CLIPS, build_detector, build_loader,  # noqa: E402
                         eval_overall, eval_per_class, load_cfg, load_det_checkpoint,
                         run_inference, split_by_clip, write_outputs)


def _fmt(x, nd=4):
    return 'n/a' if x != x else f'{x:.{nd}f}'          # NaN-safe


def _md(model_name, ck, overall, per_class, night, normal, clips) -> str:
    L = [f'# Detection breakdown — {model_name}', '',
         f'checkpoint epoch {ck.get("epoch")} · train-time metrics {ck.get("metrics")}',
         f'night clips: {", ".join(clips)}', '',
         '## Overall (mAP / mAP50 / mAP75 — repo reporting convention)', '',
         '| split | images | mAP | mAP50 | mAP75 | AP_s | AP_m | AP_l |',
         '|---|---|---|---|---|---|---|---|']
    for tag, m in (('all', overall), ('night', night), ('normal', normal)):
        if m is None:
            continue
        L.append(f'| {tag} | {m.get("n_images","-")} | {_fmt(m["AP"])} | {_fmt(m["AP50"])} | '
                 f'{_fmt(m["AP75"])} | {_fmt(m["AP_small"])} | {_fmt(m["AP_medium"])} | '
                 f'{_fmt(m["AP_large"])} |')
    if night and normal:
        L += ['', f'**night − normal: mAP50 {night["AP50"] - normal["AP50"]:+.4f} · '
                  f'mAP {night["AP"] - normal["AP"]:+.4f}** '
                  '(positive = the model holds up better in the dark)']
    L += ['', '## Per-class', '',
          '| class | n_gt(all) | AP | AP50 | AP50 night | AP50 normal | night−normal |',
          '|---|---|---|---|---|---|---|']
    nb = {r['name']: r for r in per_class['night']} if per_class.get('night') else {}
    nm = {r['name']: r for r in per_class['normal']} if per_class.get('normal') else {}
    for r in per_class['all']:
        n, m = nb.get(r['name'], {}), nm.get(r['name'], {})
        d = (n.get('AP50', float('nan')) - m.get('AP50', float('nan')))
        L.append(f'| {r["name"]} | {r["n_gt"]} | {_fmt(r["AP"])} | {_fmt(r["AP50"])} | '
                 f'{_fmt(n.get("AP50", float("nan")))} | {_fmt(m.get("AP50", float("nan")))} | '
                 f'{_fmt(d)} |')
    L += ['', '_Classes with n_gt=0 report n/a — absent from this split, not a failure._']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='output prefix (writes .json/.md)')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--lowlight-clips', default=','.join(DEFAULT_LOWLIGHT_CLIPS),
                    help='substrings identifying night clips in file_name')
    ap.add_argument('--limit', type=int, default=None, help='cap images (smoke runs)')
    ap.add_argument('--stride', type=int, default=1,
                    help='evaluate every Nth image (spans all clips; use for cheap runs)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--name', default=None, help='label for the report')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, args.mode, args.workers)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    ck = load_det_checkpoint(model, args.ckpt, dev)
    print(f"[det-analysis] loaded {args.ckpt} (missing={ck['missing']} unexpected={ck['unexpected']})")

    preds, id2file = run_inference(model, ds, loader, cfg, dev, args.score_thresh, args.limit, args.stride)
    ann = cfg['DATASET'][f'ANNOTATION_{args.mode.upper()}']
    clips = [c for c in args.lowlight_clips.split(',') if c]
    night_ids, normal_ids = split_by_clip(id2file, clips)
    print(f"[det-analysis] {len(preds)} preds · night {len(night_ids)} / normal {len(normal_ids)} imgs")

    overall = eval_overall(ann, preds, list(id2file))
    night = eval_overall(ann, preds, night_ids) if night_ids else None
    normal = eval_overall(ann, preds, normal_ids) if normal_ids else None
    per_class = {'all': eval_per_class(ann, preds, list(id2file))}
    if night_ids:
        per_class['night'] = eval_per_class(ann, preds, night_ids)
    if normal_ids:
        per_class['normal'] = eval_per_class(ann, preds, normal_ids)

    name = args.name or os.path.basename(args.cfg).replace('.yaml', '')
    payload = {'model': name, 'cfg': args.cfg, 'ckpt': args.ckpt, 'checkpoint': ck,
               'clips_night': clips, 'overall': overall, 'night': night,
               'normal': normal, 'per_class': per_class}
    write_outputs(args.out, payload, _md(name, ck, overall, per_class, night, normal, clips))
    print(f"[det-analysis] mAP50 all={overall['AP50']:.4f}"
          + (f" night={night['AP50']:.4f}" if night else '')
          + (f" normal={normal['AP50']:.4f}" if normal else ''))


if __name__ == '__main__':
    main()
