"""Ad hoc detection visualization: day/night + rare-class sample overlays for the
D1 certification deliverables (2026-07-23 det_cert task).

Reuses tools/_det_common.py (dataset/model/checkpoint plumbing, same as
det_eval_breakdown.py) and val_det.draw_detections (box overlay). Picks a handful
of images by clip substring match (night vs normal) plus rare-class targeted picks,
runs inference on just those, and saves box-overlay PNGs. Not a full-split run —
intentionally cheap (single digit images), unlike det_eval_breakdown.py's full pass.

Usage:
  python det_viz_samples.py --cfg configs/det/det_D1_recovered_yeon.yaml \
      --ckpt outputs/det_D1_recovered_yeon/det_D1_recovered_yeon/best_checkpoint.pth \
      --out analysis/det_cert_20260723/viz_D1_recovered --score-thresh 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_ROOT) == 'tools':
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from _det_common import load_cfg, build_detector, load_det_checkpoint, DEFAULT_LOWLIGHT_CLIPS  # noqa: E402
from objdet.datasets.multimodal_det import rescale_boxes_to_orig  # noqa: E402
from val_det import draw_detections  # noqa: E402


def pick_indices(dataset, night_clips, n_night, n_normal, rare_classes, cat_name_to_id):
    """Return a dict tag -> dataset index, spread across clips + rare classes."""
    n = len(dataset)
    night_idx, normal_idx = [], []
    for i in range(n):
        img_id = dataset.img_ids[i]
        fn = dataset.images[img_id]['file_name']
        (night_idx if any(c in fn for c in night_clips) else normal_idx).append(i)

    def spread(idx_list, k):
        if not idx_list or k <= 0:
            return []
        step = max(1, len(idx_list) // k)
        return [idx_list[j] for j in range(0, len(idx_list), step)][:k]

    picks = {}
    for j, i in enumerate(spread(night_idx, n_night)):
        picks[f'night_{j}'] = i
    for j, i in enumerate(spread(normal_idx, n_normal)):
        picks[f'normal_{j}'] = i

    # rare-class targeted picks: first image (in each split) whose GT contains it
    ann_by_img = dataset.img_anns  # {img_id: [ann, ...]}, built in __init__
    chosen_ids = set(picks.values())
    for cname in rare_classes:
        cid = cat_name_to_id.get(cname)
        if cid is None:
            continue
        found = None
        for i in range(n):
            if i in chosen_ids:
                continue
            img_id = dataset.img_ids[i]
            anns = ann_by_img.get(img_id, [])
            if any(a['category_id'] == cid for a in anns):
                found = i
                break
        if found is not None:
            picks[f'rare_{cname.replace(" ", "")}'] = found
            chosen_ids.add(found)
    return picks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='output directory for PNGs')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--score-thresh', type=float, default=0.3)
    ap.add_argument('--n-night', type=int, default=3)
    ap.add_argument('--n-normal', type=int, default=3)
    ap.add_argument('--rare-classes', default='Doors,Lighting')
    ap.add_argument('--lowlight-clips', default=','.join(DEFAULT_LOWLIGHT_CLIPS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_cfg(args.cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from train_det import build_dataset
    ds = build_dataset(cfg, args.mode)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    ck = load_det_checkpoint(model, args.ckpt, dev)
    print(f"[det-viz] loaded {args.ckpt} (missing={ck['missing']} unexpected={ck['unexpected']})")

    cat_name_to_id = {c['name']: c['id'] for c in ds.categories}
    clips = [c for c in args.lowlight_clips.split(',') if c]
    picks = pick_indices(ds, clips, args.n_night, args.n_normal,
                          args.rare_classes.split(','), cat_name_to_id)
    print(f"[det-viz] picked {len(picks)} samples: {picks}")

    if 'MODALITY_KEYS' in cfg['DATASET']:
        rgb_root = cfg['DATASET']['ROOT']
    else:
        rgb_root = cfg['DATASET']['MODALITIES']['img']['ROOT']
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')

    manifest = []
    for tag, idx in picks.items():
        sample = ds[idx]
        img_id = sample.get('image_id') or ds.img_ids[idx]
        file_name = sample['file_name']
        modals = [k for k in sample if isinstance(sample[k], torch.Tensor) and sample[k].dim() == 3]
        batch = {m: sample[m].unsqueeze(0).to(dev) for m in modals}
        out = model(batch)
        det = out['detections'][0]
        boxes, scores, cls_ids = det['boxes'], det['scores'], det['class_ids']
        keep = scores > args.score_thresh
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        orig_h, orig_w = sample['orig_size'].tolist() if torch.is_tensor(sample['orig_size']) else sample['orig_size']
        img_hw = batch[modals[0]].shape[-2:]
        if boxes.shape[0] > 0:
            boxes = rescale_boxes_to_orig(boxes.cpu(), orig_h, orig_w, img_hw[0], img_hw[1], resize_mode)

        rgb_path = os.path.join(rgb_root, file_name)
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
        vis = draw_detections(rgb_img, boxes.cpu(), scores.cpu(), cls_ids.cpu(),
                              ds.class_names, args.score_thresh)
        out_path = os.path.join(args.out, f'{tag}__{os.path.basename(file_name)}')
        vis.save(out_path)
        manifest.append({'tag': tag, 'image_id': int(img_id), 'file_name': file_name,
                         'n_det': int(boxes.shape[0]), 'out': out_path})
        print(f"[det-viz] {tag}: {file_name} -> {boxes.shape[0]} dets -> {out_path}")

    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[det-viz] wrote {len(manifest)} images + manifest.json to {args.out}")


if __name__ == '__main__':
    main()
