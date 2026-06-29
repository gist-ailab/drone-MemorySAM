"""
Merge per-capture COCO annotations into a single train/val pair with a
**capture-level** split (avoids temporal leakage from adjacent frames).

Dataset layout (poongsan indoor multimodal):
  BASE/<capture_*>/annotations_coco_rgb.json   — one COCO per capture
  each image entry carries:
    file_name          : rgb path, relative to DATASET ROOT
    modalities {..}    : dict of per-modality paths, relative to DATASET ROOT
                         (rgb / thermal_aligned / depth_map_lidar / ...)

Only captures that actually contain a json are merged (some have none).
image_id / annotation_id are globally re-indexed; file_name + modalities are
preserved verbatim so the loader can resolve any modality from DATASET ROOT.
A `capture` field is added to every image for traceability.

Usage:
  python -m objdet.tools.merge_coco \
    --base /drone_nas/drone/dataset/Labeled/260618_poongsan_raw_20_aligned \
    --out_dir /drone_nas/drone/dataset/Labeled/260618_poongsan_raw_20_aligned/_det_splits \
    --test_captures capture_20260618_114021 capture_20260618_115206
"""
import os
import json
import argparse
from glob import glob


def load_captures(base):
    caps = {}
    for jp in sorted(glob(os.path.join(base, 'capture_*', 'annotations_coco_rgb.json'))):
        cap = os.path.basename(os.path.dirname(jp))
        caps[cap] = json.load(open(jp))
    return caps


def merge(captures: dict, cap_names: list):
    """Merge a subset of captures into one COCO dict with re-indexed ids."""
    out = {'licenses': [], 'info': {}, 'categories': None, 'images': [], 'annotations': []}
    next_img, next_ann = 1, 1
    for cap in cap_names:
        coco = captures[cap]
        if out['categories'] is None:
            out['categories'] = coco['categories']
            out['licenses'] = coco.get('licenses', [])
            out['info'] = coco.get('info', {})
        else:
            # sanity: category ids must match across captures
            a = {(c['id'], c['name']) for c in out['categories']}
            b = {(c['id'], c['name']) for c in coco['categories']}
            assert a == b, f"category mismatch in {cap}: {b ^ a}"
        id_map = {}
        for im in coco['images']:
            new = dict(im)
            new['id'] = next_img
            new['capture'] = cap
            id_map[im['id']] = next_img
            next_img += 1
            out['images'].append(new)
        for an in coco['annotations']:
            new = dict(an)
            new['id'] = next_ann
            new['image_id'] = id_map[an['image_id']]
            next_ann += 1
            out['annotations'].append(new)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--test_captures', nargs='+', required=True)
    args = ap.parse_args()

    captures = load_captures(args.base)
    print(f"captures with json: {list(captures.keys())}")
    test_caps = list(args.test_captures)
    train_caps = [c for c in captures if c not in test_caps]
    for c in test_caps:
        assert c in captures, f"test capture {c} has no json"

    os.makedirs(args.out_dir, exist_ok=True)
    train = merge(captures, train_caps)
    val = merge(captures, test_caps)

    tp = os.path.join(args.out_dir, 'det_train.json')
    vp = os.path.join(args.out_dir, 'det_val.json')
    json.dump(train, open(tp, 'w'))
    json.dump(val, open(vp, 'w'))

    def stat(name, d, caps):
        print(f"  [{name}] captures={caps} images={len(d['images'])} anns={len(d['annotations'])}")
    print("MERGE DONE:")
    stat('train', train, train_caps)
    stat('val', val, test_caps)
    print(f"  written: {tp}\n           {vp}")


if __name__ == '__main__':
    main()
