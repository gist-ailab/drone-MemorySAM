"""GISTOLO step 2 — turn the multimodal teacher's train predictions into YOLO
labels for the RGB student, then assemble a GISTOLO training dataset that reuses
the modal-subset RGB images with distilled labels (test split stays GT).

Variants:
  pseudo : labels = teacher confident boxes only (pure cross-modal self-training)
  union  : labels = GT + teacher-only boxes (teacher boxes not matching any GT of
           the same class by IoU>=iou) — recover multimodal-visible objects the
           RGB GT-trained student would miss, without discarding GT.

  python gistolo_labels.py --preds runs_gistolo/teacher_train_preds.json \
      --modal-dir ~/dset/poongsan_v2_yolo_rgb_modal \
      --out-dir ~/dset/poongsan_v2_yolo_rgb_gistolo_union \
      --variant union --score-thresh 0.5 --iou 0.5
"""
import argparse
import json
import os


def _flat(fname):
    return fname.replace('/rgb/', '_').replace('/', '_')


def _iou_xywh(a, b):
    # a,b = [x,y,w,h] pixel
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='teacher_dump.py output json')
    ap.add_argument('--modal-dir', required=True, help='poongsan_v2_yolo_rgb_modal (RGB imgs + GT labels)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--variant', choices=['pseudo', 'union'], default='union')
    ap.add_argument('--score-thresh', type=float, default=0.5)
    ap.add_argument('--iou', type=float, default=0.5, help='union: teacher box is "new" if IoU<iou vs same-class GT')
    args = ap.parse_args()

    D = json.load(open(args.preds))
    preds, id2file = D['preds'], D['id2file']
    ann = json.load(open(D['ann']))
    wh = {im['id']: (im['width'], im['height']) for im in ann['images']}
    cats = sorted(ann['categories'], key=lambda c: c['id'])
    cat_to_idx = {c['id']: i for i, c in enumerate(cats)}

    # group teacher boxes per image
    by_img = {}
    for p in preds:
        if p['score'] < args.score_thresh:
            continue
        by_img.setdefault(p['image_id'], []).append(p)

    out_lbl = f'{args.out_dir}/labels/train'
    os.makedirs(out_lbl, exist_ok=True)
    gt_lbl = f'{args.modal_dir}/labels/train'

    n_teacher = n_gt = n_added = 0
    # stem -> image_id (from id2file; fall back to every ann image so images the
    # teacher predicted nothing on still resolve their W/H and get a label file).
    stem2imgid = {os.path.splitext(_flat(v))[0]: int(k) for k, v in id2file.items()}
    for im in ann['images']:
        stem2imgid.setdefault(os.path.splitext(_flat(im['file_name']))[0], im['id'])

    n_img = 0
    for lf in os.listdir(gt_lbl):
        if not lf.endswith('.txt'):
            continue
        n_img += 1
        stem = lf[:-4]
        img_id = stem2imgid.get(stem)
        W, H = wh.get(img_id, (640, 480)) if img_id is not None else (640, 480)
        gt_lines = [l for l in open(f'{gt_lbl}/{lf}').read().splitlines() if l.strip()]
        n_gt += len(gt_lines)

        # teacher boxes -> yolo (normalized), pixel xywh for IoU
        tb = []
        for p in by_img.get(img_id, []):
            x, y, w, h = p['bbox']
            cls = cat_to_idx.get(p['category_id'])
            if cls is None or w <= 0 or h <= 0:
                continue
            tb.append((cls, [x, y, w, h]))
        n_teacher += len(tb)

        if args.variant == 'pseudo':
            lines = [f"{c} {min(max((b[0]+b[2]/2)/W,0),1):.6f} {min(max((b[1]+b[3]/2)/H,0),1):.6f} "
                     f"{min(b[2]/W,1):.6f} {min(b[3]/H,1):.6f}" for c, b in tb]
        else:  # union: GT + teacher boxes with no same-class GT overlap
            # decode GT to pixel xywh for IoU
            gt_px = []
            for l in gt_lines:
                c, cx, cy, w, h = l.split()
                c = int(c); cx, cy, w, h = float(cx)*W, float(cy)*H, float(w)*W, float(h)*H
                gt_px.append((c, [cx - w/2, cy - h/2, w, h]))
            lines = list(gt_lines)
            for c, b in tb:
                if all(not (c == gc and _iou_xywh(b, gb) >= args.iou) for gc, gb in gt_px):
                    lines.append(f"{c} {min(max((b[0]+b[2]/2)/W,0),1):.6f} {min(max((b[1]+b[3]/2)/H,0),1):.6f} "
                                 f"{min(b[2]/W,1):.6f} {min(b[3]/H,1):.6f}")
                    n_added += 1
        open(f'{out_lbl}/{lf}', 'w').write("\n".join(lines))

    # assemble dataset: symlink images (train+test) + GT test labels + yaml
    for split in ('train', 'test'):
        os.makedirs(f'{args.out_dir}/images', exist_ok=True)
        src = os.path.abspath(f'{args.modal_dir}/images/{split}')
        dst = f'{args.out_dir}/images/{split}'
        if not os.path.exists(dst):
            os.symlink(src, dst)
    # test labels = GT (symlink)
    tl = f'{args.out_dir}/labels/test'
    if not os.path.exists(tl):
        os.symlink(os.path.abspath(f'{args.modal_dir}/labels/test'), tl)
    # yaml
    y = open(f'{args.modal_dir}/poongsan_v2_rgb.yaml').read()
    y = y.replace(os.path.abspath(args.modal_dir), os.path.abspath(args.out_dir))
    y = '\n'.join(l for l in y.splitlines() if not l.startswith('path:'))
    with open(f'{args.out_dir}/poongsan_v2_rgb.yaml', 'w') as f:
        f.write(f'path: {os.path.abspath(args.out_dir)}\n' + y + '\n')
    # copy test subset lists
    for s in ('lowlight', 'normal'):
        src = f'{args.modal_dir}/test_{s}_basenames.txt'
        if os.path.exists(src):
            open(f'{args.out_dir}/test_{s}_basenames.txt', 'w').write(open(src).read())

    print(f"[gistolo:{args.variant}] images={n_img}  GT_boxes={n_gt}  teacher_boxes(>{args.score_thresh})={n_teacher}  "
          f"added={n_added if args.variant=='union' else 'n/a'}")
    print(f"[gistolo] dataset -> {args.out_dir}")


if __name__ == '__main__':
    main()
