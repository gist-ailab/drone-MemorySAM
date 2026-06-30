"""
Detection diagnostics (model-agnostic): per-class AP + score distribution + viz.

Runs a single-GPU val pass on a det checkpoint and reports WHY AP is what it is:
  1. per-class AP / AP50 (which categories work vs collapse)
  2. score distribution (how many predictions survive thresholds; is the head confident?)
  3. visualization PNGs (pred boxes + GT) for the first N val images

Usage (single GPU, no DDP):
  CUDA_VISIBLE_DEVICES=<free> python objdet/tools/diag_det.py \
      --cfg configs/det/det_P29_indoor_jarvis.yaml \
      --det_checkpoint outputs/det/det_P29_indoor_jarvis/epoch19_checkpoint.pth \
      --out_dir outputs/det/diag --viz_n 12 --max_images 0
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_det import build_seg_model, build_dataset
from objdet.models.det_model import MemorySAMDetector
try:
    from objdet.models.det_model import MemorySAMDetectorP30
except Exception:
    MemorySAMDetectorP30 = None
from objdet.metrics import format_predictions_coco
from objdet.datasets.multimodal_det import MultiModalDetDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--det_checkpoint', required=True)
    p.add_argument('--out_dir', default='outputs/det/diag')
    p.add_argument('--viz_n', type=int, default=12)
    p.add_argument('--viz_thresh', type=float, default=0.3)
    p.add_argument('--max_images', type=int, default=0, help='0 = full val set')
    return p.parse_args()


def build_model(cfg, device, n_classes):
    seg_model = build_seg_model(cfg, device)
    name = cfg['MODEL'].get('DET_MODEL', 'MemorySAMDetector')
    common = dict(seg_model=seg_model, modals=cfg['DATASET']['MODALS'], n_classes=n_classes,
                  fpn_in_channels=cfg['MODEL'].get('FPN_CHANNELS', [32, 64, 256]),
                  fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16]),
                  freeze_backbone=cfg['MODEL'].get('FREEZE_BACKBONE', False),
                  train_memory=cfg['MODEL'].get('TRAIN_MEMORY', True),
                  n_convs=cfg['MODEL'].get('N_CONVS', 4),
                  hidden_dim=cfg['MODEL'].get('HIDDEN_DIM', 256))
    if name == 'MemorySAMDetectorP30' and MemorySAMDetectorP30 is not None:
        model = MemorySAMDetectorP30(**common, img_size=tuple(cfg['DATASET'].get('IMG_SIZE', [1024, 1024]))[0])
    else:
        model = MemorySAMDetector(**common, modality_fuse=cfg['MODEL'].get('MODALITY_FUSE', 'mean'))
    return model.to(device)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    import yaml
    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_ds = build_dataset(cfg, 'val')
    n_classes = val_ds.n_classes
    class_names = val_ds.class_names
    idx_to_cat_id = {v: k for k, v in val_ds.cat_id_to_idx.items()}
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=MultiModalDetDataset.collate_fn)

    model = build_model(cfg, device, n_classes)
    ckpt = torch.load(args.det_checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        miss, unexp = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"[ckpt] epoch={ckpt.get('epoch')} loaded model_state_dict (missing={len(miss)}, unexpected={len(unexp)})")
    else:
        model.load_detector_state_dict(ckpt['detector_state_dict'])
        print("[ckpt] loaded detector_state_dict")
    model.eval()

    all_preds, all_scores, dets_per_img = [], [], []
    img_size = tuple(cfg['DATASET'].get('IMG_SIZE', [1024, 1024]))
    viz_saved = 0

    for bi, batch in enumerate(loader):
        if args.max_images and bi >= args.max_images:
            break
        modals = [k for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device) for m in modals}
        with torch.no_grad():
            out = model(sample)
        det = out['detections'][0]
        sc = det['scores'].cpu().numpy()
        all_scores.append(sc)
        dets_per_img.append(int((sc > 0.05).sum()))
        orig_h, orig_w = batch['orig_size'][0].tolist()
        if det['boxes'].shape[0] > 0:
            boxes = det['boxes'].clone().cpu()
            boxes[:, [0, 2]] *= orig_w / img_size[1]
            boxes[:, [1, 3]] *= orig_h / img_size[0]
            all_preds.extend(format_predictions_coco(
                boxes, det['scores'].cpu(), det['class_ids'].cpu(),
                batch['image_id'][0], idx_to_cat_id))
        # viz (model-input 1024 space)
        if viz_saved < args.viz_n:
            rgb = 'img' if 'img' in sample else modals[0]
            im = (sample[rgb][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            pil = Image.fromarray(im); dr = ImageDraw.Draw(pil)
            gt = batch['bboxes'][0]
            for b in gt.tolist():
                dr.rectangle(b, outline=(0, 255, 0), width=3)  # GT green
            kb = det['boxes'].cpu(); ks = det['scores'].cpu(); kc = det['class_ids'].cpu()
            n_pred = 0
            for j in range(kb.shape[0]):
                if ks[j] < args.viz_thresh:
                    continue
                x1, y1, x2, y2 = kb[j].tolist()
                dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)  # pred red
                dr.text((x1, max(0, y1 - 10)), f"{class_names[int(kc[j])]}:{ks[j]:.2f}", fill=(255, 255, 0))
                n_pred += 1
            pil.save(os.path.join(args.out_dir, f"viz_{viz_saved:02d}_{batch['file_name'][0].split('/')[-1]}"))
            viz_saved += 1
            print(f"[viz {viz_saved}] {batch['file_name'][0].split('/')[-1]} GT={len(gt)} pred@{args.viz_thresh}={n_pred}")

    # ---------- score distribution ----------
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    print("\n================ SCORE DISTRIBUTION ================")
    print(f"total raw detections (score>0.05): {scores.size}")
    if scores.size:
        for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
            print(f"  > {t:.2f}: {int((scores > t).sum()):7d}  ({100*(scores>t).mean():.1f}%)")
        print(f"  score max={scores.max():.3f} mean={scores.mean():.3f} "
              f"p50={np.percentile(scores,50):.3f} p90={np.percentile(scores,90):.3f} p99={np.percentile(scores,99):.3f}")
    dpi = np.array(dets_per_img)
    print(f"dets/image (score>0.05): mean={dpi.mean():.1f} max={int(dpi.max()) if dpi.size else 0} "
          f"images_with_0={int((dpi==0).sum())}/{dpi.size}")

    # ---------- COCO eval (overall + per-class) ----------
    ann = cfg['DATASET']['ANNOTATION_VAL']
    coco_gt = COCO(ann)
    print("\n================ COCO EVAL ================")
    if not all_preds:
        print("NO predictions — all AP = 0"); return
    coco_dt = coco_gt.loadRes(all_preds)
    ev = COCOeval(coco_gt, coco_dt, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    overall_ap, overall_ap50 = ev.stats[0], ev.stats[1]

    # per-class AP from precision array: precision[T,R,K,A,M]; A=0(all), M=2(maxDet=100)
    prec = ev.eval['precision']
    cat_ids = coco_gt.getCatIds()
    id_to_name = {c['id']: c['name'] for c in coco_gt.loadCats(cat_ids)}
    # GT instance count per category (val)
    gt_per_cat = defaultdict(int)
    for a in coco_gt.dataset['annotations']:
        gt_per_cat[a['category_id']] += 1
    print("\n  per-class AP (IoU .50:.95 / .50)   [#GT in val]")
    rows = []
    for k, cid in enumerate(cat_ids):
        p_all = prec[:, :, k, 0, 2]
        ap = p_all[p_all > -1].mean() if (p_all > -1).any() else float('nan')
        p_50 = prec[0, :, k, 0, 2]
        ap50 = p_50[p_50 > -1].mean() if (p_50 > -1).any() else float('nan')
        rows.append((id_to_name.get(cid, str(cid)), ap, ap50, gt_per_cat.get(cid, 0)))
    for name, ap, ap50, ng in sorted(rows, key=lambda r: (-(r[1] if r[1] == r[1] else -1))):
        print(f"    {name:22s}  AP={ap:.4f}  AP50={ap50:.4f}   [{ng}]")
    print(f"\n  OVERALL  AP={overall_ap:.4f}  AP50={overall_ap50:.4f}")

    json.dump({'overall_ap': float(overall_ap), 'overall_ap50': float(overall_ap50),
               'per_class': [(n, float(a), float(a5), g) for n, a, a5, g in rows],
               'score_stats': {'n': int(scores.size), 'max': float(scores.max()) if scores.size else 0.0,
                               'mean': float(scores.mean()) if scores.size else 0.0,
                               'gt0_images': int((dpi == 0).sum()), 'n_images': int(dpi.size)}},
              open(os.path.join(args.out_dir, 'diag_summary.json'), 'w'), indent=2)
    print(f"\n[saved] {args.out_dir}/diag_summary.json + {viz_saved} viz PNGs")


if __name__ == '__main__':
    main()
