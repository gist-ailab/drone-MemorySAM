#!/usr/bin/env python
"""One-key certification evaluation for the D1 ViT-S+ detector.

Press ENTER and it runs the whole poongsan test split: per-image visualization +
inference log + GT comparison, streamed live, and at the end reports mAP50 and FPS.
Built on the repo's own analysis tools (tools/_det_common.py) so the numbers match
det_eval_breakdown.py / det_fps_bench.py exactly — this only adds the live UX.

  python certification/cert_eval.py \
      --cfg certification/configs/det_D1_vitsp_dronedemo.yaml \
      --ckpt <path>/det_D1_vitsp_20260723/best_checkpoint.pth \
      --out  runs/cert_D1

Flags: --auto (skip the ENTER prompt), --limit N (quick run), --show (cv2 window if
$DISPLAY), --stride N (every Nth image), --score-thresh (viz/log threshold).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from _det_common import (build_detector, build_loader, eval_overall,      # noqa: E402
                         eval_per_class, load_cfg, load_det_checkpoint,
                         split_ann_by_clip, DEFAULT_LOWLIGHT_CLIPS)
from objdet.datasets.multimodal_det import rescale_boxes_to_orig          # noqa: E402
from objdet.metrics import format_predictions_coco                        # noqa: E402
from val_det import draw_detections                                       # noqa: E402

BOLD, GRN, YEL, CYN, RST = '\033[1m', '\033[32m', '\033[33m', '\033[36m', '\033[0m'


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    p = np.asarray(pred, dtype=float); g = np.asarray(gt, dtype=float)
    x1 = np.maximum(p[:, None, 0], g[None, :, 0]); y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2]); y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ap[:, None] + ag[None, :] - inter + 1e-9)


def _tp_fp_fn(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=0.5):
    """Greedy per-class IoU@0.5 match — for the human-readable per-image line only.
    The certified metric is COCOeval at the end, not this."""
    tp = 0; used = set()
    iou = _iou_matrix(pred_boxes, gt_boxes)   # preds arrive score-descending
    for i in range(len(pred_boxes)):
        best, bj = iou_thr, -1
        for j in range(len(gt_boxes)):
            if j in used or gt_cls[j] != pred_cls[i]:
                continue
            if iou[i, j] >= best:
                best, bj = iou[i, j], j
        if bj >= 0:
            tp += 1; used.add(bj)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='runs/cert_eval')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--score-thresh', type=float, default=0.3, help='viz/per-image-log threshold')
    ap.add_argument('--eval-thresh', type=float, default=0.05, help='COCO scoring threshold')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--auto', action='store_true', help='skip the ENTER prompt')
    ap.add_argument('--show', action='store_true', help='cv2 window if $DISPLAY is set')
    ap.add_argument('--lowlight-clips', default=','.join(DEFAULT_LOWLIGHT_CLIPS))
    ap.add_argument('--data-root', default=None,
                    help='override DATASET.ROOT + ANNOTATION_* for this machine (poongsan_v2 mount)')
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    viz_dir = os.path.join(args.out, 'viz'); os.makedirs(viz_dir, exist_ok=True)

    cfg = load_cfg(args.cfg)
    if args.data_root:                       # retarget the dataset to this machine's mount
        r = args.data_root.rstrip('/')
        cfg['DATASET']['ROOT'] = r
        cfg['DATASET']['ANNOTATION_TRAIN'] = f'{r}/_final_ann/instances_train_egofill.json'
        cfg['DATASET']['ANNOTATION_VAL'] = f'{r}/_final_ann/instances_test_common.json'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, args.mode, workers=4)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    ck = load_det_checkpoint(model, args.ckpt, dev)

    ann = cfg['DATASET'][f'ANNOTATION_{args.mode.upper()}']
    clips = [c for c in args.lowlight_clips.split(',') if c]
    night_ids, normal_ids = split_ann_by_clip(ann, clips)

    # ---- startup banner: classes + modalities + model + data (before ENTER) -----
    mk = cfg['DATASET'].get('MODALITY_KEYS', {})
    modal_desc = ', '.join(f'{m}({mk.get(m, m)})' for m in cfg['DATASET']['MODALS'])
    print(f"\n{BOLD}╔═══ D1 ViT-S+ Detection — Certification Evaluation ═══╗{RST}")
    print(f"  {BOLD}Model{RST}      : {cfg['MODEL']['SEG_MODEL']} ({cfg['MODEL']['BACKBONE_TIMM']}) "
          f"+ {cfg['MODEL']['DET_MODEL']}")
    print(f"  {BOLD}Checkpoint{RST} : {os.path.basename(args.ckpt)}  epoch={ck.get('epoch')}  "
          f"md5={_md5(args.ckpt)[:12]}…  (load: missing={ck['missing']} unexpected={ck['unexpected']})")
    print(f"  {BOLD}Device{RST}     : {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    print(f"  {BOLD}Input{RST}      : {cfg['DATASET']['IMG_SIZE']}  |  NMS-free (RF-DETR top-k)")
    print(f"\n  {BOLD}{CYN}Classes ({len(ds.class_names)}){RST}: {', '.join(ds.class_names)}")
    print(f"  {BOLD}{CYN}Modalities used{RST}: {modal_desc}  "
          f"(REQUIRE_ALL_MODALITIES={cfg['DATASET'].get('REQUIRE_ALL_MODALITIES')})")
    print(f"  {BOLD}Images{RST}     : {len(ds)}  "
          f"(night {len(night_ids)} / normal {len(normal_ids)}, by annotation)")
    print(f"  {BOLD}Dataset{RST}    : {cfg['DATASET']['ROOT']}")
    print(f"  {BOLD}Output{RST}     : {args.out}  (overlays -> {viz_dir})")

    if not args.auto:
        try:
            input(f"\n  {YEL}Press ENTER to start the certification run…{RST} ")
        except (EOFError, KeyboardInterrupt):
            print(); return

    # ---- streaming per-image inference + viz + GT comparison --------------------
    from pycocotools.coco import COCO
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco = COCO(ann)
    cat_id_to_name = {c['id']: c['name'] for c in gt_coco.loadCats(gt_coco.getCatIds())}
    idx_to_cat = {v: k for k, v in ds.cat_id_to_idx.items()}
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')

    all_preds = []; lat = []; kept = 0
    tot_tp = tot_fp = tot_fn = 0
    log_path = os.path.join(args.out, 'inference_log.txt')
    logf = open(log_path, 'w')
    print(f"\n  {BOLD}#### / total   image                         infer    det (by class)          GT   TP/FP/FN{RST}")
    model.eval()
    with torch.no_grad():
        for n, batch in enumerate(loader):
            if args.stride > 1 and n % args.stride:
                continue
            if args.limit is not None and kept >= args.limit:
                break
            kept += 1
            modals = [k for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
            sample = {m: batch[m].to(dev) for m in modals}
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(sample)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            lat.append(dt)

            img_hw = sample[modals[0]].shape[-2:]
            det = out['detections'][0]
            image_id = int(batch['image_id'][0]); fname = batch['file_name'][0]
            keep = det['scores'] > args.eval_thresh
            boxes, scores, cls = det['boxes'][keep], det['scores'][keep], det['class_ids'][keep]
            oh, ow = batch['orig_size'][0].tolist()
            if boxes.shape[0]:
                bo = rescale_boxes_to_orig(boxes.cpu(), oh, ow, img_hw[0], img_hw[1], resize_mode)
                all_preds.extend(format_predictions_coco(bo.cpu(), scores.cpu(), cls.cpu(), image_id, idx_to_cat))

            # GT for this image (orig px)
            gts = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=[image_id]))
            gt_boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in gts]
            gt_cls = [ds.cat_id_to_idx[a['category_id']] for a in gts]

            # per-image line uses the (higher) viz threshold
            vk = scores > args.score_thresh
            vb_n = int(vk.sum())
            vb = (rescale_boxes_to_orig(boxes[vk].cpu(), oh, ow, img_hw[0], img_hw[1], resize_mode).numpy()
                  if vb_n else np.zeros((0, 4)))
            vc = cls[vk].cpu().numpy()
            tp, fp, fn = _tp_fp_fn(vb, vc, np.asarray(gt_boxes), np.asarray(gt_cls))
            tot_tp += tp; tot_fp += fp; tot_fn += fn
            by = Counter(ds.class_names[int(c)] for c in vc)
            by_s = ' '.join(f'{k}:{v}' for k, v in by.most_common(4)) or '-'
            line = (f"  [{kept:4d}/{len(ds)}] {os.path.basename(fname)[:26]:26s} {dt:6.1f}ms "
                    f"{vb_n:2d} [{by_s[:22]:22s}] GT={len(gts):2d}  {GRN}{tp}{RST}/{YEL}{fp}{RST}/{fn}")
            print(line)
            logf.write(f"{kept}\t{fname}\t{dt:.1f}\tdet={vb_n}\tGT={len(gts)}\tTP={tp}\tFP={fp}\tFN={fn}\n")

            # save overlay: predictions (per-class colour) + GT (thin, for comparison)
            try:
                rgb_key = 'img' if 'img' in sample else modals[0]
                img_np = (sample[rgb_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
                pil = draw_detections(img_np, boxes[vk].cpu(), scores[vk].cpu(), cls[vk].cpu(),
                                      ds.class_names, args.score_thresh)
                from PIL import ImageDraw
                d = ImageDraw.Draw(pil); sx, sy = img_hw[1] / ow, img_hw[0] / oh
                for gb in gt_boxes:
                    d.rectangle([gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy], outline=(0, 255, 0), width=1)
                op = os.path.join(viz_dir, f'{kept:04d}_{os.path.splitext(os.path.basename(fname))[0]}.png')
                pil.save(op)
                if args.show and os.environ.get('DISPLAY'):
                    import cv2
                    cv2.imshow('cert', cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
            except Exception as e:
                logf.write(f"  [viz warn] {e}\n")
    logf.close()

    # ---- final: mAP50 (headline) + per-class + FPS -----------------------------
    ids_all = list({int(p['image_id']) for p in all_preds}) or None
    ov = eval_overall(ann, all_preds, ids_all)
    ni = [i for i in night_ids if any(p['image_id'] == i for p in all_preds)]
    no = [i for i in normal_ids if any(p['image_id'] == i for p in all_preds)]
    ov_n = eval_overall(ann, all_preds, ni) if ni else None
    ov_m = eval_overall(ann, all_preds, no) if no else None
    pc = eval_per_class(ann, all_preds, ids_all)

    mean_lat = float(np.mean(lat)) if lat else 0.0
    fps = 1000.0 / mean_lat if mean_lat else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0.0

    print(f"\n{BOLD}╔══════════ Certification Report ══════════╗{RST}")
    print(f"  images scored : {ov.get('n_images', len(all_preds and ids_all or []))}   "
          f"detections: {len(all_preds)}   per-image TP/FP/FN: {tot_tp}/{tot_fp}/{tot_fn}")
    print(f"  {BOLD}mAP   (.50:.95){RST} : {ov['AP']:.4f}")
    print(f"  {BOLD}mAP75{RST}          : {ov['AP75']:.4f}")
    if ov_n and ov_m:
        print(f"  night mAP50 {ov_n['AP50']:.4f}  /  normal mAP50 {ov_m['AP50']:.4f}  "
              f"(gap {ov_n['AP50'] - ov_m['AP50']:+.4f})")
    print(f"  {BOLD}── per-class AP50 ──{RST}")
    for r in sorted(pc, key=lambda x: -(x['AP50'] if x['AP50'] == x['AP50'] else -1)):
        v = 'n/a' if r['AP50'] != r['AP50'] else f"{r['AP50']:.3f}"
        print(f"     {r['name'][:22]:22s} {v}")
    print(f"\n  {BOLD}Speed{RST} : {fps:.2f} FPS  (mean {mean_lat:.1f} ms/img, BS1, forward-only)   "
          f"VRAM {vram:.2f} GB")
    print(f"\n  {BOLD}{GRN}►►  mAP50 = {ov['AP50']:.4f}  |  FPS = {fps:.2f}  ◄◄{RST}")
    print(f"{BOLD}╚══════════════════════════════════════════╝{RST}")
    print(f"  logs: {log_path}   overlays: {viz_dir}")

    import json
    with open(os.path.join(args.out, 'cert_report.json'), 'w') as f:
        json.dump({'overall': ov, 'night': ov_n, 'normal': ov_m, 'per_class': pc,
                   'fps': fps, 'mean_latency_ms': mean_lat, 'vram_gb': vram,
                   'checkpoint': ck, 'cfg': args.cfg, 'ckpt': args.ckpt}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
