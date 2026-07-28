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
$DISPLAY), --stride N (every Nth image), --score-thresh (viz/log threshold),
--viz-mode panel|rgb (2-row multimodal panel, default, vs. the legacy single RGB frame).
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


# ---- visualization -----------------------------------------------------------
# The model is 3-modal (rgb + lidar depth + thermal) but the old overlay only showed
# RGB, so a night frame looked like a black square with boxes on it — no way to see
# *why* something was detected. The panel layout below puts GT|Pred on top and the
# three actual input tensors underneath.
PANEL_PX = 768                 # top row: GT / Pred, one side each
STRIP_PX = 512                 # bottom row: one modality tile each
CANVAS_W = PANEL_PX * 2        # 1536 — exactly 3 * STRIP_PX, so both rows line up
SEP = (110, 110, 110)          # panel separator lines
_MODAL_LABEL = {'img': 'RGB', 'rgb': 'RGB', 'lidar': 'LiDAR (depth)', 'thermal': 'Thermal'}
_MODAL_CMAP = {'lidar': 'COLORMAP_INFERNO', 'thermal': 'COLORMAP_JET'}


def _font(size):
    """Label font. Falls back to PIL's bitmap font so a machine without DejaVu still renders."""
    from PIL import ImageFont
    for p in ('DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _tag(pil, text, size=22):
    """Top-left panel label on an opaque backing box — the tiles underneath are often
    near-black (night RGB) or near-white (saturated thermal), so plain text disappears."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(pil); f = _font(size)
    box = d.textbbox((6, 4), text, font=f)
    d.rectangle([box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return pil


def _modal_display(t, modal):
    """(C,H,W) input tensor -> HxWx3 uint8 for display.

    RGB keeps the overlay's clamp(0,1)*255. lidar/thermal live on completely different
    value ranges (metres, raw counts), so drawn as-is they come out black; each tile gets
    its own min-max normalisation instead. This is display-only — it operates on a detached
    CPU copy and never touches the tensor the model was given.
    """
    a = t.detach().cpu().float()
    if modal in ('img', 'rgb'):
        return (a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
    g = a.mean(0).numpy() if a.dim() == 3 else a.numpy()      # 3ch depth/thermal -> 1ch
    g = (g - float(g.min())) / (float(g.max()) - float(g.min()) + 1e-6)
    g8 = (g * 255).astype('uint8')
    try:                                                      # cv2 is optional here
        import cv2
        cm = _MODAL_CMAP.get(modal)
        if cm is not None:
            return cv2.applyColorMap(g8, getattr(cv2, cm))[:, :, ::-1].copy()   # BGR->RGB
    except Exception:
        pass
    return np.repeat(g8[:, :, None], 3, axis=2)               # grayscale fallback


def _modality_strip(sample, modals, width=CANVAS_W, tile=STRIP_PX):
    """Bottom row: the tensors actually fed to the model, side by side (no re-reading files).
    Missing modality keys are skipped — draw whatever is there, centred, and move on."""
    from PIL import Image, ImageDraw
    band = Image.new('RGB', (width, tile), (0, 0, 0))
    tiles = []
    for m in modals:
        t = sample.get(m)
        if t is None:
            continue
        arr = _modal_display(t[0] if t.dim() == 4 else t, m)
        tiles.append(_tag(Image.fromarray(arr).resize((tile, tile), Image.BILINEAR),
                          _MODAL_LABEL.get(m, m), 20))
    if not tiles:
        return band
    x0 = max(0, (width - tile * len(tiles)) // 2)   # fewer than 3 modalities -> centre them
    for i, im in enumerate(tiles):
        band.paste(im, (x0 + i * tile, 0))
    d = ImageDraw.Draw(band)
    for i in range(1, len(tiles)):
        d.line([(x0 + i * tile, 0), (x0 + i * tile, tile)], fill=SEP, width=2)
    return band


def _gt_panel(img_np, gt_boxes, gt_cls, class_names, sx, sy):
    """RGB + GT only. One colour for every class (unlike the pred panel) because this
    panel exists to be eyeballed against the one next to it, not to encode class identity."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img_np)
    d = ImageDraw.Draw(pil); f = _font(16)
    for gb, c in zip(gt_boxes, gt_cls):
        x1, y1, x2, y2 = gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        ci = int(c)
        name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
        tb = d.textbbox((x1, y1), name, font=f)
        d.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=(0, 128, 0))
        d.text((x1, y1), name, fill=(255, 255, 255), font=f)
    return pil


def _compose_panel(gt_img, pred_img, strip):
    """GT | Pred (768² each) over the modality strip (512 tall) = 1536x1280."""
    from PIL import Image, ImageDraw
    h = PANEL_PX + (strip.height if strip is not None else 0)
    canvas = Image.new('RGB', (CANVAS_W, h), (0, 0, 0))
    canvas.paste(gt_img.resize((PANEL_PX, PANEL_PX)), (0, 0))
    canvas.paste(pred_img.resize((PANEL_PX, PANEL_PX)), (PANEL_PX, 0))
    if strip is not None:
        canvas.paste(strip, (0, PANEL_PX))
    d = ImageDraw.Draw(canvas)
    d.line([(PANEL_PX, 0), (PANEL_PX, PANEL_PX)], fill=SEP, width=2)
    if strip is not None:
        d.line([(0, PANEL_PX), (CANVAS_W, PANEL_PX)], fill=SEP, width=2)
    return canvas


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
    ap.add_argument('--viz-mode', default='panel', choices=['panel', 'rgb'],
                    help="overlay layout: 'panel' = GT|Pred over the 3 input modalities "
                         "(1536x1280), 'rgb' = legacy single 768 RGB frame")
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
    cfg_modals = list(cfg['DATASET'].get('MODALS') or [])   # display order for the viz strip

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

            # save overlay — outside the t0/dt window on purpose, so rendering never
            # contaminates the latency/FPS numbers
            try:
                rgb_key = 'img' if 'img' in sample else modals[0]
                img_np = (sample[rgb_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
                pil = draw_detections(img_np, boxes[vk].cpu(), scores[vk].cpu(), cls[vk].cpu(),
                                      ds.class_names, args.score_thresh)
                from PIL import ImageDraw
                sx, sy = img_hw[1] / ow, img_hw[0] / oh
                if args.viz_mode == 'rgb':
                    # legacy single frame (kept for certification-artifact compatibility):
                    # predictions in per-class colour + GT as thin green boxes on one image
                    d = ImageDraw.Draw(pil)
                    for gb in gt_boxes:
                        d.rectangle([gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy], outline=(0, 255, 0), width=1)
                    vis = pil
                else:
                    order = ([m for m in cfg_modals if m in sample]
                             + [m for m in modals if m not in cfg_modals])
                    vis = _compose_panel(
                        _tag(_gt_panel(img_np, gt_boxes, gt_cls, ds.class_names, sx, sy), 'GT'),
                        _tag(pil, 'Pred'),
                        _modality_strip(sample, order))
                op = os.path.join(viz_dir, f'{kept:04d}_{os.path.splitext(os.path.basename(fname))[0]}.png')
                vis.save(op)
                if args.show and os.environ.get('DISPLAY'):
                    import cv2
                    cv2.imshow('cert', cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)); cv2.waitKey(1)
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
