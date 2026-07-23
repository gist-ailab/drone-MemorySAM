"""GT-vs-prediction side-by-side inference video for a detection checkpoint.

Runs a detector over a test clip in temporal (file-name) order and renders each
frame as two panels — left = ground truth, right = model prediction — with
per-class colors, score labels, and a title/stat bar, then encodes an mp4. Built
for the D1 certification deliverable (2026-07-23) but model-agnostic: everything
is driven by the training cfg + checkpoint via tools/_det_common.py, so any
det P-version works by swapping --cfg/--ckpt.

Unlike det_viz_samples.py (a few still PNGs, prediction only), this is a full
temporal pass producing a GT|Pred comparison video.

Usage:
  python tools/det_viz_video.py \
      --cfg configs/det/det_D1_vitsp_jarvis.yaml \
      --ckpt /ailab_mat2/.../submission/ckpts/det_D1_vitsp_20260723/best_checkpoint.pth \
      --clip capture_20260618_114021 \
      --out /drone_nas/.../analysis_logs/det_cert_20260723/viz_video/vitsp_night.mp4 \
      --score-thresh 0.3 --fps 8 --panel-w 900
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_ROOT) == 'tools':
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from _det_common import load_cfg, build_detector, load_det_checkpoint, DEFAULT_LOWLIGHT_CLIPS  # noqa: E402
from objdet.datasets.multimodal_det import rescale_boxes_to_orig  # noqa: E402

# Distinct, high-contrast palette (tab10-ish), indexed by class idx.
PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
    (94, 53, 177), (233, 30, 99),
]
GT_COLOR = (46, 204, 113)      # ground-truth accent (green)
BG = (18, 18, 22)
FG = (235, 235, 240)
SUB = (150, 150, 160)


def _font(size: int):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _color(idx: int):
    return PALETTE[idx % len(PALETTE)]


class VideoSink:
    """Stream RGB frames to mp4 with no temp files. Prefers ffmpeg stdin pipe
    (libx264, high quality); falls back to cv2.VideoWriter (mp4v) if ffmpeg is
    missing. Streaming avoids the multi-GB PNG scratch that would pressure a
    space-tight SSD."""

    def __init__(self, path: str, fps: int, size_wh):
        import shutil
        self.path, self.fps, self.w, self.h = path, fps, size_wh[0], size_wh[1]
        self.mode = 'ffmpeg' if shutil.which('ffmpeg') else 'cv2'
        if self.mode == 'ffmpeg':
            import subprocess
            cmd = ['ffmpeg', '-y', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{self.w}x{self.h}', '-r', str(fps), '-i', '-',
                   '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            import cv2
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (self.w, self.h))

    def write(self, arr):  # arr: HxWx3 uint8 RGB, already sized (w,h)
        if arr.shape[1] != self.w or arr.shape[0] != self.h:
            import cv2
            arr = cv2.resize(arr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.mode == 'ffmpeg':
            self.p.stdin.write(arr.astype(np.uint8).tobytes())
        else:
            import cv2
            self.vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.mode == 'ffmpeg':
            self.p.stdin.close(); self.p.wait()
        else:
            self.vw.release()


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes -> [len(a), len(b)]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-6, None)


def count_matches(gt: np.ndarray, gt_lab, pred: np.ndarray, pred_lab, iou_thr=0.5):
    """Greedy TP/FP/FN at iou_thr with class agreement — footer stats only."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    ious = _iou_matrix(pred, gt)
    matched_gt = set()
    tp = 0
    for pi in range(len(pred)):        # pred already score-sorted upstream (greedy hi->lo)
        best_j, best_iou = -1, iou_thr
        for gj in range(len(gt)):
            if gj in matched_gt or gt_lab[gj] != pred_lab[pi]:
                continue
            if ious[pi, gj] >= best_iou:
                best_iou, best_j = ious[pi, gj], gj
        if best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def draw_panel(img_rgb: np.ndarray, boxes_xyxy: np.ndarray, labels, scores,
               class_names, panel_w: int, caption: str, font, font_sm):
    """Resize orig image to panel_w and draw boxes; return a PIL image with caption bar."""
    oh, ow = img_rgb.shape[:2]
    scale = panel_w / ow
    ph = int(round(oh * scale))
    im = Image.fromarray(img_rgb).resize((panel_w, ph), Image.BILINEAR).convert('RGB')
    dr = ImageDraw.Draw(im, 'RGBA')
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = (boxes_xyxy[i] * scale).tolist()
        idx = int(labels[i])
        col = GT_COLOR if scores is None else _color(idx)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        tag = name if scores is None else f'{name} {scores[i]:.2f}'
        tw = dr.textlength(tag, font=font_sm)
        ty = max(0, y1 - 15)
        dr.rectangle([x1, ty, x1 + tw + 6, ty + 15], fill=col + (230,))
        dr.text((x1 + 3, ty + 1), tag, fill=(10, 10, 12), font=font_sm)
    # caption bar
    bar_h = 30
    out = Image.new('RGB', (panel_w, ph + bar_h), BG)
    out.paste(im, (0, bar_h))
    d2 = ImageDraw.Draw(out)
    d2.text((10, 6), caption, fill=FG, font=font)
    return out, scale


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='output .mp4 path')
    ap.add_argument('--mode', default='val', choices=['val', 'test'],
                    help="which split (ANNOTATION_VAL is the cert test set here)")
    ap.add_argument('--clip', default='', help='substring filter on file_name (one clip)')
    ap.add_argument('--score-thresh', type=float, default=0.3)
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--panel-w', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0, help='0 = all frames in clip')
    ap.add_argument('--gap', type=int, default=16)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    cfg = load_cfg(args.cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from train_det import build_dataset
    ds = build_dataset(cfg, args.mode)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    ck = load_det_checkpoint(model, args.ckpt, dev)
    print(f"[viz-video] loaded {args.ckpt} "
          f"(missing={ck['missing']} unexpected={ck['unexpected']} ep={ck.get('epoch')})")

    rgb_root = cfg['DATASET']['ROOT'] if 'MODALITY_KEYS' in cfg['DATASET'] \
        else cfg['DATASET']['MODALITIES']['img']['ROOT']
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')

    # temporal order within the clip
    order = list(range(len(ds)))
    order = [i for i in order
             if (not args.clip) or args.clip in ds.images[ds.img_ids[i]]['file_name']]
    order.sort(key=lambda i: ds.images[ds.img_ids[i]]['file_name'])
    if args.limit:
        order = order[:args.limit]
    if not order:
        raise SystemExit(f"[viz-video] no frames match clip='{args.clip}' in mode={args.mode}")
    print(f"[viz-video] {len(order)} frames (clip='{args.clip or 'ALL'}')")

    font = _font(18)
    font_sm = _font(13)
    font_hd = _font(22)

    sink = None
    tot_gt = tot_pred = tot_tp = 0
    for n, idx in enumerate(order):
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
        # score-sort (for greedy match + stable label draw)
        if scores.numel():
            sidx = torch.argsort(scores, descending=True)
            boxes, scores, cls_ids = boxes[sidx], scores[sidx], cls_ids[sidx]

        orig_h, orig_w = (sample['orig_size'].tolist()
                          if torch.is_tensor(sample['orig_size']) else sample['orig_size'])
        img_hw = batch[modals[0]].shape[-2:]
        pred_xyxy = np.zeros((0, 4), np.float32)
        if boxes.shape[0] > 0:
            pred_xyxy = rescale_boxes_to_orig(
                boxes.cpu(), orig_h, orig_w, img_hw[0], img_hw[1], resize_mode).cpu().numpy()
        pred_lab = cls_ids.cpu().numpy().astype(int)
        pred_sc = scores.cpu().numpy()

        # GT (orig px, xywh -> xyxy)
        gt_xyxy, gt_lab = [], []
        for a in ds.img_anns.get(img_id, []):
            x, y, w, h = a['bbox']
            gt_xyxy.append([x, y, x + w, y + h])
            gt_lab.append(ds.cat_id_to_idx[a['category_id']])
        gt_xyxy = np.array(gt_xyxy, np.float32).reshape(-1, 4)
        gt_lab = np.array(gt_lab, int)

        rgb = np.array(Image.open(os.path.join(rgb_root, file_name)).convert('RGB'))
        left, _ = draw_panel(rgb, gt_xyxy, gt_lab, None, ds.class_names,
                             args.panel_w, 'GROUND TRUTH', font, font_sm)
        right, _ = draw_panel(rgb, pred_xyxy, pred_lab, pred_sc, ds.class_names,
                              args.panel_w, 'ViT-S+ PREDICTION', font, font_sm)

        tp, fp, fn = count_matches(gt_xyxy, gt_lab, pred_xyxy, pred_lab, 0.5)
        tot_gt += len(gt_xyxy); tot_pred += len(pred_xyxy); tot_tp += tp

        ph = max(left.height, right.height)
        hd_h, ft_h = 34, 28
        W = args.panel_w * 2 + args.gap
        canvas = Image.new('RGB', (W, ph + hd_h + ft_h), BG)
        canvas.paste(left, (0, hd_h))
        canvas.paste(right, (args.panel_w + args.gap, hd_h))
        d = ImageDraw.Draw(canvas)
        is_night = any(c in file_name for c in DEFAULT_LOWLIGHT_CLIPS)
        title = f"D1 ViT-S+ cert eval  ·  {'NIGHT' if is_night else 'NORMAL'}  ·  frame {n+1}/{len(order)}"
        d.text((10, 7), title, fill=FG, font=font_hd)
        d.text((W - 260, 12), os.path.basename(file_name), fill=SUB, font=font_sm)
        foot = f"GT boxes: {len(gt_xyxy)}   |   Pred>{args.score_thresh:.2f}: {len(pred_xyxy)}   |   matched@0.5: {tp}   (FP {fp}, FN {fn})"
        d.text((10, ph + hd_h + 5), foot, fill=SUB, font=font_sm)

        arr = np.array(canvas)
        # even dims for H.264 (libx264 + yuv420p requires both even)
        arr = arr[:arr.shape[0] // 2 * 2, :arr.shape[1] // 2 * 2]
        if sink is None:
            sink = VideoSink(args.out, args.fps, (arr.shape[1], arr.shape[0]))
            print(f"[viz-video] encoder={sink.mode} canvas={arr.shape[1]}x{arr.shape[0]}")
        sink.write(arr)
        if (n + 1) % 25 == 0:
            print(f"[viz-video] {n+1}/{len(order)} frames")

    if sink is not None:
        sink.close()

    prec = tot_tp / max(tot_pred, 1)
    rec = tot_tp / max(tot_gt, 1)
    print(f"[viz-video] DONE -> {args.out}")
    print(f"[viz-video] {len(order)} frames @ {args.fps}fps  ·  "
          f"GT {tot_gt} / Pred {tot_pred} / TP {tot_tp}  "
          f"(clip-level greedy P={prec:.3f} R={rec:.3f})")


if __name__ == '__main__':
    main()
