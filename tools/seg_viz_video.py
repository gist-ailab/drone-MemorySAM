"""GT-vs-prediction side-by-side inference VIDEO for a segmentation checkpoint.

Runs a seg model over the val/test set (dataset order) and renders each frame as
four panels — input RGB | GT | Prediction | Error — colorized with the dataset
palette, then encodes an mp4. Model-agnostic: driven by the training cfg +
checkpoint via val.load_model / val.create_dataset (the same infra as
tools/viz_features.py), so any ReliaDINO P-version works by swapping
--cfg/--model_path.

Unlike tools/viz_features.py (a few still feature-panel PNGs), this is a full
sequential pass producing an input|GT|Pred|Error comparison video with a per-frame
mIoU stat bar — meant to *watch* how a model segments a whole eval set.

Usage:
  python tools/seg_viz_video.py \
      --cfg configs/jarvis-deliver_rgbdel_P39_1_rank.yaml \
      --model_path <ckpt.pth> \
      --dataset-root /ailab_mat2/dataset/DELIVER --split val \
      --out-dir videos/inference_deliver_p39_1 --name p39_1_deliver \
      --gpu 0 --fps 4 --num 0        # num 0 = whole split

Env: needs `import val` importable (run from repo root). ReliaDINO backbone note:
if HF is flaky, export RELIADINO_LOCAL_BACKBONE=<local safetensors> (encoder.py).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.getcwd())          # so `import val` resolves from repo root
import val as V                          # reuse load_model / create_dataset / augs

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm_rgb(t):
    """(C,H,W) normalized tensor -> (H,W,3) uint8. Non-3ch modals -> grayscale."""
    a = t.detach().float().cpu().numpy().transpose(1, 2, 0)
    if a.shape[2] == 3:
        a = a * IMAGENET_STD + IMAGENET_MEAN
    else:
        a = a[..., :1]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        a = np.repeat(a, 3, axis=2)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def to_rgb(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return x[..., :3].astype(np.uint8)


def resize_h(img, h):
    im = Image.fromarray(to_rgb(img))
    w = max(1, round(im.width * h / im.height))
    return np.asarray(im.resize((w, h), Image.NEAREST))


def frame_miou(pred, gt, ignore, n_cls):
    ious = []
    for c in range(n_cls):
        p, g = (pred == c), (gt == c)
        if not g.any() and not p.any():
            continue
        inter = (p & g).sum()
        union = (p | g).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious) * 100) if ious else float('nan')


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def compose(panels_titles, header, bar_h=34, sep=4, panel_h=512):
    """panels_titles: list of (img, title). Returns one (H,W,3) uint8 frame."""
    imgs = [resize_h(im, panel_h) for im, _ in panels_titles]
    tot_w = sum(im.shape[1] for im in imgs) + sep * (len(imgs) - 1)
    canvas = np.full((panel_h + bar_h + 18, tot_w, 3), 20, np.uint8)
    x = 0
    pim = Image.fromarray(canvas)
    dr = ImageDraw.Draw(pim)
    ft = _font(15)
    fh = _font(17)
    dr.text((6, 2), header, fill=(255, 255, 255), font=fh)
    for (im, (_, title)) in zip(imgs, panels_titles):
        pim.paste(Image.fromarray(im), (x, bar_h))
        dr.text((x + 4, bar_h + panel_h + 1), title, fill=(230, 230, 230), font=ft)
        x += im.shape[1] + sep
    return np.asarray(pim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', default=None,
                    help="ReliaDINO ckpt; omit if --pred-dir is given")
    ap.add_argument('--pred-dir', default=None,
                    help="use precomputed label PNGs <pred-dir>/<stem>.png "
                         "(e.g. a competitor's dumped predictions) instead of "
                         "running a model. Stems must match the dataset (same glob).")
    ap.add_argument('--dump-pred-dir', default=None,
                    help="also save each prediction as <dump-dir>/<key>.png "
                         "(gt-resolution label map, unique-key naming) — reuse for "
                         "frame selection / cross-model comparison")
    ap.add_argument('--dataset-root', default=None, help="override DATASET.ROOT")
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--case', default=None, help="condition filter (e.g. DELIVER night)")
    ap.add_argument('--num', type=int, default=0, help="max frames (0 = whole split)")
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default='seg_infer', help="mp4 basename (no ext)")
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--panel-h', type=int, default=512)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root:
        ds_cfg['ROOT'] = args.dataset_root
    if args.case is not None:
        ds_cfg['CASE'] = args.case
    if isinstance(ds_cfg.get('PHYSAUG'), dict):
        ds_cfg['PHYSAUG']['ENABLE'] = False

    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    image_size = (cfg['EVAL']['IMAGE_SIZE'] if args.split == 'val'
                  else cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE']))
    transform = V.get_val_augmentation(image_size, dataset_cfg=ds_cfg)
    dataset, has_gt = V.create_dataset(ds_cfg, args.split, transform, args.split,
                                       macvi=False, eval_day=False)
    palette = dataset.PALETTE
    ds_cls = V._get_dataset_class(dataset)
    modals = ds_cfg.get('MODALS')
    n_cls = len(dataset.CLASSES)
    ignore = getattr(dataset, 'ignore_label', 255)

    if not args.pred_dir and not args.model_path:
        ap.error("provide --model_path (run a model) or --pred-dir (use dumped preds)")
    model = None
    if not args.pred_dir:
        model = V.load_model(cfg, Path(args.model_path), device)
        model.eval()

    n = len(dataset) if args.num <= 0 else min(args.num, len(dataset))
    print(f"[seg-video] split={args.split} case={args.case} frames={n}/{len(dataset)} "
          f"modals={modals} cls={n_cls}", flush=True)

    # --pred-dir lookup key: DELIVER reuses the same basename stem across scene
    # folders, so stems are NOT unique. Use the path after '/img/' (scene + split
    # + subdir + stem) with '/'->'__' — identical on both sides since our loader
    # and the competitor glob the same DELIVER tree. The dump script must key by
    # the same rule.
    _files = None
    if args.pred_dir or args.dump_pred_dir:
        _files = getattr(dataset, 'files', None)
        if _files is None:
            import glob as _g
            _files = sorted(_g.glob(os.path.join(ds_cfg['ROOT'], 'img', '*',
                                                 args.split, '*', '*.png')))
            if ds_cfg.get('CASE'):
                _files = [f for f in _files if ds_cfg['CASE'] in f]

    # Pred-key must be UNIQUE per image and identical on both sides (our loader
    # and the dumped preds). Two dataset layouts:
    #   DELIVER: same basename stem repeats across scene folders → NOT unique.
    #            Key = path after '/img/' (scene+split+subdir+stem), '/'->'__'.
    #   MUSES : '/img/' absent; basename stem is globally unique (verified) →
    #            plain stem, stripping the '_frame_camera' rgb suffix so it
    #            matches the competitor's inference-mode naming.
    _ds_name = type(dataset).__name__
    def _pred_key(rgb_path):
        s = str(rgb_path)
        if '/img/' in s:                                  # DELIVER-style tree
            return os.path.splitext(s.split('/img/')[-1])[0].replace('/', '__')
        stem = os.path.splitext(os.path.basename(s))[0]   # MUSES / flat trees
        for suf in ('_frame_camera', '_rgb', '_rgb_front'):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        return stem

    dump_dir = None
    if args.dump_pred_dir:
        dump_dir = Path(args.dump_pred_dir).expanduser()
        dump_dir.mkdir(parents=True, exist_ok=True)

    mp4 = out / f"{args.name}.mp4"
    sink = _VideoSink(mp4, args.fps)                      # streaming: no frame accumulation
    mious = []
    for idx in range(n):
        images, label, meta = dataset[idx]
        if args.pred_dir:                                 # use dumped competitor preds
            key = _pred_key(_files[idx])
            pp = Path(args.pred_dir) / f"{key}.png"
            if not pp.exists():
                print(f"[seg-video] WARN missing pred {key}.png; skip", flush=True)
                continue
            pred = np.array(Image.open(pp)).astype(np.uint8)
        else:
            imgs = [im.unsqueeze(0).to(device) for im in images]
            with torch.no_grad():
                m_output, _ = model(imgs, multimask_output=True)
            pred = m_output[0].argmax(0).cpu().numpy().astype(np.uint8)
        gt = np.asarray(label).astype(np.int32)
        gh, gw = gt.shape
        if pred.shape != (gh, gw):
            pred = np.asarray(torch.nn.functional.interpolate(
                torch.tensor(pred)[None, None].float(), size=(gh, gw),
                mode='nearest')[0, 0]).astype(np.uint8)
        if dump_dir is not None:                          # save pred for later reuse
            Image.fromarray(pred).save(dump_dir / f"{_pred_key(_files[idx])}.png")
        err = np.zeros((gh, gw, 3), np.uint8)
        wrong = (pred != gt) & (gt != ignore)
        err[wrong] = (220, 40, 40)
        fm = frame_miou(pred, gt, ignore, n_cls)
        mious.append(fm)

        inp = denorm_rgb(images[0])                       # first modal = RGB
        gt_c = ds_cls.decode_segmap(np.clip(gt, 0, n_cls - 1).astype(np.uint8), palette)
        pr_c = ds_cls.decode_segmap(pred, palette)
        header = (f"{args.name}  |  frame {idx + 1}/{n}"
                  + (f"  case={args.case}" if args.case else "")
                  + (f"  frameMIoU={fm:.1f}" if not math.isnan(fm) else ""))
        sink.write(compose(
            [(inp, f"input:{modals[0]}"), (gt_c, "GT"), (pr_c, "Pred"),
             (err, "Error(red=wrong)")],
            header, panel_h=args.panel_h))
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{n}", flush=True)

    sink.close()
    valid = [m for m in mious if not math.isnan(m)]
    mean_miou = float(np.mean(valid)) if valid else float('nan')
    print(f"[seg-video] wrote {mp4} ({n} frames, {args.fps}fps, {sink.backend}) "
          f"mean per-frame mIoU={mean_miou:.2f}", flush=True)


class _VideoSink:
    """Streaming mp4 writer (frames encoded one-by-one, never accumulated).
    imageio(ffmpeg libx264) preferred; cv2(mp4v) fallback."""

    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.backend = None
        self._w = None
        try:
            import imageio.v2 as imageio
            self._w = imageio.get_writer(self.path, fps=fps, macro_block_size=None,
                                         codec='libx264', quality=8)
            self.backend = 'imageio'
        except Exception as e:
            print(f"[seg-video] imageio init failed ({e}); using cv2", flush=True)

    def write(self, frame):
        if self.backend == 'imageio':
            self._w.append_data(frame)
            return
        import cv2
        if self._w is None:                              # lazy cv2 init (needs size)
            h, w = frame.shape[:2]
            self._w = cv2.VideoWriter(str(self.path),
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      self.fps, (w, h))
            self.backend = 'cv2'
        self._w.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._w is None:
            return
        if self.backend == 'imageio':
            self._w.close()
        else:
            self._w.release()


if __name__ == '__main__':
    main()
