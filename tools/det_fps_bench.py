"""Single-image (batch=1) FPS benchmark for any det model this repo defines.

Generalization of the ad hoc `fps_bench_D1.py` used for the 2026-07-23
det-certification FPS measurements (see analysis_logs/det_fps_3090_20260723.json
and analysis_logs/det_D1_certification_20260722/fps_bench.json). Config+checkpoint
driven like the rest of tools/ (_det_common.py) — no hardcoded repo path, no
hardcoded model.

Measures pure forward + postprocess (score-thresh filter + box rescale to
original image size), excluding dataloader/image-preprocessing time. Runs
`--n-warmup` untimed iterations then averages over `--n-measure` timed ones.

Usage:
  python tools/det_fps_bench.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
      --ckpt outputs/det_D1_vitsp_jarvis/det_D1_vitsp_jarvis/best_checkpoint.pth \
      --out analysis/det_D1_vitsp_fps --gpu 0

Certification protocol (reproduce the 3090 numbers in det_fps_3090_20260723.json):
  --mode val --n-warmup 15 --n-measure 150 --score-thresh 0.05
  (this is the default; the config's DATASET.IMG_SIZE, normally [768, 768],
  fixes the input resolution — override only if the cert config changes it).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'semseg/models/sam2'))
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from _det_common import load_cfg, build_detector, build_loader, load_det_checkpoint  # noqa: E402
from objdet.datasets.multimodal_det import rescale_boxes_to_orig  # noqa: E402


@torch.no_grad()
def bench(cfg_path: str, ckpt_path: str, device: torch.device, mode: str = 'val',
          n_warmup: int = 15, n_measure: int = 150, score_thresh: float = 0.05,
          workers: int = 2) -> dict:
    cfg = load_cfg(cfg_path)
    ds, loader = build_loader(cfg, mode, workers=workers)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, device, n_classes)
    ck = load_det_checkpoint(model, ckpt_path, device)
    print(f'[det-fps] loaded {ckpt_path} epoch={ck.get("epoch")} '
          f'missing={ck["missing"]} unexpected={ck["unexpected"]}')
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')

    it = iter(loader)
    times = []
    total = n_warmup + n_measure
    for i in range(total):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        modals = [k for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device, non_blocking=True) for m in modals}
        img_hw = sample[modals[0]].shape[-2:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(sample)
        for bi, det in enumerate(out['detections']):
            keep = det['scores'] > score_thresh
            boxes, scores, cls = det['boxes'][keep], det['scores'][keep], det['class_ids'][keep]
            if boxes.shape[0] > 0:
                oh, ow = batch['orig_size'][bi].tolist()
                boxes = rescale_boxes_to_orig(boxes.cpu(), oh, ow, img_hw[0], img_hw[1], resize_mode)
                _ = boxes.cpu(), scores.cpu(), cls.cpu()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append(t1 - t0)

    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times)
    median_s = statistics.median(times)
    result = {
        'gpu': torch.cuda.get_device_name(device),
        'cfg': cfg_path,
        'ckpt': ckpt_path,
        'input_res': cfg['DATASET'].get('IMG_SIZE'),
        'batch_size': 1,
        'n_warmup': n_warmup,
        'n_measured': len(times),
        'mean_ms': mean_s * 1000,
        'median_ms': median_s * 1000,
        'std_ms': std_s * 1000,
        'fps_mean': 1.0 / mean_s,
        'fps_median': 1.0 / median_s,
        'score_thresh': score_thresh,
        'note': 'forward+postprocess only (score-thresh filter + box rescale); '
                'excludes dataloader/image preprocessing.',
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='output path prefix (writes <out>.json)')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n-warmup', type=int, default=15)
    ap.add_argument('--n-measure', type=int, default=150)
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('GPU:', torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu')
    result = bench(args.cfg, args.ckpt, dev, args.mode, args.n_warmup, args.n_measure,
                    args.score_thresh, args.workers)
    print(f"n_measured={result['n_measured']} mean_ms={result['mean_ms']:.3f} "
          f"median_ms={result['median_ms']:.3f} std_ms={result['std_ms']:.3f} "
          f"fps_mean={result['fps_mean']:.3f} fps_median={result['fps_median']:.3f}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out_json = args.out if args.out.endswith('.json') else args.out + '.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[det-fps] wrote {out_json}')


if __name__ == '__main__':
    main()
