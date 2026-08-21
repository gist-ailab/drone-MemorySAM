"""Single-image (batch=1) FPS benchmark for D1 (RF-DETR + DINOv3 ReliaDINO det).

Measures pure forward + postprocess (score-thresh filter + box rescale), excluding
dataloader/preprocessing time. Warmup then average over N samples.
"""
import os
import sys
import time
import statistics

import torch

ROOT = '/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'semseg/models/sam2'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
os.chdir(ROOT)

from _det_common import load_cfg, build_detector, build_loader, load_det_checkpoint  # noqa: E402
from objdet.datasets.multimodal_det import rescale_boxes_to_orig  # noqa: E402

CFG = 'configs/det/det_D1_p37b_lowlr_yeon.yaml'
CKPT = 'outputs/det_D1_p37b_lowlr_yeon/det_D1_p37b_lowlr_yeon/best_checkpoint.pth'
N_WARMUP = 15
N_MEASURE = 150
SCORE_THRESH = 0.05

def main():
    dev = torch.device('cuda')
    print('GPU:', torch.cuda.get_device_name(dev))
    cfg = load_cfg(CFG)
    ds, loader = build_loader(cfg, 'val', workers=2)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    ck = load_det_checkpoint(model, CKPT, dev)
    print('loaded ckpt epoch', ck.get('epoch'))
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')

    it = iter(loader)
    times = []
    total = N_WARMUP + N_MEASURE
    for i in range(total):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        modals = [k for k in batch if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(dev, non_blocking=True) for m in modals}
        img_hw = sample[modals[0]].shape[-2:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(sample)
            for bi, det in enumerate(out['detections']):
                keep = det['scores'] > SCORE_THRESH
                boxes, scores, cls = det['boxes'][keep], det['scores'][keep], det['class_ids'][keep]
                if boxes.shape[0] > 0:
                    oh, ow = batch['orig_size'][bi].tolist()
                    boxes = rescale_boxes_to_orig(boxes.cpu(), oh, ow, img_hw[0], img_hw[1], resize_mode)
                    _ = boxes.cpu(), scores.cpu(), cls.cpu()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times)
    median_s = statistics.median(times)
    fps = 1.0 / mean_s
    print(f'n_measured={len(times)} mean_ms={mean_s*1000:.3f} median_ms={median_s*1000:.3f} '
          f'std_ms={std_s*1000:.3f} fps_mean={fps:.3f} fps_median={1.0/median_s:.3f}')
    import json
    with open('analysis/D1_certification_20260722/fps_bench.json', 'w') as f:
        json.dump({
            'gpu': torch.cuda.get_device_name(dev),
            'input_res': cfg['DATASET']['IMG_SIZE'],
            'batch_size': 1,
            'n_warmup': N_WARMUP,
            'n_measured': len(times),
            'mean_ms': mean_s * 1000,
            'median_ms': median_s * 1000,
            'std_ms': std_s * 1000,
            'fps_mean': fps,
            'fps_median': 1.0 / median_s,
            'score_thresh': SCORE_THRESH,
            'note': 'forward+postprocess only (score-thresh filter + box rescale); '
                    'excludes dataloader/image preprocessing. GPU shared with a '
                    'concurrent training job (P39.1-DELIVER) at ~100% util - see log for contention caveat.',
        }, f, indent=2)

if __name__ == '__main__':
    main()
