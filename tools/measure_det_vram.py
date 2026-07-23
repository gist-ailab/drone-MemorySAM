"""BS1 inference VRAM measurement for ViT-* ReliaDINO-RFDETR detectors (P37b-Det family).

Measures peak_allocated / peak_reserved for a single 768x768x3-modal forward pass
(BS1, fp32, no_grad, eval mode) — VRAM only, no training/eval/accuracy involved.
Reuses tools/_det_common.py:build_detector so the model construction is identical
to the real det pipeline (same P37b base cfg, only MODEL.BACKBONE_TIMM swapped).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/measure_det_vram.py \
        --cfg configs/det/det_P37b_classtoken_yeon.yaml \
        --backbones vit_small_patch16_dinov3,vit_small_plus_patch16_dinov3,vit_base_patch16_dinov3,vit_large_patch16_dinov3 \
        --dtypes fp32,fp16
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit('/tools/', 1)[0])
from tools._det_common import build_detector, load_cfg  # noqa: E402

N_CLASSES_DEFAULT = 10  # poongsan_v2 category count (VRAM-irrelevant: cls head is tiny)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_one(cfg: dict, backbone: str, device: torch.device, n_classes: int,
                 dtype: torch.dtype, warmup: int = 2, steady: int = 5) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg['MODEL']['BACKBONE_TIMM'] = backbone
    # RF-DETR head loads no COCO ckpt here (build_detector always passes coco_ckpt=None
    # for ReliaDINORFDETRDetector — weights would come from a det ckpt in real eval,
    # irrelevant for a VRAM-only footprint measurement).

    modals = cfg['DATASET']['MODALS']
    img_size = cfg['DATASET'].get('IMG_SIZE', [768, 768])
    h, w = img_size[0], img_size[1]

    # fp16/bf16 = AMP autocast (weights stay fp32, matmuls run in half) — full
    # .half() casting breaks on this model (some submodules build fp32 buffers
    # ad hoc, independent of input dtype -> dtype-mismatch RuntimeError). AMP
    # autocast is also what the repo's own training/eval config uses (AMP: true),
    # so it is the representative "half" inference mode, not a raw cast.
    use_amp = dtype != torch.float32

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    model = build_detector(cfg, device, n_classes)
    params_m = count_params(model)
    model.eval()
    build_s = time.time() - t0

    x = {m: torch.randn(1, 3, h, w, device=device, dtype=torch.float32) for m in modals}

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)  # peak from steady-state only
            for _ in range(steady):
                _ = model(x)
            torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    del model, x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'backbone': backbone, 'dtype': str(dtype).replace('torch.', ''),
        'params_M': round(params_m, 2),
        'peak_allocated_GB': round(peak_alloc, 3),
        'peak_reserved_GB': round(peak_reserved, 3),
        'build_s': round(build_s, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/det/det_P37b_classtoken_yeon.yaml')
    ap.add_argument('--backbones', default=(
        'vit_small_patch16_dinov3,vit_small_plus_patch16_dinov3,'
        'vit_base_patch16_dinov3,vit_large_patch16_dinov3'))
    ap.add_argument('--dtypes', default='fp32')
    ap.add_argument('--n_classes', type=int, default=N_CLASSES_DEFAULT)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    dtypes = [dtype_map[d.strip()] for d in args.dtypes.split(',')]
    backbones = [b.strip() for b in args.backbones.split(',')]

    assert torch.cuda.is_available(), "CUDA required for VRAM measurement"
    device = torch.device('cuda:0')
    print(f"[measure_det_vram] device={torch.cuda.get_device_name(device)} "
          f"cfg={args.cfg}")

    cfg = load_cfg(args.cfg)
    results = []
    for backbone in backbones:
        for dtype in dtypes:
            try:
                r = measure_one(cfg, backbone, device, args.n_classes, dtype)
            except RuntimeError as e:
                r = {'backbone': backbone, 'dtype': str(dtype).replace('torch.', ''),
                     'error': str(e)[:300]}
                torch.cuda.empty_cache()
            print(json.dumps(r))
            results.append(r)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[measure_det_vram] wrote {args.out}")


if __name__ == '__main__':
    main()
