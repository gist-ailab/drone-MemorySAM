"""Shared helpers for the detection analysis tools.

Model-agnostic by design: everything is driven by the training config + checkpoint,
so a new P-version needs no change here. The only place that knows about specific
modules is the toggle registry in det_module_ablation.py — and that auto-skips any
toggle whose attribute is absent, so it stays safe across versions.

Mirrors the seg suite's conventions (tools/module_ablation.py, seg_analysis_pipeline.py):
config+ckpt driven, JSON + Markdown output, toggles that self-skip.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# repo root on sys.path so `train_det` / `objdet` import regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from objdet.datasets.multimodal_det import MultiModalDetDataset, rescale_boxes_to_orig  # noqa: E402
from objdet.metrics import format_predictions_coco  # noqa: E402

# Default low-light clips of the poongsan `final` test split (114021 + 115624 =
# 1,768 frames; 114808 = 1,471 normal). Override with --lowlight-clips.
DEFAULT_LOWLIGHT_CLIPS = ('capture_20260618_114021', 'capture_20260618_115624')


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_detector(cfg: dict, device: torch.device, n_classes: int):
    """Build any detector this repo defines, from the config alone.

    Single place that knows the DET_MODEL dispatch, so the analysis tools never
    duplicate train_det/val_det. Backbone is frozen (eval-only use).
    """
    from train_det import build_seg_model
    seg_model = build_seg_model(cfg, device, n_classes)
    mc = cfg['MODEL']
    name = mc.get('DET_MODEL', 'MemorySAMDetector')
    modals = cfg['DATASET']['MODALS']

    if name == 'ReliaDINOM2FDetector':
        from objdet.models.det_model import ReliaDINOM2FDetector
        model = ReliaDINOM2FDetector(
            seg_model=seg_model, modals=modals, n_classes=n_classes,
            num_select=mc.get('NUM_SELECT', 100), freeze_backbone=True)
    elif name == 'ReliaDINORFDETRDetector':
        from objdet.models.det_model import ReliaDINORFDETRDetector
        model = ReliaDINORFDETRDetector(
            seg_model=seg_model, modals=modals, n_classes=n_classes,
            fpn_dim=mc.get('FPN_DIM', 256),
            fpn_strides=mc.get('FPN_STRIDES', [4, 8, 16, 32]),
            det_levels=mc.get('DET_LEVELS', [2]), freeze_backbone=True,
            num_queries=mc.get('NUM_QUERIES', 300),
            group_detr=mc.get('GROUP_DETR', 13),
            dec_layers=mc.get('DEC_LAYERS', 4),
            dec_n_points=mc.get('DEC_N_POINTS', 2),
            coco_ckpt=None,                       # weights come from the det ckpt
            num_select=mc.get('NUM_SELECT', 300))
    elif name == 'ReliaDINODetector':
        from objdet.models.det_model import ReliaDINODetector
        model = ReliaDINODetector(
            seg_model=seg_model, modals=modals, n_classes=n_classes,
            fpn_dim=mc.get('FPN_DIM', 256),
            fpn_strides=mc.get('FPN_STRIDES', [4, 8, 16, 32]),
            freeze_backbone=True, n_convs=mc.get('N_CONVS', 4),
            hidden_dim=mc.get('HIDDEN_DIM', 256))
    else:
        from objdet.models.det_model import MemorySAMDetector
        model = MemorySAMDetector(
            seg_model=seg_model, modals=modals, n_classes=n_classes,
            fpn_in_channels=mc.get('FPN_CHANNELS', [32, 64, 256]),
            fpn_strides=mc.get('FPN_STRIDES', [4, 8, 16]),
            freeze_backbone=True, train_memory=False,
            n_convs=mc.get('N_CONVS', 4), hidden_dim=mc.get('HIDDEN_DIM', 256),
            modality_fuse=mc.get('MODALITY_FUSE', 'mean'))
    return model.to(device).eval()


def load_det_checkpoint(model, ckpt_path: str, device) -> dict:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ck:
        missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
        info = {'missing': len(missing), 'unexpected': len(unexpected)}
    else:
        model.load_detector_state_dict(ck['detector_state_dict'])
        info = {'missing': 0, 'unexpected': 0}
    info['metrics'] = ck.get('metrics', {})
    info['epoch'] = ck.get('epoch', -1)
    return info


def build_loader(cfg: dict, mode: str = 'val', workers: int = 4):
    from train_det import build_dataset
    ds = build_dataset(cfg, mode)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                    collate_fn=MultiModalDetDataset.collate_fn)
    return ds, ld


@torch.no_grad()
def run_inference(model, dataset, loader, cfg: dict, device,
                  score_thresh: float = 0.05, limit: Optional[int] = None,
                  stride: int = 1) -> Tuple[List[dict], Dict[int, str]]:
    """Run the detector over the split once -> (COCO predictions, image_id->file_name)."""
    idx_to_cat = {v: k for k, v in dataset.cat_id_to_idx.items()}
    resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')
    preds: List[dict] = []
    id2file: Dict[int, str] = {}
    kept = 0
    for n, batch in enumerate(loader):
        if stride > 1 and n % stride:
            continue                      # every Nth image -> spans every clip
        if limit is not None and kept >= limit:
            break
        kept += 1
        modals = [k for k in batch
                  if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device) for m in modals}
        out = model(sample)
        img_hw = sample[modals[0]].shape[-2:]
        for i, det in enumerate(out['detections']):
            image_id = int(batch['image_id'][i])
            id2file[image_id] = batch['file_name'][i]
            if det['boxes'].shape[0] == 0:
                continue
            keep = det['scores'] > score_thresh
            boxes, scores, cls = det['boxes'][keep], det['scores'][keep], det['class_ids'][keep]
            if boxes.shape[0] == 0:
                continue
            oh, ow = batch['orig_size'][i].tolist()
            boxes = rescale_boxes_to_orig(boxes.cpu(), oh, ow, img_hw[0], img_hw[1], resize_mode)
            preds.extend(format_predictions_coco(
                boxes.cpu(), scores.cpu(), cls.cpu(), image_id, idx_to_cat))
    return preds, id2file


def _cocoeval(ann_path: str, preds: List[dict], img_ids: Optional[Sequence[int]] = None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_path)
        if not preds:
            return None, gt
        dt = gt.loadRes([dict(p) for p in preds])
        ev = COCOeval(gt, dt, 'bbox')
        if img_ids is not None:
            ev.params.imgIds = list(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev, gt


def eval_overall(ann_path: str, preds: List[dict],
                 img_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """mAP/mAP50/mAP75 triplet (+ size breakdown) — the repo's reporting convention."""
    ev, _ = _cocoeval(ann_path, preds, img_ids)
    if ev is None:
        return {k: 0.0 for k in ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large')}
    s = ev.stats
    return {'AP': float(s[0]), 'AP50': float(s[1]), 'AP75': float(s[2]),
            'AP_small': float(s[3]), 'AP_medium': float(s[4]), 'AP_large': float(s[5]),
            'n_images': len(ev.params.imgIds), 'n_preds': len(preds)}


def eval_per_class(ann_path: str, preds: List[dict],
                   img_ids: Optional[Sequence[int]] = None) -> List[dict]:
    """Per-category AP / AP50 (+ GT count) — COCOeval precision tensor, not exposed
    by objdet.metrics.evaluate_coco."""
    ev, gt = _cocoeval(ann_path, preds, img_ids)
    cats = gt.loadCats(gt.getCatIds())
    if ev is None:
        return [{'id': c['id'], 'name': c['name'], 'AP': 0.0, 'AP50': 0.0, 'n_gt': 0}
                for c in cats]
    # precision: [T(iou) x R(recall) x K(cat) x A(area) x M(maxDet)]
    prec = ev.eval['precision']
    iou50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    rows = []
    for k, c in enumerate(cats):
        ann_ids = gt.getAnnIds(catIds=[c['id']],
                               imgIds=list(img_ids) if img_ids is not None else None)
        p_all = prec[:, :, k, 0, -1]
        p_50 = prec[iou50, :, k, 0, -1]
        rows.append({
            'id': c['id'], 'name': c['name'],
            'AP': float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float('nan'),
            'AP50': float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else float('nan'),
            'n_gt': len(ann_ids),
        })
    return rows


def split_by_clip(id2file: Dict[int, str], clips: Sequence[str]
                  ) -> Tuple[List[int], List[int]]:
    """(ids whose file path matches any clip, the rest) — the night/normal split."""
    hit = [i for i, f in id2file.items() if any(c in f for c in clips)]
    rest = [i for i in id2file if i not in set(hit)]
    return sorted(hit), sorted(rest)


def detection_signature(preds: List[dict], topk: int = 10) -> Dict[int, tuple]:
    """Per-image fingerprint of the top-k detections — used to measure how much a
    module toggle actually changes the output (agreement), not just the score."""
    by_img: Dict[int, list] = {}
    for p in preds:
        by_img.setdefault(int(p['image_id']), []).append(p)
    sig = {}
    for img, ps in by_img.items():
        ps = sorted(ps, key=lambda x: -x['score'])[:topk]
        sig[img] = tuple((int(p['category_id']),) + tuple(round(float(v), 1) for v in p['bbox'])
                         for p in ps)
    return sig


def agreement(sig_a: Dict[int, tuple], sig_b: Dict[int, tuple]) -> float:
    keys = set(sig_a) | set(sig_b)
    if not keys:
        return 1.0
    same = sum(1 for k in keys if sig_a.get(k) == sig_b.get(k))
    return same / len(keys)


def write_outputs(out_prefix: str, payload: dict, markdown: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or '.', exist_ok=True)
    with open(f'{out_prefix}.json', 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    with open(f'{out_prefix}.md', 'w') as f:
        f.write(markdown)
    print(f"[det-analysis] wrote {out_prefix}.json and {out_prefix}.md")
