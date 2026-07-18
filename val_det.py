"""
Object Detection Evaluation Script for MemorySAM.

Usage:
    python val_det.py --cfg configs/det/det_P9_base.yaml \
        --det_checkpoint outputs/det/det_P9_base/best_checkpoint.pth \
        --mode val

    # Test with visualization
    python val_det.py --cfg configs/det/det_P9_base.yaml \
        --det_checkpoint outputs/det/det_P9_base/best_checkpoint.pth \
        --mode test --save_vis
"""

import os
import torch
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader

# SAM2 seg backbone is built lazily via train_det.build_seg_model; no eager SAM2
# import here so P34 (ReliaDINO/DINOv3) eval runs in envs without SAM2's dep tree.
from objdet.datasets.multimodal_det import MultiModalDetDataset, rescale_boxes_to_orig
from objdet.models.det_model import MemorySAMDetector
from objdet.metrics import evaluate_coco, format_predictions_coco
from objdet.utils.nms import batched_nms


# Distinct colors for bbox visualization
_COLORS = [
    (220, 20, 60), (0, 128, 0), (0, 0, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 0, 255), (128, 128, 0),
    (0, 128, 128), (255, 99, 71), (30, 144, 255), (50, 205, 50),
]


def parse_args():
    parser = argparse.ArgumentParser(description='MemorySAM Detection Evaluation')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--det_checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--save_vis', action='store_true', help='Save bbox visualizations')
    parser.add_argument('--save_dir', type=str, default=None)
    return parser.parse_args()


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    class_names: list,
    score_thresh: float = 0.3,
) -> Image.Image:
    """Draw bounding boxes on image."""
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    for i in range(boxes.shape[0]):
        if scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        cls_id = int(class_ids[i].item())
        score = scores[i].item()
        color = _COLORS[cls_id % len(_COLORS)]
        label = f"{class_names[cls_id]}: {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Label background
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle([text_bbox[0]-1, text_bbox[1]-1, text_bbox[2]+1, text_bbox[3]+1], fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255))

    return pil_img


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset (build first to derive n_classes from COCO categories)
    from train_det import build_dataset
    dataset = build_dataset(cfg, args.mode)
    n_classes = cfg['MODEL'].get('N_CLASSES', dataset.n_classes) or dataset.n_classes

    # Build model
    from train_det import build_seg_model
    seg_model = build_seg_model(cfg, device, n_classes)

    det_name = cfg['MODEL'].get('DET_MODEL', 'MemorySAMDetector')
    if det_name == 'ReliaDINORFDETRDetector':
        from objdet.models.det_model import ReliaDINORFDETRDetector
        model = ReliaDINORFDETRDetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            fpn_dim=cfg['MODEL'].get('FPN_DIM', 256),
            fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16, 32]),
            det_levels=cfg['MODEL'].get('DET_LEVELS', [2]),
            freeze_backbone=True,
            num_queries=cfg['MODEL'].get('NUM_QUERIES', 300),
            group_detr=cfg['MODEL'].get('GROUP_DETR', 13),
            dec_layers=cfg['MODEL'].get('DEC_LAYERS', 4),
            dec_n_points=cfg['MODEL'].get('DEC_N_POINTS', 2),
            coco_ckpt=None,   # weights come from the trained det checkpoint below
            num_select=cfg['MODEL'].get('NUM_SELECT', 300),
        ).to(device)
    elif det_name == 'ReliaDINODetector':
        from objdet.models.det_model import ReliaDINODetector
        model = ReliaDINODetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            fpn_dim=cfg['MODEL'].get('FPN_DIM', 256),
            fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16, 32]),
            freeze_backbone=True,
            n_convs=cfg['MODEL'].get('N_CONVS', 4),
            hidden_dim=cfg['MODEL'].get('HIDDEN_DIM', 256),
        ).to(device)
    else:
        model = MemorySAMDetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            fpn_in_channels=cfg['MODEL'].get('FPN_CHANNELS', [32, 64, 256]),
            fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16]),
            freeze_backbone=True,
            train_memory=False,
            n_convs=cfg['MODEL'].get('N_CONVS', 4),
            hidden_dim=cfg['MODEL'].get('HIDDEN_DIM', 256),
            modality_fuse=cfg['MODEL'].get('MODALITY_FUSE', 'mean'),
        ).to(device)

    # Load checkpoint — full model state (incl. fine-tuned backbone) if present.
    ckpt = torch.load(args.det_checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_detector_state_dict(ckpt['detector_state_dict'])
    print(f"Loaded det checkpoint: {args.det_checkpoint}")
    if 'metrics' in ckpt:
        print(f"  Checkpoint metrics: {ckpt['metrics']}")

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg['TRAIN'].get('VAL_BATCH_SIZE', 1),
        shuffle=False,
        num_workers=4,
        collate_fn=MultiModalDetDataset.collate_fn,
    )

    # Save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(args.det_checkpoint).parent / f'eval_{args.mode}'
    save_dir.mkdir(parents=True, exist_ok=True)

    idx_to_cat_id = {v: k for k, v in dataset.cat_id_to_idx.items()}
    all_predictions = []

    for batch in tqdm(loader, desc=f'Evaluating ({args.mode})'):
        modals = [k for k in batch.keys()
                  if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device) for m in modals}

        results = model(sample)

        orig_sizes = batch['orig_size']
        img_size = sample[modals[0]].shape[-2:]

        for i, det in enumerate(results['detections']):
            image_id = batch['image_id'][i]
            file_name = batch['file_name'][i]

            # Filter by score
            if det['boxes'].shape[0] > 0:
                keep = det['scores'] > args.score_thresh
                boxes = det['boxes'][keep]
                scores = det['scores'][keep]
                cls_ids = det['class_ids'][keep]
            else:
                boxes = det['boxes']
                scores = det['scores']
                cls_ids = det['class_ids']

            # Scale to original size (matches dataset resize_mode)
            orig_h, orig_w = orig_sizes[i].tolist()
            resize_mode = cfg['DATASET'].get('RESIZE_MODE', 'stretch')
            boxes_orig = boxes.clone()
            if boxes_orig.shape[0] > 0:
                boxes_orig = rescale_boxes_to_orig(
                    boxes_orig, orig_h, orig_w, img_size[0], img_size[1], resize_mode)

            # COCO format predictions
            preds = format_predictions_coco(
                boxes_orig.cpu(), scores.cpu(), cls_ids.cpu(),
                image_id, idx_to_cat_id,
            )
            all_predictions.extend(preds)

            # Save visualization
            if args.save_vis and boxes.shape[0] > 0:
                # Load RGB image for visualization (file_name is the rgb path,
                # relative to DATASET.ROOT in modalities-map mode).
                if 'MODALITY_KEYS' in cfg['DATASET']:
                    rgb_root = cfg['DATASET']['ROOT']
                else:
                    rgb_root = cfg['DATASET']['MODALITIES']['img']['ROOT']
                rgb_path = os.path.join(rgb_root, file_name)
                if os.path.exists(rgb_path):
                    rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
                    vis_img = draw_detections(
                        rgb_img, boxes_orig.cpu(), scores.cpu(), cls_ids.cpu(),
                        dataset.class_names, args.score_thresh,
                    )
                    stem = Path(file_name).stem
                    vis_img.save(save_dir / f'{stem}_det.png')

    # Save predictions JSON
    pred_path = save_dir / 'predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(all_predictions, f)
    print(f"Saved {len(all_predictions)} predictions to {pred_path}")

    # COCO evaluation (val only — test has no GT)
    ann_key = f'ANNOTATION_{args.mode.upper()}'
    if ann_key in cfg['DATASET']:
        try:
            metrics = evaluate_coco(cfg['DATASET'][ann_key], all_predictions)
            print(f"\n{'='*50}")
            print(f"Detection Results ({args.mode})")
            print(f"{'='*50}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Save metrics
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"COCO eval failed: {e}")
    else:
        print(f"No annotation file for {args.mode} — skipping COCO eval.")


if __name__ == '__main__':
    main()
