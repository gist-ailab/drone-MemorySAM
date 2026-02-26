"""
failure_multiaqua.py — Failure Case Extractor for MULTIAQUA
============================================================

val mode:
  - Reads detailed_log.json from --save_dir
  - Any image with per_class_iou < threshold is a failure
  - Copies the existing seg_viz/{stem}.png → seg_viz_failure/{stem}.png
  - Writes failure_summary.csv

test mode:
  - Reads a CSV file (e.g. frames_val.csv) with pre-computed per-class IoU
  - Filters failure images (any class IoU < threshold)
  - Loads model from --cfg and --model_path
  - Re-inferences only the failure images
  - Saves detailed visualizations (same Row 1-5 layout as val_multiaqua_detailed.py)
    into <out_dir>/seg_viz_failure/

Usage:
  # val mode (copy failures from existing seg_viz)
  python failure_multiaqua.py --mode val \\
      --save_dir outputs/.../val_pred_P13 \\
      --threshold 50

  # test mode (re-inference failures from CSV)
  python failure_multiaqua.py --mode test \\
      --cfg configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml \\
      --model_path outputs/.../best_checkpoint.pth \\
      --csv outputs/.../P13_16044_results/frames_val.csv \\
      --threshold 50

NOTE: Use the MMSS_SAM conda environment.
"""

import argparse
import csv as csv_mod
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from semseg.augmentations_mm import get_val_augmentation
from semseg.datasets import MULTIAQUA
from semseg.utils.utils import setup_cudnn

# Re-use all visualization utilities from val_multiaqua_detailed.py
from val_multiaqua_detailed import (
    MODAL_TITLES,
    REPRESENTATIVE_LAYERS,
    MoERoutingCapture,
    _add_title_to_image,
    _collate_multiaqua,
    _draw_legend,
    _load_modality_image,
    _unpad_resize_to_orig,
    build_aux_mask_row,
    build_routing_map_row,
    build_stats_row,
    load_model,
)


# ============================================================================
# Constants
# ============================================================================

# MULTIAQUA class names (same order as CLASSES)
_CLASSES = ['static_obstacle', 'dynamic_obstacle', 'water', 'sky']

# Mapping: class name → CSV column name (for frames_val.csv format)
_CSV_IOU_COLS = {
    'static_obstacle': 'IoU_static_obstacle',
    'dynamic_obstacle': 'IoU_dynamic_obstacle',
    'water': 'IoU_water',
    'sky': 'IoU_sky',
}


# ============================================================================
# Val Mode: copy from existing seg_viz
# ============================================================================

def filter_val_failures(save_dir: str, threshold: float, verbose: bool = True):
    """Val mode: read detailed_log.json and copy failure seg_viz images.

    Args:
        save_dir: Directory produced by val_multiaqua_detailed.py --mode val.
                  Must contain detailed_log.json and seg_viz/.
        threshold: Class IoU threshold in percent (0-100). Images with ANY
                   class IoU below this value are considered failures.
        verbose: Print per-image failure info.

    Returns:
        List of failure dicts {stem, failure_classes [(name, iou_pct), ...]}
    """
    save_dir = Path(save_dir)
    log_path = save_dir / "detailed_log.json"
    seg_viz_dir = save_dir / "seg_viz"
    failure_dir = save_dir / "seg_viz_failure"

    if not log_path.exists():
        raise FileNotFoundError(f"detailed_log.json not found: {log_path}")
    if not seg_viz_dir.exists():
        raise FileNotFoundError(f"seg_viz/ directory not found: {seg_viz_dir}")

    failure_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images_data = data.get('images', {})

    failures = []
    missing_viz = []

    for stem, img_log in images_data.items():
        per_class_iou = img_log.get('per_class_iou', {})
        if not per_class_iou:
            continue

        # per_class_iou values are stored as 0-1 decimals in detailed_log.json
        failure_classes = []
        for cls_name, iou_val in per_class_iou.items():
            if iou_val is None:
                continue
            iou_pct = iou_val * 100.0
            if iou_pct < threshold:
                failure_classes.append((cls_name, round(iou_pct, 2)))

        if failure_classes:
            src = seg_viz_dir / f"{stem}.png"
            if src.exists():
                dst = failure_dir / f"{stem}.png"
                shutil.copy2(str(src), str(dst))
                failures.append({'stem': stem, 'failure_classes': failure_classes})
                if verbose:
                    fc_str = ', '.join(f"{cls}:{iou:.1f}%" for cls, iou in failure_classes)
                    print(f"  [FAIL] {stem}: {fc_str}")
            else:
                missing_viz.append(stem)

    print(f"\n{'='*60}")
    print(f"Val Failure Summary: {len(failures)}/{len(images_data)} images "
          f"have class IoU < {threshold}%")
    if missing_viz:
        print(f"  (Skipped {len(missing_viz)} failures with no seg_viz image)")
    print(f"Saved to: {failure_dir}")

    # Write summary CSV
    summary_path = save_dir / "failure_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_mod.writer(f)
        writer.writerow(['image'] + _CLASSES)
        for item in failures:
            fail_map = {cls: iou for cls, iou in item['failure_classes']}
            row = [item['stem']] + [
                f"{fail_map[c]:.1f}" if c in fail_map else 'ok'
                for c in _CLASSES
            ]
            writer.writerow(row)
    print(f"Failure summary CSV: {summary_path}")

    return failures


# ============================================================================
# Test Mode: re-inference on CSV failures
# ============================================================================

def filter_csv_failures(csv_path: str, threshold: float):
    """Read CSV and return list of failure images.

    Expected columns: image, IoU_static_obstacle, IoU_dynamic_obstacle,
                      IoU_water, IoU_sky, mIoU  (as percentages 0-100).

    Args:
        csv_path: Path to CSV file (e.g. frames_val.csv).
        threshold: Failure threshold in percent.

    Returns:
        List of {stem, failure_classes [(col_name, iou_val), ...]}
    """
    iou_cols = list(_CSV_IOU_COLS.values())
    failures = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv_mod.DictReader(f)
        available_cols = reader.fieldnames or []
        iou_cols_present = [c for c in iou_cols if c in available_cols]
        if not iou_cols_present:
            # Fall back: any column starting with 'IoU_'
            iou_cols_present = [c for c in available_cols if c.startswith('IoU_')]
        if not iou_cols_present:
            raise ValueError(f"No IoU columns found in CSV: {available_cols}")

        for row in reader:
            stem = row['image']
            failure_classes = []
            for col in iou_cols_present:
                try:
                    val = float(row[col])
                except (ValueError, KeyError):
                    continue
                if val < threshold:
                    cls_name = col.replace('IoU_', '')
                    failure_classes.append((cls_name, round(val, 2)))
            if failure_classes:
                failures.append({'stem': stem, 'failure_classes': failure_classes})

    print(f"CSV failures: {len(failures)} images with class IoU < {threshold}%")
    if failures and len(failures) <= 10:
        for f in failures:
            fc_str = ', '.join(f"{c}:{v:.1f}%" for c, v in f['failure_classes'])
            print(f"  {f['stem']}: {fc_str}")
    return failures


class FilteredMultiaqua(MULTIAQUA):
    """MULTIAQUA dataset subset limited to specific stems."""

    def __init__(self, root, split, transform, modals, filter_stems):
        super().__init__(
            root, split=split, transform=transform, modals=modals,
            require_annotation=False, return_meta=True,
        )
        stem_set = set(filter_stems)
        original_count = len(self.stems)
        self.stems = [s for s in self.stems if s in stem_set]
        not_found = stem_set - set(self.stems)
        if not_found:
            print(f"Warning: {len(not_found)}/{len(filter_stems)} stems not found "
                  f"in {split}.txt: {sorted(not_found)[:5]}")
        if not self.stems:
            raise ValueError(
                f"No stems matched. filter_stems ({len(filter_stems)}) vs "
                f"{split}.txt ({original_count} entries)."
            )
        print(f"FilteredMultiaqua: {len(self.stems)} / {len(filter_stems)} failure stems loaded.")


@torch.no_grad()
def run_failure_inference(model, dataloader, device, out_dir, modals=None):
    """Re-inference failure images with the same Row 1-5 visualization layout.

    Args:
        model: Loaded model (eval mode).
        dataloader: DataLoader for FilteredMultiaqua (require_annotation=False).
        device: torch device.
        out_dir: Root output directory. Saves to out_dir/seg_viz_failure/.
        modals: List of modality names, e.g. ['img', 'lidar', 'thermal'].
    """
    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE

    out_dir = Path(out_dir)
    failure_dir = out_dir / "seg_viz_failure"
    failure_dir.mkdir(parents=True, exist_ok=True)

    modals = modals or ['img', 'lidar', 'thermal']
    core = model.module if hasattr(model, 'module') else model

    has_moe = (
        hasattr(core, 'moe_layers_q') and
        hasattr(core, 'moe_layers_v') and
        len(core.moe_layers_q) > 0
    )

    for images, _, metas in tqdm(dataloader, desc="Failure inference"):
        images = [x.to(device) for x in images]

        capture = None
        if has_moe:
            capture = MoERoutingCapture(core, viz_block_indices=REPRESENTATIVE_LAYERS)
            capture.register_hooks()
            capture.register_counter_hook()

        output, _ = model(images, multimask_output=True)

        if capture is not None:
            capture.remove_hooks()

        preds = output.softmax(dim=1)
        pred_labels = preds[:, :n_classes].argmax(dim=1)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            stem = meta["stem"]
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            pred_b = pred_labels[b]

            pred_resized = _unpad_resize_to_orig(
                pred_b, orig_h, orig_w, model_size=pred_b.shape[0]
            )
            pred_np = pred_resized.cpu().numpy().astype(np.uint8)
            colored = MULTIAQUA.decode_segmap(pred_np, palette)
            ds = dataloader.dataset

            # Row 1: [RGB | Thermal | LiDAR] (with titles)
            raw_modals = [_load_modality_image(ds, mk, stem, orig_h, orig_w) for mk in modals]
            rgb = raw_modals[0]
            if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                rgb = np.array(
                    Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                )
            overlay = (
                rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5
            ).clip(0, 255).astype(np.uint8)

            modality_cols = [
                _add_title_to_image(img, MODAL_TITLES.get(mk, mk))
                for img, mk in zip(raw_modals, modals)
            ]
            row1 = np.concatenate(modality_cols, axis=1)
            main_w = row1.shape[1]

            # Row 2: [Legend | Prediction | Overlay]
            classes = getattr(ds, 'CLASSES', MULTIAQUA.CLASSES)
            pal = getattr(ds, 'PALETTE', MULTIAQUA.PALETTE)
            legend_img = _draw_legend(classes, pal, orig_h, orig_w)
            row2 = np.concatenate([
                _add_title_to_image(legend_img, 'Legend'),
                _add_title_to_image(colored, 'Prediction'),
                _add_title_to_image(overlay, 'Overlay'),
            ], axis=1)

            rows = [row1, row2]

            # Row 3: MoE per-token stats (if model has MoE)
            if capture is not None:
                stats_row = build_stats_row(capture, modals, int(orig_h * 0.55), main_w)
                rows.append(stats_row)

                # Row 4: Spatial routing maps
                mid_block = REPRESENTATIVE_LAYERS[len(REPRESENTATIVE_LAYERS) // 2]
                row4 = build_routing_map_row(
                    capture, modals, int(orig_h * 0.6), main_w, block_idx=mid_block
                )
                rows.append(row4)

            # Row 5 (P13 only): ConfidenceAuxHead per-modality aux masks
            aux_logits = getattr(core, '_last_aux_logits', None)
            if aux_logits is not None:
                aux_row = build_aux_mask_row(
                    aux_logits, modals, b, palette,
                    orig_h, orig_w, main_w, ignore_mask=None,
                )
                if aux_row is not None:
                    rows.append(aux_row)

            viz = np.concatenate(rows, axis=0)
            Image.fromarray(viz).save(str(failure_dir / f"{stem}.png"))

    print(f"\nSaved {len(dataloader.dataset)} failure visualizations → {failure_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MULTIAQUA Failure Case Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mode', type=str, choices=['val', 'test'], default='val',
        help='val: copy from existing seg_viz  |  test: re-inference from CSV',
    )
    parser.add_argument(
        '--threshold', type=float, default=50.0,
        help='Class IoU threshold in percent (0-100). Default: 50',
    )

    # ── Val mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='[val] Directory with detailed_log.json + seg_viz/',
    )

    # ── Test mode ─────────────────────────────────────────────────────────
    parser.add_argument(
        '--cfg', type=str, default=None,
        help='[test] Config YAML file',
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='[test] Model checkpoint path',
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='[test] CSV with per-class IoU (columns: image, IoU_<class>, ...)',
    )
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='[test] Output directory for seg_viz_failure/ '
             '(default: parent of --model_path)',
    )

    args = parser.parse_args()

    # ── Val mode ──────────────────────────────────────────────────────────
    if args.mode == 'val':
        if args.save_dir is None:
            parser.error("--save_dir is required for val mode")
        filter_val_failures(args.save_dir, args.threshold)
        return

    # ── Test mode ─────────────────────────────────────────────────────────
    for flag, name in [
        (args.cfg, '--cfg'),
        (args.model_path, '--model_path'),
        (args.csv, '--csv'),
    ]:
        if flag is None:
            parser.error(f"{name} is required for test mode")

    # 1. Identify failure images from CSV
    failures = filter_csv_failures(args.csv, args.threshold)
    if not failures:
        print(f"No failures found (threshold={args.threshold}%). Nothing to do.")
        return

    failure_stems = [f['stem'] for f in failures]
    print(f"\nRe-inferencing {len(failure_stems)} failure images...")

    # 2. Load config and model
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(cfg['DEVICE'])
    setup_cudnn()

    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    image_size = eval_cfg['IMAGE_SIZE']
    modals = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])

    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    # CSV 파일명에서 split 자동 추론 (frames_test.csv → 'test', frames_val.csv → 'val')
    csv_name = Path(args.csv).stem.lower()
    if 'test' in csv_name:
        split = 'test'
    elif 'val' in csv_name:
        split = 'val'
    else:
        split = 'val'
        print(f"Warning: cannot infer split from '{csv_name}', defaulting to 'val'")
    print(f"Dataset split: '{split}' (inferred from CSV filename)")

    # 3. Build filtered dataset
    dataset = FilteredMultiaqua(
        root=dataset_cfg['ROOT'],
        split=split,
        transform=transform,
        modals=modals,
        filter_stems=failure_stems,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg.get('BATCH_SIZE', 1),
        num_workers=4,
        pin_memory=False,
        collate_fn=_collate_multiaqua,
    )

    # 4. Load model and run inference
    model = load_model(cfg, model_path, device)

    out_dir = Path(args.out_dir) if args.out_dir else model_path.parent
    run_failure_inference(model, dataloader, device, out_dir=out_dir, modals=modals)


if __name__ == '__main__':
    main()
