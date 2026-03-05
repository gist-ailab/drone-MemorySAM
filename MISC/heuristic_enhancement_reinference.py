#!/usr/bin/env python3
"""Heuristic Enhancement Re-inference Script.

성능이 낮은 test 이미지에 brightness/contrast/gamma 보정을 적용하여
재인퍼런스 후 MACVi 제출 폴더의 해당 이미지만 덮어쓰는 스크립트.

사용법:
    # 기본: mIoU < 55인 이미지에 brightness=0.11, contrast=0.9 적용
    python MISC/heuristic_enhancement_reinference.py \
        --config configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \
        --checkpoint outputs/MMSamP9/.../epoch47_94.18_checkpoint.pth \
        --frames-csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
        --macvi-dir outputs/MMSamP9/.../epoch47_94.18_eval_macvi_CV2 \
        --miou-threshold 55 \
        --brightness 0.11 \
        --contrast 0.9

    # gamma도 같이 적용
    python MISC/heuristic_enhancement_reinference.py \
        --config configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \
        --checkpoint outputs/MMSamP9/.../epoch47_94.18_checkpoint.pth \
        --frames-csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
        --macvi-dir outputs/MMSamP9/.../epoch47_94.18_eval_macvi_CV3 \
        --miou-threshold 60 \
        --brightness 0.11 --contrast 0.9 --gamma 1.5

    # dry-run: 대상 이미지만 출력 (인퍼런스 안 함)
    python MISC/heuristic_enhancement_reinference.py \
        --frames-csv outputs/MMSamP9/.../P9_15635_results/frames_test.csv \
        --miou-threshold 55 --dry-run

Enhancement 적용 순서 (interactive_gamma_viewer.py와 동일):
    1. Gamma correction: img^(1/gamma)  — gamma>1 → 밝아짐, gamma<1 → 어두워짐
    2. Brightness (additive): img + brightness  — 양수 → 밝아짐
    3. Contrast (around mean): (img - mean) * contrast + mean  — <1 → 대비 감소

모든 연산은 [0,1] float 범위에서 수행, clamp 적용.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io
import yaml
from PIL import Image
from torchvision import transforms as T

# ─── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from semseg.augmentations_mm import get_val_augmentation
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.utils.utils import setup_cudnn

# Model import map — 새 모델 추가 시 여기에 등록
_MODEL_IMPORTS = {
    'LoRA_Sam_P9': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P10': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P11': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P15': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P17': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P18': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
    'LoRA_Sam_P19': 'semseg.models.sam2.sam2.sam_lora_image_encoder_seg',
}

N_CLASSES = 4


# ─── Model loading ──────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path, device):
    """Load LoRA model from config and checkpoint."""
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']

    sam2 = build_sam2(
        "sam2_hiera_b+.yaml",
        "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false",
        ]
    )

    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P9')
    lora_r = model_cfg.get('LORA_R', 4)
    num_experts = model_cfg.get('LORA_NUM_EXPERTS', None)
    modals = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])
    if num_experts is None:
        num_experts = len(modals)

    # Dynamic import
    module_path = _MODEL_IMPORTS.get(lora_model_name)
    if module_path is None:
        raise ValueError(f"Unknown model: {lora_model_name}. "
                         f"Add it to _MODEL_IMPORTS in this script.")
    import importlib
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, lora_model_name)

    model = ModelClass(sam2, r=lora_r, num_experts=num_experts)
    model.load_lora_parameters(checkpoint_path)
    model.to(device).eval()
    return model


# ─── Data loading ────────────────────────────────────────────────────────────

def _open_img(path, H, W):
    """Load image, fallback to zeros if missing (same as MULTIAQUA._open_img)."""
    if not os.path.exists(path):
        return torch.zeros(3, H, W, dtype=torch.uint8)
    img = io.read_image(path)
    C = img.shape[0]
    if C == 4:
        img = img[:3]
    if C == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[1:] != (H, W):
        img = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)(img)
    return img


def load_sample(stem, data_root, rgb_subroot, lidar_subroot, thermal_subroot,
                modals):
    """Load a single sample (same format as MULTIAQUA.__getitem__)."""
    sample = {}
    rgb_path = os.path.join(data_root, 'data', rgb_subroot, f"{stem}.png")
    sample['img'] = io.read_image(rgb_path)[:3, ...]
    H, W = sample['img'].shape[1:]

    if 'lidar' in modals:
        lidar_path = os.path.join(data_root, 'data', lidar_subroot,
                                  f"{stem}_lidar.png")
        sample['lidar'] = _open_img(lidar_path, H, W)

    if 'thermal' in modals:
        thermal_path = os.path.join(data_root, 'data', thermal_subroot,
                                    f"{stem}_thermal.png")
        sample['thermal'] = _open_img(thermal_path, H, W)

    return sample, H, W


# ─── Enhancement ─────────────────────────────────────────────────────────────

def apply_enhancement(sample, gamma=1.0, brightness=0.0, contrast=1.0):
    """Apply gamma → brightness → contrast to RGB only.

    Args:
        sample: dict with 'img' key (uint8 tensor)
        gamma: >1 brightens, <1 darkens (default 1.0 = no change)
        brightness: additive shift on [0,1] scale (default 0.0)
        contrast: multiplier around mean (default 1.0)

    Returns:
        New sample dict with enhanced 'img'.
    """
    out = dict(sample)
    img = out['img'].float() / 255.0

    # 1) Gamma
    if gamma != 1.0:
        img = torch.clamp(img, 1e-6, 1.0) ** (1.0 / gamma)

    # 2) Brightness (additive)
    if brightness != 0.0:
        img = img + brightness
        img = torch.clamp(img, 0.0, 1.0)

    # 3) Contrast (around mean)
    if contrast != 1.0:
        mean = img.mean()
        img = (img - mean) * contrast + mean
        img = torch.clamp(img, 0.0, 1.0)

    out['img'] = (img * 255).clamp(0, 255).to(torch.uint8)
    return out


# ─── Inference ───────────────────────────────────────────────────────────────

def _unpad_resize_to_orig(pred, orig_h, orig_w, model_size=1024):
    """Undo ResizeWidthPadToSquare: crop padding then resize to original."""
    scale = model_size / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    pred_cropped = pred[:new_h, :new_w]
    pred_resized = F.interpolate(
        pred_cropped.float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode='nearest'
    ).squeeze().long()
    return pred_resized


@torch.no_grad()
def run_inference(model, sample, transform, modals, device, orig_h, orig_w):
    """Run model inference on a single sample."""
    s = dict(sample)
    s['mask'] = torch.zeros(1, orig_h, orig_w, dtype=torch.long)
    s = transform(s)
    del s['mask']
    images = [s[k].unsqueeze(0).to(device) for k in modals]
    output, _ = model(images, multimask_output=True)
    pred = output.softmax(dim=1)[:, :N_CLASSES].argmax(dim=1)[0]
    return _unpad_resize_to_orig(pred.cpu(), orig_h, orig_w,
                                 model_size=pred.shape[0])


# ─── Main ────────────────────────────────────────────────────────────────────

def find_worst_images(frames_csv, miou_threshold):
    """Read frames_test.csv and return images below threshold."""
    worst = []
    with open(frames_csv) as f:
        for r in csv.DictReader(f):
            miou = float(r['mIoU'])
            if miou < miou_threshold:
                worst.append((r['image'], miou))
    worst.sort(key=lambda x: x[1])
    return worst


def main():
    parser = argparse.ArgumentParser(
        description="Heuristic enhancement re-inference for MACVi submission")
    parser.add_argument('--config', type=str,
                        help='Eval config YAML path')
    parser.add_argument('--checkpoint', type=str,
                        help='Model checkpoint path (_checkpoint.pth)')
    parser.add_argument('--frames-csv', type=str, required=True,
                        help='frames_test.csv with per-image mIoU')
    parser.add_argument('--macvi-dir', type=str,
                        help='MACVi submission directory to overwrite')
    parser.add_argument('--miou-threshold', type=float, default=55.0,
                        help='Images with mIoU below this are enhanced')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction (>1=brighter, default=1.0)')
    parser.add_argument('--brightness', type=float, default=0.0,
                        help='Additive brightness shift (default=0.0)')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='Contrast multiplier around mean (default=1.0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only list target images, skip inference')
    args = parser.parse_args()

    # --- Find worst images ---
    worst = find_worst_images(args.frames_csv, args.miou_threshold)
    print(f"\n{'='*60}")
    print(f"Heuristic Enhancement Re-inference")
    print(f"{'='*60}")
    print(f"Threshold : mIoU < {args.miou_threshold}")
    print(f"Target    : {len(worst)} images")
    print(f"Gamma     : {args.gamma}")
    print(f"Brightness: {args.brightness} ({'brighter' if args.brightness > 0 else 'darker' if args.brightness < 0 else 'no change'})")
    print(f"Contrast  : {args.contrast} ({'increase' if args.contrast > 1 else 'decrease' if args.contrast < 1 else 'no change'})")
    print(f"{'='*60}")
    for img, m in worst:
        print(f"  {img}: mIoU={m:.2f}")

    if args.dry_run:
        print("\n[DRY-RUN] No inference performed.")
        return

    if not args.config or not args.checkpoint or not args.macvi_dir:
        parser.error("--config, --checkpoint, --macvi-dir are required "
                     "when not using --dry-run")

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('DEVICE', 'cuda'))
    setup_cudnn()
    dataset_cfg = cfg['DATASET']
    modals = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])
    eval_cfg = cfg.get('EVAL', {})
    image_size = eval_cfg.get('IMAGE_SIZE', [1024, 1024])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    # Dataset paths
    dataset_root = dataset_cfg['ROOT']
    data_root = os.path.join(dataset_root, 'MULTIAQUA_night')
    rgb_sub = dataset_cfg.get('RGB_SUBROOT', None) or 'zed'
    lidar_sub = dataset_cfg.get('LIDAR_SUBROOT', None) or 'lidar_processed'
    thermal_sub = dataset_cfg.get('THERMAL_SUBROOT', None) or 'thermal_processed'

    # --- Load model ---
    print("\nLoading model...")
    model = load_model(cfg, args.checkpoint, device)
    print("Model loaded.")

    # --- Process ---
    os.makedirs(args.macvi_dir, exist_ok=True)
    t_start = time.time()

    for i, (stem, miou) in enumerate(worst):
        print(f"\n[{i+1}/{len(worst)}] {stem} (mIoU={miou:.2f})")

        sample, H, W = load_sample(stem, data_root, rgb_sub, lidar_sub,
                                   thermal_sub, modals)
        enhanced = apply_enhancement(sample, args.gamma, args.brightness,
                                     args.contrast)
        pred = run_inference(model, enhanced, transform, modals, device, H, W)
        pred_np = pred.numpy().astype(np.uint8)

        # MACVi format: 1-indexed labels
        macvi_pred = (pred_np + 1).clip(1, 4).astype(np.uint8)
        out_path = os.path.join(args.macvi_dir, f"{stem}.png")
        Image.fromarray(macvi_pred).save(out_path)

        unique, counts = np.unique(pred_np, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  Saved → classes: {dist}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {len(worst)} images overwritten in {args.macvi_dir}")
    print(f"Total images in dir: {len(os.listdir(args.macvi_dir))}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
