"""
Sliding-window (tiled) inference for MULTIAQUA.
기존 val_multiaqua.py를 수정하지 않고, 두 가지 추론 전략을 지원:

  1) --tile_size 1024 --tile_stride 512
     원본 해상도(1242×2208)를 유지한 채 1024×1024 패치로 나눠 추론 → overlap soft-voting → 원본 크기 예측
     이점: 원본 디테일 보존, SAM2 학습 해상도(1024) 유지

  2) --image_size 512
     기존처럼 전체 이미지를 한 번에 축소 추론 (512×512)
     이점: 빠른 추론, VRAM 절약

사용법:
  # Tiled inference (원본 해상도 보존, 1024 패치)
  python val_multiaqua_tiled.py --cfg configs/eval_config/... --model_path ... --mode val \
      --tile_size 1024 --tile_stride 512

  # Small resolution inference
  python val_multiaqua_tiled.py --cfg configs/eval_config/... --model_path ... --mode val \
      --image_size 512

  # MaCVi 제출용 (tiled)
  python val_multiaqua_tiled.py --cfg configs/eval_config/... --model_path ... --mode test \
      --tile_size 1024 --tile_stride 512 --macvi
"""
import torch
import argparse
import yaml
import os
import time
import json
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import inspect

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *

from PIL import Image
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────
# Model loading (reuse from val_multiaqua.py)
# ─────────────────────────────────────────────────────────────
def load_model(cfg, model_path, device):
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    num_modalities = len(dataset_cfg['MODALS'])

    sam2_config_file = "sam2_hiera_b+.yaml"
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

    sam2 = build_sam2(
        sam2_config_file,
        checkpoint,
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false"
        ]
    )

    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P8')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K')
    lora_layer = model_cfg.get('LORA_LAYER')

    lora_model_class = eval(lora_model_name)
    model_kwargs = {
        'sam_model': sam2,
        'r': lora_r,
        'lora_layer': lora_layer,
    }
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k
    if 'use_entropy_fusion' in sig.parameters:
        model_kwargs['use_entropy_fusion'] = model_cfg.get('USE_ENTROPY_FUSION', False)

    model = lora_model_class(**model_kwargs)

    ckpt = torch.load(str(model_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Model load: {msg}")
    if hasattr(model, '_current_epoch'):
        model._current_epoch = 9999

    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Tile generation
# ─────────────────────────────────────────────────────────────
def generate_tiles(H, W, tile_size, stride):
    """
    Generate (y_start, x_start) coordinates for sliding window.
    Ensures full coverage including edges.
    Returns list of (y, x) tuples.
    """
    tiles = []
    y_positions = list(range(0, H - tile_size, stride)) + [max(0, H - tile_size)]
    x_positions = list(range(0, W - tile_size, stride)) + [max(0, W - tile_size)]
    # Deduplicate
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))
    for y in y_positions:
        for x in x_positions:
            tiles.append((y, x))
    return tiles


def pad_to_tile_size(img_tensor, tile_size):
    """
    Pad image to be at least tile_size in both dimensions.
    img_tensor: (C, H, W) or (3, H, W)
    Returns: (C, H_padded, W_padded), pad_h, pad_w
    """
    C, H, W = img_tensor.shape
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return img_tensor, pad_h, pad_w


# ─────────────────────────────────────────────────────────────
# Tiled inference core
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_tiled(model, raw_modalities, device, tile_size, stride, n_classes):
    """
    Sliding window inference on a single image at original resolution.

    Args:
        model: LoRA_Sam model
        raw_modalities: list of tensors [(C, H, W), ...] — original resolution, uint8
        device: torch.device
        tile_size: patch size (e.g. 1024)
        stride: sliding stride (e.g. 512)
        n_classes: number of classes

    Returns:
        pred_labels: (H, W) long tensor — argmax prediction at original resolution
    """
    C, H, W = raw_modalities[0].shape

    # Pad if image is smaller than tile_size
    padded_mods = []
    for mod in raw_modalities:
        mod_padded, _, _ = pad_to_tile_size(mod, tile_size)
        padded_mods.append(mod_padded)
    _, H_pad, W_pad = padded_mods[0].shape

    tiles = generate_tiles(H_pad, W_pad, tile_size, stride)

    # Accumulator: soft voting (logit sum)
    logit_sum = torch.zeros(n_classes, H_pad, W_pad, dtype=torch.float32, device=device)
    count_map = torch.zeros(1, H_pad, W_pad, dtype=torch.float32, device=device)

    for (y, x) in tiles:
        # Extract tile from each modality
        tile_mods = []
        for mod in padded_mods:
            tile = mod[:, y:y+tile_size, x:x+tile_size]  # (C, tile_size, tile_size)
            tile_mods.append(tile)

        # Normalize: convert uint8 → float32 [0, 1] → ImageNet normalize
        # SAM2 expects normalized input — same as val augmentation pipeline
        tile_inputs = []
        for t in tile_mods:
            t_float = t.float() / 255.0
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t_norm = (t_float - mean) / std
            tile_inputs.append(t_norm.unsqueeze(0).to(device))  # (1, C, H, W)

        output = model(tile_inputs, multimask_output=True)
        logits = output[0]  # (1, n_classes_or_25, tile_size, tile_size)
        logits = logits[:, :n_classes]  # (1, n_classes, tile_size, tile_size)

        logit_sum[:, y:y+tile_size, x:x+tile_size] += logits[0]
        count_map[:, y:y+tile_size, x:x+tile_size] += 1.0

    # Average
    logit_avg = logit_sum / count_map.clamp(min=1.0)

    # Remove padding, get original size
    logit_avg = logit_avg[:, :H, :W]

    pred_labels = logit_avg.argmax(dim=0)  # (H, W)
    return pred_labels


# ─────────────────────────────────────────────────────────────
# Dataset that returns RAW images (no resize, no normalize)
# ─────────────────────────────────────────────────────────────
class MULTIAQUA_Raw(MULTIAQUA):
    """
    MULTIAQUA dataset without any transform — returns original resolution uint8 tensors.
    """
    def __init__(self, *args, **kwargs):
        kwargs['transform'] = None
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        stem, rgb_dir = self.samples[index]
        rgb_path = rgb_dir / f"{stem}.png"

        from torchvision import io
        sample = {}
        sample["img"] = io.read_image(str(rgb_path))[:3, ...]
        H, W = sample["img"].shape[1:]

        if "lidar" in self.modals:
            lidar_path = self.lidar_dir / f"{stem}_lidar.png"
            sample["lidar"] = self._open_img(lidar_path, H, W)
        if "thermal" in self.modals:
            thermal_path = self.thermal_dir / f"{stem}_thermal.png"
            sample["thermal"] = self._open_img(thermal_path, H, W)

        modality_list = [sample[k] for k in self.modals]  # list of (C, H, W) uint8

        if self.require_annotation:
            lbl_path = self.ann_dir / f"{stem}.png"
            label = io.read_image(str(lbl_path))[0, ...]
            label = label.numpy().astype(np.int64)
            orig_label = np.where(
                (label >= 1) & (label <= 4),
                label - 1,
                255
            )
            orig_label = torch.from_numpy(orig_label).long()
        else:
            orig_label = torch.zeros(H, W, dtype=torch.long)

        meta = {"stem": stem, "orig_h": int(H), "orig_w": int(W), "orig_label": orig_label}
        return modality_list, orig_label, meta


def _collate_raw(batch):
    """Collate for MULTIAQUA_Raw — no batching, return list."""
    # batch size is always 1 for tiled inference
    modality_list, label, meta = batch[0]
    return modality_list, label, meta


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_tiled(model, dataset, device, tile_size, stride, n_classes,
                   save_dir=None, macvi_format=False, mode='val'):
    """Tiled inference evaluation loop."""
    metrics = Metrics(n_classes, dataset.ignore_label, device)
    palette = dataset.PALETTE

    if save_dir:
        save_dir = Path(save_dir)
        if macvi_format:
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            (save_dir / "seg").mkdir(parents=True, exist_ok=True)
            (save_dir / "seg_viz").mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    n_images = 0

    for i in tqdm(range(len(dataset)), desc=f"Tiled {mode} (tile={tile_size}, stride={stride})"):
        modality_list, orig_label, meta = dataset[i]
        stem = meta["stem"]
        orig_h, orig_w = meta["orig_h"], meta["orig_w"]

        t0 = time.perf_counter()
        pred = infer_tiled(model, modality_list, device, tile_size, stride, n_classes)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        n_images += 1

        # pred: (H, W) at original resolution
        pred_np = pred.cpu().numpy().astype(np.uint8)

        if mode == 'val':
            pred_oh = F.one_hot(pred.long().clamp(0, n_classes - 1), n_classes).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            metrics.update(pred_oh, orig_label.unsqueeze(0).to(device))

        if save_dir:
            if macvi_format:
                seg_save = (pred_np + 1).clip(1, 4).astype(np.uint8)
                Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
            else:
                Image.fromarray(pred_np).save(str(save_dir / "seg" / f"{stem}.png"))
                colored = MULTIAQUA.decode_segmap(pred_np, palette)
                ignore_mask = orig_label.cpu().numpy() == 255
                colored[ignore_mask] = [30, 30, 30]

                # Simple visualization: colored segmentation
                rgb = modality_list[0].permute(1, 2, 0).numpy()  # (H, W, 3) uint8
                overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
                overlay[ignore_mask] = [30, 30, 30]
                viz = np.concatenate([rgb, colored, overlay], axis=1)
                Image.fromarray(viz).save(str(save_dir / "seg_viz" / f"{stem}.png"))

    fps = n_images / total_time if total_time > 0 else 0.0

    if mode == 'val':
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()
        return acc, macc, f1, mf1, ious, miou, float(ious[1]), fps
    return None, None, None, None, None, None, None, fps


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tiled / multi-resolution inference for MULTIAQUA")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--macvi', action='store_true')

    # Tiled inference
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Tile size for sliding window (e.g. 1024). If set, use tiled inference at original resolution.')
    parser.add_argument('--tile_stride', type=int, default=None,
                        help='Stride for sliding window (e.g. 512). Defaults to tile_size//2.')

    # Simple resize inference
    parser.add_argument('--image_size', type=int, default=None,
                        help='Override image size (e.g. 512). Uses standard single-pass inference.')

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(cfg['DEVICE'])
    setup_cudnn()

    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    test_cfg = cfg.get('TEST', {})
    n_classes = 4  # MULTIAQUA: Static, Dynamic, Water, Sky

    model = load_model(cfg, model_path, device)

    # Determine inference mode
    use_tiled = args.tile_size is not None

    if use_tiled:
        # ── Tiled inference at original resolution ──
        tile_size = args.tile_size
        stride = args.tile_stride if args.tile_stride is not None else tile_size // 2
        print(f"[Tiled inference] tile_size={tile_size}, stride={stride}")

        split = 'val' if args.mode == 'val' else 'test'
        require_annotation = args.mode == 'val'
        night_trans = False if args.macvi else bool(dataset_cfg.get('NIGHT_TRANSLATION', False))

        dataset = MULTIAQUA_Raw(
            dataset_cfg['ROOT'],
            split=split,
            modals=dataset_cfg['MODALS'],
            require_annotation=require_annotation,
            return_meta=True,
            night_translation=night_trans,
            rgb_subroot=dataset_cfg.get('RGB_SUBROOT'),
            thermal_subroot=dataset_cfg.get('THERMAL_SUBROOT'),
            lidar_subroot=dataset_cfg.get('LIDAR_SUBROOT'),
        )

        # Save directory
        if args.save_dir:
            save_dir = args.save_dir
        elif args.macvi:
            ckpt_dir = model_path.parent
            save_dir = str(ckpt_dir / f"eval_macvi_tiled_{args.mode}")
        else:
            ckpt_dir = model_path.parent
            ckpt_stem = model_path.stem.replace('_checkpoint', '')
            lora_name = cfg['MODEL'].get('LORA_MODEL', 'P').split('_')[-1]
            save_dir = str(ckpt_dir / f"{ckpt_stem}_{args.mode}_tiled_{lora_name}")

        acc, macc, f1, mf1, ious, miou, dyn_iou, fps = evaluate_tiled(
            model, dataset, device, tile_size, stride, n_classes,
            save_dir=save_dir, macvi_format=args.macvi, mode=args.mode,
        )

        print(f"\nSaved to: {save_dir}")
        print(f"FPS: {fps:.2f}")

        if args.mode == 'val' and miou is not None:
            classes = MULTIAQUA.CLASSES
            table = [[cls, f"{iou:.2f}"] for cls, iou in zip(classes, ious)]
            table.append(["mIoU", f"{miou:.2f}"])
            table.append(["Dynamic IoU", f"{dyn_iou:.2f}"])
            print(tabulate(table, headers=["Class", "IoU"], tablefmt="github"))

    else:
        # ── Standard single-pass inference (optionally at different resolution) ──
        image_size_cfg = eval_cfg['IMAGE_SIZE'] if args.mode == 'val' else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
        if args.image_size is not None:
            image_size = [args.image_size, args.image_size]
            print(f"[Single-pass inference] image_size={args.image_size} (override)")
        else:
            image_size = image_size_cfg
            print(f"[Single-pass inference] image_size={image_size}")

        transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

        split = 'val' if args.mode == 'val' else 'test'
        require_annotation = args.mode == 'val'
        night_trans = False if args.macvi else bool(dataset_cfg.get('NIGHT_TRANSLATION', False))

        dataset = MULTIAQUA(
            dataset_cfg['ROOT'],
            split=split,
            transform=transform,
            modals=dataset_cfg['MODALS'],
            require_annotation=require_annotation,
            return_meta=True,
            night_translation=night_trans,
            rgb_subroot=dataset_cfg.get('RGB_SUBROOT'),
            thermal_subroot=dataset_cfg.get('THERMAL_SUBROOT'),
            lidar_subroot=dataset_cfg.get('LIDAR_SUBROOT'),
        )

        from val_multiaqua import _unpad_resize_to_orig, _collate_multiaqua

        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, collate_fn=_collate_multiaqua)
        metrics = Metrics(n_classes, dataset.ignore_label, device)
        palette = dataset.PALETTE

        if args.save_dir:
            save_dir = Path(args.save_dir)
        elif args.macvi:
            save_dir = Path(model_path.parent / f"eval_macvi_{args.mode}")
        else:
            ckpt_stem = model_path.stem.replace('_checkpoint', '')
            lora_name = cfg['MODEL'].get('LORA_MODEL', 'P').split('_')[-1]
            sz = args.image_size or image_size[0]
            save_dir = Path(model_path.parent / f"{ckpt_stem}_{args.mode}_sz{sz}_{lora_name}")

        if args.macvi:
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            (save_dir / "seg").mkdir(parents=True, exist_ok=True)

        total_time = 0.0
        n_images = 0

        for images, labels, metas in tqdm(dataloader, desc=f"Eval {args.mode} (sz={image_size})"):
            images = [x.to(device) for x in images]
            t0 = time.perf_counter()
            output, _ = model(images, multimask_output=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            preds = output.softmax(dim=1)
            n_images += images[0].shape[0]
            pred_labels = preds[:, :n_classes].argmax(dim=1)

            for b in range(pred_labels.shape[0]):
                meta = metas[b]
                orig_h, orig_w = meta["orig_h"], meta["orig_w"]
                pred_b = pred_labels[b]
                model_sz = pred_b.shape[0]
                pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=model_sz)
                pred_np = pred_resized.cpu().numpy().astype(np.uint8)

                if args.mode == 'val':
                    orig_label = meta["orig_label"]
                    pred_oh = F.one_hot(pred_resized.long().clamp(0, n_classes - 1), n_classes).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                    metrics.update(pred_oh, orig_label.unsqueeze(0).to(device))

                stem = meta["stem"]
                if args.macvi:
                    seg_save = (pred_np + 1).clip(1, 4).astype(np.uint8)
                    Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
                else:
                    Image.fromarray(pred_np).save(str(save_dir / "seg" / f"{stem}.png"))

        fps = n_images / total_time if total_time > 0 else 0.0
        print(f"\nSaved to: {save_dir}")
        print(f"FPS: {fps:.2f}")

        if args.mode == 'val':
            ious, miou = metrics.compute_iou()
            acc, macc = metrics.compute_pixel_acc()
            classes = MULTIAQUA.CLASSES
            table = [[cls, f"{iou:.2f}"] for cls, iou in zip(classes, ious)]
            table.append(["mIoU", f"{miou:.2f}"])
            print(tabulate(table, headers=["Class", "IoU"], tablefmt="github"))


if __name__ == "__main__":
    main()
