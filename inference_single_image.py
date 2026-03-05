"""
단일 이미지 세그멘테이션 인퍼런스 (config + weight).
입력 이미지에 gamma, brightness, contrast 조절 가능. 조절 전/후 비교를 seg_viz에 포함.

사용:
  python inference_single_image.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \\
      --model_path outputs/.../epoch47_94.18_checkpoint.pth \\
      --image /path/to/rgb.png --output_dir ./out

  # 데이터셋 stem으로 로드 (img + lidar + thermal)
  python inference_single_image.py --cfg ... --model_path ... --stem lj4_1_077210 \\
      --dataset_root /ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night --output_dir ./out

  # gamma, brightness, contrast 조절 (기본 1.0 = 변경 없음)
  python inference_single_image.py ... --gamma 1.2 --brightness 1.1 --contrast 1.05
"""
import torch
import argparse
import yaml
import os
import math
import inspect
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import io
import torchvision.transforms.functional as TF

from semseg.datasets import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import (
    LoRA_Sam_P9, LoRA_Sam_P10, LoRA_Sam_P11, LoRA_Sam_P12, LoRA_Sam_P13,
    LoRA_Sam_P14, LoRA_Sam_P15, LoRA_Sam_P16, LoRA_Sam_P17, LoRA_Sam_P18, LoRA_Sam_P19,
)


# ---------------------------------------------------------------------------
# Model loading (from val_multiaqua_detailed.py)
# ---------------------------------------------------------------------------

def load_model(cfg, model_path, device):
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_config_file = "sam2_hiera_b+.yaml"
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        sam2_config_file, checkpoint,
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false"
        ]
    )
    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P9')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS') or num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K')
    lora_layer = model_cfg.get('LORA_LAYER')

    _model_map = {
        'LoRA_Sam_P9': LoRA_Sam_P9, 'LoRA_Sam_P10': LoRA_Sam_P10, 'LoRA_Sam_P11': LoRA_Sam_P11,
        'LoRA_Sam_P12': LoRA_Sam_P12, 'LoRA_Sam_P13': LoRA_Sam_P13, 'LoRA_Sam_P14': LoRA_Sam_P14,
        'LoRA_Sam_P15': LoRA_Sam_P15, 'LoRA_Sam_P16': LoRA_Sam_P16, 'LoRA_Sam_P17': LoRA_Sam_P17,
        'LoRA_Sam_P18': LoRA_Sam_P18, 'LoRA_Sam_P19': LoRA_Sam_P19,
    }
    lora_model_class = _model_map.get(lora_model_name)
    if lora_model_class is None:
        raise ValueError(f"Unknown LORA_MODEL: {lora_model_name}")

    model_kwargs = {'sam_model': sam2, 'r': lora_r, 'lora_layer': lora_layer}
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k
    if 'num_classes' in sig.parameters:
        model_kwargs['num_classes'] = model_cfg.get('LORA_NUM_CLASSES', 4)
    if 'num_modalities' in sig.parameters:
        model_kwargs['num_modalities'] = num_modalities
    if 'use_entropy_fusion' in sig.parameters:
        model_kwargs['use_entropy_fusion'] = model_cfg.get('USE_ENTROPY_FUSION', False)

    model = lora_model_class(**model_kwargs)
    ckpt = torch.load(str(model_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    if hasattr(model, '_current_epoch'):
        model._current_epoch = 9999
    model = model.to(device)
    model.eval()
    return model


def _unpad_resize_to_orig(pred, orig_h, orig_w, model_size=1024):
    H, W = orig_h, orig_w
    t = model_size
    if W >= H:
        scale = t / W
        nH, nW = round(H * scale), t
        pad_top = (t - nH) // 2
        pred_content = pred[pad_top:pad_top + nH, :nW]
    else:
        scale = t / H
        nH, nW = t, round(W * scale)
        pad_left = (t - nW) // 2
        pred_content = pred[:nH, pad_left:pad_left + nW]
    if pred_content.shape[0] != H or pred_content.shape[1] != W:
        pred_content = pred_content.unsqueeze(0).unsqueeze(0).float()
        pred_resized = F.interpolate(pred_content, size=(H, W), mode="nearest")
        pred_resized = pred_resized.squeeze(0).squeeze(0).long()
    else:
        pred_resized = pred_content.long()
    return pred_resized


def _draw_legend(classes, palette, target_h, target_w):
    fig, ax = plt.subplots(figsize=(target_w / 80, target_h / 80), dpi=80)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('#f8f8f8')
    n = len(classes)
    patch_h = 0.9 / max(n, 1)
    for i, (cls_name, color) in enumerate(zip(classes, palette)):
        if isinstance(color, torch.Tensor):
            color = (color.cpu().numpy() / 255.0).tolist()
        else:
            color = np.asarray(color)
            color = (color / 255.0).tolist() if color.max() > 1 else color.tolist()
        y = 0.95 - (i + 0.5) * patch_h
        rect = plt.Rectangle((0.05, y - patch_h * 0.4), patch_h * 0.8, patch_h * 0.8,
                              facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.05 + patch_h + 0.02, y, cls_name,
                fontsize=min(18, int(target_h / 35)), va='center', ha='left', fontweight='bold')
    ax.set_title('Classes', fontsize=min(22, int(target_h / 30)))
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _add_title_to_image(img, title):
    h, w = img.shape[:2]
    title_h = max(56, h // 8)
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, title_h / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.5, title, fontsize=min(36, max(18, int(title_h * 0.55))),
            color='white', ha='center', va='center', fontweight='bold')
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    title_img = np.asarray(buf).reshape((fh, fw, 4))[:, :, :3].copy()
    plt.close(fig)
    title_bar = np.array(Image.fromarray(title_img).resize((w, title_h), Image.Resampling.LANCZOS))
    return np.concatenate([title_bar, img], axis=0)


def apply_rgb_adjustments(img_uint8, gamma=1.0, brightness=1.0, contrast=1.0):
    """img_uint8: (H,W,3) 0-255. Returns (H,W,3) uint8."""
    out = img_uint8.astype(np.float32) / 255.0
    if gamma != 1.0:
        out = np.power(np.clip(out, 1e-6, 1.0), 1.0 / gamma)
    if brightness != 1.0:
        out = np.clip(out * brightness, 0, 1)
    if contrast != 1.0:
        mean = out.mean()
        out = np.clip((out - mean) * contrast + mean, 0, 1)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def load_single_sample(image_path, dataset_root=None, stem=None, lidar_path=None, thermal_path=None, modals=None):
    """Load one sample as dict with keys img, lidar, thermal (if in modals). img is (3,H,W) uint8."""
    modals = modals or ['img', 'lidar', 'thermal']
    if stem is not None and dataset_root is not None:
        root = Path(dataset_root)
        data_root = root / "MULTIAQUA_night" if (root / "MULTIAQUA_night").exists() else root
        rgb_dir = data_root / "data" / "zed"
        lidar_dir = data_root / "data" / "lidar_processed"
        thermal_dir = data_root / "data" / "thermal_processed"
        img_path = rgb_dir / f"{stem}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        sample = {}
        sample["img"] = io.read_image(str(img_path))[:3, ...]
        H, W = sample["img"].shape[1:]
        if "lidar" in modals:
            lp = lidar_dir / f"{stem}_lidar.png"
            if lp.exists():
                sample["lidar"] = _open_img_tensor(lp, H, W)
            else:
                sample["lidar"] = torch.zeros(3, H, W, dtype=torch.uint8)
        if "thermal" in modals:
            tp = thermal_dir / f"{stem}_thermal.png"
            if tp.exists():
                sample["thermal"] = _open_img_tensor(tp, H, W)
            else:
                sample["thermal"] = torch.zeros(3, H, W, dtype=torch.uint8)
        return sample, H, W, stem
    else:
        if image_path is None or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        sample = {}
        sample["img"] = io.read_image(str(image_path))[:3, ...]
        H, W = sample["img"].shape[1:]
        stem = Path(image_path).stem
        if "lidar" in modals:
            sample["lidar"] = torch.zeros(3, H, W, dtype=torch.uint8) if lidar_path is None or not os.path.isfile(lidar_path) else _open_img_tensor(Path(lidar_path), H, W)
        if "thermal" in modals:
            sample["thermal"] = torch.zeros(3, H, W, dtype=torch.uint8) if thermal_path is None or not os.path.isfile(thermal_path) else _open_img_tensor(Path(thermal_path), H, W)
        return sample, H, W, stem


def _open_img_tensor(path, H, W):
    img = io.read_image(str(path))
    C, h, w = img.shape
    if C == 4:
        img = img[:3, ...]
    if C == 1:
        img = img.repeat(3, 1, 1)
    if (h, w) != (H, W):
        img = TF.resize(img, (H, W), TF.InterpolationMode.NEAREST)
    return img


@torch.no_grad()
def run_inference(model, sample_tensors, device, n_classes=4):
    """sample_tensors: list of (1,C,H,W) on device. Returns pred (H,W) long."""
    output, _ = model(sample_tensors, multimask_output=True)
    preds = output.softmax(dim=1)[:, :n_classes].argmax(dim=1)
    return preds[0]


def main():
    parser = argparse.ArgumentParser(description="Single-image segmentation inference with gamma/brightness/contrast and before/after viz")
    parser.add_argument('--cfg', type=str, required=True, help='Config YAML (eval config)')
    parser.add_argument('--model_path', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--image', type=str, default=None, help='Path to RGB image (optional if --stem given)')
    parser.add_argument('--stem', type=str, default=None, help='Stem name; load from dataset_root (with --dataset_root)')
    parser.add_argument('--dataset_root', type=str, default=None, help='Dataset root (for MULTIAQUA layout when using --stem)')
    parser.add_argument('--lidar', type=str, default=None, help='Optional lidar image path (when using --image)')
    parser.add_argument('--thermal', type=str, default=None, help='Optional thermal image path (when using --image)')
    parser.add_argument('--output_dir', type=str, default='./inference_single_out', help='Output directory (seg/ and seg_viz/)')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma on input RGB (1.0=no change, >1 brighten)')
    parser.add_argument('--brightness', type=float, default=1.0, help='Brightness factor on input RGB')
    parser.add_argument('--contrast', type=float, default=1.0, help='Contrast factor on input RGB')
    args = parser.parse_args()

    if args.stem is None and args.image is None:
        raise ValueError("Provide either --image path or --stem with --dataset_root")

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(cfg.get('DEVICE', 'cuda'))
    setup_cudnn()
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg.get('EVAL', {})
    modals = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])
    image_size = eval_cfg.get('IMAGE_SIZE', [1024, 1024])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    n_classes = 4
    palette = MULTIAQUA.PALETTE
    classes = MULTIAQUA.CLASSES

    model = load_model(cfg, args.model_path, device)
    sample, orig_h, orig_w, stem = load_single_sample(
        args.image, args.dataset_root, args.stem,
        lidar_path=args.lidar, thermal_path=args.thermal, modals=modals,
    )

    rgb_before = sample["img"].permute(1, 2, 0).numpy()  # (H,W,3) for viz
    rgb_uint8 = rgb_before.copy()

    if args.gamma != 1.0 or args.brightness != 1.0 or args.contrast != 1.0:
        rgb_adjusted = apply_rgb_adjustments(rgb_uint8, args.gamma, args.brightness, args.contrast)
        sample["img"] = torch.from_numpy(rgb_adjusted).permute(2, 0, 1)
    else:
        rgb_adjusted = rgb_uint8

    sample["mask"] = torch.zeros(1, orig_h, orig_w, dtype=torch.long)
    sample = transform(sample)
    del sample["mask"]
    images = [sample[k].unsqueeze(0).to(device) for k in modals]

    pred = run_inference(model, images, device, n_classes=n_classes)
    pred_resized = _unpad_resize_to_orig(pred.cpu(), orig_h, orig_w, model_size=pred.shape[0])
    pred_np = pred_resized.numpy().astype(np.uint8)

    save_dir = Path(args.output_dir)
    seg_dir = save_dir / "seg"
    seg_viz_dir = save_dir / "seg_viz"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_viz_dir.mkdir(parents=True, exist_ok=True)

    out_seg_path = seg_dir / f"{stem}.png"
    Image.fromarray(pred_np).save(str(out_seg_path))
    print(f"Saved seg: {out_seg_path}")

    colored = MULTIAQUA.decode_segmap(pred_np, palette)
    rgb_after = rgb_adjusted
    if rgb_after.shape[0] != orig_h or rgb_after.shape[1] != orig_w:
        rgb_after = np.array(Image.fromarray(rgb_after).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
    overlay = (rgb_after.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

    def with_title(img, title):
        return _add_title_to_image(img, title)

    main_w = orig_w * 2
    row1 = np.concatenate([
        with_title(rgb_before, 'Input (before)'),
        with_title(rgb_after, f'Input (after  gamma={args.gamma} b={args.brightness} c={args.contrast})'),
    ], axis=1)
    row2 = np.concatenate([
        with_title(colored, 'Prediction'),
        with_title(overlay, 'Overlay'),
    ], axis=1)
    if row2.shape[1] != row1.shape[1]:
        row2 = np.array(Image.fromarray(row2).resize((row1.shape[1], row2.shape[0]), Image.Resampling.LANCZOS))
    leg_h = min(100, orig_h // 4)
    legend_img = _draw_legend(classes, palette, leg_h, row1.shape[1])
    legend_bar = with_title(legend_img, 'Classes')
    viz = np.concatenate([legend_bar, row1, row2], axis=0)
    out_viz_path = seg_viz_dir / f"{stem}.png"
    Image.fromarray(viz).save(str(out_viz_path))
    print(f"Saved seg_viz: {out_viz_path}")


if __name__ == '__main__':
    main()
