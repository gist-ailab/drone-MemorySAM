"""
Interactive Image Enhancement Inference Viewer
================================================

Worst test 이미지를 순회하며 gamma/brightness/contrast/denoise를 슬라이더로 조절하고
실시간 인퍼런스 결과 확인.

사용:
  python interactive_gamma_viewer.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug4.yaml \
      --model_path outputs/.../epoch47_94.18_checkpoint.pth \
      --csv outputs/.../P9_15635_results/frames_test.csv \
      --dataset_root /ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night \
      --top_n 30

키보드:
  Left/Right arrow : 이전/다음 이미지
  r                : 모든 슬라이더 리셋
  q / Esc          : 종료
"""
import torch
import argparse
import yaml
import csv
import inspect
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torchvision import io
import torchvision.transforms.functional as TF
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image

from semseg.datasets import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import (
    LoRA_Sam_P9, LoRA_Sam_P10, LoRA_Sam_P11, LoRA_Sam_P12, LoRA_Sam_P13,
    LoRA_Sam_P14, LoRA_Sam_P15, LoRA_Sam_P16, LoRA_Sam_P17, LoRA_Sam_P18, LoRA_Sam_P19,
)


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(cfg, model_path, device):
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2 = build_sam2(
        "sam2_hiera_b+.yaml", checkpoint,
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false",
        ]
    )
    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P9')
    num_modalities = len(dataset_cfg['MODALS'])
    _model_map = {
        'LoRA_Sam_P9': LoRA_Sam_P9, 'LoRA_Sam_P10': LoRA_Sam_P10,
        'LoRA_Sam_P11': LoRA_Sam_P11, 'LoRA_Sam_P12': LoRA_Sam_P12,
        'LoRA_Sam_P13': LoRA_Sam_P13, 'LoRA_Sam_P14': LoRA_Sam_P14,
        'LoRA_Sam_P15': LoRA_Sam_P15, 'LoRA_Sam_P16': LoRA_Sam_P16,
        'LoRA_Sam_P17': LoRA_Sam_P17, 'LoRA_Sam_P18': LoRA_Sam_P18,
        'LoRA_Sam_P19': LoRA_Sam_P19,
    }
    lora_model_class = _model_map[lora_model_name]
    model_kwargs = {
        'sam_model': sam2, 'r': model_cfg.get('LORA_R', 4),
        'lora_layer': model_cfg.get('LORA_LAYER'),
    }
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = model_cfg.get('LORA_NUM_EXPERTS') or num_modalities
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = model_cfg.get('LORA_TOP_K')
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
    model.to(device).eval()
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
        pred_content = F.interpolate(
            pred_content.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode="nearest"
        ).squeeze(0).squeeze(0).long()
    return pred_content


def _open_img(path, H, W):
    if not path.exists():
        return torch.zeros(3, H, W, dtype=torch.uint8)
    img = io.read_image(str(path))
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[1:] != (H, W):
        img = TF.resize(img, (H, W), TF.InterpolationMode.NEAREST)
    return img


# ─── Data loading ────────────────────────────────────────────────────────────

def load_worst_stems(csv_path, top_n=30):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows.sort(key=lambda x: float(x['mIoU']))
    return rows[:top_n]


def load_sample(stem, dataset_root, modals):
    root = Path(dataset_root)
    rgb_dir = root / "data" / "zed"
    lidar_dir = root / "data" / "lidar_processed"
    thermal_dir = root / "data" / "thermal_processed"

    img_path = rgb_dir / f"{stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"RGB not found: {img_path}")

    rgb = io.read_image(str(img_path))[:3]
    H, W = rgb.shape[1:]
    sample = {'img': rgb}
    if 'lidar' in modals:
        sample['lidar'] = _open_img(lidar_dir / f"{stem}_lidar.png", H, W)
    if 'thermal' in modals:
        sample['thermal'] = _open_img(thermal_dir / f"{stem}_thermal.png", H, W)
    return sample, H, W


# ─── Image Enhancement ──────────────────────────────────────────────────────

def apply_enhancements(sample, gamma, brightness, contrast, denoise):
    """Apply gamma, brightness, contrast, denoise to RGB. Returns new sample."""
    out = dict(sample)
    img = out['img'].float() / 255.0

    # 1) Gamma correction
    if gamma != 1.0:
        img = torch.clamp(img, 1e-6, 1.0) ** (1.0 / gamma)

    # 2) Brightness (additive shift)
    if brightness != 0.0:
        img = img + brightness
        img = torch.clamp(img, 0.0, 1.0)

    # 3) Contrast (around mean)
    if contrast != 1.0:
        mean = img.mean()
        img = (img - mean) * contrast + mean
        img = torch.clamp(img, 0.0, 1.0)

    # Convert to uint8
    img_uint8 = (img * 255).clamp(0, 255).to(torch.uint8)

    # 4) Denoise (bilateral filter via OpenCV, preserves edges)
    if denoise > 0:
        img_np = img_uint8.permute(1, 2, 0).numpy()
        d = int(denoise)
        sigma = denoise * 25  # scale sigma with slider
        img_np = cv2.bilateralFilter(img_np, d, sigma, sigma)
        img_uint8 = torch.from_numpy(img_np).permute(2, 0, 1)

    out['img'] = img_uint8
    return out


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, sample, transform, modals, device, orig_h, orig_w):
    s = dict(sample)
    s['mask'] = torch.zeros(1, orig_h, orig_w, dtype=torch.long)
    s = transform(s)
    del s['mask']
    images = [s[k].unsqueeze(0).to(device) for k in modals]
    output, _ = model(images, multimask_output=True)
    pred = output.softmax(dim=1)[:, :4].argmax(dim=1)[0]
    return _unpad_resize_to_orig(pred.cpu(), orig_h, orig_w, model_size=pred.shape[0])


# ─── Interactive Viewer ──────────────────────────────────────────────────────

class EnhancementViewer:
    def __init__(self, model, transform, modals, device, worst_rows, dataset_root):
        self.model = model
        self.transform = transform
        self.modals = modals
        self.device = device
        self.worst_rows = worst_rows
        self.dataset_root = dataset_root
        self.idx = 0
        self.palette = MULTIAQUA.PALETTE
        self.class_names = ['Static', 'Dynamic', 'Water', 'Sky']

        # Current state
        self.sample = None
        self.baseline_pred = None
        self.orig_h = self.orig_w = 0
        self.rgb_orig = None
        self._updating = False  # prevent slider cascade

        # Setup figure: 2x2 grid
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.subplots_adjust(bottom=0.28, top=0.92, hspace=0.08, wspace=0.05)

        # Sliders
        slider_cfg = [
            # (label, left, min, max, init, step)
            ('Gamma',      0.06, 0.5, 4.0, 1.0, 0.1),
            ('Brightness',  0.11, -0.3, 0.3, 0.0, 0.01),
            ('Contrast',   0.16, 0.5, 3.0, 1.0, 0.1),
            ('Denoise',    0.21, 0, 5, 0, 1),
        ]
        self.sliders = {}
        for label, bottom, vmin, vmax, vinit, vstep in slider_cfg:
            ax = self.fig.add_axes([0.2, bottom, 0.55, 0.025])
            s = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep)
            s.on_changed(self._on_slider_change)
            self.sliders[label.lower()] = s

        # Buttons
        ax_prev = self.fig.add_axes([0.2, 0.01, 0.1, 0.04])
        ax_next = self.fig.add_axes([0.35, 0.01, 0.1, 0.04])
        ax_reset = self.fig.add_axes([0.5, 0.01, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_reset = Button(ax_reset, 'Reset All')
        self.btn_prev.on_clicked(lambda _: self._change_image(-1))
        self.btn_next.on_clicked(lambda _: self._change_image(1))
        self.btn_reset.on_clicked(lambda _: self._reset_all())

        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._load_and_show()

    def _get_params(self):
        return {
            'gamma': self.sliders['gamma'].val,
            'brightness': self.sliders['brightness'].val,
            'contrast': self.sliders['contrast'].val,
            'denoise': self.sliders['denoise'].val,
        }

    def _is_default(self, params):
        return (params['gamma'] == 1.0 and params['brightness'] == 0.0
                and params['contrast'] == 1.0 and params['denoise'] == 0)

    def _on_key(self, event):
        if event.key == 'right':
            self._change_image(1)
        elif event.key == 'left':
            self._change_image(-1)
        elif event.key == 'r':
            self._reset_all()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    def _change_image(self, delta):
        self.idx = (self.idx + delta) % len(self.worst_rows)
        self._load_and_show()

    def _reset_all(self):
        self._updating = True
        for s in self.sliders.values():
            s.reset()
        self._updating = False
        self._run_and_update()

    def _on_slider_change(self, _val):
        if not self._updating:
            self._run_and_update()

    def _load_and_show(self):
        row = self.worst_rows[self.idx]
        stem = row['image']
        self.sample, self.orig_h, self.orig_w = load_sample(
            stem, self.dataset_root, self.modals)
        self.rgb_orig = self.sample['img'].permute(1, 2, 0).numpy()
        # Baseline inference
        print(f"\n{'='*60}")
        print(f"[Image {self.idx+1}/{len(self.worst_rows)}] {stem}")
        print(f"  CSV mIoU={float(row['mIoU']):.1f}  "
              f"Static={float(row['IoU_static_obstacle']):.1f}  "
              f"Dynamic={float(row['IoU_dynamic_obstacle']):.1f}  "
              f"Water={float(row['IoU_water']):.1f}  "
              f"Sky={float(row['IoU_sky']):.1f}")
        baseline = run_inference(
            self.model, self.sample, self.transform, self.modals,
            self.device, self.orig_h, self.orig_w)
        self.baseline_pred = baseline.numpy().astype(np.uint8)
        self.baseline_colored = MULTIAQUA.decode_segmap(self.baseline_pred, self.palette)
        self._log_pred("baseline", self.baseline_pred)
        self._run_and_update()

    def _log_pred(self, tag, pred_np, compare_to=None):
        total = pred_np.size
        parts = []
        for cls_id, name in enumerate(self.class_names):
            cnt = (pred_np == cls_id).sum()
            parts.append(f"{name}={cnt/total*100:.1f}%")
        print(f"  [{tag}] {' | '.join(parts)}")
        if compare_to is not None:
            changed = (pred_np != compare_to).sum()
            print(f"  [diff vs baseline] {changed} px changed "
                  f"({changed/total*100:.2f}%)")

    def _run_and_update(self):
        row = self.worst_rows[self.idx]
        stem = row['image']
        miou = float(row['mIoU'])
        static = float(row['IoU_static_obstacle'])
        dynamic = float(row['IoU_dynamic_obstacle'])
        water = float(row['IoU_water'])
        sky = float(row['IoU_sky'])

        params = self._get_params()
        is_default = self._is_default(params)

        # Apply enhancements
        if not is_default:
            s_enhanced = apply_enhancements(
                self.sample, params['gamma'], params['brightness'],
                params['contrast'], params['denoise'])
        else:
            s_enhanced = self.sample
        rgb_enhanced = s_enhanced['img'].permute(1, 2, 0).numpy()

        # Inference
        pred = run_inference(
            self.model, s_enhanced, self.transform, self.modals,
            self.device, self.orig_h, self.orig_w)
        pred_np = pred.numpy().astype(np.uint8)

        # Log
        tag_parts = []
        if params['gamma'] != 1.0:
            tag_parts.append(f"γ={params['gamma']:.1f}")
        if params['brightness'] != 0.0:
            tag_parts.append(f"br={params['brightness']:+.2f}")
        if params['contrast'] != 1.0:
            tag_parts.append(f"ct={params['contrast']:.1f}")
        if params['denoise'] > 0:
            tag_parts.append(f"dn={params['denoise']:.0f}")
        tag = ", ".join(tag_parts) if tag_parts else "baseline"
        self._log_pred(tag, pred_np,
                       compare_to=self.baseline_pred if not is_default else None)

        colored = MULTIAQUA.decode_segmap(pred_np, self.palette)

        # Update plots
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')

        self.axes[0, 0].imshow(self.rgb_orig)
        self.axes[0, 0].set_title('Original RGB', fontsize=12)

        enhance_title = f'γ={params["gamma"]:.1f}  br={params["brightness"]:+.2f}  ct={params["contrast"]:.1f}  dn={params["denoise"]:.0f}'
        self.axes[0, 1].imshow(rgb_enhanced)
        self.axes[0, 1].set_title(enhance_title, fontsize=10)

        self.axes[1, 0].imshow(self.baseline_colored)
        self.axes[1, 0].set_title('Baseline Prediction', fontsize=12)

        self.axes[1, 1].imshow(colored)
        self.axes[1, 1].set_title('Enhanced Prediction', fontsize=12)

        self.fig.suptitle(
            f'[{self.idx+1}/{len(self.worst_rows)}] {stem}  |  '
            f'mIoU={miou:.1f}  Static={static:.1f}  Dynamic={dynamic:.1f}  '
            f'Water={water:.1f}  Sky={sky:.1f}',
            fontsize=13, fontweight='bold')

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True,
                        help='frames_test.csv from MACVi results')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Dataset root (e.g., MULTIAQUA_night)')
    parser.add_argument('--top_n', type=int, default=30,
                        help='Number of worst images to browse')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(cfg.get('DEVICE', 'cuda'))
    setup_cudnn()

    dataset_cfg = cfg['DATASET']
    modals = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])
    eval_cfg = cfg.get('EVAL', {})
    image_size = eval_cfg.get('IMAGE_SIZE', [1024, 1024])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    print("Loading model...")
    model = load_model(cfg, args.model_path, device)
    print("Model loaded.")

    worst_rows = load_worst_stems(args.csv, args.top_n)
    print(f"Loaded {len(worst_rows)} worst test images (mIoU range: "
          f"{float(worst_rows[0]['mIoU']):.1f} ~ {float(worst_rows[-1]['mIoU']):.1f})")

    viewer = EnhancementViewer(model, transform, modals, device, worst_rows, args.dataset_root)
    plt.show()


if __name__ == '__main__':
    main()
