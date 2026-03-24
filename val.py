"""
Unified Validation & Test Inference Script
==========================================
Supports MULTIAQUA, DELIVER, and future datasets via config DATASET.NAME.

Usage:
  # Basic validation (seg/ + seg_viz/ + uamm_amf_moe_log.json)
  python val.py --cfg configs/eval_config/xxx.yaml --mode val --model_path xxx.pth

  # Detailed mode: per-token MoE routing analysis + extended viz + detailed_log.json
  python val.py --cfg ... --mode val --model_path ... --detailed

  # Test inference (no GT for MULTIAQUA, with GT for DELIVER)
  python val.py --cfg ... --mode test --model_path ...

  # MACVi submission (MULTIAQUA only, 1-indexed masks)
  python val.py --cfg ... --mode val --model_path ... --macvi

  # Gamma TTA
  python val.py --cfg ... --mode val --model_path ... --test_gamma_tta "1.0,1.5,2.0"

  # Horizontal flip TTA (detailed mode only)
  python val.py --cfg ... --mode val --model_path ... --detailed --tta
"""
import torch
import argparse
import yaml
import os
import time
import json
import math
import inspect
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *
from semseg.models.sam2.sam2.sam_lola_utils import SoftMoE_LoRA_Layer


# ============================================================================
# Model Loading (generic, all P-versions via eval())
# ============================================================================

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
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K')
    lora_layer = model_cfg.get('LORA_LAYER')

    lora_model_class = eval(lora_model_name)
    model_kwargs = {'sam_model': sam2, 'r': lora_r, 'lora_layer': lora_layer}
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k
    if 'num_classes' in sig.parameters:
        model_kwargs['num_classes'] = model_cfg.get('LORA_NUM_CLASSES', dataset_cfg.get('NUM_CLASSES', 4))
    if 'num_modalities' in sig.parameters:
        model_kwargs['num_modalities'] = num_modalities
    if 'use_entropy_fusion' in sig.parameters:
        model_kwargs['use_entropy_fusion'] = model_cfg.get('USE_ENTROPY_FUSION', False)
    if 'quality_hidden_dim' in sig.parameters:
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['quality_hidden_dim'] = quality_cfg.get('HIDDEN_DIM', 64)
        model_kwargs['quality_min'] = quality_cfg.get('MIN_QUALITY', 0.1)
    if 'tau_uamm' in sig.parameters:
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['tau_uamm'] = quality_cfg.get('TAU_UAMM', 1.0)
        model_kwargs['tau_teacher'] = quality_cfg.get('TAU_TEACHER', 0.5)
        model_kwargs['memory_mod'] = quality_cfg.get('MEMORY_MOD', False)
        model_kwargs['amf_mode'] = quality_cfg.get('AMF_MODE', 'output_entropy')
        model_kwargs['multi_scale_sqg'] = quality_cfg.get('MULTI_SCALE_SQG', True)
        model_kwargs['per_modality_decoder'] = quality_cfg.get('PER_MODALITY_DECODER', True)
    if 'cond_dim' in sig.parameters:
        model_kwargs['cond_dim'] = model_cfg.get('LORA_COND_DIM', 8)

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


# ============================================================================
# Dataset Factory
# ============================================================================

def create_dataset(dataset_cfg, split, transform, mode, macvi=False, eval_day=False):
    """Create dataset based on config DATASET.NAME. Returns (dataset, has_gt)."""
    name = dataset_cfg.get('NAME', 'MULTIAQUA').upper()
    modals = dataset_cfg['MODALS']

    if name == 'MULTIAQUA':
        require_annotation = (mode == 'val')
        night_trans = False if macvi else bool(dataset_cfg.get('NIGHT_TRANSLATION', False))
        dataset = MULTIAQUA(
            dataset_cfg['ROOT'], split=split, transform=transform,
            modals=modals, require_annotation=require_annotation,
            return_meta=True, night_translation=night_trans,
            rgb_subroot=dataset_cfg.get('RGB_SUBROOT'),
            thermal_subroot=dataset_cfg.get('THERMAL_SUBROOT'),
            lidar_subroot=dataset_cfg.get('LIDAR_SUBROOT'),
            eval_day=eval_day,
        )
        has_gt = require_annotation
    elif name == 'DELIVER':
        case = dataset_cfg.get('CASE', None)
        dataset = DELIVER(
            dataset_cfg['ROOT'], split=split, transform=transform,
            modals=modals, case=case, return_meta=True,
        )
        has_gt = True  # DELIVER always has GT for all splits
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: MULTIAQUA, DELIVER")

    return dataset, has_gt


def _get_dataset_class(dataset):
    """Get the dataset class (for static methods like decode_segmap)."""
    if isinstance(dataset, MULTIAQUA):
        return MULTIAQUA
    elif isinstance(dataset, DELIVER):
        return DELIVER
    return type(dataset)


# ============================================================================
# Utility Functions
# ============================================================================

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
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
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


def _draw_bar_chart(values, labels, title, target_h, target_w=None):
    fig_w = target_w or max(320, target_h * 2)
    fig, ax = plt.subplots(figsize=(fig_w / 80, target_h / 80), dpi=80)
    n = len(values)
    y_pos = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))
    bars = ax.barh(y_pos, values, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=max(10, min(22, int(target_h / 20))))
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=max(12, min(24, int(target_h / 15))))
    ax.set_xlabel('Weight' if 'Fusion' in title else 'Score', fontsize=max(10, min(18, int(target_h / 25))))
    for bar, val in zip(bars, values):
        txt = f'{val:.3f}' if val < 0.01 or val >= 0.1 else f'{val:.2f}'
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, txt,
                va='center', ha='left', fontsize=max(10, min(18, int(target_h / 25))), fontweight='bold')
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    pil_img = Image.fromarray(img)
    out_w = target_w if target_w else int(img.shape[1] * (target_h / img.shape[0]))
    return np.array(pil_img.resize((out_w, target_h), Image.Resampling.LANCZOS))


def _add_title_to_image(img, title):
    h, w = img.shape[:2]
    title_h = max(56, h // 8)
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, title_h / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.5, 0.5, title, fontsize=min(36, max(18, int(title_h * 0.55))),
            color='white', ha='center', va='center', fontweight='bold')
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    title_img = np.asarray(buf).reshape((fh, fw, 4))[:, :, :3].copy()
    plt.close(fig)
    title_bar = np.array(Image.fromarray(title_img).resize((w, title_h), Image.Resampling.LANCZOS))
    return np.concatenate([title_bar, img], axis=0)


# ============================================================================
# Generic Modality Image Loading (for visualization)
# ============================================================================

# Display names for known modalities
MODAL_TITLES = {
    'img': 'RGB', 'lidar': 'LiDAR', 'thermal': 'Thermal',
    'depth': 'Depth (HHA)', 'event': 'Event',
}


def _load_modality_image(dataset, modal_key, meta, target_h, target_w):
    """Load modality image for visualization. Dataset-agnostic via meta['paths'] fallback."""
    path = None

    # DELIVER-style: paths stored in meta
    if 'paths' in meta and modal_key in meta['paths']:
        p = Path(meta['paths'][modal_key])
        if p.exists():
            path = str(p)

    # MULTIAQUA-style: directory-based
    if path is None:
        stem = meta['stem']
        if modal_key == 'img' and hasattr(dataset, 'rgb_dir'):
            path = str(dataset.rgb_dir / f"{stem}.png")
        elif modal_key == 'lidar' and hasattr(dataset, 'lidar_dir'):
            path = str(dataset.lidar_dir / f"{stem}_lidar.png")
        elif modal_key == 'thermal' and hasattr(dataset, 'thermal_dir'):
            path = str(dataset.thermal_dir / f"{stem}_thermal.png")

    if path is None or not Path(path).exists():
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    img = np.array(Image.open(path).convert("RGB"))
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return img


# ============================================================================
# Gamma TTA
# ============================================================================

def apply_test_gamma(rgb_tensor, gamma, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if gamma == 1.0:
        return rgb_tensor
    device = rgb_tensor.device
    mean_t = torch.tensor(mean, device=device, dtype=rgb_tensor.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=rgb_tensor.dtype).view(1, 3, 1, 1)
    x_01 = (rgb_tensor * std_t + mean_t).clamp(1e-6, 1.0)
    x_gamma = x_01 ** (1.0 / gamma)
    return (x_gamma - mean_t) / std_t


def _gamma_tta_forward(model, images, gamma_list):
    preds_accum = None
    rgb_orig = images[0].clone()
    for g in gamma_list:
        images[0] = apply_test_gamma(rgb_orig, g)
        output, _ = model(images, multimask_output=True)
        p = output.softmax(dim=1)
        preds_accum = p if preds_accum is None else preds_accum + p
    images[0] = rgb_orig
    return preds_accum / len(gamma_list)


@torch.no_grad()
def _tta_accumulate(model, images, base_output, tta_flip):
    accumulated = base_output.softmax(dim=1)
    if tta_flip:
        flipped = [torch.flip(img, dims=(3,)) for img in images]
        logits_f, _ = model(flipped, multimask_output=True)
        accumulated += torch.flip(logits_f, dims=(3,)).softmax(dim=1)
    return accumulated


# ============================================================================
# UAMM / AMF / MoE Visualization (basic mode)
# ============================================================================

def _get_uamm_amf_moe_log(model, batch_idx, modals):
    core = model.module if hasattr(model, 'module') else model
    if not hasattr(core, '_last_uamm_scores'):
        return None
    uamm = getattr(core, '_last_uamm_scores', None)
    amf = getattr(core, '_last_amf_weights', None)
    moe = getattr(core, '_last_moe_gates', None)
    modal_labels = modals if modals else [f'M{i}' for i in range(uamm.shape[1] if uamm is not None else 0)]
    log = {}
    if uamm is not None and batch_idx < uamm.shape[0]:
        log['uamm'] = {k: round(float(v), 4) for k, v in zip(modal_labels, uamm[batch_idx])}
    if amf is not None and batch_idx < amf.shape[0]:
        log['amf'] = {k: round(float(v), 4) for k, v in zip(modal_labels, amf[batch_idx])}
    if moe is not None:
        moe_arr = np.asarray(moe)
        arr = moe_arr if moe_arr.ndim == 1 else (moe_arr[batch_idx] if batch_idx < moe_arr.shape[0] else moe_arr[0])
        log['moe'] = {f'E{i}': round(float(v), 4) for i, v in enumerate(arr)}
    return log if log else None


def _get_uamm_amf_moe_viz(model, batch_idx, modals, main_h, main_w):
    core = model.module if hasattr(model, 'module') else model
    if not hasattr(core, '_last_uamm_scores'):
        return None
    uamm = getattr(core, '_last_uamm_scores', None)
    amf = getattr(core, '_last_amf_weights', None)
    moe = getattr(core, '_last_moe_gates', None)
    viz_h = int(main_h * 0.55)
    chart_w = (main_w + 2) // 3
    modal_labels = modals if modals else [f'M{i}' for i in range(uamm.shape[1] if uamm is not None else 0)]
    strips = []
    if uamm is not None and batch_idx < uamm.shape[0]:
        strips.append(_draw_bar_chart(uamm[batch_idx], modal_labels, 'UAMM (Memory Mod)', viz_h, chart_w))
    if amf is not None and batch_idx < amf.shape[0]:
        strips.append(_draw_bar_chart(amf[batch_idx], modal_labels, 'AMF (Fusion)', viz_h, chart_w))
    if moe is not None:
        moe_arr = np.asarray(moe)
        arr = np.atleast_1d(moe_arr if moe_arr.ndim == 1 else moe_arr[batch_idx] if batch_idx < moe_arr.shape[0] else moe_arr[0])
        strips.append(_draw_bar_chart(arr, [f'E{i}' for i in range(len(arr))], 'MoE LoRA (Experts)', viz_h, chart_w))
    if not strips:
        return None
    bottom = np.concatenate(strips, axis=1)
    if bottom.shape[1] != main_w:
        bottom = np.array(Image.fromarray(bottom).resize((main_w, viz_h), Image.Resampling.LANCZOS))
    return bottom


# ============================================================================
# MoE Routing Capture (detailed mode only)
# ============================================================================

REPRESENTATIVE_LAYERS = [0, 9, 18]

EXPERT_COLORS = np.array([
    [220, 50, 50], [50, 180, 50], [50, 80, 220], [220, 180, 50],
    [180, 50, 220], [50, 220, 180], [220, 130, 50], [130, 50, 220],
], dtype=np.uint8)


class MoERoutingCapture:
    """Captures per-token MoE routing data during forward pass."""

    def __init__(self, model, viz_block_indices=None):
        self.model = model
        self.viz_blocks = viz_block_indices or REPRESENTATIVE_LAYERS
        self.hooks = []
        self.call_counter = 0
        self.num_q = len(model.moe_layers_q)
        self.num_v = len(model.moe_layers_v)
        self.num_moe_layers = self.num_q + self.num_v
        self.routing_data = {'Q': {}, 'V': {}}

    def _get_modality_idx(self):
        return self.call_counter // self.num_moe_layers

    def register_hooks(self):
        self.call_counter = 0
        self.routing_data = {'Q': {}, 'V': {}}
        for block_idx, layer in enumerate(self.model.moe_layers_q):
            save_map = (block_idx in self.viz_blocks)
            h = layer.register_forward_hook(self._make_hook(block_idx, 'Q', save_map))
            self.hooks.append(h)
        for block_idx, layer in enumerate(self.model.moe_layers_v):
            h = layer.register_forward_hook(self._make_hook(block_idx, 'V', save_map=False))
            self.hooks.append(h)

    def _make_hook(self, block_idx, qv_type, save_map):
        def hook_fn(module, input, output):
            x = input[0]
            modal_idx = self._get_modality_idx()
            with torch.no_grad():
                gate_fn = getattr(module, 'gate', None) or getattr(module, '_shared_gate', None)
                gate_logits = gate_fn(x)
                gate_weights = F.softmax(gate_logits, dim=-1)
                ne = module.num_experts
                per_token_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1)
                max_entropy = math.log(ne)
                per_token_max = gate_weights.max(dim=-1).values
                argmax = gate_weights.argmax(dim=-1)
                expert_counts = [(argmax == i).float().mean().item() for i in range(ne)]
                logit_std = gate_logits.std(dim=-1).mean().cpu().item()
                logit_range = (gate_logits.max(dim=-1).values - gate_logits.min(dim=-1).values).mean().cpu().item()
                topk2 = gate_weights.topk(min(2, ne), dim=-1).values
                top2_gap = (topk2[..., 0] - topk2[..., 1]).mean().cpu().item() if ne >= 2 else 0.0

                storage = self.routing_data[qv_type]
                if block_idx not in storage:
                    storage[block_idx] = {}
                stats = {
                    'entropy_ratio': per_token_entropy.mean().cpu().item() / max_entropy,
                    'per_token_max_mean': per_token_max.mean().cpu().item(),
                    'per_token_max_std': per_token_max.std().cpu().item(),
                    'argmax_fraction': expert_counts,
                    'spatial_mean': gate_weights.mean(dim=tuple(range(gate_weights.dim() - 1))).cpu().numpy().tolist(),
                    'logit_range': logit_range,
                    'logit_std': logit_std,
                    'top2_gap': top2_gap,
                    'num_experts': ne,
                }
                if save_map:
                    stats['argmax_map'] = argmax.cpu().numpy()
                    stats['spatial_shape'] = list(argmax.shape)
                storage[block_idx][modal_idx] = stats
        return hook_fn

    def register_counter_hook(self):
        def count_hook(module, input, output):
            self.call_counter += 1
        for layer in list(self.model.moe_layers_q) + list(self.model.moe_layers_v):
            h = layer.register_forward_hook(count_hook)
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_routing_map_image(self, block_idx, modal_idx, target_h, target_w):
        q_data = self.routing_data['Q']
        if block_idx not in q_data or modal_idx not in q_data[block_idx]:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        data = q_data[block_idx][modal_idx]
        if 'argmax_map' not in data:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        argmax_map = data['argmax_map']
        ne = data['num_experts']
        shape = data['spatial_shape']
        if len(shape) == 3:
            map_2d = argmax_map[0]
        elif len(shape) == 2:
            n = shape[0]
            side = int(math.sqrt(n))
            map_2d = argmax_map.reshape(side, side) if side * side == n else argmax_map.reshape(1, -1)
        else:
            map_2d = argmax_map.reshape(-1)[:1024].reshape(32, 32)
        h, w = map_2d.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for e in range(min(ne, len(EXPERT_COLORS))):
            colored[map_2d == e] = EXPERT_COLORS[e]
        return np.array(Image.fromarray(colored).resize((target_w, target_h), Image.Resampling.NEAREST))

    def get_stats_bar_chart(self, block_idx, modals, target_h, target_w):
        q_data = self.routing_data['Q']
        if block_idx not in q_data:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        fig, axes = plt.subplots(1, 2, figsize=(target_w / 80, target_h / 80), dpi=80)
        # Entropy ratio
        ax = axes[0]
        er_values, labels = [], []
        for m_idx, mname in enumerate(modals):
            if m_idx in q_data[block_idx]:
                er_values.append(q_data[block_idx][m_idx]['entropy_ratio'])
                labels.append(mname)
        if er_values:
            bar_colors = plt.cm.Set1(np.linspace(0, 1, max(len(er_values), 3)))[:len(er_values)]
            bars = ax.barh(range(len(er_values)), er_values, color=bar_colors, height=0.6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=16)
            ax.set_xlim(0, 1.1)
            ax.set_title(f'Entropy Ratio (B{block_idx}_Q)', fontsize=16)
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
            for bar, val in zip(bars, er_values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=14, fontweight='bold')
        # Expert selection
        ax = axes[1]
        ne = 3
        for m_idx, mname in enumerate(modals):
            if m_idx in q_data[block_idx]:
                fracs = q_data[block_idx][m_idx]['argmax_fraction']
                ne = len(fracs)
                left = 0
                for e_idx, frac in enumerate(fracs):
                    c = EXPERT_COLORS[e_idx % len(EXPERT_COLORS)].astype(float) / 255.0
                    ax.barh(m_idx, frac, left=left, color=c, height=0.6, edgecolor='white', linewidth=0.5)
                    if frac > 0.08:
                        ax.text(left + frac / 2, m_idx, f'{frac:.0%}',
                                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                    left += frac
        ax.set_yticks(range(len(modals)))
        ax.set_yticklabels(modals, fontsize=16)
        ax.set_xlim(0, 1.0)
        ax.set_title(f'Expert Selection (B{block_idx}_Q)', fontsize=16)
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=EXPERT_COLORS[e % len(EXPERT_COLORS)].astype(float) / 255.0,
                                label=f'E{e}') for e in range(ne)]
        ax.legend(handles=legend_handles, fontsize=11, loc='lower right')
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
        plt.close(fig)
        return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))

    def get_log_dict(self, modals):
        log = {}
        for qv_type in ['Q', 'V']:
            for block_idx in sorted(self.routing_data[qv_type].keys()):
                block_log = {}
                for m_idx in sorted(self.routing_data[qv_type][block_idx].keys()):
                    mname = modals[m_idx] if m_idx < len(modals) else f'M{m_idx}'
                    d = self.routing_data[qv_type][block_idx][m_idx]
                    block_log[mname] = {
                        'entropy_ratio': round(d['entropy_ratio'], 4),
                        'per_token_max': round(d['per_token_max_mean'], 4),
                        'per_token_max_std': round(d['per_token_max_std'], 4),
                        'argmax_fraction': {f'E{i}': round(v, 4) for i, v in enumerate(d['argmax_fraction'])},
                        'spatial_mean': {f'E{i}': round(v, 4) for i, v in enumerate(d['spatial_mean'])},
                        'logit_range': round(d['logit_range'], 4),
                        'logit_std': round(d['logit_std'], 4),
                        'top2_gap': round(d['top2_gap'], 4),
                    }
                log[f'Block{block_idx}_{qv_type}'] = block_log
        return log

    def get_summary_stats(self, modals):
        summary = {}
        for qv_type in ['Q', 'V']:
            data = self.routing_data[qv_type]
            if not data:
                continue
            all_entropy, all_top2_gap = [], []
            expert_usage_per_modal = {m: [] for m in modals}
            for block_idx in sorted(data.keys()):
                for m_idx, mname in enumerate(modals):
                    if m_idx in data[block_idx]:
                        d = data[block_idx][m_idx]
                        all_entropy.append(d['entropy_ratio'])
                        all_top2_gap.append(d['top2_gap'])
                        expert_usage_per_modal[mname].append(d['argmax_fraction'])
            collapse_report = {}
            for mname in modals:
                usages = expert_usage_per_modal[mname]
                if not usages:
                    continue
                ne = len(usages[0])
                per_expert_min = [min(u[e] for u in usages) for e in range(ne)]
                per_expert_mean = [np.mean([u[e] for u in usages]) for e in range(ne)]
                collapsed = [e for e in range(ne) if per_expert_mean[e] < 0.05]
                collapse_report[mname] = {
                    f'E{e}': {'min': round(per_expert_min[e], 4), 'mean': round(per_expert_mean[e], 4)}
                    for e in range(ne)
                }
                if collapsed:
                    collapse_report[mname]['collapsed_experts'] = [f'E{e}' for e in collapsed]
            summary[qv_type] = {
                'avg_entropy_ratio': round(float(np.mean(all_entropy)), 4) if all_entropy else None,
                'avg_top2_gap': round(float(np.mean(all_top2_gap)), 4) if all_top2_gap else None,
                'expert_usage': collapse_report,
            }
        return summary


# ============================================================================
# Detailed Visualization Row Builders
# ============================================================================

def _build_stats_row(capture, modals, target_h, target_w):
    blocks = sorted(capture.routing_data['Q'].keys())
    if not blocks:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 240
    n_charts = min(len(blocks), 3)
    chart_w = target_w // n_charts
    strips = [capture.get_stats_bar_chart(blocks[i], modals, target_h, chart_w) for i in range(n_charts)]
    row = np.concatenate(strips, axis=1)
    if row.shape[1] != target_w:
        row = np.array(Image.fromarray(row).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_routing_map_row(capture, modals, target_h, target_w, block_idx=9):
    col_w = target_w // len(modals)
    cols = []
    for m_idx, mname in enumerate(modals):
        routing_img = capture.get_routing_map_image(block_idx, m_idx, target_h, col_w)
        cols.append(_add_title_to_image(routing_img, f'MoE Routing: {mname} (Block{block_idx})'))
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != target_w:
        row = np.array(Image.fromarray(row).resize((target_w, row.shape[0]), Image.Resampling.LANCZOS))
    return row


# ============================================================================
# P26 Fusion Visualization (detailed mode, SQG/UAMM/AMF spatial maps)
# ============================================================================

def _spatial_map_to_heatmap(spatial_map, target_h, target_w, cmap='inferno', vmin=None, vmax=None):
    """Convert a 2D spatial map (H, W) to a colored heatmap image (target_h, target_w, 3).
    Uses matplotlib colormap for visualization."""
    if spatial_map.ndim > 2:
        spatial_map = spatial_map.squeeze()
    h, w = spatial_map.shape
    if vmin is None:
        vmin = spatial_map.min()
    if vmax is None:
        vmax = spatial_map.max()
    normed = (spatial_map - vmin) / (vmax - vmin + 1e-8)
    normed = np.clip(normed, 0, 1)
    cm = plt.cm.get_cmap(cmap)
    colored = (cm(normed)[:, :, :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(colored).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _spatial_map_to_heatmap_with_colorbar(spatial_map, title, target_h, target_w,
                                           cmap='inferno', vmin=None, vmax=None):
    """Render spatial map as a matplotlib figure with colorbar and title."""
    if spatial_map.ndim > 2:
        spatial_map = spatial_map.squeeze()
    dpi = 80
    fig, ax = plt.subplots(figsize=(target_w / dpi, target_h / dpi), dpi=dpi)
    if vmin is None:
        vmin = float(spatial_map.min())
    if vmax is None:
        vmax = float(spatial_map.max())
    im = ax.imshow(spatial_map, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=min(16, max(10, int(target_h / 20))), fontweight='bold')
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=max(8, int(target_h / 30)))
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((fh, fw, 4))[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _is_p26_model(model):
    """Check if model is P26 (has P26-specific spatial buffers)."""
    core = model.module if hasattr(model, 'module') else model
    return (hasattr(core, '_last_uamm_spatial') and
            hasattr(core, '_last_per_modal_outputs') and
            hasattr(core, '_last_quality_maps'))


def _build_p26_per_modal_pred_row(model, batch_idx, modals, dataset_cls, palette,
                                   orig_h, orig_w, target_h, main_w):
    """Row: per-modality prediction masks (decoded to color)."""
    core = model.module if hasattr(model, 'module') else model
    per_modal_outputs = getattr(core, '_last_per_modal_outputs', None)
    if per_modal_outputs is None:
        return None
    m = len(per_modal_outputs)
    col_w = main_w // m
    cols = []
    for i in range(m):
        logits = per_modal_outputs[i]  # (B, C, H, W)
        if batch_idx >= logits.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        pred_i = logits[batch_idx].argmax(dim=0)  # (H, W)
        pred_resized = _unpad_resize_to_orig(pred_i, orig_h, orig_w, model_size=pred_i.shape[0])
        pred_np = pred_resized.cpu().numpy().astype(np.uint8)
        colored = dataset_cls.decode_segmap(pred_np, palette)
        colored_resized = np.array(Image.fromarray(colored).resize((col_w, target_h), Image.Resampling.LANCZOS))
        mname = modals[i] if i < len(modals) else f'M{i}'
        cols.append(_add_title_to_image(colored_resized, f'Pred: {MODAL_TITLES.get(mname, mname)}'))
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, row.shape[0]), Image.Resampling.LANCZOS))
    return row


def _build_p26_sqg_row(model, batch_idx, modals, target_h, main_w):
    """Row: SQG quality maps per modality as heatmaps."""
    core = model.module if hasattr(model, 'module') else model
    quality_maps = getattr(core, '_last_quality_maps', None)
    if quality_maps is None:
        return None
    m = len(quality_maps)
    col_w = main_w // m
    cols = []
    for i in range(m):
        qm = quality_maps[i]  # (B, 1, H_fpn, W_fpn)
        if batch_idx >= qm.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        qm_2d = qm[batch_idx, 0]  # (H_fpn, W_fpn)
        mname = modals[i] if i < len(modals) else f'M{i}'
        hm = _spatial_map_to_heatmap_with_colorbar(
            qm_2d, f'SQG: {MODAL_TITLES.get(mname, mname)}',
            target_h, col_w, cmap='viridis', vmin=0.0, vmax=1.0,
        )
        cols.append(hm)
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_p26_entropy_row(model, batch_idx, modals, target_h, main_w):
    """Row: per-modality entropy maps as heatmaps."""
    core = model.module if hasattr(model, 'module') else model
    entropy_maps = getattr(core, '_last_entropy_maps', None)
    if entropy_maps is None:
        return None
    m = len(entropy_maps)
    col_w = main_w // m
    # Find global max entropy for consistent scale
    vmax = max(em[batch_idx].max() for em in entropy_maps if batch_idx < em.shape[0])
    cols = []
    for i in range(m):
        em = entropy_maps[i]  # (B, 1, H, W)
        if batch_idx >= em.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        em_2d = em[batch_idx, 0]  # (H, W)
        mname = modals[i] if i < len(modals) else f'M{i}'
        hm = _spatial_map_to_heatmap_with_colorbar(
            em_2d, f'Entropy: {MODAL_TITLES.get(mname, mname)}',
            target_h, col_w, cmap='hot', vmin=0.0, vmax=float(vmax),
        )
        cols.append(hm)
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_p26_uamm_row(model, batch_idx, modals, target_h, main_w):
    """Row: UAMM spatial weight maps per modality as heatmaps."""
    core = model.module if hasattr(model, 'module') else model
    uamm_spatial = getattr(core, '_last_uamm_spatial', None)
    if uamm_spatial is None:
        return None
    m = len(uamm_spatial)
    col_w = main_w // m
    cols = []
    for i in range(m):
        us = uamm_spatial[i]  # (B, 1, H_fpn, W_fpn)
        if batch_idx >= us.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        us_2d = us[batch_idx, 0]  # (H_fpn, W_fpn)
        mname = modals[i] if i < len(modals) else f'M{i}'
        hm = _spatial_map_to_heatmap_with_colorbar(
            us_2d, f'UAMM Wt: {MODAL_TITLES.get(mname, mname)}',
            target_h, col_w, cmap='plasma', vmin=0.0, vmax=1.0,
        )
        cols.append(hm)
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_p26_amf_row(model, batch_idx, modals, target_h, main_w):
    """Row: AMF spatial fusion weight maps per modality as heatmaps."""
    core = model.module if hasattr(model, 'module') else model
    amf_spatial = getattr(core, '_last_amf_spatial', None)
    if amf_spatial is None:
        return None
    m = len(amf_spatial)
    col_w = main_w // m
    cols = []
    for i in range(m):
        af = amf_spatial[i]  # (B, 1, H, W)
        if batch_idx >= af.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        af_2d = af[batch_idx, 0]  # (H, W)
        mname = modals[i] if i < len(modals) else f'M{i}'
        hm = _spatial_map_to_heatmap_with_colorbar(
            af_2d, f'AMF Wt: {MODAL_TITLES.get(mname, mname)}',
            target_h, col_w, cmap='coolwarm', vmin=0.0, vmax=1.0,
        )
        cols.append(hm)
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_p26_feature_comparison_row(model, batch_idx, modals, target_h, main_w):
    """Row: per-modality backbone features (channel-mean) + fused feature comparison.
    Shows pre-UAMM features, UAMM weights, and the weighted (post-UAMM) result."""
    core = model.module if hasattr(model, 'module') else model
    per_modal_feats = getattr(core, '_last_per_modal_feats', None)
    uamm_spatial = getattr(core, '_last_uamm_spatial', None)
    if per_modal_feats is None or uamm_spatial is None:
        return None
    m = len(per_modal_feats)
    # Layout: [Pre-UAMM feat M0 | Pre-UAMM feat M1 | ... | Fused feat]
    n_cols = m + 1
    col_w = main_w // n_cols
    cols = []
    fused_feat = None
    for i in range(m):
        feat = per_modal_feats[i]  # (B, C, H_fpn, W_fpn)
        if batch_idx >= feat.shape[0]:
            cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        feat_2d = feat[batch_idx].mean(dim=0).numpy()  # (H_fpn, W_fpn) channel-mean
        # Compute weighted version for fused
        us = uamm_spatial[i][batch_idx, 0]  # (H_fpn, W_fpn)
        weighted = feat[batch_idx].numpy() * uamm_spatial[i][batch_idx]  # (C, H_fpn, W_fpn)
        if fused_feat is None:
            fused_feat = weighted.copy()
        else:
            fused_feat += weighted
        mname = modals[i] if i < len(modals) else f'M{i}'
        hm = _spatial_map_to_heatmap_with_colorbar(
            feat_2d, f'Feat: {MODAL_TITLES.get(mname, mname)}',
            target_h, col_w, cmap='magma',
        )
        cols.append(hm)
    # Fused feature
    if fused_feat is not None:
        fused_2d = fused_feat.mean(axis=0)  # (H_fpn, W_fpn)
        hm_fused = _spatial_map_to_heatmap_with_colorbar(
            fused_2d, 'Fused (post-UAMM)',
            target_h, col_w, cmap='magma',
        )
        cols.append(hm_fused)
    else:
        cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, row.shape[0]), Image.Resampling.LANCZOS))
    return row


def _spatial_map_to_heatmap_fixed_range(data_2d, title, target_h, target_w,
                                         cmap='magma', vmin=None, vmax=None):
    """Heatmap with fixed vmin/vmax range (shared across panels)."""
    dpi = 100
    fig_w = max(target_w / dpi, 1.5)
    fig_h = max(target_h / dpi, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=max(8, int(target_h / 30)), color='white', pad=4)
    im = ax.imshow(data_2d, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=max(8, int(target_h / 30)))
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((fh, fw, 4))[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


def _build_p9_uamm_feature_row(model, batch_idx, modals, target_h, main_w):
    """Row: UAMM 전/후 feature 비교 (P9/P22 전용).

    P9 UAMM은 scalar multiplication이므로 패턴은 동일하고 스케일만 다름.
    → 동일 vmin/vmax로 비교 + 증폭된 diff(×10) + UAMM score 표시.

    Layout: [Before M0 | After M0 | Diff×10 M0 | Before M1 | After M1 | Diff×10 M1 | ...]
    """
    core = model.module if hasattr(model, 'module') else model
    feats_before = getattr(core, '_last_feats_before_uamm', None)
    feats_after = getattr(core, '_last_feats_after_uamm', None)
    if feats_before is None or feats_after is None:
        return None

    uamm_scores = getattr(core, '_last_uamm_scores', None)
    m = len(feats_before)
    n_cols = m * 3
    col_w = main_w // n_cols
    cols = []

    # First pass: extract all 2D features to compute shared range
    feat_pairs = []
    for i in range(m):
        before_t = feats_before[i]
        after_t = feats_after[i]
        if before_t.dim() == 3:
            b_feat = before_t[:, batch_idx, :].mean(dim=-1).numpy()
            a_feat = after_t[:, batch_idx, :].mean(dim=-1).numpy()
            side = int(np.sqrt(b_feat.shape[0]))
            if side * side == b_feat.shape[0]:
                b_feat = b_feat.reshape(side, side)
                a_feat = a_feat.reshape(side, side)
            else:
                b_feat = b_feat.reshape(1, -1)
                a_feat = a_feat.reshape(1, -1)
        elif before_t.dim() == 4:
            b_feat = before_t[batch_idx].mean(dim=0).numpy()
            a_feat = after_t[batch_idx].mean(dim=0).numpy()
        else:
            feat_pairs.append(None)
            continue
        feat_pairs.append((b_feat, a_feat))

    all_vals = [v for pair in feat_pairs if pair is not None
                for v in [pair[0].ravel(), pair[1].ravel()]]
    if not all_vals:
        return None
    all_vals = np.concatenate(all_vals)
    shared_vmin, shared_vmax = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))

    DIFF_AMP = 10
    for i in range(m):
        if feat_pairs[i] is None:
            for _ in range(3):
                cols.append(np.zeros((target_h, col_w, 3), dtype=np.uint8))
            continue
        b_feat, a_feat = feat_pairs[i]
        diff = a_feat - b_feat
        mname = modals[i] if i < len(modals) else f'M{i}'
        mtitle = MODAL_TITLES.get(mname, mname)

        score_str = ''
        if uamm_scores is not None and batch_idx < uamm_scores.shape[0]:
            score_str = f' (s={uamm_scores[batch_idx, i]:.3f})'

        cols.append(_spatial_map_to_heatmap_fixed_range(
            b_feat, f'Before: {mtitle}', target_h, col_w,
            cmap='magma', vmin=shared_vmin, vmax=shared_vmax))
        cols.append(_spatial_map_to_heatmap_fixed_range(
            a_feat, f'After: {mtitle}{score_str}', target_h, col_w,
            cmap='magma', vmin=shared_vmin, vmax=shared_vmax))
        diff_amp = diff * DIFF_AMP
        d_abs_max = max(abs(float(diff_amp.min())), abs(float(diff_amp.max())), 1e-8)
        cols.append(_spatial_map_to_heatmap_fixed_range(
            diff_amp, f'Diff×{DIFF_AMP}: {mtitle}', target_h, col_w,
            cmap='RdBu_r', vmin=-d_abs_max, vmax=d_abs_max))

    row = np.concatenate(cols, axis=1)
    if row.shape[1] != main_w:
        row = np.array(Image.fromarray(row).resize((main_w, target_h), Image.Resampling.LANCZOS))
    return row


def _build_fusion_viz_rows(model, batch_idx, modals, dataset_cls, palette,
                            orig_h, orig_w, main_w):
    """Build fusion visualization rows for any model (P9/P22/P26/etc).

    Rows are added only when corresponding model buffers exist.
    """
    rows = []
    hmap_h = int(orig_h * 0.6)

    # Per-modality predictions (P9, P26 — any model with _last_per_modal_outputs)
    row = _build_p26_per_modal_pred_row(model, batch_idx, modals, dataset_cls, palette,
                                         orig_h, orig_w, orig_h, main_w)
    if row is not None:
        rows.append(row)

    # P26-specific rows (skipped silently when buffers absent)
    row = _build_p26_sqg_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)
    row = _build_p26_entropy_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)
    row = _build_p26_uamm_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)
    row = _build_p26_amf_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)
    row = _build_p26_feature_comparison_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)

    # P9/P22 UAMM before/after feature comparison
    row = _build_p9_uamm_feature_row(model, batch_idx, modals, hmap_h, main_w)
    if row is not None:
        rows.append(row)

    return rows


# ============================================================================
# Collate Function
# ============================================================================

def _collate_fn(batch):
    """Collate for (sample, label, meta) batches."""
    samples = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    images = [torch.stack([s[i] for s in samples]) for i in range(len(samples[0]))]
    labels = torch.stack(labels)
    return images, labels, metas


# ============================================================================
# Evaluate (val with GT)
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, macvi_format=False, modals=None,
             gamma_list=None, detailed=False, tta_flip=False):
    model.eval()
    ds = dataloader.dataset
    dataset_cls = _get_dataset_class(ds)
    n_classes = ds.n_classes
    palette = ds.PALETTE
    classes = ds.CLASSES
    metrics = Metrics(n_classes, ds.ignore_label, device)

    total_inference_time = 0.0
    num_frames = 0
    use_gamma = gamma_list is not None and len(gamma_list) > 0

    if save_dir:
        save_dir = Path(save_dir)
        if macvi_format:
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            (save_dir / "seg").mkdir(parents=True, exist_ok=True)
            (save_dir / "seg_viz").mkdir(parents=True, exist_ok=True)

    modals = modals or getattr(ds, 'modals', ['img'])
    log_data = {}
    core = model.module if hasattr(model, 'module') else model

    desc = f"Val (gamma TTA x{len(gamma_list)})" if use_gamma else "Val"
    for images, labels, metas in tqdm(dataloader, desc=desc):
        images = [x.to(device) for x in images]

        # Detailed mode: register MoE routing hooks
        capture = None
        if detailed and hasattr(core, 'moe_layers_q'):
            capture = MoERoutingCapture(core, viz_block_indices=REPRESENTATIVE_LAYERS)
            capture.register_hooks()
            capture.register_counter_hook()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        output, _ = model(images, multimask_output=True)

        if capture:
            capture.remove_hooks()

        # TTA
        if detailed and tta_flip:
            baseline_preds = _tta_accumulate(model, images, output, tta_flip=True)
        else:
            baseline_preds = output.softmax(dim=1)

        if use_gamma:
            preds = _gamma_tta_forward(model, images, gamma_list)
        else:
            preds = baseline_preds

        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0
        num_frames += images[0].shape[0]

        pred_labels = preds[:, :n_classes].argmax(dim=1)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            orig_label = meta.get("orig_label")
            pred_b = pred_labels[b]

            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])

            # Metrics (original size)
            if orig_label is not None:
                pred_softmax_orig = F.one_hot(
                    pred_resized.long().clamp(0, n_classes - 1), n_classes
                ).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                metrics.update(pred_softmax_orig, orig_label.unsqueeze(0).to(device))

            if save_dir:
                stem = meta["stem"]
                pred_np = pred_resized.cpu().numpy().astype(np.uint8)

                if macvi_format:
                    # MACVi 1-indexed (MULTIAQUA only)
                    seg_save = (pred_np + 1).clip(1, n_classes).astype(np.uint8)
                    Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
                else:
                    Image.fromarray(pred_np).save(str(save_dir / "seg" / f"{stem}.png"))

                    # Visualization
                    colored = dataset_cls.decode_segmap(pred_np, palette)
                    ignore_mask = orig_label.cpu().numpy() == ds.ignore_label if orig_label is not None else None
                    if ignore_mask is not None:
                        colored[ignore_mask] = [30, 30, 30]

                    raw_modals = [_load_modality_image(ds, mk, meta, orig_h, orig_w) for mk in modals]
                    rgb = raw_modals[0]
                    if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                        rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                    overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
                    if ignore_mask is not None:
                        overlay[ignore_mask] = [30, 30, 30]

                    if detailed:
                        # Detailed viz: titled modalities + GT + Prediction + Overlay + routing
                        modality_cols = [_add_title_to_image(img, MODAL_TITLES.get(mk, mk))
                                         for img, mk in zip(raw_modals, modals)]
                        row1 = np.concatenate(modality_cols, axis=1)
                        main_w = row1.shape[1]

                        gt_colored = dataset_cls.decode_segmap(
                            orig_label.cpu().numpy().astype(np.uint8), palette) if orig_label is not None else np.zeros_like(colored)
                        row2 = np.concatenate([
                            _add_title_to_image(gt_colored, 'GT'),
                            _add_title_to_image(colored, 'Prediction'),
                            _add_title_to_image(overlay, 'Overlay'),
                        ], axis=1)

                        rows = [row1, row2]

                        # Fusion visualization rows (P9/P22/P26 — auto-detected)
                        fusion_rows = _build_fusion_viz_rows(
                            model, b, modals, dataset_cls, palette,
                            orig_h, orig_w, main_w)
                        rows.extend(fusion_rows)

                        if capture:
                            stats_row = _build_stats_row(capture, modals, int(orig_h * 0.55), main_w)
                            mid_block = REPRESENTATIVE_LAYERS[len(REPRESENTATIVE_LAYERS) // 2]
                            map_row = _build_routing_map_row(capture, modals, int(orig_h * 0.6), main_w, block_idx=mid_block)
                            rows.extend([stats_row, map_row])

                        # Gamma comparison row
                        if use_gamma:
                            bl_b = baseline_preds[:, :n_classes].argmax(dim=1)[b]
                            bl_resized = _unpad_resize_to_orig(bl_b, orig_h, orig_w, model_size=bl_b.shape[0])
                            bl_np = bl_resized.cpu().numpy().astype(np.uint8)
                            bl_colored = dataset_cls.decode_segmap(bl_np, palette)
                            if ignore_mask is not None:
                                bl_colored[ignore_mask] = [30, 30, 30]
                            diff_mask = (bl_np != pred_np)
                            diff_img = rgb.copy()
                            diff_img[diff_mask] = [255, 50, 50]
                            diff_img[~diff_mask] = (diff_img[~diff_mask].astype(np.float32) * 0.3).astype(np.uint8)
                            if ignore_mask is not None:
                                diff_img[ignore_mask] = [30, 30, 30]
                            gamma_str = ",".join(f"{g:.1f}" for g in gamma_list)
                            row_gamma = np.concatenate([
                                _add_title_to_image(bl_colored, 'Baseline'),
                                _add_title_to_image(colored, f'Gamma TTA [{gamma_str}]'),
                                _add_title_to_image(diff_img, f'Diff ({diff_mask.sum()} px)'),
                            ], axis=1)
                            rows.append(row_gamma)

                        viz = np.concatenate(rows, axis=0)
                    else:
                        # Basic viz: Row1 [modalities], Row2 [legend|seg|overlay], Row3 [UAMM/AMF/MoE]
                        row1 = np.concatenate(raw_modals, axis=1)
                        legend_img = _draw_legend(classes, palette, orig_h, orig_w)
                        row2 = np.concatenate([legend_img, colored, overlay], axis=1)
                        viz = np.concatenate([row1, row2], axis=0)
                        viz_bottom = _get_uamm_amf_moe_viz(model, b, modals, viz.shape[0], viz.shape[1])
                        if viz_bottom is not None:
                            viz = np.concatenate([viz, viz_bottom], axis=0)

                    Image.fromarray(viz).save(str(save_dir / "seg_viz" / f"{stem}.png"))

                    # JSON logging
                    img_log = {}
                    if detailed:
                        # Detailed log: fusion + routing + per-class IoU + confidence
                        uamm = getattr(core, '_last_uamm_scores', None)
                        amf = getattr(core, '_last_amf_weights', None)
                        if uamm is not None and b < uamm.shape[0]:
                            img_log['uamm'] = {k: round(float(v), 4) for k, v in zip(modals, uamm[b])}
                        if amf is not None and b < amf.shape[0]:
                            img_log['amf'] = {k: round(float(v), 4) for k, v in zip(modals, amf[b])}

                        quality_maps = getattr(core, '_last_quality_maps', None)
                        if quality_maps is not None:
                            img_log['quality_gating'] = {}
                            for m_idx, m_name in enumerate(modals):
                                if m_idx < len(quality_maps):
                                    qm = quality_maps[m_idx]
                                    img_log['quality_gating'][m_name] = {
                                        'mean': round(float(qm.mean()), 4), 'std': round(float(qm.std()), 4),
                                        'min': round(float(qm.min()), 4), 'max': round(float(qm.max()), 4),
                                    }

                        # P26 spatial stats
                        uamm_sp = getattr(core, '_last_uamm_spatial', None)
                        if uamm_sp is not None:
                            img_log['uamm_spatial'] = {}
                            for m_idx, m_name in enumerate(modals):
                                if m_idx < len(uamm_sp) and b < uamm_sp[m_idx].shape[0]:
                                    us = uamm_sp[m_idx][b, 0]
                                    img_log['uamm_spatial'][m_name] = {
                                        'mean': round(float(us.mean()), 4), 'std': round(float(us.std()), 4),
                                        'min': round(float(us.min()), 4), 'max': round(float(us.max()), 4),
                                    }
                        amf_sp = getattr(core, '_last_amf_spatial', None)
                        if amf_sp is not None:
                            img_log['amf_spatial'] = {}
                            for m_idx, m_name in enumerate(modals):
                                if m_idx < len(amf_sp) and b < amf_sp[m_idx].shape[0]:
                                    af = amf_sp[m_idx][b, 0]
                                    img_log['amf_spatial'][m_name] = {
                                        'mean': round(float(af.mean()), 4), 'std': round(float(af.std()), 4),
                                        'min': round(float(af.min()), 4), 'max': round(float(af.max()), 4),
                                    }
                        ent_maps = getattr(core, '_last_entropy_maps', None)
                        if ent_maps is not None:
                            img_log['per_modal_entropy'] = {}
                            for m_idx, m_name in enumerate(modals):
                                if m_idx < len(ent_maps) and b < ent_maps[m_idx].shape[0]:
                                    em = ent_maps[m_idx][b, 0]
                                    img_log['per_modal_entropy'][m_name] = {
                                        'mean': round(float(em.mean()), 4), 'std': round(float(em.std()), 4),
                                        'max': round(float(em.max()), 4),
                                    }

                        # Per-class IoU
                        if orig_label is not None:
                            pred_np_i = pred_resized.cpu().numpy()
                            gt_np_i = orig_label.cpu().numpy()
                            per_class_iou = {}
                            for c_idx, c_name in enumerate(classes):
                                pred_c = (pred_np_i == c_idx)
                                gt_c = (gt_np_i == c_idx)
                                inter = (pred_c & gt_c).sum()
                                union = (pred_c | gt_c).sum()
                                per_class_iou[c_name] = round(float(inter / (union + 1e-8)), 4) if union > 0 else None
                            img_log['per_class_iou'] = per_class_iou

                        # Prediction confidence
                        softmax_b = preds[b]
                        pred_entropy = -(softmax_b * (softmax_b + 1e-8).log()).sum(dim=0)
                        img_log['pred_confidence'] = {
                            'mean_entropy': round(float(pred_entropy.mean().cpu()), 4),
                            'max_entropy': round(float(pred_entropy.max().cpu()), 4),
                            'high_uncertainty_ratio': round(float((pred_entropy > 0.5).float().mean().cpu()), 4),
                        }

                        if capture:
                            img_log['moe_routing'] = capture.get_log_dict(modals)
                            img_log['moe_summary'] = capture.get_summary_stats(modals)
                    else:
                        img_log = _get_uamm_amf_moe_log(model, b, modals)

                    if img_log:
                        log_data[stem] = img_log

    # Save JSON log
    if save_dir and not macvi_format and log_data:
        log_name = "detailed_log.json" if detailed else "uamm_amf_moe_log.json"
        log_path = save_dir / log_name
        meta_info = {"modals": modals, "split": "val", "n_images": len(log_data),
                     "dataset": type(ds).__name__}
        if detailed and hasattr(core, 'moe_layers_q'):
            meta_info.update({
                "num_moe_blocks_q": len(core.moe_layers_q),
                "num_moe_blocks_v": len(core.moe_layers_v),
                "viz_blocks": REPRESENTATIVE_LAYERS,
                "lora_model": core.__class__.__name__,
            })
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({"meta": meta_info, "images": log_data}, f, indent=2, ensure_ascii=False)
        print(f"Log saved to {log_path}")

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    fps = num_frames / total_inference_time if total_inference_time > 0 else 0.0
    return acc, macc, f1, mf1, ious, miou, fps


# ============================================================================
# Test Inference (no GT or with GT depending on dataset)
# ============================================================================

@torch.no_grad()
def run_test_inference(model, dataloader, device, save_dir, macvi_format=False, modals=None,
                       gamma_list=None, detailed=False, tta_flip=False, has_gt=False):
    model.eval()
    ds = dataloader.dataset
    dataset_cls = _get_dataset_class(ds)
    n_classes = ds.n_classes
    palette = ds.PALETTE
    classes = ds.CLASSES
    use_gamma = gamma_list is not None and len(gamma_list) > 0

    metrics = Metrics(n_classes, ds.ignore_label, device) if has_gt else None

    save_dir = Path(save_dir)
    if macvi_format:
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        (save_dir / "seg").mkdir(parents=True, exist_ok=True)
        (save_dir / "seg_viz").mkdir(parents=True, exist_ok=True)

    modals = modals or getattr(ds, 'modals', ['img'])
    log_data = {}
    core = model.module if hasattr(model, 'module') else model
    idx = 0
    total_inference_time = 0.0

    desc = f"Test (gamma TTA x{len(gamma_list)})" if use_gamma else "Test inference"
    for images, labels, metas in tqdm(dataloader, desc=desc):
        images = [x.to(device) for x in images]

        capture = None
        if detailed and hasattr(core, 'moe_layers_q'):
            capture = MoERoutingCapture(core, viz_block_indices=REPRESENTATIVE_LAYERS)
            capture.register_hooks()
            capture.register_counter_hook()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, _ = model(images, multimask_output=True)
        if capture:
            capture.remove_hooks()

        if detailed and tta_flip:
            baseline_preds = _tta_accumulate(model, images, output, tta_flip=True)
        else:
            baseline_preds = output.softmax(dim=1)

        if use_gamma:
            preds = _gamma_tta_forward(model, images, gamma_list)
        else:
            preds = baseline_preds

        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0

        pred_labels = preds[:, :n_classes].argmax(dim=1)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            stem, orig_h, orig_w = meta["stem"], meta["orig_h"], meta["orig_w"]
            orig_label = meta.get("orig_label")
            pred_b = pred_labels[b]
            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])
            pred_np = pred_resized.cpu().numpy().astype(np.uint8)

            # Metrics if GT available
            if has_gt and orig_label is not None and metrics is not None:
                pred_softmax_orig = F.one_hot(
                    pred_resized.long().clamp(0, n_classes - 1), n_classes
                ).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                metrics.update(pred_softmax_orig, orig_label.unsqueeze(0).to(device))

            if macvi_format:
                seg_save = (pred_np + 1).clip(1, n_classes).astype(np.uint8)
                Image.fromarray(seg_save).save(str(save_dir / f"{stem}.png"))
            else:
                Image.fromarray(pred_np).save(str(save_dir / "seg" / f"{stem}.png"))
                colored = dataset_cls.decode_segmap(pred_np, palette)
                ignore_mask = orig_label.cpu().numpy() == ds.ignore_label if orig_label is not None else None
                raw_modals = [_load_modality_image(ds, mk, meta, orig_h, orig_w) for mk in modals]
                rgb = raw_modals[0]
                if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                    rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

                if detailed:
                    if ignore_mask is not None:
                        colored[ignore_mask] = [30, 30, 30]
                        overlay[ignore_mask] = [30, 30, 30]
                    modality_cols = [_add_title_to_image(img, MODAL_TITLES.get(mk, mk))
                                     for img, mk in zip(raw_modals, modals)]
                    row1 = np.concatenate(modality_cols, axis=1)
                    main_w = row1.shape[1]

                    # Row 2: GT (if available) or Legend + Prediction + Overlay
                    if has_gt and orig_label is not None:
                        gt_colored = dataset_cls.decode_segmap(
                            orig_label.cpu().numpy().astype(np.uint8), palette)
                        row2 = np.concatenate([
                            _add_title_to_image(gt_colored, 'GT'),
                            _add_title_to_image(colored, 'Prediction'),
                            _add_title_to_image(overlay, 'Overlay'),
                        ], axis=1)
                    else:
                        legend_img = _draw_legend(classes, palette, orig_h, orig_w)
                        row2 = np.concatenate([
                            _add_title_to_image(legend_img, 'Legend'),
                            _add_title_to_image(colored, 'Prediction'),
                            _add_title_to_image(overlay, 'Overlay'),
                        ], axis=1)

                    rows = [row1, row2]

                    # Fusion visualization rows (P9/P22/P26 — auto-detected)
                    fusion_rows = _build_fusion_viz_rows(
                        model, b, modals, dataset_cls, palette,
                        orig_h, orig_w, main_w)
                    rows.extend(fusion_rows)

                    if capture:
                        rows.append(_build_stats_row(capture, modals, int(orig_h * 0.55), main_w))
                        mid_block = REPRESENTATIVE_LAYERS[len(REPRESENTATIVE_LAYERS) // 2]
                        rows.append(_build_routing_map_row(capture, modals, int(orig_h * 0.6), main_w, block_idx=mid_block))
                    viz = np.concatenate(rows, axis=0)
                else:
                    if ignore_mask is not None:
                        colored[ignore_mask] = [30, 30, 30]
                    row1 = np.concatenate(raw_modals, axis=1)
                    legend_img = _draw_legend(classes, palette, orig_h, orig_w)
                    row2 = np.concatenate([legend_img, colored, overlay], axis=1)
                    viz = np.concatenate([row1, row2], axis=0)
                    viz_bottom = _get_uamm_amf_moe_viz(model, b, modals, viz.shape[0], viz.shape[1])
                    if viz_bottom is not None:
                        viz = np.concatenate([viz, viz_bottom], axis=0)

                Image.fromarray(viz).save(str(save_dir / "seg_viz" / f"{stem}.png"))

                # JSON log
                img_log = {}
                if detailed:
                    uamm = getattr(core, '_last_uamm_scores', None)
                    amf = getattr(core, '_last_amf_weights', None)
                    if uamm is not None and b < uamm.shape[0]:
                        img_log['uamm'] = {k: round(float(v), 4) for k, v in zip(modals, uamm[b])}
                    if amf is not None and b < amf.shape[0]:
                        img_log['amf'] = {k: round(float(v), 4) for k, v in zip(modals, amf[b])}

                    # P26 spatial stats
                    uamm_sp = getattr(core, '_last_uamm_spatial', None)
                    if uamm_sp is not None:
                        img_log['uamm_spatial'] = {}
                        for m_idx, m_name in enumerate(modals):
                            if m_idx < len(uamm_sp) and b < uamm_sp[m_idx].shape[0]:
                                us = uamm_sp[m_idx][b, 0]
                                img_log['uamm_spatial'][m_name] = {
                                    'mean': round(float(us.mean()), 4), 'std': round(float(us.std()), 4),
                                    'min': round(float(us.min()), 4), 'max': round(float(us.max()), 4),
                                }
                    amf_sp = getattr(core, '_last_amf_spatial', None)
                    if amf_sp is not None:
                        img_log['amf_spatial'] = {}
                        for m_idx, m_name in enumerate(modals):
                            if m_idx < len(amf_sp) and b < amf_sp[m_idx].shape[0]:
                                af = amf_sp[m_idx][b, 0]
                                img_log['amf_spatial'][m_name] = {
                                    'mean': round(float(af.mean()), 4), 'std': round(float(af.std()), 4),
                                    'min': round(float(af.min()), 4), 'max': round(float(af.max()), 4),
                                }
                    ent_maps = getattr(core, '_last_entropy_maps', None)
                    if ent_maps is not None:
                        img_log['per_modal_entropy'] = {}
                        for m_idx, m_name in enumerate(modals):
                            if m_idx < len(ent_maps) and b < ent_maps[m_idx].shape[0]:
                                em = ent_maps[m_idx][b, 0]
                                img_log['per_modal_entropy'][m_name] = {
                                    'mean': round(float(em.mean()), 4), 'std': round(float(em.std()), 4),
                                    'max': round(float(em.max()), 4),
                                }

                    softmax_b = preds[b]
                    pred_entropy = -(softmax_b * (softmax_b + 1e-8).log()).sum(dim=0)
                    img_log['pred_confidence'] = {
                        'mean_entropy': round(float(pred_entropy.mean().cpu()), 4),
                        'max_entropy': round(float(pred_entropy.max().cpu()), 4),
                        'high_uncertainty_ratio': round(float((pred_entropy > 0.5).float().mean().cpu()), 4),
                    }
                    if capture:
                        img_log['moe_routing'] = capture.get_log_dict(modals)
                        img_log['moe_summary'] = capture.get_summary_stats(modals)
                else:
                    img_log = _get_uamm_amf_moe_log(model, b, modals)

                if img_log:
                    log_data[stem] = img_log

            idx += 1

    # Save JSON log
    if not macvi_format and log_data:
        log_name = "detailed_log.json" if detailed else "uamm_amf_moe_log.json"
        log_path = save_dir / log_name
        meta_info = {"modals": modals, "split": "test", "n_images": len(log_data),
                     "dataset": type(ds).__name__}
        if detailed and hasattr(core, 'moe_layers_q'):
            meta_info.update({
                "num_moe_blocks_q": len(core.moe_layers_q),
                "num_moe_blocks_v": len(core.moe_layers_v),
                "viz_blocks": REPRESENTATIVE_LAYERS,
                "lora_model": core.__class__.__name__,
            })
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({"meta": meta_info, "images": log_data}, f, indent=2, ensure_ascii=False)
        print(f"Log saved to {log_path}")

    fps = idx / total_inference_time if total_inference_time > 0 else 0.0
    if macvi_format:
        print(f"Saved {idx} masks to {save_dir} (MaCVi 1-indexed)")
    else:
        print(f"Saved {idx} predictions to {save_dir}")
    print(f"FPS: {fps:.2f}")

    # Return metrics if GT available
    if has_gt and metrics is not None:
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()
        return acc, macc, f1, mf1, ious, miou, fps
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified validation script (MULTIAQUA, DELIVER, ...)')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--detailed', action='store_true',
                        help='Enable detailed MoE routing analysis + extended viz')
    parser.add_argument('--macvi', action='store_true',
                        help='MACVi submission format (MULTIAQUA only, 1-indexed)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable horizontal flip TTA (detailed mode only)')
    parser.add_argument('--test_gamma', type=float, default=None,
                        help='Single gamma correction (e.g., 2.0 = brighten)')
    parser.add_argument('--test_gamma_tta', type=str, default=None,
                        help='Multi-gamma TTA: comma-separated (e.g., "1.0,1.5,2.0")')
    parser.add_argument('--eval_day', action='store_true',
                        help='MULTIAQUA test: use zed_day RGB')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                        help='Representative block indices for routing map (default: 0, 9, 18)')
    args = parser.parse_args()

    if args.blocks:
        global REPRESENTATIVE_LAYERS
        REPRESENTATIVE_LAYERS = args.blocks

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
    dataset_name = dataset_cfg.get('NAME', 'MULTIAQUA').upper()

    image_size = eval_cfg['IMAGE_SIZE'] if args.mode == 'val' else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    eval_day = (args.mode == 'test') and (args.eval_day or bool(eval_cfg.get('EVAL_DAY', False)))

    dataset, has_gt = create_dataset(dataset_cfg, split, transform, args.mode,
                                     macvi=args.macvi, eval_day=eval_day)

    dataloader = DataLoader(
        dataset, batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=4, pin_memory=False, collate_fn=_collate_fn,
    )

    model = load_model(cfg, model_path, device)

    # Gamma TTA
    gamma_list = None
    if args.test_gamma_tta:
        gamma_list = [float(x.strip()) for x in args.test_gamma_tta.split(',')]
        print(f"Gamma TTA: {gamma_list}")
    elif args.test_gamma is not None:
        gamma_list = [args.test_gamma]
        print(f"Single gamma: {args.test_gamma}")

    tta_flip = args.tta and args.detailed
    if tta_flip:
        print("TTA: horizontal flip (2 passes/image)")
    if args.detailed:
        print(f"Detailed mode: routing blocks = {REPRESENTATIVE_LAYERS}")

    # Checkpoint prefix for save dir naming
    lora_model_name = cfg['MODEL'].get('LORA_MODEL', 'LoRA_Sam_P9')
    short_name = lora_model_name.replace('LoRA_Sam_', '')
    ckpt_prefix = model_path.stem.replace("_checkpoint", "")
    tta_suffix = "_tta" if tta_flip else ""
    gamma_suffix = ""
    if gamma_list and gamma_list != [1.0]:
        gamma_suffix = "_gamma" + "_".join(f"{g:.1f}" for g in gamma_list)
    detailed_suffix = f"_{short_name}" if args.detailed else ""

    modals = dataset_cfg.get('MODALS')

    if args.mode == 'val':
        default_name = (f"{ckpt_prefix}_eval_macvi{gamma_suffix}" if args.macvi
                        else f"{ckpt_prefix}_val_pred{detailed_suffix}{tta_suffix}{gamma_suffix}")
        save_dir = args.save_dir or (model_path.parent / default_name)
        result = evaluate(
            model, dataloader, device, save_dir=save_dir,
            macvi_format=args.macvi, modals=modals,
            gamma_list=gamma_list, detailed=args.detailed, tta_flip=tta_flip,
        )
        acc, macc, f1, mf1, ious, miou, fps = result
        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.2f}" for iou in ious] + [f"{miou:.2f}"],
            'Acc': [f"{a:.2f}" for a in acc] + [f"{macc:.2f}"],
        }
        mode_tag = " [detailed]" if args.detailed else ""
        tta_tag = " [TTA-flip]" if tta_flip else ""
        gamma_tag = f" [Gamma {gamma_list}]" if gamma_list else ""
        print("\n" + "=" * 60)
        print(f"{dataset_name} {short_name} Validation ({len(dataset)} images){mode_tag}{tta_tag}{gamma_tag}")
        print("=" * 60)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU: {miou:.2f}  mAcc: {macc:.2f}  FPS: {fps:.2f}")
        if save_dir:
            print(f"Saved to {save_dir}")

        # Save results text
        out_txt = model_path.parent / f"eval_{split}_{ckpt_prefix}{detailed_suffix}{tta_suffix}{gamma_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(out_txt, 'w') as f:
            f.write(f"Model: {model_path}\nDataset: {dataset_name}\nSplit: {split}  N={len(dataset)}\n")
            if gamma_list:
                f.write(f"Gamma: {gamma_list}\n")
            f.write(tabulate(table, headers='keys') + "\n")
            f.write(f"\nmIoU: {miou:.2f}  mAcc: {macc:.2f}  FPS: {fps:.2f}\n")
        print(f"Results saved to {out_txt}")

    else:  # test
        default_name = (f"{ckpt_prefix}_eval_macvi{gamma_suffix}" if args.macvi
                        else f"{ckpt_prefix}_test_pred{detailed_suffix}{tta_suffix}{gamma_suffix}")
        save_dir = args.save_dir or (model_path.parent / default_name)
        result = run_test_inference(
            model, dataloader, device, save_dir,
            macvi_format=args.macvi, modals=modals,
            gamma_list=gamma_list, detailed=args.detailed, tta_flip=tta_flip,
            has_gt=has_gt,
        )
        # If GT available (e.g., DELIVER test), print metrics
        if result is not None:
            acc, macc, f1, mf1, ious, miou, fps = result
            table = {
                'Class': list(dataset.CLASSES) + ['Mean'],
                'IoU': [f"{iou:.2f}" for iou in ious] + [f"{miou:.2f}"],
                'Acc': [f"{a:.2f}" for a in acc] + [f"{macc:.2f}"],
            }
            print("\n" + "=" * 60)
            print(f"{dataset_name} {short_name} Test ({len(dataset)} images)")
            print("=" * 60)
            print(tabulate(table, headers='keys', tablefmt='grid'))
            print(f"\nmIoU: {miou:.2f}  mAcc: {macc:.2f}  FPS: {fps:.2f}")


if __name__ == '__main__':
    main()
