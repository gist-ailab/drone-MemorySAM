"""
MULTIAQUA P9 Validation & Per-Token MoE Routing Visualization
==============================================================

P9 전용 평가 + 시각화 스크립트. val_multiaqua.py 기반이지만 MoE 분석을 대폭 강화.

기존 `_last_moe_gates`가 공간 평균만 저장하여 per-token routing 다양성을 숨기는 문제를 해결.
Per-token entropy, argmax fraction, spatial routing map을 시각화.

저장 구조:
  save_dir/seg/         : raw segmentation (원본 크기)
  save_dir/seg_viz/     : 4-Row Layout
    Row 1: [RGB | Thermal | LiDAR]
    Row 2: [Legend | Segmentation | Overlay]
    Row 3: [UAMM Bar | AMF Bar | MoE Per-Token Stats]
    Row 4: [MoE Map (img) | MoE Map (lidar) | MoE Map (thermal)]
  save_dir/uamm_amf_moe_log.json : 확장된 per-token MoE 통계

사용:
  python val_multiaqua_P9.py --cfg configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml \\
      --mode val --model_path outputs/MMSamP9/.../epoch47_94.18.pth

NOTE: Use the MMSS_SAM conda environment.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from semseg.datasets import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import (
    LoRA_Sam_P9, LoRA_Sam_P10, LoRA_Sam_P11
)
from semseg.models.sam2.sam2.sam_lola_utils import SoftMoE_LoRA_Layer


# ============================================================================
# Model Loading (from val_multiaqua.py)
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

    _model_map = {
        'LoRA_Sam_P9': LoRA_Sam_P9,
        'LoRA_Sam_P10': LoRA_Sam_P10,
        'LoRA_Sam_P11': LoRA_Sam_P11,
    }
    lora_model_class = _model_map.get(lora_model_name)
    if lora_model_class is None:
        raise ValueError(f"Unknown LORA_MODEL: {lora_model_name}. Supported: {list(_model_map.keys())}")

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

    model = lora_model_class(**model_kwargs)
    ckpt = torch.load(str(model_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Model load: {msg}")
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Utility Functions (from val_multiaqua.py)
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


def _load_modality_image(dataset, modal_key, stem, target_h, target_w):
    if modal_key == 'img':
        path = dataset.rgb_dir / f"{stem}.png" if hasattr(dataset, 'rgb_dir') else None
    elif modal_key == 'lidar':
        path = dataset.lidar_dir / f"{stem}_lidar.png" if hasattr(dataset, 'lidar_dir') else None
    elif modal_key == 'thermal':
        path = dataset.thermal_dir / f"{stem}_thermal.png" if hasattr(dataset, 'thermal_dir') else None
    else:
        path = None
    if path is None or not path.exists():
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    img = np.array(Image.open(str(path)).convert("RGB"))
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return img


def _draw_bar_chart(values, labels, title, target_h, target_w=None):
    fig_w = target_w or max(320, target_h * 2)
    fig, ax = plt.subplots(figsize=(fig_w / 80, target_h / 80), dpi=80)
    n = len(values)
    y_pos = np.arange(n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))
    bars = ax.barh(y_pos, values, color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=22)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=24)
    ax.set_xlabel('Weight' if 'Fusion' in title else 'Score', fontsize=18)
    for i, (bar, val) in enumerate(zip(bars, values)):
        txt = f'{val:.3f}' if val < 0.01 or val >= 0.1 else f'{val:.2f}'
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, txt,
                va='center', ha='left', fontsize=18, fontweight='bold')
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    pil_img = Image.fromarray(img)
    out_w = target_w if target_w else int(img.shape[1] * (target_h / img.shape[0]))
    pil_img = pil_img.resize((out_w, target_h), Image.Resampling.LANCZOS)
    return np.array(pil_img)


# ============================================================================
# MoE Per-Token Routing Analysis
# ============================================================================

# Representative layers: early, mid, late
REPRESENTATIVE_LAYERS = [0, 9, 18]  # Block indices

# Expert colors for routing map
EXPERT_COLORS = np.array([
    [220, 50, 50],    # E0: Red
    [50, 180, 50],    # E1: Green
    [50, 80, 220],    # E2: Blue
    [220, 180, 50],   # E3: Yellow (if 4 experts)
], dtype=np.uint8)


class MoERoutingCapture:
    """Captures per-token MoE routing data during forward pass."""

    def __init__(self, model, target_block_indices=None):
        self.model = model
        self.target_blocks = target_block_indices or REPRESENTATIVE_LAYERS
        self.hooks = []
        self.call_counter = 0
        self.num_moe_layers = len(model.moe_layers_q) + len(model.moe_layers_v)
        self.num_layers_per_modal = self.num_moe_layers  # all hooks fire per modality

        # Storage: {block_idx: {modality_idx: {stats dict}}}
        self.routing_data = {}

        # Only hook Q layers at target block indices for spatial maps
        self._layer_to_block = {}
        for idx, layer in enumerate(model.moe_layers_q):
            self._layer_to_block[id(layer)] = idx

    def _get_modality_idx(self):
        """Determine which modality is being processed based on call count."""
        return self.call_counter // self.num_moe_layers

    def _get_layer_idx_in_modal(self):
        """Get layer index within the current modality's forward pass."""
        return self.call_counter % self.num_moe_layers

    def register_hooks(self):
        """Register forward hooks on target Q layers."""
        self.call_counter = 0
        self.routing_data = {}

        for block_idx in self.target_blocks:
            if block_idx < len(self.model.moe_layers_q):
                layer = self.model.moe_layers_q[block_idx]
                h = layer.register_forward_hook(self._make_hook(block_idx))
                self.hooks.append(h)

    def _make_hook(self, block_idx):
        def hook_fn(module, input, output):
            x = input[0]
            modal_idx = self._get_modality_idx()

            with torch.no_grad():
                gate_logits = module.gate(x)  # (..., E)
                gate_weights = F.softmax(gate_logits, dim=-1)  # (..., E)
                ne = module.num_experts

                # Per-token statistics
                per_token_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1)
                max_entropy = math.log(ne)
                per_token_max = gate_weights.max(dim=-1).values
                argmax = gate_weights.argmax(dim=-1)
                expert_counts = [(argmax == i).float().mean().item() for i in range(ne)]

                # Spatial routing map: reshape argmax to 2D
                # Input shape could be (B, H, W, E) or (B*nw, wh, ww, E)
                argmax_2d = argmax.cpu().numpy()

                if block_idx not in self.routing_data:
                    self.routing_data[block_idx] = {}

                self.routing_data[block_idx][modal_idx] = {
                    'entropy_ratio': per_token_entropy.mean().cpu().item() / max_entropy,
                    'per_token_max_mean': per_token_max.mean().cpu().item(),
                    'per_token_max_std': per_token_max.std().cpu().item(),
                    'argmax_fraction': expert_counts,
                    'spatial_mean': gate_weights.mean(
                        dim=tuple(range(gate_weights.dim() - 1))
                    ).cpu().numpy().tolist(),
                    'logit_range': (gate_logits.max(dim=-1).values - gate_logits.min(dim=-1).values).mean().cpu().item(),
                    # For spatial map: store argmax in original spatial shape
                    'argmax_map': argmax_2d,
                    'spatial_shape': list(argmax.shape),
                    'num_experts': ne,
                }
        return hook_fn

    def register_counter_hook(self):
        """Register a lightweight hook on ALL MoE layers just to count calls."""
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
        """Generate a colored spatial routing map for one block+modality."""
        if block_idx not in self.routing_data or modal_idx not in self.routing_data[block_idx]:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        data = self.routing_data[block_idx][modal_idx]
        argmax_map = data['argmax_map']  # Could be (B*nw, wh, ww) or (B, H, W)
        ne = data['num_experts']

        # Flatten all but last dims to get per-token expert assignment
        # For window-partitioned input: (B*nw, wh, ww) → collapse to approx spatial map
        shape = data['spatial_shape']

        if len(shape) == 3:
            # (B or B*nw, H, W) — take first batch/window group
            map_2d = argmax_map[0]  # (H, W)
        elif len(shape) == 2:
            # (N, ) — try to reshape to square
            n = shape[0]
            side = int(math.sqrt(n))
            if side * side == n:
                map_2d = argmax_map.reshape(side, side)
            else:
                map_2d = argmax_map.reshape(1, -1)
        else:
            map_2d = argmax_map.reshape(-1)[:1024].reshape(32, 32)

        # Color map
        h, w = map_2d.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for e in range(min(ne, len(EXPERT_COLORS))):
            mask = map_2d == e
            colored[mask] = EXPERT_COLORS[e]

        # Resize to target
        pil_img = Image.fromarray(colored).resize((target_w, target_h), Image.Resampling.NEAREST)
        return np.array(pil_img)

    def get_stats_bar_chart(self, block_idx, modals, target_h, target_w):
        """Draw per-token stats as a bar chart for one representative block."""
        if block_idx not in self.routing_data:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(target_w / 80, target_h / 80), dpi=80)

        # Left: Entropy ratio per modality
        ax = axes[0]
        er_values = []
        labels = []
        for m_idx, mname in enumerate(modals):
            if m_idx in self.routing_data[block_idx]:
                er_values.append(self.routing_data[block_idx][m_idx]['entropy_ratio'])
                labels.append(mname)
        if er_values:
            colors = ['#e74c3c', '#2ecc71', '#3498db'][:len(er_values)]
            bars = ax.barh(range(len(er_values)), er_values, color=colors, height=0.6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=16)
            ax.set_xlim(0, 1.1)
            ax.set_title(f'Entropy Ratio (B{block_idx})', fontsize=16)
            ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='uniform')
            for bar, val in zip(bars, er_values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=14, fontweight='bold')

        # Right: Argmax fraction (stacked horizontal bar)
        ax = axes[1]
        ne = 3
        for m_idx, mname in enumerate(modals):
            if m_idx in self.routing_data[block_idx]:
                fracs = self.routing_data[block_idx][m_idx]['argmax_fraction']
                ne = len(fracs)
                left = 0
                for e_idx, frac in enumerate(fracs):
                    c = EXPERT_COLORS[e_idx].astype(float) / 255.0
                    ax.barh(m_idx, frac, left=left, color=c, height=0.6, edgecolor='white', linewidth=0.5)
                    if frac > 0.08:
                        ax.text(left + frac / 2, m_idx, f'{frac:.0%}',
                                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
                    left += frac
        ax.set_yticks(range(len(modals)))
        ax.set_yticklabels(modals, fontsize=16)
        ax.set_xlim(0, 1.0)
        ax.set_title(f'Expert Selection (B{block_idx})', fontsize=16)
        # Legend
        for e in range(ne):
            ax.barh([], [], color=EXPERT_COLORS[e].astype(float) / 255.0, label=f'E{e}')
        ax.legend(fontsize=11, loc='lower right')

        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
        plt.close(fig)
        return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))

    def get_log_dict(self, modals):
        """Get JSON-serializable routing statistics."""
        log = {}
        for block_idx in sorted(self.routing_data.keys()):
            block_log = {}
            for m_idx in sorted(self.routing_data[block_idx].keys()):
                mname = modals[m_idx] if m_idx < len(modals) else f'M{m_idx}'
                d = self.routing_data[block_idx][m_idx]
                block_log[mname] = {
                    'entropy_ratio': round(d['entropy_ratio'], 4),
                    'per_token_max': round(d['per_token_max_mean'], 4),
                    'argmax_fraction': {f'E{i}': round(v, 4) for i, v in enumerate(d['argmax_fraction'])},
                    'spatial_mean': {f'E{i}': round(v, 4) for i, v in enumerate(d['spatial_mean'])},
                }
            log[f'Block{block_idx}_Q'] = block_log
        return log


# ============================================================================
# Visualization Row Builders
# ============================================================================

def build_uamm_amf_row(model, batch_idx, modals, main_h, main_w):
    """Row 3: [UAMM Bar | AMF Bar | MoE Stats Bar] — similar to original but with per-token MoE."""
    core = model.module if hasattr(model, 'module') else model
    uamm = getattr(core, '_last_uamm_scores', None)
    amf = getattr(core, '_last_amf_weights', None)

    viz_h = int(main_h * 0.55)
    chart_w = (main_w + 2) // 3
    modal_labels = modals

    strips = []
    if uamm is not None and batch_idx < uamm.shape[0]:
        strips.append(_draw_bar_chart(uamm[batch_idx], modal_labels, 'UAMM (Memory Mod)', viz_h, chart_w))
    if amf is not None and batch_idx < amf.shape[0]:
        strips.append(_draw_bar_chart(amf[batch_idx], modal_labels, 'AMF (Fusion)', viz_h, chart_w))

    # Third panel: use routing capture stats for representative block
    if len(strips) < 3:
        # Placeholder
        strips.append(np.ones((viz_h, chart_w, 3), dtype=np.uint8) * 240)

    bottom = np.concatenate(strips, axis=1)
    if bottom.shape[1] != main_w:
        bottom = np.array(Image.fromarray(bottom).resize((main_w, viz_h), Image.Resampling.LANCZOS))
    return bottom


def build_routing_map_row(capture, modals, target_h, target_w, block_idx=9):
    """Row 4: [MoE Map (modal0) | MoE Map (modal1) | MoE Map (modal2)]"""
    map_h = target_h
    col_w = target_w // len(modals)

    cols = []
    for m_idx, mname in enumerate(modals):
        routing_img = capture.get_routing_map_image(block_idx, m_idx, map_h, col_w)
        # Add title overlay
        titled = _add_title_to_image(routing_img, f'MoE Routing: {mname} (Block{block_idx})')
        cols.append(titled)

    row = np.concatenate(cols, axis=1)
    if row.shape[1] != target_w:
        row = np.array(Image.fromarray(row).resize((target_w, map_h), Image.Resampling.LANCZOS))
    return row


def _add_title_to_image(img, title):
    """Add a title bar on top of an image."""
    h, w = img.shape[:2]
    title_h = max(30, h // 15)
    title_bar = np.ones((title_h, w, 3), dtype=np.uint8) * 40  # Dark gray

    # Use matplotlib to render text
    fig, ax = plt.subplots(figsize=(w / 80, title_h / 80), dpi=80)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_facecolor('#282828')
    ax.text(0.5, 0.5, title, fontsize=min(16, int(title_h * 0.6)),
            color='white', ha='center', va='center', fontweight='bold')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    title_img = np.asarray(buf).reshape((fh, fw, 4))[:, :, :3].copy()
    plt.close(fig)
    title_bar = np.array(Image.fromarray(title_img).resize((w, title_h), Image.Resampling.LANCZOS))

    return np.concatenate([title_bar, img], axis=0)


def build_stats_row(capture, modals, target_h, target_w):
    """Row 3 alternative: per-block routing stats bars."""
    blocks = sorted(capture.routing_data.keys())
    if not blocks:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 240

    n_charts = min(len(blocks), 3)
    chart_w = target_w // n_charts
    strips = []
    for i in range(n_charts):
        block_idx = blocks[i]
        chart = capture.get_stats_bar_chart(block_idx, modals, target_h, chart_w)
        strips.append(chart)

    row = np.concatenate(strips, axis=1)
    if row.shape[1] != target_w:
        row = np.array(Image.fromarray(row).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return row


# ============================================================================
# Collate Function
# ============================================================================

def _collate_multiaqua(batch):
    samples = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    images = [torch.stack([s[i] for s in samples]) for i in range(len(samples[0]))]
    labels = torch.stack(labels)
    return images, labels, metas


# ============================================================================
# Evaluation Loop
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, save_dir=None, modals=None):
    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    total_inference_time = 0.0
    num_frames = 0

    if save_dir:
        save_dir = Path(save_dir)
        seg_dir = save_dir / "seg"
        seg_viz_dir = save_dir / "seg_viz"
        seg_dir.mkdir(parents=True, exist_ok=True)
        seg_viz_dir.mkdir(parents=True, exist_ok=True)

    modals = modals or ['img', 'lidar', 'thermal']
    uamm_amf_moe_log = {}

    core = model.module if hasattr(model, 'module') else model

    for images, labels, metas in tqdm(dataloader, desc="Val"):
        images = [x.to(device) for x in images]

        # Set up routing capture
        capture = MoERoutingCapture(core, target_block_indices=REPRESENTATIVE_LAYERS)
        capture.register_hooks()
        capture.register_counter_hook()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, _ = model(images, multimask_output=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0
        num_frames += images[0].shape[0]

        capture.remove_hooks()

        preds = output.softmax(dim=1)
        pred_labels = preds[:, :n_classes].argmax(dim=1)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            orig_label = meta["orig_label"]
            pred_b = pred_labels[b]

            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])
            pred_softmax_orig = F.one_hot(
                pred_resized.long().clamp(0, n_classes - 1), n_classes
            ).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
            metrics.update(pred_softmax_orig, orig_label.unsqueeze(0).to(device))

            if save_dir:
                stem = meta["stem"]
                pred_np = pred_resized.cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(str(seg_dir / f"{stem}.png"))

                # Build visualization
                colored = MULTIAQUA.decode_segmap(pred_np, palette)
                ds = dataloader.dataset

                # Row 1: modality images
                modality_cols = [_load_modality_image(ds, mk, stem, orig_h, orig_w) for mk in modals]
                rgb = modality_cols[0]
                if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                    rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

                classes = getattr(ds, 'CLASSES', MULTIAQUA.CLASSES)
                pal = getattr(ds, 'PALETTE', MULTIAQUA.PALETTE)
                legend_img = _draw_legend(classes, pal, orig_h, orig_w)

                row1 = np.concatenate(modality_cols, axis=1)
                row2 = np.concatenate([legend_img, colored, overlay], axis=1)
                main_w = row1.shape[1]

                # Row 3: UAMM + AMF + Per-Token Stats
                row3_h = int(orig_h * 0.55)
                row3_left = build_uamm_amf_row(model, b, modals, orig_h, main_w)

                # Replace the third bar with routing stats
                stats_row = build_stats_row(capture, modals, row3_left.shape[0], main_w)

                # Row 4: Spatial routing maps
                map_h = int(orig_h * 0.6)
                # Use the mid-level block for spatial map
                mid_block = REPRESENTATIVE_LAYERS[len(REPRESENTATIVE_LAYERS) // 2]
                row4 = build_routing_map_row(capture, modals, map_h, main_w, block_idx=mid_block)

                viz = np.concatenate([row1, row2, stats_row, row4], axis=0)
                Image.fromarray(viz).save(str(seg_viz_dir / f"{stem}.png"))

                # JSON log with per-token MoE stats
                img_log = {}
                uamm = getattr(core, '_last_uamm_scores', None)
                amf = getattr(core, '_last_amf_weights', None)
                if uamm is not None and b < uamm.shape[0]:
                    img_log['uamm'] = {k: round(float(v), 4) for k, v in zip(modals, uamm[b])}
                if amf is not None and b < amf.shape[0]:
                    img_log['amf'] = {k: round(float(v), 4) for k, v in zip(modals, amf[b])}

                # Per-token MoE routing stats
                img_log['moe_routing'] = capture.get_log_dict(modals)

                # Legacy: spatial mean for comparison
                moe = getattr(core, '_last_moe_gates', None)
                if moe is not None:
                    moe_arr = np.asarray(moe)
                    arr = moe_arr if moe_arr.ndim == 1 else moe_arr[b]
                    img_log['moe_spatial_mean'] = {f'E{i}': round(float(v), 4) for i, v in enumerate(arr)}

                uamm_amf_moe_log[stem] = img_log

    # Save JSON
    if save_dir and uamm_amf_moe_log:
        log_path = save_dir / "uamm_amf_moe_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "meta": {
                    "modals": modals,
                    "split": "val",
                    "n_images": len(uamm_amf_moe_log),
                    "representative_blocks": REPRESENTATIVE_LAYERS,
                    "note": "moe_routing shows per-token stats; moe_spatial_mean is the misleading average"
                },
                "images": uamm_amf_moe_log,
            }, f, indent=2, ensure_ascii=False)
        print(f"Enhanced MoE log saved to {log_path}")

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    dynamic_iou = float(ious[1])
    fps = num_frames / total_inference_time if total_inference_time > 0 else 0.0
    return acc, macc, f1, mf1, ious, miou, dynamic_iou, fps


@torch.no_grad()
def run_test_inference(model, dataloader, device, save_dir, modals=None):
    """Test inference with routing visualization."""
    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE

    save_dir = Path(save_dir)
    seg_dir = save_dir / "seg"
    seg_viz_dir = save_dir / "seg_viz"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_viz_dir.mkdir(parents=True, exist_ok=True)

    modals = modals or ['img', 'lidar', 'thermal']
    uamm_amf_moe_log = {}
    core = model.module if hasattr(model, 'module') else model
    idx = 0
    total_inference_time = 0.0

    for images, _, metas in tqdm(dataloader, desc="Test inference"):
        images = [x.to(device) for x in images]

        capture = MoERoutingCapture(core, target_block_indices=REPRESENTATIVE_LAYERS)
        capture.register_hooks()
        capture.register_counter_hook()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output, _ = model(images, multimask_output=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_inference_time += time.perf_counter() - t0

        capture.remove_hooks()

        preds = output.softmax(dim=1)
        pred_labels = preds[:, :n_classes].argmax(dim=1)

        for b in range(pred_labels.shape[0]):
            meta = metas[b]
            stem, orig_h, orig_w = meta["stem"], meta["orig_h"], meta["orig_w"]
            pred_b = pred_labels[b]
            pred_resized = _unpad_resize_to_orig(pred_b, orig_h, orig_w, model_size=pred_b.shape[0])
            pred_np = pred_resized.cpu().numpy().astype(np.uint8)

            Image.fromarray(pred_np).save(str(seg_dir / f"{stem}.png"))
            colored = MULTIAQUA.decode_segmap(pred_np, palette)
            ds = dataloader.dataset
            modality_cols = [_load_modality_image(ds, mk, stem, orig_h, orig_w) for mk in modals]
            rgb = modality_cols[0]
            if rgb.shape[0] != orig_h or rgb.shape[1] != orig_w:
                rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
            overlay = (rgb.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
            classes = getattr(ds, 'CLASSES', MULTIAQUA.CLASSES)
            pal = getattr(ds, 'PALETTE', MULTIAQUA.PALETTE)
            legend_img = _draw_legend(classes, pal, orig_h, orig_w)

            row1 = np.concatenate(modality_cols, axis=1)
            row2 = np.concatenate([legend_img, colored, overlay], axis=1)
            main_w = row1.shape[1]
            stats_row = build_stats_row(capture, modals, int(orig_h * 0.55), main_w)
            mid_block = REPRESENTATIVE_LAYERS[len(REPRESENTATIVE_LAYERS) // 2]
            row4 = build_routing_map_row(capture, modals, int(orig_h * 0.6), main_w, block_idx=mid_block)
            viz = np.concatenate([row1, row2, stats_row, row4], axis=0)
            Image.fromarray(viz).save(str(seg_viz_dir / f"{stem}.png"))

            # JSON log
            img_log = {}
            uamm = getattr(core, '_last_uamm_scores', None)
            amf = getattr(core, '_last_amf_weights', None)
            if uamm is not None and b < uamm.shape[0]:
                img_log['uamm'] = {k: round(float(v), 4) for k, v in zip(modals, uamm[b])}
            if amf is not None and b < amf.shape[0]:
                img_log['amf'] = {k: round(float(v), 4) for k, v in zip(modals, amf[b])}
            img_log['moe_routing'] = capture.get_log_dict(modals)
            moe = getattr(core, '_last_moe_gates', None)
            if moe is not None:
                moe_arr = np.asarray(moe)
                arr = moe_arr if moe_arr.ndim == 1 else moe_arr[b]
                img_log['moe_spatial_mean'] = {f'E{i}': round(float(v), 4) for i, v in enumerate(arr)}
            uamm_amf_moe_log[stem] = img_log
            idx += 1

    if uamm_amf_moe_log:
        log_path = save_dir / "uamm_amf_moe_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "meta": {"modals": modals, "split": "test", "n_images": len(uamm_amf_moe_log),
                         "representative_blocks": REPRESENTATIVE_LAYERS},
                "images": uamm_amf_moe_log,
            }, f, indent=2, ensure_ascii=False)
        print(f"Enhanced MoE log saved to {log_path}")

    fps = idx / total_inference_time if total_inference_time > 0 else 0.0
    print(f"Saved {idx} predictions: seg/ and seg_viz/ under {save_dir}")
    print(f"Inference FPS: {fps:.2f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
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

    image_size = eval_cfg['IMAGE_SIZE'] if args.mode == 'val' else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    dataset = MULTIAQUA(
        dataset_cfg['ROOT'], split=split, transform=transform,
        modals=dataset_cfg['MODALS'],
        require_annotation=(args.mode == 'val'),
        return_meta=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=4, pin_memory=False, collate_fn=_collate_multiaqua,
    )

    model = load_model(cfg, model_path, device)
    print(f"Representative blocks for routing map: {REPRESENTATIVE_LAYERS}")

    if args.mode == 'val':
        save_dir = args.save_dir or (model_path.parent / "val_pred_P9")
        acc, macc, f1, mf1, ious, miou, dynamic_iou, fps = evaluate(
            model, dataloader, device, save_dir=save_dir,
            modals=dataset_cfg.get('MODALS')
        )
        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.2f}" for iou in ious] + [f"{miou:.2f}"],
            'Acc': [f"{a:.2f}" for a in acc] + [f"{macc:.2f}"],
        }
        print("\n" + "=" * 60)
        print(f"MULTIAQUA P9 Validation ({len(dataset)} images)")
        print("=" * 60)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU: {miou:.2f}  mAcc: {macc:.2f}")
        print(f"Dynamic IoU: {dynamic_iou:.2f}")
        print(f"FPS: {fps:.2f}")
        if save_dir:
            print(f"Saved to {save_dir}")
    else:
        save_dir = args.save_dir or (model_path.parent / "test_pred_P9")
        run_test_inference(
            model, dataloader, device, save_dir=save_dir,
            modals=dataset_cfg.get('MODALS')
        )


if __name__ == '__main__':
    main()
