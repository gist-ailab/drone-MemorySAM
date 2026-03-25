#!/usr/bin/env python3
"""
viz_memory_attention.py — Memory Attention Cross-Modal Weight 시각화

SAM2 memory attention의 cross-attention weight를 이미지별로 시각화.
각 modality가 다른 modality의 memory에 어떻게 attend하는지 직관적으로 볼 수 있다.

============================================================================
사용법 (기본: 순수 attention 시각화, degradation 없음):
============================================================================

  # MMSamBase — Val (주간)
  python MISC/viz_memory_attention.py \
    --cfg configs/eval_config/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml \
    --model_path outputs/MMSamBase/.../best_checkpoint.pth \
    --mode val --model_name MMSamBase

  # MMSamBase — Test (야간)
  python MISC/viz_memory_attention.py \
    --cfg configs/eval_config/levine-multiaqua_rgbtl_LoRASam_hardaug4.yaml \
    --model_path outputs/MMSamBase/.../best_checkpoint.pth \
    --mode test --model_name MMSamBase

  # P9 — Val (주간)
  python MISC/viz_memory_attention.py \
    --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml \
    --model_path outputs/MMSamP9/.../epoch131_94.41_top1_checkpoint.pth \
    --mode val --model_name P9

  # P9 — Test (야간)
  python MISC/viz_memory_attention.py \
    --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml \
    --model_path outputs/MMSamP9/.../epoch131_94.41_top1_checkpoint.pth \
    --mode test --model_name P9

  # model_name 생략 시 config의 SAVE_DIR에서 자동 추출 (예: MMSamBase, MMSamP9)

  # 특정 이미지만 + Degradation 비교
  python MISC/viz_memory_attention.py ... --indices 0 5 10 --degradations dark_rgb zero_lidar

============================================================================
출력 구조:
============================================================================

  outputs/viz_memory_attn/
  ├── MMSamBase/
  │   ├── val/
  │   │   ├── per_image/000_<stem>.png ...
  │   │   ├── aggregate_summary.png
  │   │   └── aggregate_stats.json
  │   └── test/
  │       ├── per_image/000_<stem>.png ...
  │       ├── aggregate_summary.png
  │       └── aggregate_stats.json
  └── P9/
      ├── val/ ...
      └── test/ ...

============================================================================
결과 해석:
============================================================================

■ Per-image figure 레이아웃:
  Row 0: [RGB 입력] [LiDAR 입력] [Thermal 입력] [GT 라벨 (있으면)]
  Row 1: LiDAR(Q) → RGB memory attention, Layer 0~3
  Row 2: Thermal(Q) → RGB memory attention, Layer 0~3
  Row 3: Thermal(Q) → LiDAR memory attention, Layer 0~3
  Row 4: Preference map (빨강=RGB 선호, 파랑=LiDAR 선호)

■ Attention map 읽는 법:
  - 밝은 영역 = 해당 위치의 query가 해당 memory에 강하게 attend
  - mass = 해당 modality memory에 배정된 전체 attention 비율 (0~1)
    - mass=0.6 → query의 attention 60%가 해당 memory로 감
  - Preference map: 각 위치에서 RGB vs LiDAR 중 어디에 더 attend하는지
    - 빨강 = RGB 선호, 파랑 = LiDAR 선호, 흰색 = 균등

■ 핵심 관찰 포인트 (1단계 문제 증명):
  1. Val(주간)에서 attention 분포가 어떤가? (baseline)
  2. Test(야간)에서 어떻게 달라지는가?
  3. RGB가 어두울 때 모델이 LiDAR/Thermal memory로 shift하는가?
     → shift 안하면 = "열화된 modality에 여전히 attend" = 문제 존재 증명
  4. Layer별 차이: 보통 앞 layer는 균등, 뒤 layer는 더 discriminative
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.datasets.multiaqua import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation
from MISC.diagnose_memory_attention import (
    MemoryAttentionProbe,
    apply_degradation,
    load_model,
)


# ============================================================================
# Colormaps
# ============================================================================
# RGB(red) vs LiDAR(blue) preference
_PREF_COLORS = [(0.2, 0.4, 0.9), (0.95, 0.95, 0.95), (0.9, 0.2, 0.2)]
CMAP_PREF = LinearSegmentedColormap.from_list('img_vs_lidar', _PREF_COLORS, N=256)

# Label colormap (MULTIAQUA: Static=0, Dynamic=1, Water=2, Sky=3)
LABEL_COLORS = np.array([
    [128, 128, 128],  # Static — gray
    [255, 0, 0],      # Dynamic — red
    [0, 0, 255],      # Water — blue
    [135, 206, 235],  # Sky — sky blue
], dtype=np.uint8)


# ============================================================================
# Utilities
# ============================================================================

def _denormalize_image(tensor):
    """Normalized tensor (C,H,W) → displayable (H,W,3) numpy"""
    img = tensor.cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def _to_spatial(attn_flat, N_q):
    """(N_q,) flat attention → (H, W) 2D"""
    h = int(math.sqrt(N_q))
    w = N_q // h
    return attn_flat[:h * w].reshape(h, w)


def _upsample_map(attn_2d, target_h, target_w):
    """(h, w) attention map → (target_h, target_w) bilinear upsampled"""
    t = torch.from_numpy(attn_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def _label_to_color(label_tensor):
    """Label tensor (H,W) → RGB image (H,W,3)"""
    label = label_tensor.cpu().numpy().astype(np.int32)
    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(LABEL_COLORS)):
        mask = label == cls_id
        color[mask] = LABEL_COLORS[cls_id]
    # ignore (255) → black
    return color


# ============================================================================
# Attention extraction
# ============================================================================

def extract_attention_data(model, images_gpu, probe, modal_names):
    """
    Forward pass + attention weight 추출.

    Returns:
        attn_data: {
            frame_idx: {
                layer_idx: {
                    'per_mod': {mod_name: (N_q,) array},  # query-side: 각 query가 해당 mod에 보내는 attention 합
                    'per_mod_received': {mod_name: (N_mod_tokens,) array},  # key-side: 각 key가 받는 attention 평균
                    'obj_ptr': (N_q,) array or None,
                    'N_q': int,
                    'N_k': int,
                }
            }
        }
        output: model output (predictions)
    """
    m = len(modal_names)
    probe.clear()

    with torch.no_grad(), probe.probe_enabled():
        output, _ = model(images_gpu, multimask_output=True)

    num_layers = probe.num_layers
    attn_data = {}

    non_init_count = 0
    for frame_idx in range(m):
        mem_info = probe.memory_token_info.get(frame_idx, {})
        if mem_info.get('is_init', True) and frame_idx == 0:
            attn_data[frame_idx] = {'skip': True}
            continue

        token_counts = mem_info.get('token_counts', {})
        memory_frames = sorted(mem_info.get('memory_frames', []))

        frame_data = {}
        for layer_idx in range(num_layers):
            entry_idx = non_init_count * num_layers + layer_idx
            if entry_idx >= len(probe.cross_attn_weights):
                continue

            entry = probe.cross_attn_weights[entry_idx]
            attn_w = entry['weights']  # (B, H, N_q, N_k)
            B, H, N_q, N_k = attn_w.shape

            # Memory token boundaries
            boundaries = {}
            offset = 0
            for mf_idx in memory_frames:
                if mf_idx in token_counts:
                    n_tok = token_counts[mf_idx]
                    boundaries[mf_idx] = (offset, offset + n_tok)
                    offset += n_tok
            obj_ptr_range = (offset, N_k)

            # Per-modality query-side: sum attention to each modality's tokens → (N_q,)
            per_mod = {}
            # Per-modality key-side: mean attention received by each key token → (N_mod_tokens,)
            per_mod_received = {}
            for mf_idx, (start, end) in boundaries.items():
                if end <= start:
                    continue
                mod_name = modal_names[mf_idx] if mf_idx < len(modal_names) else f"mod{mf_idx}"
                mod_slice = attn_w[:, :, :, start:end]  # (B, H, N_q, N_mod)
                # Query-side: 각 query가 이 mod에 보내는 attention 합
                per_mod[mod_name] = mod_slice.sum(dim=-1).mean(dim=(0, 1)).numpy()  # (N_q,)
                # Key-side: 각 key 위치가 전체 query로부터 받는 attention 평균
                per_mod_received[mod_name] = mod_slice.mean(dim=(0, 1, 2)).numpy()  # (N_mod_tokens,)

            obj_ptr_map = None
            if obj_ptr_range[1] > obj_ptr_range[0]:
                obj_ptr_map = attn_w[:, :, :, obj_ptr_range[0]:obj_ptr_range[1]].sum(dim=-1).mean(dim=(0, 1)).numpy()

            frame_data[layer_idx] = {
                'per_mod': per_mod,
                'per_mod_received': per_mod_received,
                'obj_ptr': obj_ptr_map,
                'N_q': N_q,
                'N_k': N_k,
            }

        attn_data[frame_idx] = frame_data
        non_init_count += 1

    return attn_data, output


# ============================================================================
# Per-image figure
# ============================================================================

def create_per_image_figure(images_gpu, attn_data, modal_names,
                            label=None, image_idx=0, stem='', save_path=None,
                            attn_data_deg_dict=None, model_name='', mode=''):
    """
    한 장의 이미지에 대한 attention map 시각화.

    기본 레이아웃 (degradation 없을 때):
      Row 0: [RGB] [LiDAR] [Thermal] [GT label]
      Row 1: LiDAR(Q) → RGB memory (query-side), Layer 0~3
      Row 2: Thermal(Q) → RGB memory (query-side), Layer 0~3
      Row 3: Thermal(Q) → LiDAR memory (query-side), Layer 0~3
      Row 4: Preference (RGB vs LiDAR), Layer 0~3
      Row 5: RGB memory ← Thermal가 참조 (key-side, RGB 위에 overlay)
      Row 6: LiDAR memory ← Thermal가 참조 (key-side, LiDAR 위에 overlay)

    degradation 있으면 아래에 추가.
    """
    m = len(modal_names)
    num_layers = 4

    # Input images
    input_imgs = [_denormalize_image(images_gpu[i][0]) for i in range(m)]
    img_h, img_w = input_imgs[0].shape[:2]

    # Label image
    label_img = None
    if label is not None:
        try:
            lbl = label[0] if label.dim() > 2 else label
            label_img = _label_to_color(lbl)
        except Exception:
            pass

    # Degradation rows
    deg_names = sorted(attn_data_deg_dict.keys()) if attn_data_deg_dict else []
    n_base_rows = 7  # 0:input, 1:f1→img, 2:f2→img, 3:f2→lidar, 4:pref, 5:rgb_recv, 6:lidar_recv
    n_deg_rows = len(deg_names) * 3  # each: →img, →lidar, pref
    n_rows = n_base_rows + n_deg_rows

    ncols = num_layers + 1  # +1 for label column
    fig = plt.figure(figsize=(ncols * 3.5, n_rows * 3))
    gs = gridspec.GridSpec(n_rows, ncols,
                           width_ratios=[0.18] + [1] * num_layers,
                           wspace=0.08, hspace=0.35)

    def _label(row, text):
        ax = fig.add_subplot(gs[row, 0])
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=8,
                rotation=90, fontweight='bold', wrap=True)
        ax.axis('off')

    def _plot_attn(row, col, attn_flat, N_q, cmap='hot', vmin=0, vmax=None,
                   title='', overlay_img=None):
        ax = fig.add_subplot(gs[row, col + 1])
        attn_2d = _to_spatial(attn_flat, N_q)
        if overlay_img is not None:
            attn_up = _upsample_map(attn_2d, img_h, img_w)
            ax.imshow(overlay_img, alpha=0.4)
            im = ax.imshow(attn_up, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(attn_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_title(title, fontsize=7)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_pref(row, col, per_mod, N_q, mod_a, mod_b, title='', overlay_img=None):
        ax = fig.add_subplot(gs[row, col + 1])
        a0 = per_mod.get(mod_a, np.zeros(N_q))
        a1 = per_mod.get(mod_b, np.zeros(N_q))
        ratio = a0 / (a0 + a1 + 1e-8)  # 1.0=all mod_a, 0.0=all mod_b
        ratio_2d = _to_spatial(ratio, N_q)
        if overlay_img is not None:
            ratio_up = _upsample_map(ratio_2d, img_h, img_w)
            ax.imshow(overlay_img, alpha=0.3)
            im = ax.imshow(ratio_up, cmap=CMAP_PREF, alpha=0.7, vmin=0, vmax=1)
        else:
            im = ax.imshow(ratio_2d, cmap=CMAP_PREF, vmin=0, vmax=1, interpolation='bilinear')
        ax.set_title(title, fontsize=7)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels([mod_b, '50/50', mod_a], fontsize=6)

    def _draw_attn_rows(attn, base_row, condition_label, overlay_img_idx=2):
        """Frame 2 attention rows: →img, →lidar, preference"""
        frame2 = attn.get(2, {})
        if isinstance(frame2, dict) and frame2.get('skip'):
            return

        overlay = input_imgs[overlay_img_idx]

        # →img
        _label(base_row, f'{modal_names[2]}\n→{modal_names[0]}\n({condition_label})')
        if isinstance(frame2, dict):
            for li in range(num_layers):
                if li not in frame2:
                    continue
                ld = frame2[li]
                if modal_names[0] in ld['per_mod']:
                    mass = ld['per_mod'][modal_names[0]].mean()
                    _plot_attn(base_row, li, ld['per_mod'][modal_names[0]], ld['N_q'],
                              cmap='Reds', title=f'L{li} →{modal_names[0]} ({mass:.3f})',
                              overlay_img=overlay)

        # →lidar
        _label(base_row + 1, f'{modal_names[2]}\n→{modal_names[1]}\n({condition_label})')
        if isinstance(frame2, dict):
            for li in range(num_layers):
                if li not in frame2:
                    continue
                ld = frame2[li]
                if modal_names[1] in ld['per_mod']:
                    mass = ld['per_mod'][modal_names[1]].mean()
                    _plot_attn(base_row + 1, li, ld['per_mod'][modal_names[1]], ld['N_q'],
                              cmap='Blues', title=f'L{li} →{modal_names[1]} ({mass:.3f})',
                              overlay_img=overlay)

        # preference
        _label(base_row + 2, f'Preference\n({condition_label})')
        if isinstance(frame2, dict):
            for li in range(num_layers):
                if li not in frame2:
                    continue
                ld = frame2[li]
                _plot_pref(base_row + 2, li, ld['per_mod'], ld['N_q'],
                          modal_names[0], modal_names[1],
                          title=f'L{li} pref',
                          overlay_img=overlay)

    # ── Row 0: Input images ──
    _label(0, 'Input')
    for i in range(m):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(input_imgs[i])
        ax.set_title(f'{modal_names[i]} (frame {i})', fontsize=9, fontweight='bold')
        ax.axis('off')
    # GT label in 4th column
    if label_img is not None and num_layers >= 4:
        ax = fig.add_subplot(gs[0, m + 1]) if m < num_layers else fig.add_subplot(gs[0, num_layers])
        ax.imshow(label_img)
        ax.set_title('GT label', fontsize=9, fontweight='bold')
        ax.axis('off')

    # ── Row 1: Frame 1 (LiDAR → RGB memory) ──
    _label(1, f'{modal_names[1]}\n→{modal_names[0]}')
    frame1 = attn_data.get(1, {})
    if isinstance(frame1, dict) and not frame1.get('skip'):
        for li in range(num_layers):
            if li not in frame1:
                continue
            ld = frame1[li]
            if modal_names[0] in ld['per_mod']:
                mass = ld['per_mod'][modal_names[0]].mean()
                _plot_attn(1, li, ld['per_mod'][modal_names[0]], ld['N_q'],
                          cmap='hot', title=f'L{li} →{modal_names[0]} ({mass:.3f})',
                          overlay_img=input_imgs[1])

    # ── Row 2-4: Frame 2 (Normal) — Query-side ──
    _draw_attn_rows(attn_data, 2, 'Normal')

    # ── Row 5-6: Key-side (memory가 받는 attention) ──
    # "LiDAR memory의 어느 위치에 thermal이 attention을 주는가" → LiDAR 이미지 위에 overlay
    frame2_data = attn_data.get(2, {})
    if isinstance(frame2_data, dict) and not frame2_data.get('skip'):
        for row_offset, mod_name, mod_idx, cmap in [
            (5, modal_names[0], 0, 'Reds'),   # RGB memory receives
            (6, modal_names[1], 1, 'Blues'),   # LiDAR memory receives
        ]:
            _label(row_offset, f'{mod_name} mem\n← {modal_names[2]}\n(key-side)')
            for li in range(num_layers):
                if li not in frame2_data:
                    continue
                ld = frame2_data[li]
                recv = ld.get('per_mod_received', {})
                if mod_name in recv:
                    recv_flat = recv[mod_name]
                    N_mod = len(recv_flat)
                    _plot_attn(row_offset, li, recv_flat, N_mod,
                              cmap=cmap,
                              title=f'L{li} {mod_name} receives (max={recv_flat.max():.4f})',
                              overlay_img=input_imgs[mod_idx])

    # ── Degradation rows ──
    for di, deg_name in enumerate(deg_names):
        base_row = n_base_rows + di * 3
        _draw_attn_rows(attn_data_deg_dict[deg_name], base_row, deg_name)

    title_parts = [f'Memory Attention — {model_name}' if model_name else 'Memory Attention']
    if mode:
        title_parts[0] += f' / {mode}'
    title_parts.append(f'[{stem}] (idx={image_idx})')
    fig.suptitle(' — '.join(title_parts),
                 fontsize=12, fontweight='bold', y=0.998)

    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# Aggregate summary
# ============================================================================

def collect_stats(attn_data, modal_names):
    """한 이미지의 attn_data에서 frame별/layer별 mass 통계 추출 → dict"""
    stats = {}
    m = len(modal_names)
    for frame_idx in range(m):
        fd = attn_data.get(frame_idx, {})
        if not isinstance(fd, dict) or fd.get('skip'):
            continue
        for layer_idx, ld in fd.items():
            if not isinstance(layer_idx, int):
                continue
            per_mod = ld['per_mod']
            entry = {'frame': frame_idx, 'layer': layer_idx}
            for mod_name, attn_flat in per_mod.items():
                entry[f'mass_{mod_name}'] = float(attn_flat.mean())
            if ld.get('obj_ptr') is not None:
                entry['mass_obj_ptr'] = float(ld['obj_ptr'].mean())
            key = (frame_idx, layer_idx)
            stats[key] = entry
    return stats


def plot_aggregate_summary(all_stats_list, modal_names, mode, save_path):
    """
    여러 이미지의 통계를 box plot으로 요약.

    Layout: 2 rows × 4 cols
      Row 0: Frame 1 (LiDAR query) → Layer 0-3
      Row 1: Frame 2 (Thermal query) → Layer 0-3
    """
    num_layers = 4
    frames_to_show = [1, 2]  # skip frame 0 (init)

    fig, axes = plt.subplots(len(frames_to_show), num_layers,
                              figsize=(4.5 * num_layers, 4.5 * len(frames_to_show)))

    for fi, frame_idx in enumerate(frames_to_show):
        for li in range(num_layers):
            ax = axes[fi, li]
            key = (frame_idx, li)

            # Collect masses for each modality across images
            mod_data = defaultdict(list)
            for img_stats in all_stats_list:
                if key in img_stats:
                    entry = img_stats[key]
                    for mod in modal_names:
                        mass_key = f'mass_{mod}'
                        if mass_key in entry:
                            mod_data[mod].append(entry[mass_key])
                    if 'mass_obj_ptr' in entry:
                        mod_data['obj_ptr'].append(entry['mass_obj_ptr'])

            if not mod_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
                continue

            mods = [m for m in modal_names if m in mod_data]
            if 'obj_ptr' in mod_data:
                mods.append('obj_ptr')

            data = [mod_data[m] for m in mods]
            colors = []
            for m in mods:
                if m == 'img':
                    colors.append('#e74c3c')  # red
                elif m == 'lidar':
                    colors.append('#3498db')  # blue
                elif m == 'thermal':
                    colors.append('#2ecc71')  # green
                else:
                    colors.append('#95a5a6')  # gray

            bp = ax.boxplot(data, labels=mods, patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='black', markersize=4))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)

            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Attention Mass' if li == 0 else '')
            q_name = modal_names[frame_idx]
            ax.set_title(f'Frame {frame_idx} ({q_name} Q) — Layer {li}', fontsize=10)

            # Annotate mean values
            for i, (m, vals) in enumerate(zip(mods, data)):
                if vals:
                    mean_val = np.mean(vals)
                    ax.text(i + 1, -0.03, f'{mean_val:.3f}', ha='center', fontsize=7,
                            color='black', fontweight='bold')

    fig.suptitle(
        f'Memory Attention Mass Distribution — {mode} (N={len(all_stats_list)} images)\n'
        f'mass = query가 해당 memory에 배정한 attention 비율 (높을수록 더 많이 참조)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved aggregate summary: {save_path}")


def save_aggregate_json(all_stats_list, modal_names, mode, save_path):
    """통계를 JSON으로 저장"""
    agg = defaultdict(lambda: defaultdict(list))

    for img_stats in all_stats_list:
        for (frame_idx, layer_idx), entry in img_stats.items():
            key = f'frame{frame_idx}_layer{layer_idx}'
            for mod in modal_names:
                mass_key = f'mass_{mod}'
                if mass_key in entry:
                    agg[key][mod].append(entry[mass_key])
            if 'mass_obj_ptr' in entry:
                agg[key]['obj_ptr'].append(entry['mass_obj_ptr'])

    # Compute mean/std
    summary = {'mode': mode, 'num_images': len(all_stats_list), 'modalities': modal_names}
    per_layer = {}
    for key, mod_data in agg.items():
        layer_summary = {}
        for mod, vals in mod_data.items():
            layer_summary[mod] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }
        per_layer[key] = layer_summary
    summary['per_layer'] = per_layer

    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate stats: {save_path}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Memory Attention Cross-Modal Weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Val (주간) 전체
  python MISC/viz_memory_attention.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml --model_path <ckpt> --mode val

  # Test (야간) 전체
  python MISC/viz_memory_attention.py --cfg configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml --model_path <ckpt> --mode test

  # 특정 이미지 + degradation 비교
  python MISC/viz_memory_attention.py ... --indices 0 5 10 --degradations dark_rgb zero_lidar
        """)
    parser.add_argument('--cfg', required=True, help='Eval config YAML')
    parser.add_argument('--model_path', default='', help='Checkpoint path (overrides config)')
    parser.add_argument('--model_name', default=None,
                        help='모델 이름 (폴더명으로 사용, 예: MMSamBase, P9). '
                             '미지정 시 config SAVE_DIR에서 자동 추출')
    parser.add_argument('--mode', choices=['val', 'test'], default='val',
                        help='val=주간(145장), test=야간')
    parser.add_argument('--num_samples', type=int, default=999,
                        help='최대 이미지 수 (default: 전부)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='특정 이미지 인덱스만 시각화')
    parser.add_argument('--degradations', nargs='+', default=None,
                        help='Degradation 비교 (예: dark_rgb zero_lidar). 없으면 순수 attention만')
    parser.add_argument('--save_dir', default=None,
                        help='저장 경로 (default: outputs/viz_memory_attn/{model_name}/{mode})')
    parser.add_argument('--skip_per_image', action='store_true',
                        help='이미지별 figure 생략, aggregate만')

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_cfg = cfg['DATASET']
    modal_names = dataset_cfg['MODALS']

    # Model name: --model_name > SAVE_DIR에서 추출 > LORA_MODEL에서 추출
    model_name = args.model_name
    if model_name is None:
        save_dir_cfg = cfg.get('SAVE_DIR', '')
        # e.g. './outputs/MMSamP9/levine_...' → 'MMSamP9'
        # e.g. './outputs/MMSamBase/levine_...' → 'MMSamBase'
        parts = Path(save_dir_cfg).parts
        for i, p in enumerate(parts):
            if p == 'outputs' and i + 1 < len(parts):
                model_name = parts[i + 1]
                break
    if model_name is None:
        lora_model = cfg.get('MODEL', {}).get('LORA_MODEL', 'unknown')
        model_name = lora_model.replace('LoRA_Sam_', 'P').replace('LoRA_Sam', 'MMSamBase')

    # Save directory: outputs/viz_memory_attn/{model_name}/{mode}/
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path('outputs') / 'viz_memory_attn' / model_name / args.mode
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'per_image').mkdir(exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Mode: {args.mode}")
    print(f"Modalities: {modal_names}")
    print(f"Degradations: {args.degradations or 'none (pure attention)'}")
    print(f"Save dir: {save_dir}")

    # Load model
    model_path = args.model_path or cfg.get('EVAL', {}).get('MODEL_PATH', '')
    if not model_path:
        print("ERROR: --model_path required")
        return
    model = load_model(cfg, model_path, device)

    # Probe
    probe = MemoryAttentionProbe(model)
    probe.hook_memory_composition()

    # Load dataset
    dataset_root = dataset_cfg['ROOT']
    test_cfg = cfg.get('TEST', {})
    if args.mode == 'test':
        dataset_root = test_cfg.get('FILE', dataset_root)

    eval_cfg = cfg.get('EVAL', {})
    image_size = eval_cfg.get('IMAGE_SIZE', [1024, 1024])
    if args.mode == 'test':
        image_size = test_cfg.get('IMAGE_SIZE', image_size)

    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    dataset = MULTIAQUA(
        root=dataset_root,
        split=split,
        transform=transform,
        modals=modal_names,
        require_annotation=(args.mode == 'val'),
        return_meta=True,
        rgb_subroot=dataset_cfg.get('RGB_SUBROOT'),
        thermal_subroot=dataset_cfg.get('THERMAL_SUBROOT'),
        lidar_subroot=dataset_cfg.get('LIDAR_SUBROOT'),
    )

    # Indices
    if args.indices:
        indices = [i for i in args.indices if i < len(dataset)]
    else:
        indices = list(range(min(args.num_samples, len(dataset))))

    print(f"Processing {len(indices)} images...")

    all_stats = []

    try:
        for count, idx in enumerate(indices):
            sample_data = dataset[idx]
            images, label, meta = sample_data
            stem = meta.get('stem', str(idx))
            print(f"[{count+1}/{len(indices)}] idx={idx}, stem={stem}", end='')

            # Batch + GPU
            images_gpu = [img.unsqueeze(0).to(device) for img in images]

            # Normal attention
            attn_data, output = extract_attention_data(model, images_gpu, probe, modal_names)

            # Collect stats
            stats = collect_stats(attn_data, modal_names)
            all_stats.append(stats)

            # Quick print: frame 2 attention mass
            key2 = (2, 2)  # frame 2, layer 2
            if key2 in stats:
                masses = {k.replace('mass_', ''): v for k, v in stats[key2].items()
                          if k.startswith('mass_')}
                mass_str = ' '.join(f'{m}={v:.3f}' for m, v in masses.items())
                print(f'  L2: {mass_str}')
            else:
                print()

            # Degradation (optional)
            attn_deg_dict = None
            if args.degradations:
                attn_deg_dict = {}
                for deg_mode in args.degradations:
                    images_deg = apply_degradation(images_gpu, deg_mode, modal_names)
                    attn_deg, _ = extract_attention_data(model, images_deg, probe, modal_names)
                    attn_deg_dict[deg_mode] = attn_deg

            # Per-image figure
            if not args.skip_per_image:
                fig_path = save_dir / 'per_image' / f'{idx:03d}_{stem}.png'
                create_per_image_figure(
                    images_gpu, attn_data, modal_names,
                    label=label, image_idx=idx, stem=stem,
                    save_path=fig_path,
                    attn_data_deg_dict=attn_deg_dict,
                    model_name=model_name, mode=args.mode,
                )

    finally:
        probe.unhook_memory_composition()

    # Aggregate summary
    if all_stats:
        plot_aggregate_summary(all_stats, modal_names,
                              f'{model_name} / {args.mode}',
                              save_dir / 'aggregate_summary.png')
        summary = save_aggregate_json(all_stats, modal_names,
                                       f'{model_name}_{args.mode}',
                                       save_dir / 'aggregate_stats.json')

        # Print summary table
        print("\n" + "=" * 70)
        print(f"AGGREGATE SUMMARY — {model_name} / {args.mode} ({len(all_stats)} images)")
        print("=" * 70)
        print(f"{'Frame':>8} {'Layer':>6} ", end='')
        for mod in modal_names:
            print(f'{mod:>12}', end='')
        print(f'{"obj_ptr":>12}')
        print("-" * 70)

        for frame_idx in [1, 2]:
            for layer_idx in range(4):
                key = f'frame{frame_idx}_layer{layer_idx}'
                if key not in summary['per_layer']:
                    continue
                layer_data = summary['per_layer'][key]
                q_name = modal_names[frame_idx]
                print(f'{q_name:>8} L{layer_idx:>4} ', end='')
                for mod in modal_names:
                    if mod in layer_data:
                        d = layer_data[mod]
                        print(f' {d["mean"]:.3f}±{d["std"]:.3f}', end='')
                    else:
                        print(f'{"---":>12}', end='')
                if 'obj_ptr' in layer_data:
                    d = layer_data['obj_ptr']
                    print(f' {d["mean"]:.3f}±{d["std"]:.3f}', end='')
                print()

        print("=" * 70)
        print(f"\n모든 결과: {save_dir}")
        print("""
┌─────────────────────────────────────────────────────────────┐
│ 해석 가이드                                                 │
├─────────────────────────────────────────────────────────────┤
│ mass = query가 해당 memory에 배정한 attention 비율           │
│                                                             │
│ Frame 1 (LiDAR Q):                                          │
│   memory에는 RGB만 있음 → mass_img ≈ 1.0이 정상             │
│                                                             │
│ Frame 2 (Thermal Q):                                        │
│   memory에 RGB + LiDAR 있음                                 │
│   → mass_img vs mass_lidar 비율이 핵심                      │
│   → val(주간)과 test(야간) 비교:                            │
│     - 야간에도 mass_img가 비슷하면 = 열화 RGB에 여전히 attend│
│       → 문제 존재 (memory attention이 quality 무시)          │
│     - 야간에 mass_lidar가 올라가면 = 적응적 shift            │
│       → 문제 없음 (이미 잘 작동)                            │
│                                                             │
│ 이미지별 figure의 Preference map:                           │
│   빨강 = RGB memory 선호, 파랑 = LiDAR memory 선호          │
│   → 공간적으로 어디서 어떤 modality를 참조하는지 확인        │
└─────────────────────────────────────────────────────────────┘
""")


if __name__ == '__main__':
    main()
