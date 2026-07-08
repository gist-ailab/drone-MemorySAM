#!/usr/bin/env python3
"""
diagnose_memory_attention.py — Memory Attention Cross-Modal Weight 분석

[1단계] 문제 존재 증명:
  degraded modality → SAM2 memory attention weight이 실제로 이상해지는가?

분석:
  - 각 modality를 "frame"으로 처리할 때, cross-attention에서 memory의 각 modality
    영역에 대한 attention weight 분포를 추출
  - 정상 입력 vs degraded 입력 비교 (synthetic degradation)
  - frame 2 (3번째 modality)의 cross-attention이 핵심: memory에 2개 modality가 있어
    어느 쪽에 더 attention하는지 분석 가능

사용법:
  python MISC/diagnose_memory_attention.py \
    --cfg configs/eval/levine-multiaqua_rgbtl_P9_hardaug8_physaug.yaml \
    --model_path <checkpoint_path> \
    --mode val \
    --num_samples 10 \
    --degradation dark_rgb \
    --save_dir outputs/diagnose_memory_attn

Degradation 모드:
  - none:        원본 (baseline)
  - dark_rgb:    RGB × 0.05 (야간 극저조도 시뮬레이션)
  - zero_thermal: thermal 전부 0
  - zero_lidar:  lidar 전부 0
  - noise_rgb:   RGB에 강한 가우시안 노이즈 추가
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.datasets.multiaqua import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation


# ============================================================================
# 1. Attention Weight Extraction Hook
# ============================================================================

class MemoryAttentionProbe:
    """
    RoPEAttention의 cross-attention에서 attention weight를 추출하는 probe.

    F.scaled_dot_product_attention은 fused kernel이라 weight를 반환하지 않으므로,
    RoPEAttention.forward를 monkey-patch해서 manual 계산으로 교체한다.
    """

    def __init__(self, model):
        """
        Args:
            model: LoRA_Sam_P9 (또는 유사) 모델 인스턴스
        """
        self.model = model
        self.sam = model.sam  # SAM2Base instance

        # memory_attention 모듈의 layers에 접근
        self.mem_attn = self.sam.memory_attention  # MemoryAttention
        self.num_layers = self.mem_attn.num_layers

        # 저장소
        self.cross_attn_weights = []  # list of (layer_idx, attn_weights_tensor)
        self.self_attn_weights = []
        self.memory_token_info = {}   # frame_idx → memory composition info

        # 원본 forward 백업
        self._original_forwards = {}
        self._patched = False

    def _make_manual_rope_attn_forward(self, rope_attn, layer_idx, attn_type):
        """
        RoPEAttention.forward를 manual attention weight 계산으로 교체하는 closure.
        """
        original_forward = rope_attn.forward
        probe = self

        def manual_forward(q, k, v, num_k_exclude_rope=0):
            # Input projections (동일)
            q_proj = rope_attn.q_proj(q)
            k_proj = rope_attn.k_proj(k)
            v_proj = rope_attn.v_proj(v)

            # Separate into heads
            q_heads = rope_attn._separate_heads(q_proj, rope_attn.num_heads)
            k_heads = rope_attn._separate_heads(k_proj, rope_attn.num_heads)
            v_heads = rope_attn._separate_heads(v_proj, rope_attn.num_heads)

            # Apply RoPE encoding
            w = h = math.sqrt(q_heads.shape[-2])
            rope_attn.freqs_cis = rope_attn.freqs_cis.to(q_heads.device)
            if rope_attn.freqs_cis.shape[0] != q_heads.shape[-2]:
                rope_attn.freqs_cis = rope_attn.compute_cis(end_x=w, end_y=h).to(q_heads.device)

            num_k_rope = k_heads.size(-2) - num_k_exclude_rope

            from sam2.modeling.position_encoding import apply_rotary_enc
            q_heads, k_heads[:, :, :num_k_rope] = apply_rotary_enc(
                q_heads,
                k_heads[:, :, :num_k_rope],
                freqs_cis=rope_attn.freqs_cis,
                repeat_freqs_k=rope_attn.rope_k_repeat,
            )

            # === Manual attention weight computation ===
            scale = q_heads.shape[-1] ** -0.5
            attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Store attention weights
            probe._store_attn_weights(layer_idx, attn_type, attn_weights)

            # Compute output
            out = torch.matmul(attn_weights, v_heads)
            out = rope_attn._recombine_heads(out)
            out = rope_attn.out_proj(out)

            return out

        return manual_forward

    def _store_attn_weights(self, layer_idx, attn_type, weights):
        """attention weight 저장"""
        entry = {
            'layer_idx': layer_idx,
            'attn_type': attn_type,
            'weights': weights.detach().cpu(),  # (B, num_heads, N_q, N_k)
        }
        if attn_type == 'cross':
            self.cross_attn_weights.append(entry)
        else:
            self.self_attn_weights.append(entry)

    @contextmanager
    def probe_enabled(self):
        """
        Context manager: memory attention의 RoPEAttention을 monkey-patch해서
        attention weight를 추출한다. 종료 시 원복.
        """
        # Patch each layer's self_attn and cross_attn_image
        for layer_idx, layer in enumerate(self.mem_attn.layers):
            # Cross-attention (핵심)
            ca_module = layer.cross_attn_image
            ca_key = (layer_idx, 'cross')
            self._original_forwards[ca_key] = ca_module.forward
            ca_module.forward = self._make_manual_rope_attn_forward(
                ca_module, layer_idx, 'cross'
            )

            # Self-attention (참고용)
            sa_module = layer.self_attn
            sa_key = (layer_idx, 'self')
            self._original_forwards[sa_key] = sa_module.forward
            sa_module.forward = self._make_manual_rope_attn_forward(
                sa_module, layer_idx, 'self'
            )

        self._patched = True
        try:
            yield self
        finally:
            # Restore originals
            for layer_idx, layer in enumerate(self.mem_attn.layers):
                ca_key = (layer_idx, 'cross')
                sa_key = (layer_idx, 'self')
                layer.cross_attn_image.forward = self._original_forwards[ca_key]
                layer.self_attn.forward = self._original_forwards[sa_key]
            self._original_forwards.clear()
            self._patched = False

    def clear(self):
        """수집된 데이터 초기화"""
        self.cross_attn_weights.clear()
        self.self_attn_weights.clear()
        self.memory_token_info.clear()

    def hook_memory_composition(self):
        """
        _prepare_memory_conditioned_features 내부의 memory 구성을 추적하기 위한 hook.
        SAM2Base의 메서드를 wrap해서 memory token boundary를 기록한다.
        """
        original_fn = self.sam._prepare_memory_conditioned_features
        probe = self

        def wrapped_fn(frame_idx, is_init_cond_frame, current_vision_feats,
                       current_vision_pos_embeds, feat_sizes, output_dict,
                       num_frames, track_in_reverse=False):
            # 호출 전에 output_dict의 cond_frame_outputs를 확인해서
            # 어떤 frame이 memory에 있는지 기록
            memory_frames = list(output_dict["cond_frame_outputs"].keys())
            non_cond_frames = list(output_dict["non_cond_frame_outputs"].keys())

            # memory token 구성 추적
            token_counts = {}
            for mf_idx in sorted(memory_frames):
                out = output_dict["cond_frame_outputs"][mf_idx]
                if out.get("maskmem_features") is not None:
                    maskmem = out["maskmem_features"]
                    # maskmem_features: (B, C, H_mem, W_mem)
                    h_mem, w_mem = maskmem.shape[-2:]
                    token_counts[mf_idx] = h_mem * w_mem

            probe.memory_token_info[frame_idx] = {
                'is_init': is_init_cond_frame,
                'memory_frames': memory_frames,
                'non_cond_frames': non_cond_frames,
                'token_counts': token_counts,
            }

            return original_fn(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )

        self.sam._prepare_memory_conditioned_features = wrapped_fn
        self._original_prepare = original_fn

    def unhook_memory_composition(self):
        if hasattr(self, '_original_prepare'):
            self.sam._prepare_memory_conditioned_features = self._original_prepare
            del self._original_prepare


# ============================================================================
# 2. Degradation Functions
# ============================================================================

def apply_degradation(images, mode, modal_names):
    """
    입력 images(list of tensors)에 synthetic degradation 적용.

    Args:
        images: list of tensors [rgb_batch, lidar_batch, thermal_batch], 각 (B, C, H, W)
        mode: degradation mode string
        modal_names: list of modality names (e.g. ['img', 'lidar', 'thermal'])

    Returns:
        degraded images list (새 list, 원본 불변)
    """
    images = [x.clone() for x in images]

    if mode == 'none':
        pass

    elif mode == 'dark_rgb':
        # RGB를 극저조도로 (×0.05)
        idx = modal_names.index('img') if 'img' in modal_names else 0
        images[idx] = images[idx] * 0.05

    elif mode == 'very_dark_rgb':
        # RGB 거의 제로 (×0.01)
        idx = modal_names.index('img') if 'img' in modal_names else 0
        images[idx] = images[idx] * 0.01

    elif mode == 'zero_thermal':
        if 'thermal' in modal_names:
            idx = modal_names.index('thermal')
            images[idx] = torch.zeros_like(images[idx])

    elif mode == 'zero_lidar':
        if 'lidar' in modal_names:
            idx = modal_names.index('lidar')
            images[idx] = torch.zeros_like(images[idx])

    elif mode == 'noise_rgb':
        idx = modal_names.index('img') if 'img' in modal_names else 0
        noise = torch.randn_like(images[idx]) * 0.2  # normalized scale
        images[idx] = (images[idx] + noise).clamp(0, 1)

    elif mode == 'zero_rgb':
        idx = modal_names.index('img') if 'img' in modal_names else 0
        images[idx] = torch.zeros_like(images[idx])

    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    return images


# ============================================================================
# 3. Analysis Functions
# ============================================================================

def analyze_cross_attention_per_modality(probe, num_modalities, modal_names):
    """
    수집된 cross-attention weights를 modality별로 분리 분석.

    frame 0 (첫 modality): memory = dummy token → 건너뜀
    frame 1: memory = frame 0 tokens + obj_ptrs
    frame 2: memory = frame 0 tokens + frame 1 tokens + obj_ptrs

    Returns:
        dict: frame_idx → layer_idx → {
            'per_modality_attn': {mod_idx: mean_attention_weight},
            'obj_ptr_attn': mean_attention_to_obj_ptrs,
            'attn_entropy': spatial entropy of attention,
        }
    """
    m = num_modalities
    results = {}

    # Group cross-attention weights by frame (track_step call)
    # Memory attention has num_layers layers. Each track_step call produces
    # num_layers cross-attention entries.
    num_layers = probe.num_layers

    # Frame 0 (init): directly_add_no_mem_embed=True → memory attention NOT called → 0 entries
    # Frame 1+: memory attention called → num_layers entries each
    # So cross_attn_weights index = (non_init_frame_order) * num_layers + layer_idx
    non_init_frame_count = 0

    for frame_idx in range(m):
        frame_results = {}
        mem_info = probe.memory_token_info.get(frame_idx, {})

        if mem_info.get('is_init', True) and frame_idx == 0:
            # First frame: no memory attention called (directly_add_no_mem_embed)
            frame_results['skip'] = True
            frame_results['reason'] = 'init frame, no memory attention'
            results[frame_idx] = frame_results
            continue

        token_counts = mem_info.get('token_counts', {})
        memory_frames = sorted(mem_info.get('memory_frames', []))

        for layer_idx in range(num_layers):
            entry_idx = non_init_frame_count * num_layers + layer_idx
            if entry_idx >= len(probe.cross_attn_weights):
                continue

            entry = probe.cross_attn_weights[entry_idx]
            attn_w = entry['weights']  # (B, num_heads, N_q, N_k)

            B, H, N_q, N_k = attn_w.shape

            # memory token boundaries 계산
            # memory = [frame0_maskmem (H*W tokens), frame1_maskmem (H*W), ..., obj_ptrs]
            boundaries = {}
            offset = 0
            for mf_idx in memory_frames:
                if mf_idx in token_counts:
                    n_tokens = token_counts[mf_idx]
                    boundaries[mf_idx] = (offset, offset + n_tokens)
                    offset += n_tokens

            # 나머지는 obj_ptr tokens
            obj_ptr_range = (offset, N_k)

            # Per-modality attention 계산
            # mean over query tokens and heads: attention 분포
            attn_mean = attn_w.mean(dim=(0, 1, 2))  # (N_k,) — 각 key token의 평균 attention

            per_mod_attn = {}
            for mf_idx, (start, end) in boundaries.items():
                if end > start:
                    mod_name = modal_names[mf_idx] if mf_idx < len(modal_names) else f"mod{mf_idx}"
                    per_mod_attn[mod_name] = {
                        'mean': attn_mean[start:end].mean().item(),
                        'sum': attn_mean[start:end].sum().item(),
                        'n_tokens': end - start,
                        'total_mass': attn_w[:, :, :, start:end].sum(dim=-1).mean().item(),
                    }

            obj_ptr_attn = None
            if obj_ptr_range[1] > obj_ptr_range[0]:
                obj_ptr_attn = {
                    'mean': attn_mean[obj_ptr_range[0]:obj_ptr_range[1]].mean().item(),
                    'total_mass': attn_w[:, :, :, obj_ptr_range[0]:obj_ptr_range[1]].sum(dim=-1).mean().item(),
                    'n_tokens': obj_ptr_range[1] - obj_ptr_range[0],
                }

            # Attention entropy (query별 entropy 평균)
            # attn_w shape: (B, H, N_q, N_k)
            eps = 1e-8
            ent = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # (B, H, N_q)
            max_ent = math.log(N_k)
            ent_ratio = (ent / max_ent).mean().item()

            # Spatial attention map: query token별로 어떤 modality에 집중하는지
            # attn_w: (B, H, N_q, N_k) → reshape query to spatial (H_q, W_q)
            spatial_mod_maps = {}
            for mf_idx, (start, end) in boundaries.items():
                if end > start:
                    mod_name = modal_names[mf_idx] if mf_idx < len(modal_names) else f"mod{mf_idx}"
                    # sum attention over modality's key tokens → (B, H, N_q)
                    mod_attn = attn_w[:, :, :, start:end].sum(dim=-1)
                    spatial_mod_maps[mod_name] = mod_attn.mean(dim=(0, 1)).numpy()  # (N_q,)

            frame_results[layer_idx] = {
                'per_modality_attn': per_mod_attn,
                'obj_ptr_attn': obj_ptr_attn,
                'entropy_ratio': ent_ratio,
                'spatial_mod_maps': spatial_mod_maps,
                'N_q': N_q,
                'N_k': N_k,
            }

        results[frame_idx] = frame_results
        non_init_frame_count += 1

    return results


def summarize_results(results, modal_names):
    """분석 결과를 간단한 summary dict로 변환"""
    summary = {}
    for frame_idx, frame_data in results.items():
        if frame_data.get('skip'):
            continue
        frame_summary = {}
        for layer_idx, layer_data in frame_data.items():
            if not isinstance(layer_idx, int):
                continue
            per_mod = layer_data['per_modality_attn']
            layer_summary = {
                'attn_mass': {mod: info['total_mass'] for mod, info in per_mod.items()},
                'entropy_ratio': layer_data['entropy_ratio'],
            }
            if layer_data['obj_ptr_attn']:
                layer_summary['obj_ptr_mass'] = layer_data['obj_ptr_attn']['total_mass']
            frame_summary[f'layer_{layer_idx}'] = layer_summary
        summary[f'frame_{frame_idx}'] = frame_summary
    return summary


# ============================================================================
# 4. Visualization
# ============================================================================

def plot_attention_comparison(results_normal, results_degraded,
                              degradation_mode, modal_names, save_path):
    """
    정상 vs degraded attention weight 비교 시각화.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    num_modalities = len(modal_names)

    # frame 2 (마지막 modality)의 cross-attention 분석이 핵심
    # memory에 2개 modality가 있어서 분배 비교 가능
    target_frame = num_modalities - 1

    normal_data = results_normal.get(target_frame, {})
    degraded_data = results_degraded.get(target_frame, {})

    if not normal_data or normal_data.get('skip'):
        print(f"No data for frame {target_frame}")
        return

    num_layers = len([k for k in normal_data.keys() if isinstance(k, int)])

    fig, axes = plt.subplots(2, num_layers, figsize=(5 * num_layers, 8))
    if num_layers == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f'Cross-Attention Weight Distribution (Frame {target_frame}: {modal_names[target_frame]})\n'
        f'Normal vs {degradation_mode}',
        fontsize=14
    )

    for layer_idx in range(num_layers):
        if layer_idx not in normal_data or layer_idx not in degraded_data:
            continue

        normal_layer = normal_data[layer_idx]
        degraded_layer = degraded_data[layer_idx]

        # Bar chart: attention mass per modality
        mods = list(normal_layer['per_modality_attn'].keys())
        normal_masses = [normal_layer['per_modality_attn'][m]['total_mass'] for m in mods]
        degraded_masses = [degraded_layer['per_modality_attn'][m]['total_mass'] for m in mods]

        # Add obj_ptr if present
        if normal_layer['obj_ptr_attn']:
            mods.append('obj_ptr')
            normal_masses.append(normal_layer['obj_ptr_attn']['total_mass'])
            degraded_masses.append(degraded_layer['obj_ptr_attn']['total_mass'])

        x = np.arange(len(mods))
        width = 0.35

        ax = axes[0, layer_idx]
        bars1 = ax.bar(x - width/2, normal_masses, width, label='Normal', color='steelblue')
        bars2 = ax.bar(x + width/2, degraded_masses, width, label=degradation_mode, color='coral')
        ax.set_xlabel('Memory Source')
        ax.set_ylabel('Attention Mass')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xticks(x)
        ax.set_xticklabels(mods, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.0)

        # Entropy comparison
        ax2 = axes[1, layer_idx]
        ent_data = [normal_layer['entropy_ratio'], degraded_layer['entropy_ratio']]
        colors = ['steelblue', 'coral']
        bars = ax2.bar(['Normal', degradation_mode], ent_data, color=colors)
        ax2.set_ylabel('Entropy Ratio (0=focused, 1=uniform)')
        ax2.set_title(f'Layer {layer_idx} Attention Entropy')
        ax2.set_ylim(0, 1.0)
        for bar, val in zip(bars, ent_data):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_spatial_attention_maps(results, modal_names, title_prefix, save_path):
    """
    Spatial attention map 시각화: query token 위치별로 어떤 modality에 집중하는지.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    target_frame = len(modal_names) - 1
    frame_data = results.get(target_frame, {})
    if not frame_data or frame_data.get('skip'):
        return

    num_layers = len([k for k in frame_data.keys() if isinstance(k, int)])
    mem_mods = list(frame_data[0]['per_modality_attn'].keys()) if 0 in frame_data else []

    if not mem_mods:
        return

    fig, axes = plt.subplots(len(mem_mods), num_layers,
                              figsize=(4 * num_layers, 4 * len(mem_mods)))
    if num_layers == 1 and len(mem_mods) == 1:
        axes = np.array([[axes]])
    elif num_layers == 1:
        axes = axes.reshape(-1, 1)
    elif len(mem_mods) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{title_prefix} — Spatial Attention to Each Modality (Frame {target_frame})', fontsize=12)

    for layer_idx in range(num_layers):
        if layer_idx not in frame_data:
            continue
        layer_data = frame_data[layer_idx]
        spatial_maps = layer_data.get('spatial_mod_maps', {})
        N_q = layer_data['N_q']
        h_q = int(math.sqrt(N_q))
        w_q = N_q // h_q

        for mod_i, mod_name in enumerate(mem_mods):
            if mod_name not in spatial_maps:
                continue
            attn_map = spatial_maps[mod_name]  # (N_q,)
            attn_2d = attn_map[:h_q * w_q].reshape(h_q, w_q)

            ax = axes[mod_i, layer_idx]
            im = ax.imshow(attn_2d, cmap='hot', interpolation='bilinear')
            ax.set_title(f'L{layer_idx} → {mod_name}\nmean={attn_2d.mean():.3f}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_multi_image_summary(all_summaries_normal, all_summaries_degraded,
                              degradation_mode, modal_names, save_path):
    """
    여러 이미지에 대한 attention mass 통계 (box plot).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    target_frame = len(modal_names) - 1

    # 각 이미지에서 layer별 modality attention mass 수집
    normal_data = defaultdict(lambda: defaultdict(list))  # mod → layer → [values]
    degraded_data = defaultdict(lambda: defaultdict(list))

    for summ in all_summaries_normal:
        frame_key = f'frame_{target_frame}'
        if frame_key not in summ:
            continue
        for layer_key, layer_data in summ[frame_key].items():
            layer_idx = int(layer_key.split('_')[1])
            for mod, mass in layer_data.get('attn_mass', {}).items():
                normal_data[mod][layer_idx].append(mass)

    for summ in all_summaries_degraded:
        frame_key = f'frame_{target_frame}'
        if frame_key not in summ:
            continue
        for layer_key, layer_data in summ[frame_key].items():
            layer_idx = int(layer_key.split('_')[1])
            for mod, mass in layer_data.get('attn_mass', {}).items():
                degraded_data[mod][layer_idx].append(mass)

    if not normal_data:
        return

    mods = sorted(normal_data.keys())
    layers = sorted(next(iter(normal_data.values())).keys())

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 6))
    if len(layers) == 1:
        axes = [axes]

    fig.suptitle(
        f'Attention Mass Distribution: Normal vs {degradation_mode}\n'
        f'(Frame {target_frame}: {modal_names[target_frame]}, N={len(all_summaries_normal)} images)',
        fontsize=13
    )

    for li, layer_idx in enumerate(layers):
        ax = axes[li]
        positions = []
        data_pairs = []
        labels = []
        colors = []

        for mi, mod in enumerate(mods):
            norm_vals = normal_data[mod].get(layer_idx, [])
            deg_vals = degraded_data[mod].get(layer_idx, [])

            pos_n = mi * 3
            pos_d = mi * 3 + 1

            if norm_vals:
                bp1 = ax.boxplot([norm_vals], positions=[pos_n], widths=0.6,
                               patch_artist=True, showmeans=True)
                bp1['boxes'][0].set_facecolor('steelblue')
                bp1['boxes'][0].set_alpha(0.7)

            if deg_vals:
                bp2 = ax.boxplot([deg_vals], positions=[pos_d], widths=0.6,
                               patch_artist=True, showmeans=True)
                bp2['boxes'][0].set_facecolor('coral')
                bp2['boxes'][0].set_alpha(0.7)

            positions.append(mi * 3 + 0.5)
            labels.append(mod)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Attention Mass')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_ylim(0, 1.0)

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='steelblue', alpha=0.7, label='Normal'),
            Patch(facecolor='coral', alpha=0.7, label=degradation_mode),
        ])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# 5. Main
# ============================================================================

def load_model(cfg, model_path, device):
    """Config 기반 LoRA 모델 로드 (val_multiaqua.py와 동일 방식)"""
    import inspect
    from semseg.models.sam2.sam2.build_sam import build_sam2
    import semseg.models.sam2.sam2.sam_lora_image_encoder_seg as seg_module

    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        "sam2_hiera_b+.yaml",
        "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false",
        ],
    )

    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P9')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_layer = model_cfg.get('LORA_LAYER')

    lora_model_class = getattr(seg_module, lora_model_name)
    model_kwargs = {
        'sam_model': sam2,
        'r': lora_r,
        'lora_layer': lora_layer,
    }
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'num_modalities' in sig.parameters:
        model_kwargs['num_modalities'] = num_modalities
    if 'quality_hidden_dim' in sig.parameters:
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['quality_hidden_dim'] = quality_cfg.get('HIDDEN_DIM', 64)
        model_kwargs['quality_min'] = quality_cfg.get('MIN_QUALITY', 0.1)
    if 'tau_uamm' in sig.parameters:
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['tau_uamm'] = quality_cfg.get('TAU_UAMM', 1.0)
        model_kwargs['tau_teacher'] = quality_cfg.get('TAU_TEACHER', 0.5)
        model_kwargs['memory_mod'] = quality_cfg.get('MEMORY_MOD', False)
        model_kwargs['amf_mode'] = quality_cfg.get('AMF_MODE', 'sqg_quality')
        model_kwargs['multi_scale_sqg'] = quality_cfg.get('MULTI_SCALE_SQG', True)
        model_kwargs['per_modality_decoder'] = quality_cfg.get('PER_MODALITY_DECODER', True)

    model = lora_model_class(**model_kwargs)

    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"  Missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

    model.to(device)
    model.eval()

    return model


def run_diagnosis(args):
    """메인 진단 루프"""
    # Load config
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modal_names = cfg['DATASET']['MODALS']
    num_modalities = len(modal_names)

    print(f"Modalities: {modal_names}")
    print(f"Degradation: {args.degradation}")
    print(f"Mode: {args.mode}")

    # Load model
    model_path = args.model_path or cfg.get('EVAL', {}).get('MODEL_PATH', '')
    if not model_path:
        print("ERROR: --model_path required")
        return

    model = load_model(cfg, model_path, device)

    # Create probe
    probe = MemoryAttentionProbe(model)

    # Load dataset
    dataset_cfg = cfg['DATASET']
    dataset_root = dataset_cfg['ROOT']
    if args.mode == 'test':
        dataset_root = cfg.get('TEST', {}).get('FILE', dataset_root)

    image_size = cfg.get('EVAL', {}).get('IMAGE_SIZE', [1024, 1024])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    require_annotation = (args.mode == 'val')

    dataset = MULTIAQUA(
        root=dataset_root,
        split=split,
        transform=transform,
        modals=modal_names,
        require_annotation=require_annotation,
        return_meta=True,
        rgb_subroot=dataset_cfg.get('RGB_SUBROOT'),
        thermal_subroot=dataset_cfg.get('THERMAL_SUBROOT'),
        lidar_subroot=dataset_cfg.get('LIDAR_SUBROOT'),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_summaries_normal = []
    all_summaries_degraded = []
    all_results = []

    num_samples = min(args.num_samples, len(dataset))
    print(f"Analyzing {num_samples} samples...")

    probe.hook_memory_composition()

    try:
        for idx, batch in enumerate(loader):
            if idx >= num_samples:
                break

            print(f"\n[{idx+1}/{num_samples}] ", end='')

            # Unpack batch: (images_list, labels, metas)
            images, labels, metas = batch
            images_gpu = [x.to(device) for x in images]

            # === Normal forward ===
            probe.clear()
            with torch.no_grad(), probe.probe_enabled():
                output, _ = model(images_gpu, multimask_output=True)

            results_normal = analyze_cross_attention_per_modality(
                probe, num_modalities, modal_names
            )
            summary_normal = summarize_results(results_normal, modal_names)
            all_summaries_normal.append(summary_normal)

            # === Degraded forward ===
            if args.degradation != 'none':
                images_deg = apply_degradation(images_gpu, args.degradation, modal_names)

                probe.clear()
                with torch.no_grad(), probe.probe_enabled():
                    output_deg, _ = model(images_deg, multimask_output=True)

                results_degraded = analyze_cross_attention_per_modality(
                    probe, num_modalities, modal_names
                )
                summary_degraded = summarize_results(results_degraded, modal_names)
                all_summaries_degraded.append(summary_degraded)
            else:
                results_degraded = results_normal
                summary_degraded = summary_normal
                all_summaries_degraded.append(summary_degraded)

            # Per-image results
            stem = metas['stem'][0] if 'stem' in metas else str(idx)
            img_result = {
                'index': idx,
                'stem': stem,
                'normal': summary_normal,
                'degraded': summary_degraded,
            }
            all_results.append(img_result)

            # Print quick summary for frame 2
            target_frame = f'frame_{num_modalities - 1}'
            if target_frame in summary_normal:
                for lk, ld in summary_normal[target_frame].items():
                    masses = ld.get('attn_mass', {})
                    mass_str = ', '.join(f'{m}={v:.3f}' for m, v in masses.items())
                    print(f"  Normal {lk}: {mass_str} | ent={ld['entropy_ratio']:.3f}")
            if args.degradation != 'none' and target_frame in summary_degraded:
                for lk, ld in summary_degraded[target_frame].items():
                    masses = ld.get('attn_mass', {})
                    mass_str = ', '.join(f'{m}={v:.3f}' for m, v in masses.items())
                    print(f"  Degraded {lk}: {mass_str} | ent={ld['entropy_ratio']:.3f}")

            # Per-image visualizations (first few only)
            if idx < 5:
                plot_attention_comparison(
                    results_normal, results_degraded,
                    args.degradation, modal_names,
                    save_dir / f'attn_comparison_{idx:03d}.png'
                )
                plot_spatial_attention_maps(
                    results_normal, modal_names, 'Normal',
                    save_dir / f'spatial_normal_{idx:03d}.png'
                )
                if args.degradation != 'none':
                    plot_spatial_attention_maps(
                        results_degraded, modal_names, args.degradation,
                        save_dir / f'spatial_degraded_{idx:03d}.png'
                    )

    finally:
        probe.unhook_memory_composition()

    # === Aggregate statistics ===
    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)

    target_frame = f'frame_{num_modalities - 1}'

    # Collect per-layer, per-modality stats
    agg_stats = {
        'config': args.cfg,
        'model_path': model_path,
        'mode': args.mode,
        'degradation': args.degradation,
        'num_samples': num_samples,
        'modalities': modal_names,
        'per_layer': {},
    }

    for condition, summaries in [('normal', all_summaries_normal), ('degraded', all_summaries_degraded)]:
        for summ in summaries:
            if target_frame not in summ:
                continue
            for layer_key, layer_data in summ[target_frame].items():
                if layer_key not in agg_stats['per_layer']:
                    agg_stats['per_layer'][layer_key] = {'normal': defaultdict(list), 'degraded': defaultdict(list)}
                for mod, mass in layer_data.get('attn_mass', {}).items():
                    agg_stats['per_layer'][layer_key][condition][mod].append(mass)
                agg_stats['per_layer'][layer_key][condition]['entropy'].append(layer_data['entropy_ratio'])

    # Print aggregate
    for layer_key in sorted(agg_stats['per_layer'].keys()):
        layer_stats = agg_stats['per_layer'][layer_key]
        print(f"\n{layer_key}:")
        for condition in ['normal', 'degraded']:
            cond_data = layer_stats[condition]
            parts = []
            for mod in modal_names:
                if mod in cond_data:
                    vals = cond_data[mod]
                    parts.append(f"{mod}={np.mean(vals):.4f}±{np.std(vals):.4f}")
            ent_vals = cond_data.get('entropy', [])
            ent_str = f"ent={np.mean(ent_vals):.4f}" if ent_vals else ""
            print(f"  {condition:10s}: {', '.join(parts)} | {ent_str}")

    # Save JSON results
    # Convert defaultdict to dict for JSON serialization
    json_stats = json.loads(json.dumps(agg_stats, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))

    json_path = save_dir / 'aggregate_stats.json'
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Per-image results
    per_image_path = save_dir / 'per_image_results.json'
    with open(per_image_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {per_image_path}")

    # Summary box plot
    if args.degradation != 'none':
        plot_multi_image_summary(
            all_summaries_normal, all_summaries_degraded,
            args.degradation, modal_names,
            save_dir / 'summary_boxplot.png'
        )

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Diagnose Memory Attention Cross-Modal Weights')
    parser.add_argument('--cfg', required=True, help='Eval config YAML path')
    parser.add_argument('--model_path', default='', help='Checkpoint path (overrides config)')
    parser.add_argument('--mode', choices=['val', 'test'], default='val',
                        help='Dataset mode: val (daytime) or test (nighttime)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of images to analyze')
    parser.add_argument('--degradation', default='dark_rgb',
                        choices=['none', 'dark_rgb', 'very_dark_rgb', 'zero_thermal',
                                 'zero_lidar', 'noise_rgb', 'zero_rgb'],
                        help='Synthetic degradation mode')
    parser.add_argument('--save_dir', default='outputs/diagnose_memory_attn',
                        help='Output directory')

    args = parser.parse_args()
    run_diagnosis(args)


if __name__ == '__main__':
    main()
