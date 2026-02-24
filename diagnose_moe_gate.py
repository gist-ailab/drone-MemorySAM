"""
MoE Gate Uniform Routing Diagnostic Script
==========================================

Soft-MoE LoRA gate가 uniform으로 수렴하는 근본 원인을 진단하기 위한 스크립트.

4가지 가설 검증:
  H1: Zero-Init Expert Symmetry (experts_b=0 → gate gradient=0)
  H2: Gate Weight ≈ 0 after training (gate didn't move from init)
  H3: LayerNorm removes modality distinguishability
  H4: Expert outputs converge to same function

사용법:
  # Static analysis only (CPU, no data)
  python diagnose_moe_gate.py --checkpoint outputs/MMSamP9/.../epoch47_94.18.pth \
      --config configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml --static-only

  # Full analysis with forward pass + gradient (GPU + data)
  python diagnose_moe_gate.py --checkpoint outputs/MMSamP9/.../epoch47_94.18.pth \
      --config configs/levine-multiaqua_rgbtl_P9_hardaug4.yaml

NOTE: Use the MMSS_SAM conda environment to run this script.
"""

import os
import sys
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict

# Project imports
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import (
    LoRA_Sam_P9, LoRA_Sam_P10, LoRA_Sam_P11
)
from semseg.models.sam2.sam2.sam_lola_utils import SoftMoE_LoRA_Layer
from semseg.datasets import MULTIAQUA
from semseg.augmentations_mm import get_val_augmentation


# ============================================================================
# Part 0: Setup
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='MoE Gate Diagnostic')
    parser.add_argument('--config', type=str, required=True,
                        help='Config YAML file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint (.pth) path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--static-only', action='store_true',
                        help='Only run static analysis (no GPU/data needed)')
    parser.add_argument('--skip-gradient', action='store_true',
                        help='Skip gradient analysis (Section 3)')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples for forward/gradient analysis')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg, device='cpu'):
    """Build SAM2 + LoRA model from config."""
    sam2_config_file = "sam2_hiera_b+.yaml"
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        sam2_config_file,
        checkpoint,
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
    lora_top_k = model_cfg.get('LORA_TOP_K', 2)
    lora_layer = model_cfg.get('LORA_LAYER', None)

    import inspect
    _model_map = {
        'LoRA_Sam_P9': LoRA_Sam_P9,
        'LoRA_Sam_P10': LoRA_Sam_P10,
        'LoRA_Sam_P11': LoRA_Sam_P11,
    }
    if lora_model_name not in _model_map:
        raise ValueError(f"Unknown model: {lora_model_name}. Supported: {list(_model_map.keys())}")
    lora_model_class = _model_map[lora_model_name]

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
    if 'num_classes' in sig.parameters:
        model_kwargs['num_classes'] = model_cfg.get('LORA_NUM_CLASSES', 4)
    if 'num_modalities' in sig.parameters:
        model_kwargs['num_modalities'] = num_modalities

    model = lora_model_class(**model_kwargs).cpu()

    print(f"Built model: {lora_model_name}")
    print(f"  num_experts={lora_num_experts}, lora_r={lora_r}, lora_layer={lora_layer}")

    return model


def load_checkpoint(model, ckpt_path):
    """Load checkpoint into model."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Handle both formats: raw state_dict or dict with 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded checkpoint (epoch {epoch}) from: {ckpt_path}")
    else:
        state_dict = ckpt
        print(f"Loaded raw state_dict from: {ckpt_path}")

    # Remove 'module.' prefix if DDP checkpoint
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('module.', '')] = v

    msg = model.load_state_dict(cleaned, strict=False)
    if msg.missing_keys:
        print(f"  Missing keys: {len(msg.missing_keys)} (expected for frozen SAM2 params)")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys[:5]}...")

    return model


def get_moe_layers(model):
    """Extract all SoftMoE_LoRA_Layer instances from model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, SoftMoE_LoRA_Layer):
            layers.append((name, module))
    return layers


def build_val_dataloader(cfg, num_samples=4):
    """Build a small validation dataloader."""
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']

    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg)

    _dataset_map = {'MULTIAQUA': MULTIAQUA}
    dataset_cls = _dataset_map[dataset_cfg['NAME']]
    valset = dataset_cls(
        dataset_cfg['ROOT'],
        'val',
        transform,
        dataset_cfg['MODALS']
    )

    # Use a subset
    subset = torch.utils.data.Subset(valset, list(range(min(num_samples, len(valset)))))

    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, num_workers=0, pin_memory=False, shuffle=False
    )
    return loader


# ============================================================================
# Part 1: Static Parameter Analysis
# ============================================================================

def analyze_static(model):
    """Run all static analyses on model parameters."""
    print("\n" + "=" * 80)
    print("SECTION 1: STATIC PARAMETER ANALYSIS")
    print("=" * 80)

    moe_layers = get_moe_layers(model)
    print(f"\nFound {len(moe_layers)} SoftMoE_LoRA_Layer instances")

    analyze_gate_weights(moe_layers)
    analyze_expert_b_weights(moe_layers)
    analyze_expert_cosine_similarity(moe_layers)
    estimate_logit_range(moe_layers)


def analyze_gate_weights(moe_layers):
    """1.1: Gate Weight/Bias magnitude after training [H2]"""
    print("\n" + "-" * 60)
    print("1.1 Gate Weight/Bias Analysis [H2: Gate Weight ≈ 0?]")
    print("-" * 60)

    rows = []
    for name, layer in moe_layers:
        w = layer.gate.weight.data  # [num_experts, dim]
        b = layer.gate.bias.data    # [num_experts]

        dim = w.shape[1]
        num_e = w.shape[0]

        trained_norm = w.norm().item()
        init_norm = 0.01 * math.sqrt(num_e * dim)
        ratio = trained_norm / init_norm if init_norm > 0 else float('inf')

        # Per-expert weight norms
        per_expert_norms = [w[i].norm().item() for i in range(num_e)]

        # Short name: extract layer index and Q/V info
        short = name.split('.')[-1] if len(name) > 30 else name
        # Try to extract block index
        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p == 'moe_layers_q' or p == 'moe_layers_v':
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break
        if not block_info:
            block_info = name[-30:]

        rows.append([
            block_info,
            f"{dim}",
            f"{trained_norm:.4f}",
            f"{init_norm:.4f}",
            f"{ratio:.2f}x",
            f"[{', '.join(f'{x:.4f}' for x in per_expert_norms)}]",
            f"[{', '.join(f'{x:.4f}' for x in b.tolist())}]"
        ])

    headers = ["Layer", "Dim", "W_norm", "Init_norm", "Ratio", "Per-E W_norm", "Bias"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Summary statistics
    ratios = [float(r[4].replace('x', '')) for r in rows]
    print(f"\n  Summary: norm ratio min={min(ratios):.2f}x, max={max(ratios):.2f}x, mean={np.mean(ratios):.2f}x")
    if np.mean(ratios) < 1.5:
        print("  >>> H2 LIKELY: Gate weights barely moved from initialization!")
    elif np.mean(ratios) < 3.0:
        print("  >>> H2 PARTIAL: Gate weights moved moderately")
    else:
        print("  >>> H2 UNLIKELY: Gate weights moved significantly from init")


def analyze_expert_b_weights(moe_layers):
    """1.2: Expert B weight magnitude [H1]"""
    print("\n" + "-" * 60)
    print("1.2 Expert B Weight Analysis [H1: Zero-Init Symmetry?]")
    print("-" * 60)

    rows = []
    for name, layer in moe_layers:
        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break
        if not block_info:
            block_info = name[-25:]

        b_norms = []
        for i in range(layer.num_experts):
            b_norms.append(layer.experts_b[i].weight.data.norm().item())

        a_norms = []
        for i in range(layer.num_experts):
            a_norms.append(layer.experts_a[i].weight.data.norm().item())

        # Max ratio between experts
        max_b = max(b_norms)
        min_b = min(b_norms)
        b_ratio = max_b / min_b if min_b > 1e-10 else float('inf')

        rows.append([
            block_info,
            f"[{', '.join(f'{x:.4f}' for x in b_norms)}]",
            f"{b_ratio:.2f}",
            f"[{', '.join(f'{x:.4f}' for x in a_norms)}]",
        ])

    headers = ["Layer", "B_norms (per E)", "B_ratio", "A_norms (per E)"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Check if experts_b are still near zero or very similar
    all_b_norms = []
    for _, layer in moe_layers:
        for i in range(layer.num_experts):
            all_b_norms.append(layer.experts_b[i].weight.data.norm().item())

    mean_b = np.mean(all_b_norms)
    print(f"\n  Summary: mean experts_b norm = {mean_b:.6f}")
    if mean_b < 0.01:
        print("  >>> H1 STRONG: experts_b are still near-zero!")
    elif mean_b < 0.1:
        print("  >>> H1 MODERATE: experts_b are small")
    else:
        print("  >>> experts_b have grown. Check cosine similarity for H4.")


def analyze_expert_cosine_similarity(moe_layers):
    """1.3: Expert pairwise cosine similarity [H4]"""
    print("\n" + "-" * 60)
    print("1.3 Expert Cosine Similarity [H4: Experts Converged?]")
    print("-" * 60)

    rows_a = []
    rows_b = []

    all_cos_a = []
    all_cos_b = []

    for name, layer in moe_layers:
        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break
        if not block_info:
            block_info = name[-25:]

        ne = layer.num_experts

        # Expert A cosine similarities
        cos_a = []
        for i in range(ne):
            for j in range(i + 1, ne):
                wa = layer.experts_a[i].weight.data.flatten()
                wb = layer.experts_a[j].weight.data.flatten()
                cs = F.cosine_similarity(wa.unsqueeze(0), wb.unsqueeze(0)).item()
                cos_a.append(cs)
                all_cos_a.append(cs)

        # Expert B cosine similarities
        cos_b = []
        for i in range(ne):
            for j in range(i + 1, ne):
                wa = layer.experts_b[i].weight.data.flatten()
                wb = layer.experts_b[j].weight.data.flatten()
                cs = F.cosine_similarity(wa.unsqueeze(0), wb.unsqueeze(0)).item()
                cos_b.append(cs)
                all_cos_b.append(cs)

        rows_a.append([block_info] + [f"{c:.3f}" for c in cos_a])
        rows_b.append([block_info] + [f"{c:.3f}" for c in cos_b])

    ne = moe_layers[0][1].num_experts
    pair_labels = [f"E{i}-E{j}" for i in range(ne) for j in range(i + 1, ne)]

    print("\nExperts_A cosine similarity:")
    headers_a = ["Layer"] + pair_labels
    # Print only every 6th layer for brevity
    step = max(1, len(rows_a) // 8)
    print(tabulate(rows_a[::step], headers=headers_a, tablefmt="simple"))

    print(f"\n  A cosim summary: mean={np.mean(all_cos_a):.4f}, min={min(all_cos_a):.4f}, max={max(all_cos_a):.4f}")

    print("\nExperts_B cosine similarity:")
    headers_b = ["Layer"] + pair_labels
    print(tabulate(rows_b[::step], headers=headers_b, tablefmt="simple"))

    print(f"\n  B cosim summary: mean={np.mean(all_cos_b):.4f}, min={min(all_cos_b):.4f}, max={max(all_cos_b):.4f}")

    if np.mean(all_cos_b) > 0.8:
        print("  >>> H4 LIKELY: experts_b have converged to similar functions!")
    elif np.mean(all_cos_b) > 0.5:
        print("  >>> H4 PARTIAL: experts_b are moderately similar")
    else:
        print("  >>> H4 UNLIKELY: experts_b are differentiated")


def estimate_logit_range(moe_layers):
    """1.4: Analytical logit range estimation"""
    print("\n" + "-" * 60)
    print("1.4 Logit Range Estimation (Analytical)")
    print("-" * 60)

    rows = []
    for name, layer in moe_layers:
        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break
        if not block_info:
            block_info = name[-25:]

        w = layer.gate.weight.data  # [E, dim]
        b = layer.gate.bias.data    # [E]
        ne = w.shape[0]

        # Expected logit for expert i: w[i] @ x + b[i]
        # With LayerNorm x ~ N(0, 1/dim), ||x|| ~ sqrt(dim)
        # Expected logit std ≈ ||w[i]||₂ (since x elements are ~N(0,1))
        per_expert_logit_std = [w[i].norm().item() for i in range(ne)]

        # Expected max logit diff: difference between two expert logits
        # std of (w_i - w_j) @ x = ||w_i - w_j||₂
        max_logit_diff = 0
        for i in range(ne):
            for j in range(i + 1, ne):
                diff_norm = (w[i] - w[j]).norm().item()
                max_logit_diff = max(max_logit_diff, diff_norm)

        # Expected softmax deviation: if logit diff < 0.1, softmax ≈ uniform
        rows.append([
            block_info,
            f"[{', '.join(f'{x:.4f}' for x in per_expert_logit_std)}]",
            f"[{', '.join(f'{x:.4f}' for x in b.tolist())}]",
            f"{max_logit_diff:.4f}",
            "UNIFORM" if max_logit_diff < 0.5 else "some spread"
        ])

    headers = ["Layer", "Logit_std/E", "Bias", "Max_logit_diff", "Prediction"]
    step = max(1, len(rows) // 8)
    print(tabulate(rows[::step], headers=headers, tablefmt="simple"))

    diffs = [float(r[3]) for r in rows]
    print(f"\n  Max logit diff: mean={np.mean(diffs):.4f}, max={max(diffs):.4f}")
    if np.mean(diffs) < 0.5:
        print("  >>> Pre-softmax logits are too close → softmax will be near-uniform")


# ============================================================================
# Part 2: Forward Pass Analysis
# ============================================================================

def analyze_forward(model, dataloader, device, cfg):
    """Run forward pass analysis."""
    print("\n" + "=" * 80)
    print("SECTION 2: FORWARD PASS ANALYSIS")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    moe_layers = get_moe_layers(model)

    # Prepare hooks
    hook_data = defaultdict(list)
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            x = input[0]  # input to SoftMoE_LoRA_Layer
            with torch.no_grad():
                gate_logits = module.gate(x)
                gate_weights = F.softmax(gate_logits, dim=-1)

                # Store statistics (detached, on CPU)
                hook_data[layer_name].append({
                    'gate_logits_mean': gate_logits.mean(dim=tuple(range(gate_logits.dim()-1))).cpu(),
                    'gate_logits_std': gate_logits.std(dim=tuple(range(gate_logits.dim()-1))).cpu(),
                    'gate_weights_mean': gate_weights.mean(dim=tuple(range(gate_weights.dim()-1))).cpu(),
                    'gate_logit_range': (gate_logits.max(dim=-1).values - gate_logits.min(dim=-1).values).mean().cpu(),
                    'input_mean': x.mean(dim=tuple(range(x.dim()-1))).cpu(),
                    'input_norm': x.norm(dim=-1).mean().cpu(),
                    'input_std': x.std(dim=-1).mean().cpu(),
                })
        return hook_fn

    for name, layer in moe_layers:
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

    modals = cfg['DATASET']['MODALS']

    # Run forward pass for each modality separately
    print(f"\nRunning forward passes for {len(modals)} modalities...")

    # Get one batch
    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]

    modal_hook_data = {}

    for m_idx, modal_name in enumerate(modals):
        hook_data.clear()

        # Forward pass with single modality repeated (to isolate per-modal tokens)
        # Actually, the model processes modalities sequentially in a for loop,
        # so we need to run the full model and look at the sequential hook calls.
        pass

    # Remove hooks
    for h in hooks:
        h.remove()
    hooks.clear()

    # Instead, let's use a more targeted approach:
    # Run full forward and capture hooks per step
    analyze_gate_distributions(model, dataloader, device, moe_layers, modals)
    analyze_expert_outputs(model, dataloader, device, moe_layers)
    analyze_sensitivity(model, dataloader, device, moe_layers)


def analyze_gate_distributions(model, dataloader, device, moe_layers, modals):
    """2.1 + 2.2: Gate logits/weights per modality"""
    print("\n" + "-" * 60)
    print("2.1/2.2 Gate Distributions Per Modality")
    print("-" * 60)

    model.eval()
    num_modals = len(modals)

    # For each MoE layer, we'll collect per-call data
    # The model processes modalities sequentially, so hooks fire num_modals times per forward
    call_data = defaultdict(list)

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            x = input[0]
            with torch.no_grad():
                gate_logits = module.gate(x)
                gate_weights = F.softmax(gate_logits, dim=-1)

                # Per-token entropy: how decisive is each token's routing?
                per_token_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1)  # (...,)
                max_entropy = math.log(module.num_experts)  # uniform entropy

                # Per-token max gate weight: how peaked is each token?
                per_token_max = gate_weights.max(dim=-1).values  # (...,)

                # Argmax distribution: which expert gets most tokens?
                argmax = gate_weights.argmax(dim=-1)  # (...,)
                expert_counts = [(argmax == i).float().mean().item() for i in range(module.num_experts)]

                call_data[layer_name].append({
                    'logits_mean_per_e': gate_logits.mean(dim=tuple(range(gate_logits.dim()-1))).cpu().numpy(),
                    'logits_std': gate_logits.std().cpu().item(),
                    'weights_mean_per_e': gate_weights.mean(dim=tuple(range(gate_weights.dim()-1))).cpu().numpy(),
                    'logit_range': (gate_logits.max(dim=-1).values - gate_logits.min(dim=-1).values).mean().cpu().item(),
                    'input_mean_vec': x.mean(dim=tuple(range(x.dim()-1))).cpu().numpy(),
                    'input_l2': x.norm(dim=-1).mean().cpu().item(),
                    # NEW: per-token analysis
                    'per_token_entropy_mean': per_token_entropy.mean().cpu().item(),
                    'per_token_entropy_std': per_token_entropy.std().cpu().item(),
                    'entropy_ratio': per_token_entropy.mean().cpu().item() / max_entropy,
                    'per_token_max_mean': per_token_max.mean().cpu().item(),
                    'per_token_max_std': per_token_max.std().cpu().item(),
                    'expert_argmax_frac': expert_counts,
                })
        return hook_fn

    hooks = []
    for name, layer in moe_layers:
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

    # Forward pass
    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]

    call_data.clear()
    with torch.no_grad():
        model(images, multimask_output=True)

    for h in hooks:
        h.remove()

    # Analyze: each layer should have num_modals calls
    layer_names = list(call_data.keys())

    # Print table for sampled layers
    step = max(1, len(layer_names) // 8)
    sampled = layer_names[::step]

    for lname in sampled:
        calls = call_data[lname]
        if len(calls) < num_modals:
            continue

        parts = lname.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break

        print(f"\n  {block_info}:")
        for m_idx in range(min(num_modals, len(calls))):
            d = calls[m_idx]
            w_str = ', '.join(f'{x:.4f}' for x in d['weights_mean_per_e'])
            l_str = ', '.join(f'{x:.4f}' for x in d['logits_mean_per_e'])
            argmax_str = ', '.join(f'{x:.2f}' for x in d['expert_argmax_frac'])
            print(f"    {modals[m_idx]:>8s}: weights=[{w_str}]  logits=[{l_str}]  logit_range={d['logit_range']:.4f}")
            print(f"              per_token: entropy_ratio={d['entropy_ratio']:.4f}  max_weight={d['per_token_max_mean']:.4f}±{d['per_token_max_std']:.4f}  argmax_frac=[{argmax_str}]")

    # Cross-modality input comparison (H3)
    print("\n" + "-" * 60)
    print("2.2 Cross-Modality Input Comparison [H3: LayerNorm homogenizes?]")
    print("-" * 60)

    cos_sims_all = []
    for lname in sampled:
        calls = call_data[lname]
        if len(calls) < num_modals:
            continue

        parts = lname.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break

        sims = []
        for i in range(num_modals):
            for j in range(i + 1, num_modals):
                vi = torch.tensor(calls[i]['input_mean_vec'])
                vj = torch.tensor(calls[j]['input_mean_vec'])
                cs = F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
                sims.append(cs)
                cos_sims_all.append(cs)

        pair_labels = [f"{modals[i]}-{modals[j]}" for i in range(num_modals) for j in range(i + 1, num_modals)]
        sim_str = ', '.join(f'{pair_labels[k]}={sims[k]:.4f}' for k in range(len(sims)))
        print(f"  {block_info}: {sim_str}")

    if cos_sims_all:
        mean_sim = np.mean(cos_sims_all)
        print(f"\n  Mean cross-modal cosine similarity: {mean_sim:.4f}")
        if mean_sim > 0.95:
            print("  >>> H3 LIKELY: LayerNorm makes all modalities look nearly identical!")
        elif mean_sim > 0.8:
            print("  >>> H3 PARTIAL: Modalities are somewhat similar after LayerNorm")
        else:
            print("  >>> H3 UNLIKELY: Modalities retain distinguishable features")

    # Per-token routing analysis summary
    print("\n" + "-" * 60)
    print("Per-Token Routing Analysis Summary")
    print("-" * 60)
    print("(entropy_ratio=1.0 means uniform, <0.8 means meaningful routing)")
    print("(per_token_max > 0.5 means tokens are routed decisively)\n")

    all_entropy_ratios = []
    all_max_weights = []
    for lname in layer_names:
        calls = call_data[lname]
        for d in calls:
            all_entropy_ratios.append(d['entropy_ratio'])
            all_max_weights.append(d['per_token_max_mean'])

    if all_entropy_ratios:
        mean_er = np.mean(all_entropy_ratios)
        mean_mw = np.mean(all_max_weights)
        print(f"  Overall entropy ratio: {mean_er:.4f} (1.0 = perfectly uniform)")
        print(f"  Overall mean max-weight: {mean_mw:.4f} (1/{moe_layers[0][1].num_experts} = {1/moe_layers[0][1].num_experts:.4f} for uniform)")

        if mean_er > 0.95:
            print("  >>> Per-token routing IS uniform — NOT a spatial averaging artifact!")
            print("  >>> This means each individual token gets near-equal weights for all experts.")
        elif mean_er > 0.8:
            print("  >>> Per-token routing has MILD specialization")
            print("  >>> Some tokens route differently, but not strongly.")
        else:
            print("  >>> Per-token routing is DIVERSE — the spatial average hides real routing!")
            print("  >>> The gate IS working at per-token level.")


def analyze_expert_outputs(model, dataloader, device, moe_layers):
    """2.3: Expert output analysis"""
    print("\n" + "-" * 60)
    print("2.3 Expert Output Analysis [H1/H4]")
    print("-" * 60)

    model.eval()

    # Capture inputs to MoE layers
    input_data = defaultdict(list)

    def make_input_hook(layer_name):
        def hook_fn(module, input, output):
            # Only capture from first modality call
            if len(input_data[layer_name]) < 1:
                input_data[layer_name].append(input[0].detach())
        return hook_fn

    hooks = []
    for name, layer in moe_layers:
        h = layer.register_forward_hook(make_input_hook(name))
        hooks.append(h)

    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]

    with torch.no_grad():
        model(images, multimask_output=True)

    for h in hooks:
        h.remove()

    # Now compute expert outputs manually
    step = max(1, len(moe_layers) // 8)
    sampled_layers = moe_layers[::step]

    rows = []
    for name, layer in sampled_layers:
        if name not in input_data or not input_data[name]:
            continue

        x = input_data[name][0]  # (B, H, W, C) or (B*nw, wh, ww, C)

        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break

        with torch.no_grad():
            expert_outs = []
            for i in range(layer.num_experts):
                out = layer.experts_b[i](layer.experts_a[i](x))
                expert_outs.append(out)

            # Expert output norms
            e_norms = [eo.norm().item() / max(eo.numel(), 1) * 1000 for eo in expert_outs]

            # Pairwise cosine similarity of flattened expert outputs
            cos_sims = []
            for i in range(layer.num_experts):
                for j in range(i + 1, layer.num_experts):
                    cs = F.cosine_similarity(
                        expert_outs[i].flatten().unsqueeze(0),
                        expert_outs[j].flatten().unsqueeze(0)
                    ).item()
                    cos_sims.append(cs)

            # Compare to QKV output
            # The original QKV is not directly accessible here, but we can compare to input norm
            input_norm = x.norm().item() / max(x.numel(), 1) * 1000
            lora_ratio = np.mean(e_norms) / input_norm if input_norm > 0 else 0

        rows.append([
            block_info,
            f"[{', '.join(f'{x:.4f}' for x in e_norms)}]",
            f"[{', '.join(f'{x:.4f}' for x in cos_sims)}]",
            f"{lora_ratio:.4f}"
        ])

    ne = moe_layers[0][1].num_experts
    pair_labels = [f"E{i}-E{j}" for i in range(ne) for j in range(i + 1, ne)]
    headers = ["Layer", "Expert_out_norms(×1k/numel)", f"Cosine_sim({','.join(pair_labels)})", "LoRA/Input ratio"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    all_cos = []
    for r in rows:
        vals = r[2].strip('[]').split(', ')
        all_cos.extend(float(v) for v in vals)
    if all_cos:
        print(f"\n  Expert output cosine sim: mean={np.mean(all_cos):.4f}")
        if np.mean(all_cos) > 0.9:
            print("  >>> H4 CONFIRMED: Expert outputs are nearly identical!")
        elif np.mean(all_cos) > 0.7:
            print("  >>> H4 PARTIAL: Expert outputs are quite similar")


def analyze_sensitivity(model, dataloader, device, moe_layers):
    """2.4: Sensitivity test — how much does expert selection matter?"""
    print("\n" + "-" * 60)
    print("2.4 Sensitivity Test (Gate Perturbation)")
    print("-" * 60)

    model.eval()

    # Pick a few representative layers
    test_layers = []
    indices = [0, len(moe_layers) // 4, len(moe_layers) // 2, -1]
    for idx in indices:
        test_layers.append(moe_layers[idx])

    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]

    # Capture input to test layers
    input_data = {}

    def make_input_hook(layer_name):
        def hook_fn(module, input, output):
            if layer_name not in input_data:
                input_data[layer_name] = input[0].detach()
        return hook_fn

    hooks = []
    for name, layer in test_layers:
        h = layer.register_forward_hook(make_input_hook(name))
        hooks.append(h)

    with torch.no_grad():
        model(images, multimask_output=True)

    for h in hooks:
        h.remove()

    # For each test layer, compute output with different forced gates
    rows = []
    for name, layer in test_layers:
        if name not in input_data:
            continue

        x = input_data[name]
        ne = layer.num_experts

        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{idx}_{qv}"
                break

        with torch.no_grad():
            # Normal output
            normal_out = layer(x)
            normal_norm = normal_out.norm().item()

            # Force each expert exclusively
            forced_outputs = []
            for e in range(ne):
                forced_out = layer.experts_b[e](layer.experts_a[e](x))
                forced_outputs.append(forced_out)

            # L2 distance between forced outputs
            dists = []
            for i in range(ne):
                for j in range(i + 1, ne):
                    d = (forced_outputs[i] - forced_outputs[j]).norm().item()
                    dists.append(d)

            # Relative difference: dist / normal_norm
            rel_dists = [d / normal_norm if normal_norm > 0 else 0 for d in dists]

        pair_labels = [f"E{i}-E{j}" for i in range(ne) for j in range(i + 1, ne)]
        rows.append([
            block_info,
            f"{normal_norm:.4f}",
            f"[{', '.join(f'{d:.4f}' for d in dists)}]",
            f"[{', '.join(f'{d:.4f}' for d in rel_dists)}]",
        ])

    headers = ["Layer", "Normal_out_norm", "L2_dist(forced pairs)", "Relative_dist"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    all_rel = []
    for r in rows:
        vals = r[3].strip('[]').split(', ')
        all_rel.extend(float(v) for v in vals)
    if all_rel:
        print(f"\n  Relative distance: mean={np.mean(all_rel):.4f}")
        if np.mean(all_rel) < 0.1:
            print("  >>> Expert selection DOESN'T MATTER — all experts produce nearly identical output!")


# ============================================================================
# Part 3: Gradient Flow Analysis
# ============================================================================

def analyze_gradients(model, dataloader, device, cfg, moe_layers):
    """Section 3: Gradient flow analysis."""
    print("\n" + "=" * 80)
    print("SECTION 3: GRADIENT FLOW ANALYSIS")
    print("=" * 80)

    model = model.to(device)
    model.train()

    analyze_gate_gradients(model, dataloader, device, cfg, moe_layers)
    analyze_counterfactual(model, dataloader, device, cfg, moe_layers)


def analyze_gate_gradients(model, dataloader, device, cfg, moe_layers):
    """3.1: Gate gradient magnitude during one training step"""
    print("\n" + "-" * 60)
    print("3.1 Gate Gradient Magnitude [H1 Core Test]")
    print("-" * 60)

    model.train()

    # Zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Get one batch
    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]
    labels = labels.to(device)

    # Forward pass
    model_out = model(images, multimask_output=True)

    # Extract output
    if isinstance(model_out, (tuple, list)):
        if len(model_out) >= 2:
            output = model_out[0]
        else:
            output = model_out
    else:
        output = model_out

    # Simple cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    logits = F.interpolate(output, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    loss = loss_fn(logits, labels)

    print(f"  Loss value: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Collect gradient norms
    rows = []
    all_gate_grads = []
    all_expert_a_grads = []
    all_expert_b_grads = []

    step = max(1, len(moe_layers) // 8)

    for idx, (name, layer) in enumerate(moe_layers):
        parts = name.split('.')
        block_info = ""
        for i, p in enumerate(parts):
            if p in ('moe_layers_q', 'moe_layers_v'):
                qv = 'Q' if 'q' in p else 'V'
                layer_idx = parts[i + 1] if i + 1 < len(parts) else '?'
                block_info = f"Block{layer_idx}_{qv}"
                break

        # Gate gradient
        gate_w_grad = layer.gate.weight.grad
        gate_b_grad = layer.gate.bias.grad
        gate_w_norm = gate_w_grad.norm().item() if gate_w_grad is not None else 0.0
        gate_b_norm = gate_b_grad.norm().item() if gate_b_grad is not None else 0.0

        # Expert gradients
        ea_grads = []
        eb_grads = []
        for i in range(layer.num_experts):
            ea_g = layer.experts_a[i].weight.grad
            eb_g = layer.experts_b[i].weight.grad
            ea_grads.append(ea_g.norm().item() if ea_g is not None else 0.0)
            eb_grads.append(eb_g.norm().item() if eb_g is not None else 0.0)

        all_gate_grads.append(gate_w_norm)
        all_expert_a_grads.extend(ea_grads)
        all_expert_b_grads.extend(eb_grads)

        mean_ea = np.mean(ea_grads)
        mean_eb = np.mean(eb_grads)
        ratio = gate_w_norm / mean_ea if mean_ea > 0 else float('inf')

        if idx % step == 0:
            rows.append([
                block_info,
                f"{gate_w_norm:.6f}",
                f"{gate_b_norm:.6f}",
                f"{mean_ea:.6f}",
                f"{mean_eb:.6f}",
                f"{ratio:.4f}"
            ])

    headers = ["Layer", "Gate_W_grad", "Gate_B_grad", "Mean_EA_grad", "Mean_EB_grad", "Gate/EA ratio"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    mean_gate = np.mean(all_gate_grads)
    mean_ea = np.mean(all_expert_a_grads)
    mean_eb = np.mean(all_expert_b_grads)

    print(f"\n  Overall: gate_grad={mean_gate:.6f}, expert_a_grad={mean_ea:.6f}, expert_b_grad={mean_eb:.6f}")
    print(f"  Gate/Expert_A ratio: {mean_gate / mean_ea if mean_ea > 0 else 'inf':.4f}")

    if mean_gate < mean_ea * 0.01:
        print("  >>> H1 CONFIRMED: Gate gradient is >100x smaller than expert gradient!")
        print("  >>> The gate has essentially no learning signal from the main loss.")
    elif mean_gate < mean_ea * 0.1:
        print("  >>> H1 PARTIALLY CONFIRMED: Gate gradient is ~10x smaller")

    # Zero gradients for next test
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def analyze_counterfactual(model, dataloader, device, cfg, moe_layers):
    """3.2: Counterfactual test — break expert symmetry, check gate gradient"""
    print("\n" + "-" * 60)
    print("3.2 Counterfactual Test (Break Expert Symmetry)")
    print("-" * 60)

    model.train()

    # Save original experts_b weights
    saved_weights = {}
    for name, layer in moe_layers:
        saved_weights[name] = []
        for i in range(layer.num_experts):
            saved_weights[name].append(layer.experts_b[i].weight.data.clone())

    # Replace with different random values
    print("  Temporarily replacing experts_b with random asymmetric values...")
    torch.manual_seed(42)
    for name, layer in moe_layers:
        dim = layer.in_features
        rank = layer.rank
        for i in range(layer.num_experts):
            scale = 0.1 * (i + 1)  # Different scale per expert
            layer.experts_b[i].weight.data = scale * torch.randn(dim, rank).to(device)

    # Zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # Forward + backward
    batch = next(iter(dataloader))
    images, labels = batch
    images = [x.to(device) for x in images]
    labels = labels.to(device)

    model_out = model(images, multimask_output=True)
    output = model_out[0] if isinstance(model_out, (tuple, list)) else model_out

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    logits = F.interpolate(output, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    loss = loss_fn(logits, labels)
    loss.backward()

    # Collect gate gradients with broken symmetry
    counterfactual_grads = []
    step = max(1, len(moe_layers) // 8)
    rows = []

    for idx, (name, layer) in enumerate(moe_layers):
        gate_w_grad = layer.gate.weight.grad
        gate_w_norm = gate_w_grad.norm().item() if gate_w_grad is not None else 0.0
        counterfactual_grads.append(gate_w_norm)

        if idx % step == 0:
            parts = name.split('.')
            block_info = ""
            for i, p in enumerate(parts):
                if p in ('moe_layers_q', 'moe_layers_v'):
                    qv = 'Q' if 'q' in p else 'V'
                    layer_idx = parts[i + 1] if i + 1 < len(parts) else '?'
                    block_info = f"Block{layer_idx}_{qv}"
                    break
            rows.append([block_info, f"{gate_w_norm:.6f}"])

    headers = ["Layer", "Gate_W_grad (counterfactual)"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Compare with original
    mean_cf = np.mean(counterfactual_grads)
    print(f"\n  Counterfactual mean gate gradient: {mean_cf:.6f}")
    print("  (Compare with original gate gradient from 3.1)")

    # Restore original weights
    print("\n  Restoring original experts_b weights...")
    for name, layer in moe_layers:
        for i in range(layer.num_experts):
            layer.experts_b[i].weight.data = saved_weights[name][i].to(device)

    # Zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ============================================================================
# Part 4: Main Execution and Summary
# ============================================================================

def print_summary():
    """Print diagnostic summary."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print("""
Review the results above and check:

H1 (Expert Symmetry):
  - Section 1.2: Are experts_b norms small or similar across experts?
  - Section 2.3: Are expert output cosine similarities > 0.9?
  - Section 3.1: Is gate gradient << expert gradient?
  - Section 3.2: Does counterfactual test show much larger gate gradients?

H2 (Gate Weight ≈ 0):
  - Section 1.1: Is trained gate weight norm < 1.5x init norm?
  - Section 1.4: Are predicted logit ranges near 0?

H3 (LayerNorm Homogenizes):
  - Section 2.2: Is cross-modal cosine similarity > 0.95?

H4 (Expert Output Similarity):
  - Section 1.3: Is experts_b cosine similarity > 0.8?
  - Section 2.3: Are expert output cosine similarities > 0.9?
  - Section 2.4: Is relative distance between forced expert outputs < 0.1?

Root Cause Identification:
  If H1 + H4 → Expert symmetry never broken (structural issue)
  If H2 alone → Gate didn't learn (LR or init issue)
  If H3 → Gate has no distinguishing signal (architecture issue)
""")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 80)
    print(f"MoE Gate Diagnostic: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # Build and load model
    model = build_model(cfg)
    model = load_checkpoint(model, args.checkpoint)

    moe_layers = get_moe_layers(model)
    print(f"\nTotal SoftMoE_LoRA_Layer count: {len(moe_layers)}")

    # Print layer dimensions
    dims = set()
    for name, layer in moe_layers:
        dims.add(layer.in_features)
    print(f"Gate dimensions found: {sorted(dims)}")
    print(f"Num experts: {moe_layers[0][1].num_experts}")
    print(f"LoRA rank: {moe_layers[0][1].rank}")

    # Section 1: Static Analysis (always runs)
    analyze_static(model)

    if args.static_only:
        print("\n[Static analysis complete. Use --skip-gradient or full mode for more.]")
        print_summary()
        return

    # Section 2: Forward Pass Analysis
    print("\nBuilding validation dataloader...")
    dataloader = build_val_dataloader(cfg, args.num_samples)
    analyze_forward(model, dataloader, args.device, cfg)

    # Section 3: Gradient Analysis
    if not args.skip_gradient:
        analyze_gradients(model, dataloader, args.device, cfg, moe_layers)

    print_summary()


if __name__ == '__main__':
    main()
