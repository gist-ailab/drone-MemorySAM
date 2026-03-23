"""
Feature Analysis Visualization for P9 / P22 / P25.

각 모달리티의 UAMM 이전/이후 backbone feature (PCA), 단일 모달리티 mask,
AMF 이후 fused mask를 비교 시각화.

출력 레이아웃 (이미지당 1장):
  Row 0: [Input: RGB | Mod1 | Mod2 | ...]
  Row 1: [Backbone Feature PCA: Mod0 | Mod1 | Mod2 | ...]           ← UAMM 이전
  Row 2: [Per-modal Mask: Mod0 | Mod1 | Mod2 | ...]                 ← UAMM 이후 (memory-attended)
  Row 3: [AMF Fused Mask | GT | Overlay | UAMM/AMF weights bar]

사용:
  python vis_feature_analysis.py \
    --cfg configs/eval_config/<config>.yaml \
    --model_path <checkpoint_path> \
    --mode val \
    --save_dir outputs/vis_feature_analysis \
    --max_images 20

  # DELIVER 데이터셋
  python vis_feature_analysis.py \
    --cfg configs/eval_config/<deliver_config>.yaml \
    --model_path <checkpoint_path> \
    --mode val --dataset DELIVER \
    --save_dir outputs/vis_feature_analysis
"""
import torch
import argparse
import yaml
import os
import json
import inspect
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *


# ─── Model Loading (val_multiaqua.py 와 동일) ─────────────────────────
def load_model(cfg, model_path, device):
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_config_file = "sam2_hiera_b+.yaml"
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        sam2_config_file, checkpoint,
        hydra_overrides_extra=["++model._target_=semseg.models.sam2.sam2.sam2_image_predictor_seg.SAM2ImagePredictor"],
    )
    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P9')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K', 2)
    lora_layer = model_cfg.get('LORA_LAYER', None)

    lora_model_class = globals()[lora_model_name]
    model_kwargs = {'sam_model': sam2, 'r': lora_r, 'lora_layer': lora_layer}

    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k
    if 'num_classes' in sig.parameters:
        n_cls = dataset_cfg.get('NUM_CLASSES', 4)
        model_kwargs['num_classes'] = n_cls
    # P24/P25/P26 quality gate
    if 'quality_hidden_dim' in sig.parameters:
        qg_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['quality_hidden_dim'] = qg_cfg.get('HIDDEN_DIM', 64)
        model_kwargs['quality_min'] = qg_cfg.get('MIN_QUALITY', 0.1)
    if 'tau_uamm' in sig.parameters:
        qg_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['tau_uamm'] = qg_cfg.get('TAU_UAMM', 1.0)
        model_kwargs['tau_teacher'] = qg_cfg.get('TAU_TEACHER', 0.5)
        model_kwargs['memory_mod'] = qg_cfg.get('MEMORY_MOD', False)
        model_kwargs['amf_mode'] = qg_cfg.get('AMF_MODE', 'output_entropy')
        model_kwargs['multi_scale_sqg'] = qg_cfg.get('MULTI_SCALE_SQG', True)
        model_kwargs['per_modality_decoder'] = qg_cfg.get('PER_MODALITY_DECODER', True)
    if 'cond_dim' in sig.parameters:
        model_kwargs['cond_dim'] = model_cfg.get('LORA_COND_DIM', 8)

    model = lora_model_class(**model_kwargs)

    # Load checkpoint
    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Model load: {msg}")
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = model.to(device)
    model.eval()
    return model


# ─── Feature PCA Visualization ──────────────────────────────────────
def feat_to_pca_rgb(feat_tensor, target_h=None, target_w=None):
    """
    (C, H, W) feature map → (H, W, 3) RGB via PCA.
    Top-3 PCA components normalized to [0, 255].
    """
    C, H, W = feat_tensor.shape
    feat_flat = feat_tensor.reshape(C, -1).T.numpy()  # (HW, C)
    n_components = min(3, C)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feat_flat)  # (HW, 3)
    # Normalize each component to [0, 1]
    for i in range(n_components):
        col = pca_result[:, i]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-8:
            pca_result[:, i] = (col - mn) / (mx - mn)
        else:
            pca_result[:, i] = 0.5
    if n_components < 3:
        pca_result = np.concatenate([pca_result, np.zeros((pca_result.shape[0], 3 - n_components))], axis=1)
    rgb = (pca_result.reshape(H, W, 3) * 255).clip(0, 255).astype(np.uint8)
    if target_h and target_w:
        rgb = np.array(Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.LANCZOS))
    return rgb


# ─── Mask Colorization ──────────────────────────────────────────────
def colorize_mask(mask_np, palette, n_classes):
    """(H, W) class indices → (H, W, 3) RGB."""
    h, w = mask_np.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(n_classes):
        if isinstance(palette, torch.Tensor):
            color = palette[cls_idx].cpu().numpy().astype(np.uint8)
        else:
            color = np.array(palette[cls_idx], dtype=np.uint8)
        colored[mask_np == cls_idx] = color
    return colored


# ─── Bar Chart Drawing ──────────────────────────────────────────────
def draw_weight_bars(values_dict, target_h, target_w):
    """
    Draw grouped horizontal bar chart.
    values_dict: {'UAMM': [v0, v1, ...], 'AMF': [v0, v1, ...]}
    """
    fig, axes = plt.subplots(1, len(values_dict), figsize=(target_w / 80, target_h / 80), dpi=80)
    if len(values_dict) == 1:
        axes = [axes]
    for ax, (title, vals) in zip(axes, values_dict.items()):
        labels, values = zip(*vals)
        y_pos = np.arange(len(values))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(values)))
        bars = ax.barh(y_pos, values, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=14)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, fontsize=16, fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


# ─── Quality Map Heatmap (P25) ──────────────────────────────────────
def quality_map_to_heatmap(qmap, target_h, target_w):
    """(1, H, W) or (H, W) quality map → (target_h, target_w, 3) heatmap."""
    if qmap.ndim == 3:
        qmap = qmap[0]
    fig, ax = plt.subplots(figsize=(target_w / 80, target_h / 80), dpi=80)
    im = ax.imshow(qmap, cmap='jet', vmin=0, vmax=1, aspect='auto')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf).reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    return np.array(Image.fromarray(img).resize((target_w, target_h), Image.Resampling.LANCZOS))


# ─── Unpad/Resize (val_multiaqua.py와 동일) ─────────────────────────
def _unpad_resize_to_orig(pred, orig_h, orig_w, model_size=1024):
    H, W = orig_h, orig_w
    t = model_size
    if W >= H:
        scale = t / W
        nH = round(H * scale)
        pad_top = (t - nH) // 2
        pred_content = pred[pad_top:pad_top + nH, :t]
    else:
        scale = t / H
        nW = round(W * scale)
        pad_left = (t - nW) // 2
        pred_content = pred[:t, pad_left:pad_left + nW]
    if pred_content.shape[0] != H or pred_content.shape[1] != W:
        pred_content = pred_content.unsqueeze(0).unsqueeze(0).float()
        pred_resized = F.interpolate(pred_content, size=(H, W), mode="nearest")
        pred_resized = pred_resized.squeeze(0).squeeze(0).long()
    else:
        pred_resized = pred_content.long()
    return pred_resized


# ─── Input image denormalize ─────────────────────────────────────────
def denorm_tensor(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """(3, H, W) normalized tensor → (H, W, 3) uint8."""
    t = tensor.cpu().float()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return (t * 255).astype(np.uint8)


def modal_tensor_to_rgb(tensor):
    """(C, H, W) modality tensor → (H, W, 3) uint8. Handles 1-ch or 3-ch."""
    t = tensor.cpu().float()
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] > 3:
        t = t[:3]
    # Normalize to [0, 1]
    mn, mx = t.min(), t.max()
    if mx - mn > 1e-6:
        t = (t - mn) / (mx - mn)
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


# ─── Collate ─────────────────────────────────────────────────────────
def _collate_with_meta(batch):
    """For datasets that return (sample, label, meta)."""
    samples = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    images = [torch.stack([s[i] for s in samples]) for i in range(len(samples[0]))]
    labels = torch.stack(labels)
    return images, labels, metas


def _collate_no_meta(batch):
    """For datasets that return (sample, label) without meta."""
    samples = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    images = [torch.stack([s[i] for s in samples]) for i in range(len(samples[0]))]
    labels = torch.stack(labels)
    return images, labels, [{}] * len(labels)


# ─── Main Visualization ─────────────────────────────────────────────
@torch.no_grad()
def run_visualization(model, dataloader, device, save_dir, modals, max_images=0,
                      dataset_name='MULTIAQUA'):
    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE
    is_multiaqua = (dataset_name == 'MULTIAQUA')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vis_count = 0
    summary_log = {}

    for batch in tqdm(dataloader, desc="Feature visualization"):
        images, labels, metas = batch
        images = [x.to(device) for x in images]

        output, feat = model(images, multimask_output=True)

        core = model.module if hasattr(model, 'module') else model
        per_modal_outputs = getattr(core, '_last_per_modal_outputs', None)
        per_modal_feats = getattr(core, '_last_per_modal_feats', None)
        uamm_scores = getattr(core, '_last_uamm_scores', None)
        amf_weights = getattr(core, '_last_amf_weights', None)
        quality_maps = getattr(core, '_last_quality_maps', None)  # P25 only

        if per_modal_outputs is None or per_modal_feats is None:
            print("[WARN] Model does not store per-modal outputs. Skipping.")
            return

        m = len(modals)
        fused_pred = output.softmax(dim=1)[:, :n_classes].argmax(dim=1).cpu()  # (B, H, W)

        for b in range(images[0].shape[0]):
            if max_images > 0 and vis_count >= max_images:
                return

            meta = metas[b] if metas[b] else {}
            stem = meta.get('stem', f'img_{vis_count:05d}')
            orig_h = meta.get('orig_h', images[0].shape[-2])
            orig_w = meta.get('orig_w', images[0].shape[-1])
            vis_h, vis_w = 256, 256  # cell size for feature/mask panels

            # ── Row 0: Input modalities ──
            input_panels = []
            for i, mk in enumerate(modals):
                if mk == 'img':
                    panel = denorm_tensor(images[i][b])
                else:
                    panel = modal_tensor_to_rgb(images[i][b])
                panel = np.array(Image.fromarray(panel).resize((vis_w, vis_h), Image.Resampling.LANCZOS))
                input_panels.append(panel)
            row0 = np.concatenate(input_panels, axis=1)

            # ── Row 1: Backbone Feature PCA (UAMM 이전) ──
            feat_panels = []
            for i in range(m):
                feat_i = per_modal_feats[i][b]  # (C, H_fpn, W_fpn)
                pca_rgb = feat_to_pca_rgb(feat_i, vis_h, vis_w)
                feat_panels.append(pca_rgb)
            row1 = np.concatenate(feat_panels, axis=1)

            # ── Row 2: Per-modal Mask (UAMM 이후, memory-attended) ──
            mask_panels = []
            for i in range(m):
                modal_pred = per_modal_outputs[i][b]  # (n_classes, H, W)
                modal_mask = modal_pred[:n_classes].argmax(dim=0).numpy()  # (H, W)
                colored = colorize_mask(modal_mask, palette, n_classes)
                colored = np.array(Image.fromarray(colored).resize((vis_w, vis_h), Image.Resampling.NEAREST))
                mask_panels.append(colored)
            row2 = np.concatenate(mask_panels, axis=1)

            # ── Row 3: Fused mask | GT | Overlay | Weights ──
            # Fused mask
            fused_mask = fused_pred[b].numpy()  # (model_size, model_size)
            fused_colored = colorize_mask(fused_mask, palette, n_classes)
            fused_colored = np.array(Image.fromarray(fused_colored).resize((vis_w, vis_h), Image.Resampling.NEAREST))

            # GT (test set에서는 dummy label일 수 있음)
            gt = labels[b].numpy()
            has_gt = not (gt == 0).all()  # dummy label 체크
            if has_gt:
                gt_display = gt.copy()
                gt_display[gt == 255] = 0
                gt_colored = colorize_mask(gt_display, palette, n_classes)
                gt_colored[gt == 255] = [30, 30, 30]  # ignore region
            else:
                gt_colored = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.uint8) * 128  # gray placeholder
            gt_colored = np.array(Image.fromarray(gt_colored).resize((vis_w, vis_h), Image.Resampling.NEAREST))

            # Overlay (fused mask on first input)
            base_img = input_panels[0] if input_panels else np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
            overlay = (base_img.astype(np.float32) * 0.5 + fused_colored.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

            # Weights bar
            weight_info = {}
            if uamm_scores is not None and b < uamm_scores.shape[0]:
                weight_info['UAMM'] = list(zip(modals, uamm_scores[b].tolist()))
            if amf_weights is not None and b < amf_weights.shape[0]:
                weight_info['AMF'] = list(zip(modals, amf_weights[b].tolist()))

            remaining_w = row0.shape[1] - vis_w * 3
            if remaining_w > 0 and weight_info:
                weight_bar = draw_weight_bars(weight_info, vis_h, remaining_w)
            else:
                weight_bar = None

            row3_panels = [fused_colored, gt_colored, overlay]
            if weight_bar is not None:
                row3_panels.append(weight_bar)
            else:
                # fill remaining cols
                for _ in range(m - 3):
                    row3_panels.append(np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 240)
            row3 = np.concatenate(row3_panels, axis=1)

            # Ensure all rows same width
            target_w = row0.shape[1]
            for i, row in enumerate([row1, row2, row3]):
                if row.shape[1] != target_w:
                    pil_row = Image.fromarray(row).resize((target_w, vis_h), Image.Resampling.LANCZOS)
                    if i == 0: row1 = np.array(pil_row)
                    elif i == 1: row2 = np.array(pil_row)
                    else: row3 = np.array(pil_row)

            # ── Row 4 (P25 only): Spatial Quality Maps ──
            row4 = None
            if quality_maps is not None:
                qmap_panels = []
                for i in range(m):
                    qmap_i = quality_maps[i][b]  # (1, H_fpn, W_fpn) numpy
                    heatmap = quality_map_to_heatmap(qmap_i, vis_h, vis_w)
                    qmap_panels.append(heatmap)
                row4 = np.concatenate(qmap_panels, axis=1)
                if row4.shape[1] != target_w:
                    row4 = np.array(Image.fromarray(row4).resize((target_w, vis_h), Image.Resampling.LANCZOS))

            # ── Add row labels ──
            label_w = 120
            row_labels = ['Input', 'Backbone\n(pre-UAMM)', 'Per-modal\nMask\n(post-UAMM)', 'Fused|GT|Overlay']
            rows = [row0, row1, row2, row3]
            if row4 is not None:
                rows.append(row4)
                row_labels.append('Quality\nMap (P25)')

            labeled_rows = []
            for row, label in zip(rows, row_labels):
                label_panel = np.ones((row.shape[0], label_w, 3), dtype=np.uint8) * 255
                # Draw label text
                fig_lbl, ax_lbl = plt.subplots(figsize=(label_w / 80, row.shape[0] / 80), dpi=80)
                ax_lbl.text(0.5, 0.5, label, ha='center', va='center', fontsize=11, fontweight='bold',
                            transform=ax_lbl.transAxes)
                ax_lbl.axis('off')
                fig_lbl.tight_layout(pad=0)
                fig_lbl.canvas.draw()
                buf = fig_lbl.canvas.buffer_rgba()
                lw, lh = fig_lbl.canvas.get_width_height()
                lbl_img = np.asarray(buf).reshape((lh, lw, 4))[:, :, :3].copy()
                plt.close(fig_lbl)
                lbl_img = np.array(Image.fromarray(lbl_img).resize((label_w, row.shape[0]), Image.Resampling.LANCZOS))
                labeled_rows.append(np.concatenate([lbl_img, row], axis=1))

            # ── Column headers (modality names) ──
            header_h = 30
            header = np.ones((header_h, label_w + target_w, 3), dtype=np.uint8) * 255
            fig_h, ax_h = plt.subplots(figsize=((label_w + target_w) / 80, header_h / 80), dpi=80)
            for i, mk in enumerate(modals):
                x_pos = (label_w + i * vis_w + vis_w / 2) / (label_w + target_w)
                ax_h.text(x_pos, 0.5, mk.upper(), ha='center', va='center', fontsize=12, fontweight='bold',
                         transform=ax_h.transAxes)
            ax_h.axis('off')
            fig_h.tight_layout(pad=0)
            fig_h.canvas.draw()
            buf_h = fig_h.canvas.buffer_rgba()
            hw, hh = fig_h.canvas.get_width_height()
            header = np.asarray(buf_h).reshape((hh, hw, 4))[:, :, :3].copy()
            header = np.array(Image.fromarray(header).resize((label_w + target_w, header_h), Image.Resampling.LANCZOS))
            plt.close(fig_h)

            # ── Assemble final image ──
            final = np.concatenate([header] + labeled_rows, axis=0)
            Image.fromarray(final).save(str(save_dir / f"{stem}.png"))

            # ── JSON log ──
            img_log = {'stem': stem}
            if uamm_scores is not None and b < uamm_scores.shape[0]:
                img_log['uamm'] = {k: round(float(v), 4) for k, v in zip(modals, uamm_scores[b])}
            if amf_weights is not None and b < amf_weights.shape[0]:
                img_log['amf'] = {k: round(float(v), 4) for k, v in zip(modals, amf_weights[b])}
            summary_log[stem] = img_log

            vis_count += 1

    # Save summary
    with open(save_dir / 'analysis_log.json', 'w') as f:
        json.dump({'n_images': vis_count, 'modals': modals, 'images': summary_log}, f, indent=2)
    print(f"Saved {vis_count} visualizations to {save_dir}")


# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Analysis Visualization")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--max_images', type=int, default=0, help='0 = all')
    parser.add_argument('--dataset', type=str, default=None, help='Override dataset name (e.g., DELIVER)')
    parser.add_argument('--cell_size', type=int, default=256, help='Cell size for each panel')
    args = parser.parse_args()

    setup_cudnn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    dataset_name = args.dataset or dataset_cfg['NAME']
    modals = dataset_cfg['MODALS']

    # Override model path
    model = load_model(cfg, args.model_path, device)

    # Build dataset
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)
    ds_kwargs = {}
    if dataset_name == 'MULTIAQUA':
        night_trans = bool(dataset_cfg.get('NIGHT_TRANSLATION', False))
        ds_kwargs['night_translation'] = night_trans
        ds_kwargs['return_meta'] = True
        # test set은 annotation이 없을 수 있음
        if args.mode == 'test':
            ds_kwargs['require_annotation'] = False
        if 'NUM_CLASSES' in dataset_cfg:
            ds_kwargs['n_classes'] = dataset_cfg['NUM_CLASSES']

    dataset = eval(dataset_name)(dataset_cfg['ROOT'], args.mode, valtransform, modals, **ds_kwargs)

    # Detect collate type
    sample = dataset[0]
    has_meta = len(sample) == 3
    collate_fn = _collate_with_meta if has_meta else _collate_no_meta

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                            collate_fn=collate_fn)

    save_dir = args.save_dir
    if save_dir is None:
        model_dir = Path(args.model_path).parent
        save_dir = model_dir / f"vis_feature_{args.mode}"

    run_visualization(model, dataloader, device, save_dir, modals,
                      max_images=args.max_images, dataset_name=dataset_name)
