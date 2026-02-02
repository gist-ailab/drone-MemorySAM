import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.models.sam2.sam2.build_sam import build_sam2 as build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg_bkup import LoRA_Sam
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import LoRA_Sam_P3, LoRA_Sam_P2, LoRA_Sam_P1, LoRA_Sam_P5, LoRA_Sam_P4, LoRA_Sam_P6
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

# Global storage for hooks
moe_gating_data = defaultdict(list)
feature_maps_before = []
feature_maps_after = []
confidence_scores = []

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def patch_moe_layer_for_analysis(model):
    """Patch MoE_LoRA_Layer forward to capture gate_probs"""
    from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import MoE_LoRA_Layer
    
    original_forward = MoE_LoRA_Layer.forward
    
    def patched_forward(self, x):
        # Call original forward
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Calculate Routing Logits
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Store gate_probs for analysis
        self._gate_probs = gate_probs.detach()
        
        # Continue with original logic
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        final_output = torch.zeros_like(x_flat)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, indices, 1.0)
        masked_probs = gate_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        for i in range(self.num_experts):
            expert_weight = masked_probs[:, i].unsqueeze(-1)
            if expert_weight.sum() == 0:
                continue
            expert_out = self.experts_b[i](self.experts_a[i](x_flat))
            final_output += expert_weight * expert_out
        
        return final_output.view(original_shape)
    
    # Patch the forward method
    MoE_LoRA_Layer.forward = patched_forward
    return original_forward

def restore_moe_layer(model, original_forward):
    """Restore original MoE_LoRA_Layer forward"""
    from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import MoE_LoRA_Layer
    MoE_LoRA_Layer.forward = original_forward

# Wrapper to capture intermediate outputs
class AnalysisWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.modality_logits_list = []
        self.vision_feats_before = []
        self.vision_feats_after = []
        self.image_embeddings_list = []
        self.moe_gating_per_modality = defaultdict(list)
        
    def forward(self, batched_input, multimask_output=True):
        if isinstance(self.model, (LoRA_Sam_P5, LoRA_Sam_P6)):
            m = len(batched_input)
            image_embedding, vision_feats = [], []
            
            # Extract features before modulation
            for i in range(m):
                img_emb = self.model.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                _, v_feats, _, _ = self.model.sam._prepare_backbone_features(img_emb)
                vision_feats.append(v_feats)
                
                # Store before modulation
                if len(v_feats) > 0:
                    self.vision_feats_before.append(v_feats[0].detach().cpu())
            
            # Store image embeddings for visualization
            self.image_embeddings_list.append([img_emb['backbone_fpn'][0].detach().cpu() for img_emb in image_embedding])
            
            # Forward through model
            output, m_feat = self.model(batched_input, multimask_output)
            
            # Extract confidence scores by re-running confidence head
            modality_logits = []
            for i in range(m):
                score_source = image_embedding[i]['backbone_fpn'][0]
                logits = self.model.confidence_head(score_source)
                modality_logits.append(logits.detach().cpu())
            
            self.modality_logits_list.append(torch.cat(modality_logits, dim=1))
            self.vision_feats_after.append(m_feat.detach().cpu())
            
            # Capture MoE gating for each modality
            if hasattr(self.model, 'moe_layers_q'):
                for mod_idx in range(m):
                    gate_probs_list = []
                    for layer in self.model.moe_layers_q:
                        if hasattr(layer, '_gate_probs'):
                            gate_probs = layer._gate_probs.detach().cpu().numpy()
                            # Average across tokens
                            if len(gate_probs.shape) > 1:
                                gate_probs = np.mean(gate_probs, axis=0)
                            gate_probs_list.append(gate_probs)
                    if gate_probs_list:
                        avg_gate_probs = np.mean(gate_probs_list, axis=0)
                        self.moe_gating_per_modality[mod_idx].append(avg_gate_probs)
            
            return output, m_feat
        else:
            return self.model(batched_input, multimask_output)

@torch.no_grad()
def evaluate_with_analysis(model, dataloader, device, modals, save_dir):
    """Evaluate model and collect analysis data"""
    print('Evaluating with detailed analysis...')
    
    # Wrap model for analysis
    wrapped_model = AnalysisWrapper(model)
    wrapped_model.eval()
    wrapped_model = wrapped_model.to(device)
    
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    
    # Clear global storage
    global moe_gating_data, feature_maps_before, feature_maps_after, confidence_scores
    moe_gating_data = defaultdict(list)
    feature_maps_before = []
    feature_maps_after = []
    confidence_scores = []
    
    # Patch MoE layers to capture gate_probs
    original_forward = patch_moe_layer_for_analysis(model)
    
    # Register hooks for MoE gating
    moe_hooks = []
    if hasattr(model, 'moe_layers_q'):
        for layer in model.moe_layers_q:
            def make_hook(layer_ref=layer):
                def hook(module, input, output):
                    # Store gate_probs if available
                    if hasattr(module, '_gate_probs'):
                        gate_probs = module._gate_probs.detach().cpu().numpy()
                        # Average across tokens (B*N, num_experts) -> (num_experts,)
                        if len(gate_probs.shape) > 1:
                            gate_probs = np.mean(gate_probs, axis=0)
                        moe_gating_data['q'].append(gate_probs)
                return hook
            hook_handle = layer.register_forward_hook(make_hook())
            moe_hooks.append(hook_handle)
    
    batch_idx = 0
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        
        # Forward pass
        output, m_feat = wrapped_model(images, multimask_output=True)
        preds = output.softmax(dim=1)
        metrics.update(preds, labels)
        
        batch_idx += 1
        # Limit to first 10 batches for visualization
        if batch_idx >= 10:
            break
    
    # Extract collected data
    if wrapped_model.modality_logits_list:
        all_logits = torch.cat(wrapped_model.modality_logits_list, dim=0)
        conf_scores = torch.sigmoid(all_logits).numpy()
        confidence_scores.extend(conf_scores)
    
    if wrapped_model.vision_feats_before:
        feature_maps_before.extend([f.numpy() for f in wrapped_model.vision_feats_before])
    if wrapped_model.vision_feats_after:
        feature_maps_after.extend([f.numpy() for f in wrapped_model.vision_feats_after])
    
    # Remove hooks and restore original forward
    for hook in moe_hooks:
        hook.remove()
    restore_moe_layer(model, original_forward)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

def plot_confidence_scores(save_dir, modals):
    """Plot confidence scores for each modality"""
    if not confidence_scores:
        print("Warning: No confidence scores collected")
        return
    
    # Aggregate confidence scores
    all_scores = np.array(confidence_scores)  # (num_batches, num_modals, batch_size, 1)
    # Average across batches and batch dimension
    avg_scores = np.mean(all_scores, axis=(0, 2, 3))  # (num_modals,)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(modals))
    bars = ax.bar(x_pos, avg_scores, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(modals)])
    
    ax.set_xlabel('Modality', fontsize=12)
    ax.set_ylabel('Average Confidence Score', fontsize=12)
    ax.set_title('Modality Confidence Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modals)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confidence_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence scores plot saved to {save_dir / 'confidence_scores.png'}")

def plot_moe_gating_heatmap(save_dir, modals):
    """Plot MoE gating pattern heatmap"""
    if not moe_gating_data:
        print("Warning: No MoE gating data collected. MoE gating requires model modification to capture gate_probs.")
        print("Creating placeholder heatmap...")
        # Create placeholder
        num_modals = len(modals)
        num_experts = 4
        heatmap_data = np.ones((num_modals, num_experts)) / num_experts
    else:
        # Aggregate gating probabilities
        num_modals = len(modals)
        if 'q' in moe_gating_data and moe_gating_data['q']:
            num_experts = len(moe_gating_data['q'][0])
            # Average across all batches
            avg_probs = np.mean(moe_gating_data['q'], axis=0)
            # Create heatmap data: (num_modals, num_experts)
            # For now, use same pattern for all modalities
            heatmap_data = np.tile(avg_probs, (num_modals, 1))
        else:
            num_experts = 4
            heatmap_data = np.ones((num_modals, num_experts)) / num_experts
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=[f'Expert {i}' for i in range(num_experts)],
                yticklabels=modals,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Selection Probability'})
    
    ax.set_xlabel('Expert ID', fontsize=12)
    ax.set_ylabel('Modality', fontsize=12)
    ax.set_title('MoE Gating Pattern Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'moe_gating_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MoE gating heatmap saved to {save_dir / 'moe_gating_heatmap.png'}")

def plot_feature_maps(save_dir, modals, num_samples=3):
    """Plot feature maps before and after modulation"""
    if not feature_maps_before or not feature_maps_after:
        print("Warning: No feature maps collected")
        return
    
    # Select a few samples to visualize
    num_samples = min(num_samples, len(feature_maps_before))
    
    for sample_idx in range(num_samples):
        if sample_idx >= len(feature_maps_before):
            break
        
        feat_before = feature_maps_before[sample_idx]
        feat_after = feature_maps_after[sample_idx]
        
        # Average across channels for visualization
        if len(feat_before.shape) == 3:  # (HW, B, C)
            feat_before = np.mean(feat_before, axis=0)  # (B, C)
            feat_before = np.mean(feat_before, axis=0)  # (C,)
        elif len(feat_before.shape) == 4:  # (B, C, H, W)
            feat_before = np.mean(feat_before, axis=(0, 1))  # (H, W)
        
        if len(feat_after.shape) == 4:  # (B, C, H, W)
            feat_after = np.mean(feat_after, axis=(0, 1))  # (H, W)
        elif len(feat_after.shape) == 2:  # (H, W)
            feat_after = feat_after
        
        # Reshape to 2D if needed
        if len(feat_before.shape) == 1:
            # Assume square feature map
            side_len = int(np.sqrt(len(feat_before)))
            feat_before = feat_before[:side_len*side_len].reshape(side_len, side_len)
        
        if len(feat_after.shape) == 1:
            side_len = int(np.sqrt(len(feat_after)))
            feat_after = feat_after[:side_len*side_len].reshape(side_len, side_len)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].imshow(feat_before, cmap='viridis')
        axes[0].set_title(f'Feature Map Before Modulation (Sample {sample_idx+1})', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(feat_after, cmap='viridis')
        axes[1].set_title(f'Feature Map After Modulation (Sample {sample_idx+1})', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(save_dir / f'feature_maps_sample_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Feature maps saved to {save_dir / 'feature_maps_sample_*.png'}")

def denormalize_image(img_tensor, modal_type="RGB"):
    """Denormalize image tensor for visualization"""
    if modal_type == "RGB" or modal_type == "img":
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
    else:
        # Depth, Event, etc. - min-max normalization
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
    
    img_tensor = torch.clamp(img_tensor, 0, 1)
    if img_tensor.shape[0] == 1:
        # Single channel - convert to 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1)
    return img_tensor.permute(1, 2, 0).cpu().numpy()

def visualize_inference_sample(images, labels, predictions, modals, dataset, 
                                confidence_scores, moe_gating, feat_before, feat_after,
                                save_path, sample_idx):
    """Create comprehensive inference visualization"""
    batch_size = images[0].shape[0]
    
    for b in range(batch_size):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
        
        # Get palette
        palette = dataset.PALETTE.numpy() / 255.0
        
        # Row 1: Input images, GT mask, Prediction
        # Modality images
        num_modals = len(modals)
        for i, modal in enumerate(modals):
            ax = fig.add_subplot(gs[0, i])
            img = images[i][b].cpu()
            img_vis = denormalize_image(img, modal)
            ax.imshow(img_vis)
            ax.set_title(f'{modal.upper()}', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Ground Truth Mask
        ax = fig.add_subplot(gs[0, num_modals])
        label = labels[b].cpu().numpy()
        label_colored = np.zeros((label.shape[0], label.shape[1], 3))
        for cls_id in range(len(palette)):
            mask = label == cls_id
            label_colored[mask] = palette[cls_id]
        ax.imshow(label_colored)
        ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Prediction Mask
        ax = fig.add_subplot(gs[0, num_modals + 1])
        pred = predictions[b].cpu().numpy()
        pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3))
        for cls_id in range(len(palette)):
            mask = pred == cls_id
            pred_colored[mask] = palette[cls_id]
        ax.imshow(pred_colored)
        ax.set_title('Prediction', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Row 2: Confidence Scores, MoE Gating, Feature Maps
        # Confidence Scores
        ax = fig.add_subplot(gs[1, 0])
        if confidence_scores is not None and len(confidence_scores) > 0:
            conf_scores = confidence_scores[b] if len(confidence_scores.shape) > 1 else confidence_scores
            if isinstance(conf_scores, torch.Tensor):
                conf_scores = conf_scores.numpy()
            x_pos = np.arange(len(modals))
            bars = ax.bar(x_pos, conf_scores, alpha=0.7, color=['red', 'blue', 'green', 'orange'][:len(modals)])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(modals, rotation=45, ha='right')
            ax.set_ylabel('Confidence Score', fontsize=9)
            ax.set_title('Confidence Scores', fontsize=10, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            for bar, score in zip(bars, conf_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence Scores', fontsize=10, fontweight='bold')
        
        # MoE Gating (for first modality as example)
        ax = fig.add_subplot(gs[1, 1])
        if moe_gating is not None and len(moe_gating) > 0:
            gate_probs = moe_gating[0] if isinstance(moe_gating, list) else moe_gating
            if isinstance(gate_probs, torch.Tensor):
                gate_probs = gate_probs.numpy()
            num_experts = len(gate_probs)
            x_pos = np.arange(num_experts)
            bars = ax.bar(x_pos, gate_probs, alpha=0.7, color='purple')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'E{i}' for i in range(num_experts)], fontsize=8)
            ax.set_ylabel('Selection Prob', fontsize=9)
            ax.set_title('MoE Expert Selection', fontsize=10, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            for bar, prob in zip(bars, gate_probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('MoE Expert Selection', fontsize=10, fontweight='bold')
        
        # Feature Map Before Modulation
        ax = fig.add_subplot(gs[1, 2])
        if feat_before is not None:
            if isinstance(feat_before, torch.Tensor):
                feat_before = feat_before.numpy()
            # Handle different shapes: (HW, B, C), (B, C, H, W), (C, H, W), (H, W)
            if len(feat_before.shape) == 3:
                if feat_before.shape[0] > feat_before.shape[1]:  # (HW, B, C)
                    # Reshape to (H, W, C) then average
                    hw, b, c = feat_before.shape
                    h = w = int(np.sqrt(hw))
                    feat_before = feat_before[:h*w].reshape(h, w, c)
                    feat_vis = np.mean(feat_before, axis=2)
                else:  # (C, H, W) or (B, C, H)
                    feat_vis = np.mean(feat_before, axis=0)
            elif len(feat_before.shape) == 4:  # (B, C, H, W)
                feat_vis = np.mean(feat_before[b], axis=0)
            elif len(feat_before.shape) == 2:  # (H, W)
                feat_vis = feat_before
            else:
                feat_vis = feat_before.flatten()[:64*64].reshape(64, 64)  # Fallback
            im = ax.imshow(feat_vis, cmap='viridis')
            ax.set_title('Feature Before UDMM', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Before UDMM', fontsize=10, fontweight='bold')
        
        # Feature Map After Modulation
        ax = fig.add_subplot(gs[1, 3])
        if feat_after is not None:
            if isinstance(feat_after, torch.Tensor):
                feat_after = feat_after.numpy()
            # Handle different shapes
            if len(feat_after.shape) == 3:
                if feat_after.shape[0] > feat_after.shape[1]:  # (HW, B, C)
                    hw, b, c = feat_after.shape
                    h = w = int(np.sqrt(hw))
                    feat_after = feat_after[:h*w].reshape(h, w, c)
                    feat_vis = np.mean(feat_after, axis=2)
                else:  # (C, H, W)
                    feat_vis = np.mean(feat_after, axis=0)
            elif len(feat_after.shape) == 4:  # (B, C, H, W)
                feat_vis = np.mean(feat_after[b], axis=0)
            elif len(feat_after.shape) == 2:  # (H, W)
                feat_vis = feat_after
            else:
                feat_vis = feat_after.flatten()[:64*64].reshape(64, 64)  # Fallback
            im = ax.imshow(feat_vis, cmap='viridis')
            ax.set_title('Feature After UDMM', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature After UDMM', fontsize=10, fontweight='bold')
        
        # Difference map
        ax = fig.add_subplot(gs[1, 4])
        if feat_before is not None and feat_after is not None:
            if isinstance(feat_before, torch.Tensor):
                feat_before = feat_before.numpy()
            if isinstance(feat_after, torch.Tensor):
                feat_after = feat_after.numpy()
            
            if len(feat_before.shape) == 3:
                feat_before_vis = np.mean(feat_before, axis=0)
            elif len(feat_before.shape) == 4:
                feat_before_vis = np.mean(feat_before[b], axis=0)
            else:
                feat_before_vis = feat_before
                
            if len(feat_after.shape) == 3:
                feat_after_vis = np.mean(feat_after, axis=0)
            elif len(feat_after.shape) == 4:
                feat_after_vis = np.mean(feat_after[b], axis=0)
            else:
                feat_after_vis = feat_after
            
            # Resize if needed
            if feat_before_vis.shape != feat_after_vis.shape and zoom is not None:
                zoom_factor = (feat_after_vis.shape[0] / feat_before_vis.shape[0],
                              feat_after_vis.shape[1] / feat_before_vis.shape[1])
                feat_before_vis = zoom(feat_before_vis, zoom_factor, order=1)
            elif feat_before_vis.shape != feat_after_vis.shape:
                # Fallback: simple resize using interpolation
                import torch.nn.functional as F
                feat_before_tensor = torch.from_numpy(feat_before_vis).unsqueeze(0).unsqueeze(0)
                feat_before_tensor = F.interpolate(feat_before_tensor, size=feat_after_vis.shape, mode='bilinear', align_corners=False)
                feat_before_vis = feat_before_tensor.squeeze().numpy()
            
            diff = feat_after_vis - feat_before_vis
            im = ax.imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            ax.set_title('Feature Difference', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Difference', fontsize=10, fontweight='bold')
        
        # IoU per class (if available)
        ax = fig.add_subplot(gs[1, 5])
        ax.axis('off')
        ax.text(0.1, 0.9, 'Sample Info', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.7, f'Sample: {sample_idx}_{b}', fontsize=10, transform=ax.transAxes)
        ax.text(0.1, 0.5, f'Modalities: {", ".join(modals)}', fontsize=10, transform=ax.transAxes)
        
        plt.suptitle(f'Inference Analysis - Sample {sample_idx}_{b}', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(save_path / f'inference_sample_{sample_idx}_{b}.png', dpi=150, bbox_inches='tight')
        plt.close()

@torch.no_grad()
def evaluate_with_visualization(model, dataloader, device, modals, dataset, save_dir, num_samples=5):
    """Evaluate model and create inference visualizations"""
    print('Evaluating with inference visualization...')
    
    # Wrap model for analysis
    wrapped_model = AnalysisWrapper(model)
    wrapped_model.eval()
    wrapped_model = wrapped_model.to(device)
    
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    
    # Patch MoE layers
    original_forward = patch_moe_layer_for_analysis(model)
    
    sample_idx = 0
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        
        # Forward pass
        output, m_feat = wrapped_model(images, multimask_output=True)
        preds = output.softmax(dim=1)
        predictions = preds.argmax(dim=1)
        metrics.update(preds, labels)
        
        # Visualize samples
        if sample_idx < num_samples:
            # Get data for visualization
            batch_size = images[0].shape[0]
            for b in range(min(batch_size, num_samples - sample_idx)):
                # Get confidence scores
                conf_scores = None
                if wrapped_model.modality_logits_list:
                    all_logits = torch.cat(wrapped_model.modality_logits_list, dim=0)
                    if len(all_logits) > 0:
                        conf_scores = torch.sigmoid(all_logits[-1][b]).numpy()
                
                # Get MoE gating (for first modality)
                moe_gating = None
                if 0 in wrapped_model.moe_gating_per_modality and wrapped_model.moe_gating_per_modality[0]:
                    moe_gating = wrapped_model.moe_gating_per_modality[0][-1]
                
                # Get feature maps
                feat_before = None
                feat_after = None
                if wrapped_model.vision_feats_before:
                    feat_before = wrapped_model.vision_feats_before[-batch_size + b]
                if wrapped_model.vision_feats_after:
                    feat_after = wrapped_model.vision_feats_after[-batch_size + b]
                
                # Create visualization
                visualize_inference_sample(
                    images=[img.cpu() for img in images],
                    labels=labels,
                    predictions=predictions,
                    modals=modals,
                    dataset=dataset,
                    confidence_scores=conf_scores,
                    moe_gating=moe_gating,
                    feat_before=feat_before,
                    feat_after=feat_after,
                    save_path=save_dir,
                    sample_idx=sample_idx
                )
                sample_idx += 1
        
        # Clear lists to save memory (keep only last batch)
        if len(wrapped_model.modality_logits_list) > 1:
            wrapped_model.modality_logits_list = wrapped_model.modality_logits_list[-1:]
        if len(wrapped_model.vision_feats_before) > batch_size:
            wrapped_model.vision_feats_before = wrapped_model.vision_feats_before[-batch_size:]
        if len(wrapped_model.vision_feats_after) > batch_size:
            wrapped_model.vision_feats_after = wrapped_model.vision_feats_after[-batch_size:]
    
    # Restore original forward
    restore_moe_layer(model, original_forward)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    dataset_cfg = cfg['DATASET']
    model_cfg = cfg['MODEL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError(f"Model path not found: {model_path}")
    print(f"Evaluating with detailed analysis: {model_path}...")

    # Create save directory for analysis results
    save_dir = Path(os.path.dirname(eval_cfg['MODEL_PATH'])) / 'analysis'
    save_dir.mkdir(exist_ok=True)

    for case in cases:
        dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', transform, dataset_cfg['MODALS'], case)

        checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        sam2_config_file = "sam2_hiera_b+.yaml"
        num_modalities = len(dataset_cfg['MODALS'])
        modals = dataset_cfg['MODALS']

        sam2 = build_sam2(
            sam2_config_file,
            checkpoint,
            hydra_overrides_extra=[
                "++model.pred_obj_scores=false",
                "++model.fixed_no_obj_ptr=false",
                "++model.pred_obj_scores_mlp=false"
            ]
        )
        
        # Get LoRA model configuration from config
        lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P6')
        lora_r = model_cfg.get('LORA_R', 4)
        lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
        if lora_num_experts is None:
            lora_num_experts = num_modalities
        lora_top_k = model_cfg.get('LORA_TOP_K', 2)
        lora_layer = model_cfg.get('LORA_LAYER', None)
        
        # Dynamically load LoRA model class
        lora_model_class = eval(lora_model_name)
        
        # Build model with config parameters
        model_kwargs = {
            'sam_model': sam2,
            'r': lora_r,
            'lora_layer': lora_layer,
        }
        
        # Add optional parameters if they exist in the model signature
        sig = inspect.signature(lora_model_class.__init__)
        if 'num_experts' in sig.parameters:
            model_kwargs['num_experts'] = lora_num_experts
        if 'top_k' in sig.parameters:
            model_kwargs['top_k'] = lora_top_k
        
        model = lora_model_class(**model_kwargs).cpu()
        print(f"Using LoRA model: {lora_model_name}")
        print(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")
        
        # Load model weights
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'), strict=False)
        print(f"Model loading message: {msg}")
        model = model.to(device)
        model.eval()
        
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=4, pin_memory=False, sampler=sampler_val)
        
        # Evaluate with visualization
        print("\n" + "="*80)
        print("Running inference visualization...")
        print("="*80)
        acc, macc, f1, mf1, ious, miou = evaluate_with_visualization(
            model, dataloader, device, modals, dataset, save_dir, num_samples=10
        )
        
        # Also run standard analysis
        print("\n" + "="*80)
        print("Running detailed analysis...")
        print("="*80)
        acc2, macc2, f1_2, mf1_2, ious2, miou2 = evaluate_with_analysis(model, dataloader, device, modals, save_dir)
        
        # Generate visualizations
        print("\nGenerating analysis visualizations...")
        plot_confidence_scores(save_dir, modals)
        plot_moe_gating_heatmap(save_dir, modals)
        plot_feature_maps(save_dir, modals)
        
        # Print results
        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.4f}" for iou in ious] + [f"{miou:.4f}"],
            'F1': [f"{f:.4f}" for f in f1] + [f"{mf1:.4f}"],
            'Acc': [f"{a:.4f}" for a in acc] + [f"{macc:.4f}"]
        }
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU: {miou:.4f}")
        print(f"\nAnalysis results saved in {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
