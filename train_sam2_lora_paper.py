import os
import torch
import argparse
import yaml
import time
import math
import random
import multiprocessing as mp
import numpy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation, get_nightval_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm_sam import evaluate
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg_bkup import LoRA_Sam
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *
from semseg.models.sam2.sam2.lora_sam import get_model
# torch.autograd.set_detect_anomaly(True)


# Models that consume gt_mask in forward() and emit a quality-gate dict.
# (P24/P25/P26/P26_AblB share the same train-loop contract.)
QUALITY_GATE_MODELS = (
    'LoRA_Sam_P24',
    'LoRA_Sam_P25',
    'LoRA_Sam_P26',
    'LoRA_Sam_P26_AblB',
    'LoRA_Sam_P27',
    'LoRA_Sam_P28',
    'LoRA_Sam_P29',
    'LoRA_Sam_P30',
    'LoRA_Sam_P31',
    'LoRA_Sam_P32',
    'LoRA_Sam_P33',
)


def _unpack_model_output(model_out):
    """Normalize the variable-arity tuples returned by different LoRA_Sam_P* models
    into a uniform dict so the train loop can stay flat.

    Returns dict with keys:
      output, m_feat,
      aux_outputs, amf_weights, gate_dists  → P10/P11
      aux_logits_list                       → P13/P14/P15/P16
      gate_loss_data                        → P24/P25/P26/P26_AblB
    """
    parsed = {
        'output': None, 'm_feat': None,
        'aux_outputs': None, 'amf_weights': None, 'gate_dists': None,
        'aux_logits_list': None, 'gate_loss_data': None,
    }
    n = len(model_out)
    if n == 5:
        # P11: (output, m_feat, aux_outputs, amf_weights, gate_dists)
        parsed['output'], parsed['m_feat'], parsed['aux_outputs'], \
            parsed['amf_weights'], parsed['gate_dists'] = model_out
    elif n == 4:
        # P10: (output, m_feat, aux_outputs, amf_weights)
        parsed['output'], parsed['m_feat'], parsed['aux_outputs'], parsed['amf_weights'] = model_out
    elif n == 3:
        # P13/14/15/16: (output, m_feat, aux_logits_list)
        # P24/25/26:    (output, m_feat, gate_loss_data:dict)
        parsed['output'], parsed['m_feat'], third = model_out
        if isinstance(third, dict):
            parsed['gate_loss_data'] = third
        else:
            parsed['aux_logits_list'] = third
    else:
        # Default: (output, m_feat)
        parsed['output'], parsed['m_feat'] = model_out
    return parsed


def _compute_quality_gate_loss(gate_loss_data, quality_cfg, device):
    """Compute spatial quality-gate loss for P24/P25/P26 family.

    Two regimes:
      - 'kl' (P26): KL divergence between predicted SQG distribution and CE-derived target.
      - else (P24/P25): per-modality BCE between raw quality logit and exp(-CE) target.
    """
    if gate_loss_data is None:
        return torch.tensor(0.0, device=device)

    ignore_mask = gate_loss_data.get('ignore_mask')

    if gate_loss_data.get('loss_type') == 'kl':
        pred_logits = gate_loss_data['predicted_logits']        # list of (B,1,H,W)
        target_dist = gate_loss_data['quality_target_dist']     # (m,B,1,H,W)
        tau_uamm = quality_cfg.get('TAU_UAMM', 1.0)

        pred_stack = torch.stack(pred_logits, dim=0)            # (m,B,1,H,W)
        pred_log_dist = F.log_softmax(pred_stack / tau_uamm, dim=0)
        kl_raw = F.kl_div(pred_log_dist, target_dist.detach(), reduction='none')

        if ignore_mask is None:
            return kl_raw.mean()
        valid = ~ignore_mask                                    # (B,1,H,W)
        kl_raw = kl_raw * valid.unsqueeze(0).float()
        n_valid = valid.float().sum().clamp(min=1.0) * len(pred_logits)
        return kl_raw.sum() / n_valid

    # P24/P25: per-modality BCE
    predicted_list = gate_loss_data['predicted']                # raw logits
    target_list = gate_loss_data['target']                      # exp(-CE) ∈ (0,1]
    loss = torch.tensor(0.0, device=device)
    for pred_q, tgt_q in zip(predicted_list, target_list):
        bce = F.binary_cross_entropy_with_logits(
            pred_q, tgt_q.detach(), reduction='none')           # (B,1,H,W)
        if ignore_mask is None:
            loss = loss + bce.mean()
        else:
            valid = ~ignore_mask
            n_valid = valid.float().sum().clamp(min=1.0)
            loss = loss + (bce * valid.float()).sum() / n_valid
    return loss / len(predicted_list)


def _compute_p13_aux_loss(aux_logits_list, lbl, device):
    """Average per-modality CE for P13/14/15/16 aux heads."""
    if aux_logits_list is None:
        return torch.tensor(0.0, device=device)
    lbl_size = lbl.shape[-2:]
    total = torch.tensor(0.0, device=device)
    for al in aux_logits_list:
        al_up = F.interpolate(al, size=lbl_size, mode='bilinear', align_corners=False)
        total = total + F.cross_entropy(al_up, lbl, ignore_index=255)
    return total / len(aux_logits_list)


def _compute_p10_gating_loss(aux_outputs, amf_weights, lbl, loss_fn):
    """P10/P11: Oracle KL on AMF weights + auxiliary segmentation loss."""
    if aux_outputs is None or amf_weights is None:
        return None
    lbl_size = lbl.shape[-2:]
    aux_losses_per_img = []
    for ao in aux_outputs:
        ao_up = F.interpolate(ao, size=lbl_size, mode='bilinear', align_corners=False)
        loss_per_img = F.cross_entropy(
            ao_up, lbl, ignore_index=255, reduction='none'
        ).mean(dim=[-2, -1])  # (B,)
        aux_losses_per_img.append(loss_per_img)

    aux_losses_stacked = torch.stack(aux_losses_per_img, dim=1)  # (B, m)
    with torch.no_grad():
        oracle = F.softmax(-aux_losses_stacked.detach(), dim=1)  # (B, m)

    gating_kl = F.kl_div((amf_weights + 1e-8).log(), oracle, reduction='batchmean')
    aux_seg = sum(
        loss_fn(F.interpolate(ao, size=lbl_size, mode='bilinear', align_corners=False), lbl)
        for ao in aux_outputs
    ) / len(aux_outputs)
    return gating_kl + 0.3 * aux_seg


def _compute_p11_mi_loss(gate_dists, device):
    """P11: MI loss = conditional entropy − marginal entropy (push experts apart)."""
    if gate_dists is None or len(gate_dists) <= 1:
        return torch.tensor(0.0, device=device)
    stacked = torch.stack(gate_dists, dim=0)                    # (m, E)
    cond_entropy = -(stacked * (stacked + 1e-8).log()).sum(dim=-1).mean()
    marginal = stacked.mean(dim=0)                              # (E,)
    marg_entropy = -(marginal * (marginal + 1e-8).log()).sum(dim=-1)
    return cond_entropy - marg_entropy


class PrototypeSegmentation:
    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        # Initialize global prototypes for each class
        self.global_prototypes = torch.zeros((num_classes, feature_dim), requires_grad=False).to('cuda') 

    def update_global_prototypes(self, current_prototypes):
        # Update global prototypes with current prototypes
        self.global_prototypes.data = 0.8 * self.global_prototypes.data + 0.2 * current_prototypes.data
        # Using `.data` avoids creating new computation graphs for the updated values.

    def compute_loss(self, features, labels):
        # Calculate per-class prototypes from current batch
        batch_prototypes = self.calculate_batch_prototypes(features, labels)

        # Update global prototypes
        self.update_global_prototypes(batch_prototypes)

        # Compute the prototype matching loss
        prototype_loss = self.prototype_loss(batch_prototypes)

        # Optionally, you can also compute segmentation loss here (e.g., CrossEntropyLoss)
        # segmentation_loss = F.cross_entropy(logits, labels)

        # Combine the losses
        total_loss = prototype_loss
        return total_loss

    def calculate_batch_prototypes(self, features, labels):
        # Initialize prototypes
        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device)
        count = torch.zeros(self.num_classes, device=features.device)
        
        # Ensure labels are on the same device as features
        labels = labels.to(features.device)
        
        labels = labels.unsqueeze(1)
        # Resize labels to match feature map size
        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(1)

        # Flatten features and resized labels
        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels_resized = labels_resized.view(-1)

        for i in range(self.num_classes):
            mask = (labels_resized == i)
            if mask.sum() > 0:
                batch_prototypes[i] = features[mask].mean(dim=0)
                count[i] = mask.sum()
        
        # Avoid division by zero
        count = count.clamp(min=1)
        return batch_prototypes / count.unsqueeze(1)

    def prototype_loss(self, batch_prototypes):
        # Calculate the loss between batch prototypes and global prototypes
        loss = F.mse_loss(batch_prototypes, self.global_prototypes)
        return loss

def plot_training_curves(save_dir, epochs, losses, proto_losses, lrs,
                         val_losses=None):
    """Plot and save training curves for Loss, Proto Loss, Val Loss, and LR"""
    n_plots = 4 if val_losses else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    # Plot Loss
    axes[0].plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot Proto Loss
    axes[1].plot(epochs, proto_losses, 'r-', linewidth=2, label='Proto Loss')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Proto Loss', fontsize=12)
    axes[1].set_title('Prototype Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot Val Loss (if available)
    ax_lr = 2
    if val_losses:
        val_epochs = [e for e, v in val_losses]
        val_vals = [v for e, v in val_losses]
        axes[2].plot(val_epochs, val_vals, 'm-', linewidth=2, label='Val Loss')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        ax_lr = 3

    # Plot LR
    axes[ax_lr].plot(epochs, lrs, 'g-', linewidth=2, label='Learning Rate')
    axes[ax_lr].set_xlabel('Epoch', fontsize=12)
    axes[ax_lr].set_ylabel('Learning Rate', fontsize=12)
    axes[ax_lr].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[ax_lr].grid(True, alpha=0.3)
    axes[ax_lr].legend()
    axes[ax_lr].set_yscale('log')  # Use log scale for LR

    plt.tight_layout()
    plot_path = save_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also save individual plots
    # Combined Loss plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, proto_losses, 'r-', linewidth=2, label='Proto Loss')
    if val_losses:
        val_epochs = [e for e, v in val_losses]
        val_vals = [v for e, v in val_losses]
        ax.plot(val_epochs, val_vals, 'm--', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = save_dir / 'training_losses.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_p24_quality_vis(save_dir, epoch, gate_loss_data, sample_img, modal_names,
                         mode='train', max_size=128):
    """
    P24 quality map visualization — saves predicted/target quality maps per modality.
    Lightweight: resizes to max_size, single PNG per epoch.

    Args:
        save_dir: output directory
        epoch: current epoch number
        gate_loss_data: dict with 'predicted' and optionally 'target' lists
        sample_img: list of input tensors (for reference thumbnail)
        modal_names: list of modality names e.g. ['img', 'lidar', 'thermal']
        mode: 'train' (pred+target) or 'val' (pred only)
        max_size: resize long edge to this for space efficiency
    """
    vis_dir = Path(save_dir) / 'quality_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    predicted = gate_loss_data['predicted']
    target = gate_loss_data.get('target')
    has_target = target is not None and len(target) > 0
    m = len(predicted)
    n_rows = 3 if has_target else 2

    fig, axes = plt.subplots(n_rows, m, figsize=(3 * m, 3 * n_rows))
    if m == 1:
        axes = axes[:, None]

    for i in range(m):
        # Row 0: Input thumbnail
        img_np = sample_img[i][0].detach().cpu().float()
        if img_np.shape[0] == 3:
            img_np = img_np.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        else:
            img_np = img_np[0].numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0, i].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        axes[0, i].set_title(f'{modal_names[i] if i < len(modal_names) else f"mod{i}"}', fontsize=10)
        axes[0, i].axis('off')

        # Row 1: Predicted quality map (apply sigmoid if raw logits)
        pred_raw = predicted[i][0, 0].detach().cpu().float()
        pred_q = torch.sigmoid(pred_raw).numpy() if pred_raw.min() < 0 or pred_raw.max() > 1 else pred_raw.numpy()
        im1 = axes[1, i].imshow(pred_q, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'Pred Q [{pred_q.min():.2f},{pred_q.max():.2f}]', fontsize=9)
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Row 2: Target quality map (train only)
        if has_target:
            tgt_q = target[i][0, 0].detach().float().cpu().numpy()
            im2 = axes[2, i].imshow(tgt_q, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'Target Q [{tgt_q.min():.2f},{tgt_q.max():.2f}]', fontsize=9)
            axes[2, i].axis('off')
            plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel('Input', fontsize=10, rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('Predicted', fontsize=10, rotation=0, labelpad=40)
    if has_target:
        axes[2, 0].set_ylabel('Target', fontsize=10, rotation=0, labelpad=40)

    suffix = 'train' if mode == 'train' else 'val'
    fig.suptitle(f'P24 Quality Maps ({suffix}) — Epoch {epoch + 1}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / f'epoch{epoch+1:03d}_{suffix}.png', dpi=100, bbox_inches='tight')
    plt.close()


def _update_topk_checkpoints(topk_list, new_miou, new_epoch, save_dir, prefix,
                             ckpt_dict, k=5):
    """Top-K 체크포인트 관리. rank를 postfix로 파일명에 기재.

    Args:
        topk_list: 현재 top-k 목록 List[(miou, epoch)], 내림차순 정렬 유지
        new_miou, new_epoch: 새로 저장할 값
        save_dir: 저장 디렉토리 (Path)
        prefix: 파일명 prefix ('' = day-val, 'night_' = night-val)
        ckpt_dict: torch.save에 넘길 dict
        k: 유지할 최대 개수

    Returns:
        updated topk_list (내림차순)
    """
    topk_list = list(topk_list) + [(new_miou, new_epoch)]
    topk_list.sort(key=lambda x: x[0], reverse=True)

    # k 초과 항목의 파일 삭제
    if len(topk_list) > k:
        for miou, ep in topk_list[k:]:
            for f in save_dir.glob(f"{prefix}epoch{ep}_{miou}_top*_checkpoint.pth"):
                f.unlink(missing_ok=True)
        topk_list = topk_list[:k]

    # 전체 파일명을 현재 순위에 맞게 저장/rename
    for rank, (miou, ep) in enumerate(topk_list, 1):
        target = save_dir / f"{prefix}epoch{ep}_{miou}_top{rank}_checkpoint.pth"
        if (miou, ep) == (new_miou, new_epoch):
            torch.save(ckpt_dict, target)
        else:
            # 순위 변경으로 rename 필요한 경우
            for old_f in save_dir.glob(f"{prefix}epoch{ep}_{miou}_top*_checkpoint.pth"):
                if old_f != target:
                    old_f.rename(target)
                    break

    return topk_list


# ── Weights & Biases inference-image logging ─────────────────────────────────
# ImageNet stats used by Normalize() in semseg/augmentations_mm.py — needed to
# de-normalize the RGB modality back to a viewable image.
_VIS_MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
_VIS_STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)


def _select_fixed_vis_indices(n_total, k=10):
    """Deterministic, evenly-spaced indices so the SAME samples are visualized
    every eval — lets the W&B media slider show qualitative progress over epochs."""
    if n_total <= 0:
        return []
    k = min(k, n_total)
    if k == 1:
        return [0]
    step = (n_total - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def _denorm_rgb(rgb_tensor):
    """(3,H,W) normalized float tensor -> (H,W,3) uint8 RGB for visualization."""
    img = rgb_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * _VIS_STD + _VIS_MEAN
    return numpy.clip(img * 255.0, 0, 255).astype(numpy.uint8)


@torch.no_grad()
def log_wandb_inference_samples(model, dataset, indices, device, palette, step,
                                key="val_samples"):
    """Run inference on a FIXED set of val samples and log [RGB | GT | Pred]
    panels to wandb under one key, so each eval adds a frame to the media slider."""
    if not (HAS_WANDB and wandb.run is not None) or not indices:
        return
    was_training = model.training
    model.eval()
    images = []
    for idx in indices:
        item = dataset[idx]
        sample, label = item[0], item[1]
        modal_inputs = [m.unsqueeze(0).to(device, non_blocking=True) for m in sample]
        out = model(modal_inputs, True)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        pred = logits.argmax(dim=1)[0].cpu()
        rgb = _denorm_rgb(sample[0])
        gt_color = dataset.decode_segmap(label, palette)
        pred_color = dataset.decode_segmap(pred, palette)
        panel = numpy.concatenate([rgb, gt_color, pred_color], axis=1)
        images.append(wandb.Image(panel, caption=f"idx{idx} | RGB | GT | Pred"))
    wandb.log({key: images}, step=step)
    if was_training:
        model.train()


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    best_night_mIoU = 0.0   # [Night-Val] 야간 시뮬 기준 best
    best_night_epoch = 0
    best_test_mIoU = 0.0    # [Test] test set 기준 best
    best_test_epoch = 0
    top_day_ckpts = []       # List[(miou, epoch)] 내림차순, 상위 5개 유지
    top_night_ckpts = []     # [Night-Val] 상위 5개
    top_test_ckpts = []      # [Test] 상위 5개
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_enable = model_cfg.get('RESUME_ENABLE', False)
    resume_path = model_cfg.get('RESUME_PATH', '')
    gpus = int(os.environ['WORLD_SIZE'])

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'], dataset_cfg=dataset_cfg)
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)

    # [Night-Val] NIGHT_AUG.ENABLE 시 야간 시뮬 val set 별도 생성 (ISSUE-001 대응)
    night_aug_enabled = dataset_cfg.get('NIGHT_AUG', {}).get('ENABLE', False)
    if night_aug_enabled:
        nightvaltransform = get_nightval_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)

    ds_kwargs = {}
    if dataset_cfg.get('NAME') == 'MULTIAQUA' and 'NUM_CLASSES' in dataset_cfg:
        ds_kwargs['n_classes'] = dataset_cfg['NUM_CLASSES']
    night_trans = bool(dataset_cfg.get('NIGHT_TRANSLATION', False))
    # night_translation은 MULTIAQUA 전용 — DELIVER 등 다른 데이터셋에는 전달하지 않음
    if dataset_cfg.get('NAME') == 'MULTIAQUA':
        ds_kwargs['night_translation'] = night_trans
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'], **ds_kwargs)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'], **ds_kwargs)
    nightvalset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', nightvaltransform, dataset_cfg['MODALS'], **ds_kwargs) if night_aug_enabled else None
    # DELIVER 등 test split이 있는 데이터셋: test set 평가 활성화
    eval_test_enabled = dataset_cfg.get('NAME') != 'MULTIAQUA'
    testset = None
    if eval_test_enabled:
        try:
            testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', valtransform, dataset_cfg['MODALS'], **ds_kwargs)
        except Exception as e:
            print(f"[INFO] Test set not available: {e}")
            eval_test_enabled = False
    class_names = trainset.CLASSES

    # model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    # SAM2 backbone size is config-driven (default Hiera-B+). Set MODEL.SAM2_CHECKPOINT +
    # MODEL.SAM2_CONFIG to use a bigger backbone, e.g. Hiera-Large for higher mIoU.
    checkpoint = model_cfg.get("SAM2_CHECKPOINT", "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    sam2_config_file = model_cfg.get("SAM2_CONFIG", "sam2_hiera_b+.yaml")
    print(f"[SAM2] backbone config={sam2_config_file} ckpt={checkpoint}")
    num_modalities = len(dataset_cfg['MODALS'])
    
    # Load resume checkpoint if enabled
    resume_checkpoint = None
    if resume_enable and resume_path and os.path.isfile(resume_path):
        print(f"Loading checkpoint from: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location='cpu')
        print(f"Resuming from epoch: {resume_checkpoint.get('epoch', 0)}")
    elif resume_enable and resume_path:
        print(f"Warning: Resume enabled but checkpoint not found at: {resume_path}")
        print("Starting training from scratch...")

    # sam2 = build_sam2(sam2_config_file, checkpoint)
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
        lora_num_experts = num_modalities  # Auto from num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K', 2)
    lora_layer = model_cfg.get('LORA_LAYER', None)
    
    # Dynamically load LoRA model class
    lora_model_class = get_model(lora_model_name)  # registry lookup (구 eval() 대체)
    
    # Build model with config parameters
    model_kwargs = {
        'sam_model': sam2,
        'r': lora_r,
        'lora_layer': lora_layer,
    }
    
    # Add optional parameters if they exist in the model signature
    # Check if model accepts num_experts and top_k (for P3, P4, P5, P6)
    import inspect
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k
    if 'num_classes' in sig.parameters:
        # P10+: num_classes는 config의 LORA_NUM_CLASSES 또는 기본값 4
        model_kwargs['num_classes'] = model_cfg.get('LORA_NUM_CLASSES', 4)
    if 'aux_warmup_epochs' in sig.parameters:
        # [P16] Aux Warmup: 초기 N epoch uniform weights
        model_kwargs['aux_warmup_epochs'] = train_cfg.get('AUX_WARMUP_EPOCHS', 10)
    if 'use_entropy_fusion' in sig.parameters:
        # [P18] P18-A(False): P9-style 고정상수, P18-B(True): entropy fusion
        model_kwargs['use_entropy_fusion'] = model_cfg.get('USE_ENTROPY_FUSION', False)
    if 'gate_hidden_ratio' in sig.parameters:
        # [P20] SharedGateMLP hidden ratio (default 4 → C//4)
        model_kwargs['gate_hidden_ratio'] = model_cfg.get('GATE_HIDDEN_RATIO', 4)
    if 'deba_bottleneck_dim' in sig.parameters:
        # [P21/P22/P23] DeBA bottleneck dimension (default 64)
        deba_cfg = model_cfg.get('DEBA', {})
        model_kwargs['deba_bottleneck_dim'] = deba_cfg.get('BOTTLENECK_DIM', 64)
        model_kwargs['deba_kernel_size'] = deba_cfg.get('KERNEL_SIZE', 3)
    if 'deba_scales' in sig.parameters:
        # [P23] MoE DeBA-BB multi-scale factors (default [1, 2])
        deba_cfg = model_cfg.get('DEBA', {})
        model_kwargs['deba_scales'] = deba_cfg.get('SCALES', [1, 2])
        model_kwargs['deba_gate_noise_std'] = deba_cfg.get('GATE_NOISE_STD', 0.1)
    if 'quality_hidden_dim' in sig.parameters:
        # [P24/P25/P26] SpatialQualityGating parameters
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['quality_hidden_dim'] = quality_cfg.get('HIDDEN_DIM', 64)
        model_kwargs['quality_min'] = quality_cfg.get('MIN_QUALITY', 0.1)
    if 'tau_uamm' in sig.parameters:
        # [P26] Quality gate + architecture parameters
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['tau_uamm'] = quality_cfg.get('TAU_UAMM', 1.0)
        model_kwargs['tau_teacher'] = quality_cfg.get('TAU_TEACHER', 0.5)
        model_kwargs['memory_mod'] = quality_cfg.get('MEMORY_MOD', False)
        model_kwargs['amf_mode'] = quality_cfg.get('AMF_MODE', 'sqg_quality')
        model_kwargs['multi_scale_sqg'] = quality_cfg.get('MULTI_SCALE_SQG', True)
        model_kwargs['per_modality_decoder'] = quality_cfg.get('PER_MODALITY_DECODER', True)
    if 'cond_dim' in sig.parameters:
        # [P26] Modality-conditioned MoE LoRA gate
        model_kwargs['cond_dim'] = model_cfg.get('LORA_COND_DIM', 8)
    if 'sdc_enable' in sig.parameters:
        # [P29] Self-Derived Condition (SDC) routing config (MODEL.SDC)
        sdc_cfg = model_cfg.get('SDC', {}) or {}
        model_kwargs['sdc_enable'] = sdc_cfg.get('ENABLE', True)
        model_kwargs['sdc_K'] = sdc_cfg.get('K', 6)
        model_kwargs['sdc_latent'] = sdc_cfg.get('LATENT_DIM', 32)
    if 'class_token_decoder' in sig.parameters:
        # [P30] class-token decoder + reliability-anchored learned router
        ctd = model_cfg.get('CLASS_TOKEN_DECODER', {}) or {}
        rtr = model_cfg.get('LEARNED_ROUTER', {}) or {}
        model_kwargs['class_token_decoder'] = ctd.get('ENABLE', False)
        model_kwargs['ctd_dim'] = ctd.get('DIM', 128)
        model_kwargs['learned_router'] = rtr.get('ENABLE', False)
        model_kwargs['router_per_class'] = rtr.get('PER_CLASS', False)
        model_kwargs['router_anchor_lambda'] = rtr.get('ANCHOR_LAMBDA', 1.0)
        model_kwargs['num_classes'] = trainset.n_classes
    if 'ctd_multi_scale' in sig.parameters:
        # [P31] calibrated dual-reliability RBMA + multi-scale HR class-token decoder
        ctd = model_cfg.get('CLASS_TOKEN_DECODER', {}) or {}
        calib = model_cfg.get('RBMA_CALIB', {}) or {}
        rtr = model_cfg.get('LEARNED_ROUTER', {}) or {}
        model_kwargs['ctd_multi_scale'] = ctd.get('MULTI_SCALE', False)
        model_kwargs['ctd_up'] = ctd.get('UP', 2)
        model_kwargs['ctd_aux_ce'] = ctd.get('AUX_CE', True)
        # [P31.1] CTD 강등: 최종 출력은 SAM decoder 유지, CTD는 training-only aux loss
        # (P30-seg 실측 붕괴 val −13.4/test −10.2 + det E0.1 query-head 유죄에 근거)
        model_kwargs['ctd_aux_only'] = ctd.get('AUX_ONLY', False)
        model_kwargs['rbma_calibrate'] = calib.get('ENABLE', False)
        model_kwargs['consistency_bias'] = calib.get('CONSISTENCY_BIAS', False)
        model_kwargs['lambda_cons_init'] = calib.get('LAMBDA_CONS_INIT', 0.5)
        model_kwargs['amf_reliability'] = calib.get('AMF_RELIABILITY', False)
        model_kwargs['amf_rel_tau'] = calib.get('AMF_REL_TAU', 0.25)
        model_kwargs['unfreeze_last_n_blocks'] = model_cfg.get('UNFREEZE_LAST_N_BLOCKS', 0)
        model_kwargs['router_reg_mode'] = rtr.get('REG_MODE', 'diversity')
    if 'corroboration_bias' in sig.parameters:
        # [P32] CoRB — corroboration-biased memory attention: RBMA 신뢰도 신호를
        # self-entropy → cross-modal corroboration(corr_veto)으로 교체 (무학습, λ만 학습).
        corrb = model_cfg.get('CORROBORATION', {}) or {}
        model_kwargs['corroboration_bias'] = corrb.get('ENABLE', False)
        model_kwargs['corrb_veto'] = corrb.get('VETO', True)
    if 'competence_fusion' in sig.parameters:
        # [P33] M1 competence-weighted fusion (calibrated self-entropy, NOT corr_veto) +
        #       M2 asymmetric modality dropout. OFF → P32 byte-identical.
        comp = model_cfg.get('COMPETENCE_FUSION', {}) or {}
        model_kwargs['competence_fusion'] = comp.get('ENABLE', False)
        model_kwargs['comp_tau'] = comp.get('TAU', 0.25)
        model_kwargs['comp_topk'] = comp.get('TOPK', 0)
        model_kwargs['comp_entropy_reg'] = comp.get('ENTROPY_REG', 0.0)
        mdrop = model_cfg.get('MODAL_DROPOUT', {}) or {}
        model_kwargs['modal_dropout'] = mdrop.get('ENABLE', False)
        model_kwargs['modal_dropout_p'] = mdrop.get('P', 0.3)
        model_kwargs['modal_dropout_warmup_ep'] = mdrop.get('WARMUP_EP', 20)
        # TARGETS = 모달 이름 리스트(예: [img, depth]) → DATASET.MODALS 순서로 인덱스 해석.
        modals = cfg['DATASET']['MODALS']
        raw_targets = mdrop.get('TARGETS', ['img', 'depth'])
        tgt_idx = [modals.index(t) if isinstance(t, str) else int(t)
                   for t in raw_targets if (not isinstance(t, str)) or (t in modals)]
        model_kwargs['modal_dropout_targets'] = tuple(tgt_idx) if tgt_idx else (0, 1)
    if 'lambda_bias_init' in sig.parameters:
        # [P27] Learnable attention-bias scalar initial value
        quality_cfg = model_cfg.get('QUALITY_GATE', {})
        model_kwargs['lambda_bias_init'] = quality_cfg.get('LAMBDA_BIAS_INIT', 1.0)
    if 'num_modalities' in sig.parameters:
        # All LoRA models: pass actual number of modalities from config
        model_kwargs['num_modalities'] = num_modalities

    model = lora_model_class(**model_kwargs).cpu()
    
    # Load model weights from checkpoint if resuming
    if resume_checkpoint:
        model_state = resume_checkpoint.get('model_state_dict', resume_checkpoint.get('model_state_dict'))
        if model_state:
            model.load_state_dict(model_state, strict=False)
            print("Model weights loaded from checkpoint")
    
    print(f"Using LoRA model: {lora_model_name}")
    print(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")

    # Encoder gradient checkpointing (saves VRAM by recomputing activations during backward)
    if train_cfg.get('GRADIENT_CHECKPOINT', False):
        model.sam.image_encoder.trunk.gradient_checkpointing = True
        print("Encoder gradient checkpointing enabled (saves VRAM, ~30% slower training)")

    if train_cfg['DDP']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("Using SyncBatchNorm to handle small batch size.")

    model = model.to(device)
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    print(model)

    for layer in model.sam.sam_mask_decoder.iou_prediction_head.layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
            
    for param in model.sam.sam_mask_decoder.parameters():
        param.requires_grad = True
    for param in model.sam.obj_ptr_proj.parameters():
        param.requires_grad = False
    for param in model.sam.sam_mask_decoder.iou_prediction_head.parameters():
        param.requires_grad = False
    # pred_obj_score_head only exists when pred_obj_scores=True
    if hasattr(model.sam.sam_mask_decoder, 'pred_obj_score_head'):
        for param in model.sam.sam_mask_decoder.pred_obj_score_head.parameters():
            param.requires_grad = False
    for param in model.sam.memory_attention.parameters():
        param.requires_grad = True
    for param in model.sam.memory_encoder.parameters():
        param.requires_grad = True
    for param in model.sam.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))


    # Auto calculate accumulation steps
    purposed_batch_size = 16
    # accumulation_steps = 2  # 8 GPUs * Batch 1 * 2 steps = Effective Batch 16
    accumulation_steps = math.ceil(purposed_batch_size / (train_cfg['BATCH_SIZE'] * gpus))
    effective_batch_size = train_cfg['BATCH_SIZE'] * gpus * accumulation_steps
    updates_per_epoch = len(trainset) // effective_batch_size
    iters_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * gpus)

    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    # [P10] Gating aux loss 가중치: config에서 읽거나 기본값 사용
    lambda_gate = train_cfg.get('LAMBDA_GATE', 0.5)
    # [P11] MI routing loss 가중치
    lambda_mi = train_cfg.get('LAMBDA_MI', 1.0)
    # [P13] Aux loss 가중치
    lambda_aux = train_cfg.get('LAMBDA_AUX', 0.3)
    # [P26 v6] Per-modal auxiliary CE loss 가중치
    quality_gate_cfg = cfg['MODEL'].get('QUALITY_GATE', {})
    lambda_aux_ce = quality_gate_cfg.get('AUX_CE_WEIGHT', 0.5)
    # [P29] SDC label-free clustering loss weight (MODEL.SDC.LAMBDA)
    lambda_sdc = (cfg['MODEL'].get('SDC', {}) or {}).get('LAMBDA', 0.1)
    # [P30] router diversity reg weight (encourage modality mixing; MODEL.LEARNED_ROUTER.REG_LAMBDA)
    lambda_router = (cfg['MODEL'].get('LEARNED_ROUTER', {}) or {}).get('REG_LAMBDA', 0.01)
    # [P31] RBMA calibration loss weight (MODEL.RBMA_CALIB.LAMBDA)
    lambda_cal = (cfg['MODEL'].get('RBMA_CALIB', {}) or {}).get('LAMBDA', 0.1)
    # [P31] class-token-decoder aux CE weight @H/4 (MODEL.CLASS_TOKEN_DECODER.AUX_CE_WEIGHT)
    lambda_ctd_aux = (cfg['MODEL'].get('CLASS_TOKEN_DECODER', {}) or {}).get('AUX_CE_WEIGHT', 0.4)
    # [P31.1] 강등된 CTD의 full-res aux seg CE weight (AUX_ONLY 모드에서만 발생)
    lambda_ctd_seg = (cfg['MODEL'].get('CLASS_TOKEN_DECODER', {}) or {}).get('SEG_CE_WEIGHT', 0.4)
    start_epoch = 0
    # [P31] unfrozen backbone blocks train at a reduced LR (full LR would wreck pretrained Hiera)
    unfreeze_lr_scale = cfg['MODEL'].get('UNFREEZE_LR_SCALE', 0.1) \
        if cfg['MODEL'].get('UNFREEZE_LAST_N_BLOCKS', 0) > 0 else 1.0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'],
                              backbone_lr_scale=unfreeze_lr_scale,
                              backbone_prefix='sam.image_encoder.trunk.blocks')
    scheduler = get_scheduler(
        sched_cfg['NAME'], 
        optimizer, 
        int((epochs + 1) * updates_per_epoch), # 총 업데이트 횟수
        sched_cfg['POWER'], 
        updates_per_epoch * sched_cfg['WARMUP'], 
        sched_cfg['WARMUP_RATIO']
        )


    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None

        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None
    
    # Restore training state from checkpoint
    if resume_checkpoint:
        start_epoch = resume_checkpoint.get('epoch', 0)
        best_mIoU = resume_checkpoint.get('best_miou', 0.0)
        best_epoch = resume_checkpoint.get('best_epoch', 0)
        best_night_mIoU = resume_checkpoint.get('best_night_miou', 0.0)
        best_night_epoch = resume_checkpoint.get('best_night_epoch', 0)
        top_day_ckpts = resume_checkpoint.get('top_day_ckpts', [])
        top_night_ckpts = resume_checkpoint.get('top_night_ckpts', [])
        
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            print("Optimizer state restored")
        
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            print("Scheduler state restored")
        
        # NOTE: scaler state is restored after `scaler` is created below (it does
        # not exist yet here). Loading it here raised UnboundLocalError on resume.
        print(f"Resuming training from epoch {start_epoch + 1}, best mIoU: {best_mIoU:.4f}")
        

    # Data loader tuning (pin_memory + persistent_workers + prefetch_factor)
    # persistent/prefetch only valid when num_workers > 0.
    _loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        _loader_kwargs['persistent_workers'] = True
        _loader_kwargs['prefetch_factor'] = 4
    # Eval loaders run infrequently — avoid keeping (val+night+test) × num_workers
    # persistent processes alive for the whole run (RAM/shm/fd pressure, train-loader starvation).
    _eval_loader_kwargs = {
        'num_workers': min(num_workers, 4),
        'pin_memory': True,
    }
    if _eval_loader_kwargs['num_workers'] > 0:
        _eval_loader_kwargs['prefetch_factor'] = 4
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], drop_last=True, sampler=sampler, **_loader_kwargs)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], sampler=sampler_val, **_eval_loader_kwargs)
    # [Night-Val] 야간 시뮬 val loader (NIGHT_AUG.ENABLE 시에만 생성)
    night_valloader = DataLoader(nightvalset, batch_size=eval_cfg['BATCH_SIZE'], sampler=sampler_val, **_eval_loader_kwargs) if nightvalset is not None else None
    # [Test] test set loader (DELIVER 등 test split이 있는 데이터셋)
    testloader = DataLoader(testset, batch_size=eval_cfg['BATCH_SIZE'], sampler=sampler_val, **_eval_loader_kwargs) if testset is not None else None


    # AMP dtype: 'float16' (default, GradScaler 필수) 또는 'bfloat16' (Ampere+/B200 권장, GradScaler 불필요).
    _amp_dtype_str = str(train_cfg.get('AMP_DTYPE', 'float16')).lower()
    AMP_DTYPE = torch.bfloat16 if _amp_dtype_str in ('bf16', 'bfloat16') else torch.float16
    # GradScaler는 fp16에서만 필요. bf16은 range가 fp32와 동일해 overflow 위험 거의 없음.
    scaler = GradScaler(enabled=(train_cfg['AMP'] and AMP_DTYPE == torch.float16))
    # Restore GradScaler state on resume (only meaningful when the scaler is
    # enabled, i.e. fp16 AMP; for bf16 the scaler is disabled and has no state).
    if resume_checkpoint and 'scaler_state_dict' in resume_checkpoint and scaler.is_enabled():
        scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
        print("Scaler state restored")


    wandb_vis_indices = []
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        # ── Weights & Biases init ────────────────────────────────────────────
        # Per-server setup is manual & one-time: `pip install wandb` then
        # `wandb login` (paste your API key). No key is stored in the repo.
        # Disable per-run via `WANDB: {ENABLE: false}` in the config or
        # `WANDB_DISABLED=1` in the env.
        wandb_cfg = cfg.get('WANDB', {}) or {}
        wandb_enabled = (HAS_WANDB and wandb_cfg.get('ENABLE', True)
                         and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'))
        if wandb_enabled:
            modals_str = ''.join([m[0] for m in dataset_cfg['MODALS']])
            cfg_name = cfg.get('_CFG_NAME', '') or save_dir.name
            night_on = dataset_cfg.get('NIGHT_AUG', {}).get('ENABLE', False)
            # Tags group runs by model / dataset / key hyperparameters.
            run_tags = [
                f"model:{lora_model_name}",
                f"backbone:{model_cfg['BACKBONE']}",
                f"dataset:{dataset_cfg['NAME']}",
                f"modals:{modals_str}",
                f"loss:{loss_cfg['NAME']}",
                f"lr:{lr:g}",
                f"bs:{train_cfg['BATCH_SIZE']}",
                f"lora_r:{lora_r}",
                f"cfg:{cfg_name}",
            ]
            if night_on:
                run_tags.append("night_aug")
            # Repo-scoped account: if a gitignored `.wandb_key` sits next to this
            # script, use it for THIS process only (set the env var, never touch
            # ~/.netrc). On shared boxes (e.g. B200) this means runs from this
            # repo log to your account without a global `wandb login` that would
            # affect other users. Falls back to the machine's `wandb login`.
            if not os.environ.get('WANDB_API_KEY'):
                _key_file = Path(__file__).resolve().parent / '.wandb_key'
                if _key_file.is_file():
                    _k = _key_file.read_text().strip()
                    if _k:
                        os.environ['WANDB_API_KEY'] = _k
                        print(f"[wandb] using repo-local key from {_key_file}")
            try:
                wandb.init(
                    project=wandb_cfg.get('PROJECT', 'MemorySAM'),
                    entity=wandb_cfg.get('ENTITY', None),
                    name=wandb_cfg.get('NAME', None) or cfg_name,
                    dir=str(save_dir),
                    tags=run_tags,
                    config={
                        "model": lora_model_name,
                        "backbone": model_cfg['BACKBONE'],
                        "lora_r": lora_r,
                        "num_experts": lora_num_experts,
                        "epochs": epochs,
                        "batch_size": train_cfg['BATCH_SIZE'],
                        "lr": lr,
                        "weight_decay": optim_cfg['WEIGHT_DECAY'],
                        "scheduler": sched_cfg['NAME'],
                        "loss": loss_cfg['NAME'],
                        "lambda_gate": lambda_gate,
                        "lambda_mi": lambda_mi,
                        "lambda_aux": lambda_aux,
                        "amp": train_cfg['AMP'],
                        "ddp": train_cfg['DDP'],
                        "night_aug": night_on,
                        "night_sim_p": dataset_cfg.get('NIGHT_AUG', {}).get('NIGHT_SIM_P', 0),
                        "aux_warmup_epochs": train_cfg.get('AUX_WARMUP_EPOCHS', 0),
                        "save_dir": str(save_dir),
                        "dataset": dataset_cfg['NAME'],
                        "modals": dataset_cfg['MODALS'],
                        "config_file": cfg_name,
                    },
                )
                # Fixed sample set for qualitative inference panels (same every eval).
                wandb_vis_indices = _select_fixed_vis_indices(
                    len(valset), k=int(wandb_cfg.get('NUM_VIS', 10)))
            except Exception as e:
                print(f"[wandb] init failed ({e}); continuing without wandb logging. "
                      f"Run `wandb login` on this server to enable it.")
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info(f"Using LoRA model: {lora_model_name}")
        logger.info(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")
        logger.info(f"Loss weights: λ_gate={lambda_gate}, λ_mi={lambda_mi}, λ_aux={lambda_aux}")
        _m_info = model.module if hasattr(model, 'module') else model
        if hasattr(_m_info, 'aux_warmup_epochs'):
            logger.info(f"[P16] Aux Warmup: {_m_info.aux_warmup_epochs} epochs uniform → 5 epoch ramp → full entropy")
    
    num_classes = trainset.n_classes
    feature_dim = 32
    prototypeseg = PrototypeSegmentation(num_classes, feature_dim)
    
    # Initialize lists for tracking training metrics
    # Restore from checkpoint if resuming
    if resume_checkpoint and 'train_losses' in resume_checkpoint:
        train_losses = resume_checkpoint['train_losses']
        train_proto_losses = resume_checkpoint['train_proto_losses']
        train_lrs = resume_checkpoint['train_lrs']
        epochs_list = resume_checkpoint['epochs_list']
        val_losses = resume_checkpoint.get('val_losses', [])
        print(f"Restored training metrics: {len(epochs_list)} epochs")
    else:
        train_losses = []
        train_proto_losses = []
        train_lrs = []
        epochs_list = []
        val_losses = []  # list of (epoch, loss) tuples
    
    for epoch in range(start_epoch, epochs):
        model.train()
        # [P16] Aux warmup: 현재 epoch 전달
        _m = model.module if hasattr(model, 'module') else model
        if hasattr(_m, '_current_epoch'):
            _m._current_epoch = epoch
        if train_cfg['DDP']: sampler.set_epoch(epoch)
        train_loss = 0.0
        proto_loss = 0.0
        gate_loss_accum = 0.0
        mi_loss_accum = 0.0
        aux_loss_accum = 0.0
        aux_ce_loss_accum = 0.0
        # [P31] S2 게이트 모니터링: per-modal reliability AUROC/μ/σ + router 평균 가중치
        p31_cal_accum = 0.0
        p31_ctd_accum = 0.0
        p31_auroc_rows = []
        p31_relmu_rows = []
        p31_relsd_rows = []
        p31_routerw_rows = []
       
    
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        
        for iter, (sample, lbl) in pbar:
            # optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device, non_blocking=True) for x in sample]
            lbl = lbl.to(device, non_blocking=True)
            
            with autocast(enabled=train_cfg['AMP'], dtype=AMP_DTYPE):
                # Quality-gate models (P24/P25/P26/P26_AblB) need gt_mask in forward.
                uses_quality_gate = (lora_model_name in QUALITY_GATE_MODELS)
                if uses_quality_gate:
                    model_out = model(sample, multimask_output=True, gt_mask=lbl)
                else:
                    model_out = model(sample, multimask_output=True)

                parsed = _unpack_model_output(model_out)
                output = parsed['output']
                m_feat = parsed['m_feat']
                aux_outputs = parsed['aux_outputs']            # P10/P11
                amf_weights = parsed['amf_weights']            # P10/P11
                gate_dists = parsed['gate_dists']              # P11
                aux_logits_list = parsed['aux_logits_list']    # P13/14/15/16
                gate_loss_data = parsed['gate_loss_data']      # P24/P25/P26

                loss_orig = loss_fn(output, lbl)
                protoloss = prototypeseg.compute_loss(m_feat, lbl) * 256 * 256

                # [P10/P11] Gating auxiliary (oracle KL + aux seg)
                gating_aux = _compute_p10_gating_loss(aux_outputs, amf_weights, lbl, loss_fn)
                gating_loss = gating_aux if gating_aux is not None else torch.tensor(0.0, device=device)

                # [P11] MI routing loss
                mi_loss = _compute_p11_mi_loss(gate_dists, device)

                # [P13/14/15/16] Aux segmentation CE
                p13_aux_loss = _compute_p13_aux_loss(aux_logits_list, lbl, device)

                # [P24/P25/P26] Spatial quality-gate loss
                quality_cfg = cfg['MODEL'].get('QUALITY_GATE', {})
                quality_gate_loss = _compute_quality_gate_loss(gate_loss_data, quality_cfg, device)

                # [P26 v6] Per-modal auxiliary CE loss
                if gate_loss_data is not None and 'aux_ce_losses' in gate_loss_data:
                    aux_ce_list = gate_loss_data['aux_ce_losses']
                    aux_ce_loss = sum(aux_ce_list) / len(aux_ce_list)
                else:
                    aux_ce_loss = torch.tensor(0.0, device=device)

                # [P29] SDC label-free clustering loss (model stashes _sdc_loss)
                _core = model.module if hasattr(model, 'module') else model
                _sdc = getattr(_core, '_sdc_loss', None)
                sdc_loss = _sdc if _sdc is not None else torch.tensor(0.0, device=device)
                # [P30] router diversity reg: maximize modality-mixing entropy → subtract
                _rreg = getattr(_core, '_router_reg', None)
                router_loss = (-_rreg) if _rreg is not None else torch.tensor(0.0, device=device)
                # [P31] RBMA calibration loss + class-token-decoder aux CE (model stashes
                # both into gate_loss_data; absent for P24–P30)
                _zero = torch.tensor(0.0, device=device)
                rbma_cal_loss = gate_loss_data.get('rbma_cal_loss', _zero) if gate_loss_data else _zero
                ctd_aux_loss = gate_loss_data.get('ctd_aux_ce', _zero) if gate_loss_data else _zero
                ctd_seg_loss = gate_loss_data.get('ctd_seg_ce', _zero) if gate_loss_data else _zero
                # [P33] M1 competence-fusion anti-collapse entropy reg (coeff already applied
                # in-model via comp_entropy_reg; absent for P24–P32). Zero unless enabled.
                comp_entropy_loss = gate_loss_data.get('comp_entropy', _zero) if gate_loss_data else _zero

                # Aggregate
                total_loss_unscaled = (loss_orig + protoloss
                                       + lambda_gate * gating_loss
                                       + lambda_mi * mi_loss
                                       + lambda_aux * p13_aux_loss
                                       + lambda_gate * quality_gate_loss
                                       + lambda_aux_ce * aux_ce_loss
                                       + lambda_sdc * sdc_loss
                                       + lambda_router * router_loss
                                       + lambda_cal * rbma_cal_loss
                                       + lambda_ctd_aux * ctd_aux_loss
                                       + lambda_ctd_seg * ctd_seg_loss
                                       + comp_entropy_loss)
                loss = total_loss_unscaled / accumulation_steps

            # [P24/P25/P26] Save quality map visualization (1st iter per epoch, rank 0 only)
            is_rank0 = (not train_cfg['DDP']) or torch.distributed.get_rank() == 0
            if uses_quality_gate and gate_loss_data is not None and iter == 0 and is_rank0:
                try:
                    # P26 KL-mode emits different dict keys; adapt to vis function format.
                    if gate_loss_data.get('loss_type') == 'kl':
                        pred_logits = gate_loss_data['predicted_logits']
                        target_dist = gate_loss_data['quality_target_dist']
                        vis_data = {
                            'predicted': pred_logits,
                            'target': [target_dist[i] for i in range(len(pred_logits))],
                        }
                    else:
                        vis_data = gate_loss_data
                    save_p24_quality_vis(
                        save_dir, epoch, vis_data, sample,
                        dataset_cfg.get('MODALS', [f'mod{i}' for i in range(len(sample))]),
                        mode='train',
                    )
                except Exception as e:
                    print(f"[quality-gate vis] Warning: {e}")

            scaler.scale(loss).backward()
            if (iter + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            # [removed] torch.cuda.synchronize() — DDP all-reduce + scaler.step은
            # 이미 동기화를 제공. 매 iter sync는 H2D overlap을 막아 GPU util을 떨어뜨림.

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr.real <= 1e-8:
                lr = 1e-8 # minimum of lr
                lr = float(lr.real)
            else:
                lr = float(lr.real) if hasattr(lr, 'real') else float(lr)
                
            # train_loss += loss.item()
            train_loss += total_loss_unscaled.item()
            proto_loss += protoloss.item()
            gate_loss_accum += gating_loss.item()
            mi_loss_accum += mi_loss.item()
            aux_loss_accum += p13_aux_loss.item() + quality_gate_loss.item()
            aux_ce_loss_accum += aux_ce_loss.item()
            # [P31] reliability AUROC(게이트) + router 가중치 수집 (모델이 stash)
            p31_cal_accum += float(rbma_cal_loss)
            p31_ctd_accum += float(ctd_aux_loss) + float(ctd_seg_loss)
            _p31_auroc = getattr(_core, '_last_rel_auroc', None)
            if _p31_auroc is not None:
                p31_auroc_rows.append(_p31_auroc)
                _mu, _sd = _core._last_rel_stats
                p31_relmu_rows.append(_mu)
                p31_relsd_rows.append(_sd)
            _p31_rt = getattr(_core, 'router', None)
            if _p31_rt is not None and getattr(_p31_rt, '_last_w_mean', None) is not None:
                p31_routerw_rows.append(_p31_rt._last_w_mean.tolist())

            desc = (
                f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] "
                f"LR: {lr:.8f} Loss: {train_loss/(iter+1):.6f} "
                f"Proto: {proto_loss/(iter+1):.6f}"
            )
            # 활성 loss만 표시 (P13+에서 Gate/MI는 항상 0)
            if gate_loss_accum > 0:
                desc += f" Gate: {gate_loss_accum/(iter+1):.6f}"
            if mi_loss_accum > 0:
                desc += f" MI: {mi_loss_accum/(iter+1):.6f}"
            if aux_loss_accum > 0:
                desc += f" Aux: {aux_loss_accum/(iter+1):.6f}"
            if aux_ce_loss_accum > 0:
                desc += f" AuxCE: {aux_ce_loss_accum/(iter+1):.6f}"
            # [P16] Warmup 상태 표시
            if hasattr(_m, 'aux_warmup_epochs'):
                wu = _m.aux_warmup_epochs
                ramp_end = wu + 5
                if epoch < wu:
                    desc += f" [Warmup {epoch+1}/{wu}]"
                elif epoch < ramp_end:
                    ramp = (epoch - wu) / 5.0
                    desc += f" [Ramp {ramp:.1f}]"
            pbar.set_description(desc)
        
        train_loss /= iter+1
        proto_loss /= iter+1
        avg_lr = scheduler.get_lr()
        avg_lr = sum(avg_lr) / len(avg_lr)
        avg_lr = float(avg_lr.real) if hasattr(avg_lr, 'real') else float(avg_lr)
        
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            # Save metrics for plotting
            train_losses.append(train_loss)
            train_proto_losses.append(proto_loss)
            train_lrs.append(avg_lr)
            epochs_list.append(epoch + 1)
            
            # Add to TensorBoard
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/proto_loss', proto_loss, epoch)
            writer.add_scalar('train/aux_loss', aux_loss_accum / (iter + 1), epoch)
            if aux_ce_loss_accum > 0:
                writer.add_scalar('train/aux_ce_loss', aux_ce_loss_accum / (iter + 1), epoch)
            writer.add_scalar('train/lr', avg_lr, epoch)

            # [P31] S2 게이트 로깅: per-modal reliability AUROC(>0.5 목표)·σ(상수화=퇴화 감지)
            # + router 평균 가중치(uniform 감지). doc 20 "AUROC 선행 게이트"의 측정 구현.
            _p31_modals = dataset_cfg.get('MODALS', [])
            if p31_auroc_rows:
                _auroc_ep = np.nanmean(np.array(p31_auroc_rows, dtype=np.float64), axis=0)
                _relmu_ep = np.array(p31_relmu_rows, dtype=np.float64).mean(axis=0)
                _relsd_ep = np.array(p31_relsd_rows, dtype=np.float64).mean(axis=0)
                _names = (_p31_modals + [f'mod{i}' for i in range(len(_auroc_ep))])[:len(_auroc_ep)]
                for _i, _n in enumerate(_names):
                    writer.add_scalar(f'p31/rel_auroc_{_n}', _auroc_ep[_i], epoch)
                    writer.add_scalar(f'p31/rel_std_{_n}', _relsd_ep[_i], epoch)
                writer.add_scalar('p31/cal_loss', p31_cal_accum / (iter + 1), epoch)
                print(f"[P31] rel AUROC {[f'{n}:{a:.3f}' for n, a in zip(_names, _auroc_ep)]} "
                      f"| rel μ {np.round(_relmu_ep, 3).tolist()} σ {np.round(_relsd_ep, 4).tolist()} "
                      f"| cal {p31_cal_accum/(iter+1):.4f} ctd {p31_ctd_accum/(iter+1):.4f}")
            if p31_routerw_rows:
                _wbar = np.array(p31_routerw_rows, dtype=np.float64).mean(axis=0)
                _names = (_p31_modals + [f'mod{i}' for i in range(len(_wbar))])[:len(_wbar)]
                for _i, _n in enumerate(_names):
                    writer.add_scalar(f'p31/router_w_{_n}', _wbar[_i], epoch)
                print(f"[P31] router w̄ {[f'{n}:{w:.3f}' for n, w in zip(_names, _wbar)]}")

            # wandb: 학습 메트릭 로깅
            if HAS_WANDB and wandb.run is not None:
                wandb_train = {
                    "epoch": epoch + 1,
                    "train/total_loss": train_loss,
                    "train/seg_loss": (train_loss - proto_loss
                                       - lambda_gate * gate_loss_accum / (iter + 1)
                                       - lambda_mi * mi_loss_accum / (iter + 1)
                                       - lambda_aux * aux_loss_accum / (iter + 1)),
                    "train/proto_loss": proto_loss,
                    "train/aux_loss": aux_loss_accum / (iter + 1),
                    "train/gate_loss": gate_loss_accum / (iter + 1),
                    "train/mi_loss": mi_loss_accum / (iter + 1),
                    "train/aux_ce_loss": aux_ce_loss_accum / (iter + 1),
                    "train/lr": avg_lr,
                }
                # P16 warmup ramp 값
                if hasattr(_m, 'aux_warmup_epochs'):
                    wu = _m.aux_warmup_epochs
                    if epoch < wu:
                        wandb_train["train/warmup_ramp"] = 0.0
                    elif epoch < wu + 5:
                        wandb_train["train/warmup_ramp"] = (epoch - wu) / 5.0
                    else:
                        wandb_train["train/warmup_ramp"] = 1.0
                # [P31] AUROC 게이트 + router 가중치 wandb 로깅
                if p31_auroc_rows:
                    for _i, _n in enumerate(_names[:len(_auroc_ep)]):
                        wandb_train[f"p31/rel_auroc_{_n}"] = float(_auroc_ep[_i])
                        wandb_train[f"p31/rel_std_{_n}"] = float(_relsd_ep[_i])
                    wandb_train["p31/cal_loss"] = p31_cal_accum / (iter + 1)
                    wandb_train["p31/ctd_loss"] = p31_ctd_accum / (iter + 1)
                if p31_routerw_rows:
                    for _i, _n in enumerate((_p31_modals + [f'mod{_j}' for _j in range(len(_wbar))])[:len(_wbar)]):
                        wandb_train[f"p31/router_w_{_n}"] = float(_wbar[_i])
                wandb.log(wandb_train, step=epoch)
            
            # Plot and save graphs
            plot_training_curves(save_dir, epochs_list, train_losses, train_proto_losses, train_lrs,
                                 val_losses=val_losses if val_losses else None)
            
            # Save last checkpoint after each epoch
            last_ckp_path = save_dir / 'last_checkpoint.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if train_cfg['AMP'] else None,
                'loss': train_loss,
                'proto_loss': proto_loss,
                'best_miou': best_mIoU,
                'best_epoch': best_epoch,
                'best_night_miou': best_night_mIoU,   # [Night-Val]
                'best_night_epoch': best_night_epoch,  # [Night-Val]
                'top_day_ckpts': top_day_ckpts,
                'top_night_ckpts': top_night_ckpts,
                'train_losses': train_losses,
                'train_proto_losses': train_proto_losses,
                'train_lrs': train_lrs,
                'epochs_list': epochs_list,
                'val_losses': val_losses,
            }, last_ckp_path)

            # 5 epoch 단위 주기 저장 (test set proxy 없이 최적 epoch 탐색용)
            if (epoch + 1) % 5 == 0:
                periodic_path = save_dir / f'periodic_epoch{epoch+1}_checkpoint.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if train_cfg['AMP'] else None,
                    'loss': train_loss,
                    'proto_loss': proto_loss,
                    'best_miou': best_mIoU,
                    'best_night_miou': best_night_mIoU,
                }, periodic_path)

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            # free cached blocks only before eval (was called every epoch, which reset the
            # caching allocator and forced re-cudaMalloc next epoch for no memory benefit).
            torch.cuda.empty_cache()
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, f1, mf1, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                # Compute validation loss (CE + proto)
                model.eval()
                val_loss_sum = 0.0
                val_n = 0
                quality_vis_done = False
                with torch.no_grad():
                    for val_imgs, val_lbls in valloader:
                        val_imgs = [x.to(device, non_blocking=True) for x in val_imgs]
                        val_lbls = val_lbls.to(device, non_blocking=True)
                        val_out = model(val_imgs, True)
                        val_logits = val_out[0]
                        val_feat = val_out[1]
                        vl_ce = loss_fn(val_logits, val_lbls)
                        vl_proto = prototypeseg.compute_loss(val_feat, val_lbls) * 256 * 256
                        val_loss_sum += (vl_ce + vl_proto).item()
                        # [P24/P25/P26] Val quality vis: 1st batch only
                        if uses_quality_gate and not quality_vis_done:
                            quality_vis_done = True
                            model_unwrapped = model.module if hasattr(model, 'module') else model
                            qmaps = getattr(model_unwrapped, '_last_quality_maps', None)
                            if qmaps is not None:
                                try:
                                    val_qmaps = [torch.from_numpy(q) for q in qmaps]
                                    save_p24_quality_vis(
                                        save_dir, epoch,
                                        {'predicted': val_qmaps},
                                        val_imgs,
                                        dataset_cfg.get('MODALS', [f'mod{i}' for i in range(len(val_imgs))]),
                                        mode='val',
                                    )
                                except Exception as e:
                                    print(f"[quality-gate val vis] Warning: {e}")
                        val_n += 1
                val_loss_avg = val_loss_sum / max(val_n, 1)
                val_losses.append((epoch + 1, val_loss_avg))
                writer.add_scalar('val/loss', val_loss_avg, epoch)

                # wandb: Day-Val 전체 메트릭 로깅
                if HAS_WANDB and wandb.run is not None:
                    wandb_val = {
                        "epoch": epoch + 1,
                        "val/mIoU": miou,
                        "val/pixel_acc": macc,
                        "val/mean_f1": mf1,
                        "val/best_mIoU": best_mIoU,
                    }
                    for c, v in zip(class_names, ious):
                        wandb_val[f"val_iou/{c}"] = v
                    for c, v in zip(class_names, acc):
                        wandb_val[f"val_acc/{c}"] = v
                    for c, v in zip(class_names, f1):
                        wandb_val[f"val_f1/{c}"] = v
                    wandb.log(wandb_val, step=epoch)

                    # Fixed-sample qualitative inference panels (same imgs every
                    # eval) — model is already in eval() here.
                    try:
                        log_wandb_inference_samples(
                            model, valset, wandb_vis_indices, device,
                            valset.PALETTE, step=epoch)
                    except Exception as e:
                        print(f"[wandb] sample image logging failed: {e}")

                # Top-5 유지: 현재 top_day_ckpts 최하위보다 높거나 5개 미만이면 저장
                worst_day = top_day_ckpts[-1][0] if len(top_day_ckpts) >= 5 else -1.0
                if miou > worst_day:
                    new_epoch_day = epoch + 1
                    top_day_ckpts = _update_topk_checkpoints(
                        top_day_ckpts, miou, new_epoch_day, save_dir, prefix='',
                        ckpt_dict={
                            'epoch': new_epoch_day,
                            'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_miou': miou,
                            'best_night_miou': best_night_mIoU,
                        }, k=5)
                    if miou > best_mIoU:
                        best_mIoU = miou
                        best_epoch = new_epoch_day
                        logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                # Per-class IoU 요약
                iou_str = " | ".join([f"{c}: {v:.2f}" for c, v in zip(class_names, ious)])
                aux_str = f"  Aux: {aux_loss_accum/(iter+1):.4f}" if aux_loss_accum > 0 else ""
                logger.info(
                    f"[Day-Val] epoch:{epoch+1}  mIoU: {miou:.4f}  Best: {best_mIoU:.4f} (ep{best_epoch})"
                    f"  Loss: {train_loss:.6f}  Proto: {proto_loss:.6f}{aux_str}"
                    f"\n         IoU: {iou_str}"
                )

                # ── [Night-Val] 야간 시뮬 조건 평가 (ISSUE-001) ──────────────────────
                if night_valloader is not None:
                    night_acc, night_macc, night_f1, night_mf1, night_ious, night_miou = evaluate(model, night_valloader, device)
                    writer.add_scalar('val_night/mIoU', night_miou, epoch)

                    # wandb: Night-Val 전체 메트릭 로깅
                    if HAS_WANDB and wandb.run is not None:
                        wandb_night = {
                            "epoch": epoch + 1,
                            "val_night/mIoU": night_miou,
                            "val_night/pixel_acc": night_macc,
                            "val_night/mean_f1": night_mf1,
                            "val_night/best_mIoU": best_night_mIoU,
                        }
                        for c, v in zip(class_names, night_ious):
                            wandb_night[f"val_night_iou/{c}"] = v
                        for c, v in zip(class_names, night_acc):
                            wandb_night[f"val_night_acc/{c}"] = v
                        for c, v in zip(class_names, night_f1):
                            wandb_night[f"val_night_f1/{c}"] = v
                        wandb.log(wandb_night, step=epoch)

                    worst_night = top_night_ckpts[-1][0] if len(top_night_ckpts) >= 5 else -1.0
                    if night_miou > worst_night:
                        new_epoch_night = epoch + 1
                        top_night_ckpts = _update_topk_checkpoints(
                            top_night_ckpts, night_miou, new_epoch_night, save_dir, prefix='night_',
                            ckpt_dict={
                                'epoch': new_epoch_night,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                'best_night_miou': night_miou,
                            }, k=5)
                        if night_miou > best_night_mIoU:
                            best_night_mIoU = night_miou
                            best_night_epoch = new_epoch_night
                            logger.info(f"[Night-Val] NEW BEST  epoch{best_night_epoch}  Night mIoU: {best_night_mIoU:.4f}")
                            logger.info(print_iou(epoch, night_ious, night_miou, night_acc, night_macc, class_names))

                    night_iou_str = " | ".join([f"{c}: {v:.2f}" for c, v in zip(class_names, night_ious)])
                    logger.info(
                        f"[Night-Val] epoch:{epoch+1}  mIoU: {night_miou:.4f}  Best: {best_night_mIoU:.4f} (ep{best_night_epoch})"
                        f"\n            IoU: {night_iou_str}"
                    )

                # ── [Test] test set 평가 (DELIVER 등) ──────────────────────
                if testloader is not None:
                    test_acc, test_macc, test_f1, test_mf1, test_ious, test_miou = evaluate(model, testloader, device)
                    writer.add_scalar('test/mIoU', test_miou, epoch)

                    if HAS_WANDB and wandb.run is not None:
                        wandb_test = {
                            "epoch": epoch + 1,
                            "test/mIoU": test_miou,
                            "test/pixel_acc": test_macc,
                            "test/mean_f1": test_mf1,
                            "test/best_mIoU": best_test_mIoU,
                        }
                        for c, v in zip(class_names, test_ious):
                            wandb_test[f"test_iou/{c}"] = v
                        wandb.log(wandb_test, step=epoch)

                    worst_test = top_test_ckpts[-1][0] if len(top_test_ckpts) >= 5 else -1.0
                    if test_miou > worst_test:
                        new_epoch_test = epoch + 1
                        top_test_ckpts = _update_topk_checkpoints(
                            top_test_ckpts, test_miou, new_epoch_test, save_dir, prefix='test_',
                            ckpt_dict={
                                'epoch': new_epoch_test,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                'best_test_miou': test_miou,
                            }, k=5)
                        if test_miou > best_test_mIoU:
                            best_test_mIoU = test_miou
                            best_test_epoch = new_epoch_test
                            logger.info(f"[Test] NEW BEST  epoch{best_test_epoch}  Test mIoU: {best_test_mIoU:.4f}")
                            logger.info(print_iou(epoch, test_ious, test_miou, test_acc, test_macc, class_names))

                    test_iou_str = " | ".join([f"{c}: {v:.2f}" for c, v in zip(class_names, test_ious)])
                    logger.info(
                        f"[Test] epoch:{epoch+1}  mIoU: {test_miou:.4f}  Best: {best_test_mIoU:.4f} (ep{best_test_epoch})"
                        f"\n       IoU: {test_iou_str}"
                    )

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
        if HAS_WANDB and wandb.run is not None:
            wandb.finish()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best Day-Val mIoU',   f"{best_mIoU:.2f}  (epoch {best_epoch})"],
        ['Best Night-Val mIoU', f"{best_night_mIoU:.2f}  (epoch {best_night_epoch})" if best_night_mIoU > 0 else "N/A (NIGHT_AUG disabled)"],
        ['Best Test mIoU',      f"{best_test_mIoU:.2f}  (epoch {best_test_epoch})" if best_test_mIoU > 0 else "N/A (no test set)"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver/deliver_rgbdel_sam.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg['_CFG_NAME'] = Path(args.cfg).stem  # used for wandb run name / cfg tag

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)

    resume_enable = cfg['MODEL'].get('RESUME_ENABLE', False)
    resume_path = cfg['MODEL'].get('RESUME_PATH', '')

    # AUTO_RESUME: rerun the SAME command after a crash/kill and continue automatically.
    # If enabled and no explicit RESUME_PATH is set, pick up save_dir/last_checkpoint.pth
    # (saved every epoch). To start FRESH: set AUTO_RESUME false, or delete that file.
    # NOTE: this resumes weights AS-IS — after changing model code, start fresh instead
    # (resuming stale weights silently continues the old run).
    if cfg['MODEL'].get('AUTO_RESUME', False) and not (resume_enable and resume_path):
        auto_ckpt = save_dir / 'last_checkpoint.pth'
        if auto_ckpt.is_file():
            resume_enable = True
            resume_path = str(auto_ckpt)
            cfg['MODEL']['RESUME_ENABLE'] = True
            cfg['MODEL']['RESUME_PATH'] = resume_path
            print(f"[AUTO_RESUME] found checkpoint -> resuming: {auto_ckpt}")
        else:
            print(f"[AUTO_RESUME] no last_checkpoint.pth in {save_dir} -> starting fresh")

    # If resuming, set save_dir from resume path
    if resume_enable and resume_path and os.path.isfile(resume_path):
        save_dir = Path(os.path.dirname(resume_path))
        print(f"Resume enabled: Using save_dir from checkpoint: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()
