import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pathlib import Path
try:
    import trackio
    HAS_TRACKIO = True
except ImportError:
    HAS_TRACKIO = False
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation, get_nightval_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm_sam import evaluate
import numpy
import random
import torch
from semseg.models.sam2.sam2.build_sam import build_sam2 as build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg_bkup import LoRA_Sam
# from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import LoRA_Sam_P3, LoRA_Sam_P2, LoRA_Sam_P1, LoRA_Sam_P5, LoRA_Sam_P4, LoRA_Sam_P6, LoRA_Sam_P7
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import matplotlib
import math
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)


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


def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    best_night_mIoU = 0.0   # [Night-Val] 야간 시뮬 기준 best
    best_night_epoch = 0
    top_day_ckpts = []       # List[(miou, epoch)] 내림차순, 상위 5개 유지
    top_night_ckpts = []     # [Night-Val] 상위 5개
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
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'], night_translation=night_trans, **ds_kwargs)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'], night_translation=night_trans, **ds_kwargs)
    nightvalset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', nightvaltransform, dataset_cfg['MODALS'], night_translation=night_trans, **ds_kwargs) if night_aug_enabled else None
    class_names = trainset.CLASSES

    # model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_config_file = "sam2_hiera_b+.yaml"
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
    lora_model_class = eval(lora_model_name)
    
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

    model = lora_model_class(**model_kwargs).cpu()
    
    # Load model weights from checkpoint if resuming
    if resume_checkpoint:
        model_state = resume_checkpoint.get('model_state_dict', resume_checkpoint.get('model_state_dict'))
        if model_state:
            model.load_state_dict(model_state, strict=False)
            print("Model weights loaded from checkpoint")
    
    print(f"Using LoRA model: {lora_model_name}")
    print(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")

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
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
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
        
        if 'scaler_state_dict' in resume_checkpoint and train_cfg['AMP']:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
            print("Scaler state restored")
        
        print(f"Resuming training from epoch {start_epoch + 1}, best mIoU: {best_mIoU:.4f}")
        

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val)
    # [Night-Val] 야간 시뮬 val loader (NIGHT_AUG.ENABLE 시에만 생성)
    night_valloader = DataLoader(nightvalset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val) if nightvalset is not None else None


    scaler = GradScaler(enabled=train_cfg['AMP'])
    


    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        # Trackio init
        if HAS_TRACKIO:
            trackio.init(
                project="MemorySAM",
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
                    "night_aug": dataset_cfg.get('NIGHT_AUG', {}).get('ENABLE', False),
                    "night_sim_p": dataset_cfg.get('NIGHT_AUG', {}).get('NIGHT_SIM_P', 0),
                    "aux_warmup_epochs": train_cfg.get('AUX_WARMUP_EPOCHS', 0),
                    "save_dir": str(save_dir),
                    "dataset": dataset_cfg['NAME'],
                    "modals": dataset_cfg['MODALS'],
                },
            )
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
       
    
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        
        for iter, (sample, lbl) in pbar:
            # optimizer.zero_grad(set_to_none=True)
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(param_group['lr'])

            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):
                # P11: forward가 (output, m_feat, aux_outputs, amf_weights, gate_dists) 리턴
                # P10: forward가 (output, m_feat, aux_outputs, amf_weights) 리턴
                # P13/P14/P15/P16: forward가 (output, m_feat, aux_logits_list) 리턴
                # 그 외: (output, m_feat) 리턴
                model_out = model(sample, multimask_output=True)
                if len(model_out) == 5:
                    output, m_feat, aux_outputs, amf_weights, gate_dists = model_out
                    p13_aux_logits = None
                elif len(model_out) == 4:
                    output, m_feat, aux_outputs, amf_weights = model_out
                    gate_dists = None
                    p13_aux_logits = None
                elif len(model_out) == 3:
                    # [P13] (output, m_feat, aux_logits_list)
                    output, m_feat, p13_aux_logits = model_out
                    aux_outputs, amf_weights = None, None
                    gate_dists = None
                else:
                    output, m_feat = model_out
                    aux_outputs, amf_weights = None, None
                    gate_dists = None
                    p13_aux_logits = None
                logits = output

                loss_orig = loss_fn(logits, lbl)
                protoloss = prototypeseg.compute_loss(m_feat, lbl) * 256 * 256

                # [P10/P11] Gating Auxiliary Loss (oracle KL + aux seg)
                gating_loss = torch.tensor(0.0, device=device)
                if aux_outputs is not None and amf_weights is not None:
                    lbl_size = lbl.shape[-2:]
                    aux_losses_per_img = []
                    for ao in aux_outputs:
                        ao_up = F.interpolate(ao, size=lbl_size,
                                              mode='bilinear', align_corners=False)
                        loss_per_img = F.cross_entropy(
                            ao_up, lbl, ignore_index=255, reduction='none'
                        ).mean(dim=[-2, -1])  # (B,)
                        aux_losses_per_img.append(loss_per_img)

                    aux_losses_stacked = torch.stack(aux_losses_per_img, dim=1)  # (B, m)
                    with torch.no_grad():
                        oracle = F.softmax(-aux_losses_stacked.detach(), dim=1)  # (B, m)

                    amf = amf_weights  # (B, m), gradient 유지
                    gating_kl = F.kl_div(
                        (amf + 1e-8).log(), oracle, reduction='batchmean'
                    )
                    aux_seg = sum(
                        loss_fn(F.interpolate(ao, size=lbl_size,
                                              mode='bilinear', align_corners=False), lbl)
                        for ao in aux_outputs
                    ) / len(aux_outputs)

                    gating_loss = gating_kl + 0.3 * aux_seg

                # [P11] MI Routing Loss: MoE expert specialization
                mi_loss = torch.tensor(0.0, device=device)
                if gate_dists is not None and len(gate_dists) > 1:
                    stacked = torch.stack(gate_dists, dim=0)  # (m, E)

                    cond_ent = -(stacked * (stacked + 1e-8).log()).sum(dim=-1)  # (m,)
                    cond_entropy = cond_ent.mean()

                    marginal = stacked.mean(dim=0)  # (E,)
                    marg_entropy = -(marginal * (marginal + 1e-8).log()).sum(dim=-1)

                    # MI = marg_entropy - cond_entropy → maximize → minimize (cond - marg)
                    mi_loss = cond_entropy - marg_entropy

                # [P13] Aux CE Loss: aux head가 각 modality에서 segmentation 학습
                p13_aux_loss = torch.tensor(0.0, device=device)
                if p13_aux_logits is not None:
                    lbl_size = lbl.shape[-2:]
                    for al in p13_aux_logits:
                        al_up = F.interpolate(al, size=lbl_size,
                                              mode='bilinear', align_corners=False)
                        p13_aux_loss = p13_aux_loss + F.cross_entropy(
                            al_up, lbl, ignore_index=255)
                    p13_aux_loss = p13_aux_loss / len(p13_aux_logits)

                # 전체 Loss
                total_loss_unscaled = (loss_orig + protoloss
                                       + lambda_gate * gating_loss
                                       + lambda_mi * mi_loss
                                       + lambda_aux * p13_aux_loss)
                loss = total_loss_unscaled / accumulation_steps

            scaler.scale(loss).backward()
            if (iter + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            torch.cuda.synchronize()

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
            aux_loss_accum += p13_aux_loss.item()

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
            writer.add_scalar('train/lr', avg_lr, epoch)

            # Trackio: 학습 메트릭 로깅
            if HAS_TRACKIO:
                trackio_train = {
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
                    "train/lr": avg_lr,
                }
                # P16 warmup ramp 값
                if hasattr(_m, 'aux_warmup_epochs'):
                    wu = _m.aux_warmup_epochs
                    if epoch < wu:
                        trackio_train["train/warmup_ramp"] = 0.0
                    elif epoch < wu + 5:
                        trackio_train["train/warmup_ramp"] = (epoch - wu) / 5.0
                    else:
                        trackio_train["train/warmup_ramp"] = 1.0
                trackio.log(trackio_train)
            
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

        torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, f1, mf1, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                # Compute validation loss (CE + proto)
                model.eval()
                val_loss_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for val_imgs, val_lbls in valloader:
                        val_imgs = [x.to(device) for x in val_imgs]
                        val_lbls = val_lbls.to(device)
                        val_out = model(val_imgs, True)
                        val_logits = val_out[0]
                        val_feat = val_out[1]
                        vl_ce = loss_fn(val_logits, val_lbls)
                        vl_proto = prototypeseg.compute_loss(val_feat, val_lbls) * 256 * 256
                        val_loss_sum += (vl_ce + vl_proto).item()
                        val_n += 1
                val_loss_avg = val_loss_sum / max(val_n, 1)
                val_losses.append((epoch + 1, val_loss_avg))
                writer.add_scalar('val/loss', val_loss_avg, epoch)

                # Trackio: Day-Val 전체 메트릭 로깅
                if HAS_TRACKIO:
                    trackio_val = {
                        "epoch": epoch + 1,
                        "val/mIoU": miou,
                        "val/pixel_acc": macc,
                        "val/mean_f1": mf1,
                        "val/best_mIoU": best_mIoU,
                    }
                    for c, v in zip(class_names, ious):
                        trackio_val[f"val_iou/{c}"] = v
                    for c, v in zip(class_names, acc):
                        trackio_val[f"val_acc/{c}"] = v
                    for c, v in zip(class_names, f1):
                        trackio_val[f"val_f1/{c}"] = v
                    trackio.log(trackio_val)

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

                    # Trackio: Night-Val 전체 메트릭 로깅
                    if HAS_TRACKIO:
                        trackio_night = {
                            "epoch": epoch + 1,
                            "val_night/mIoU": night_miou,
                            "val_night/pixel_acc": night_macc,
                            "val_night/mean_f1": night_mf1,
                            "val_night/best_mIoU": best_night_mIoU,
                        }
                        for c, v in zip(class_names, night_ious):
                            trackio_night[f"val_night_iou/{c}"] = v
                        for c, v in zip(class_names, night_acc):
                            trackio_night[f"val_night_acc/{c}"] = v
                        for c, v in zip(class_names, night_f1):
                            trackio_night[f"val_night_f1/{c}"] = v
                        trackio.log(trackio_night)

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

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
        if HAS_TRACKIO:
            trackio.finish()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best Day-Val mIoU',   f"{best_mIoU:.2f}  (epoch {best_epoch})"],
        ['Best Night-Val mIoU', f"{best_night_mIoU:.2f}  (epoch {best_night_epoch})" if best_night_mIoU > 0 else "N/A (NIGHT_AUG disabled)"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver_rgbdel.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    
    # If resuming, set save_dir from resume path
    resume_enable = cfg['MODEL'].get('RESUME_ENABLE', False)
    resume_path = cfg['MODEL'].get('RESUME_PATH', '')
    if resume_enable and resume_path and os.path.isfile(resume_path):
        save_dir = Path(os.path.dirname(resume_path))
        print(f"Resume enabled: Using save_dir from checkpoint: {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()
