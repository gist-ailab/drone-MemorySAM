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
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
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

def plot_training_curves(save_dir, epochs, losses, proto_losses, lrs):
    """Plot and save training curves for Loss, Proto Loss, and LR"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
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
    
    # Plot LR
    axes[2].plot(epochs, lrs, 'g-', linewidth=2, label='Learning Rate')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_yscale('log')  # Use log scale for LR
    
    plt.tight_layout()
    plot_path = save_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual plots
    # Combined Loss plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, proto_losses, 'r-', linewidth=2, label='Proto Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = save_dir / 'training_losses.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
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

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
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


    scaler = GradScaler(enabled=train_cfg['AMP'])
    


    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        # logger.info('================== model complexity =====================')
        # cal_flops(model, dataset_cfg['MODALS'], logger)
        # logger.info('================== model structure =====================')
        # logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info(f"Using LoRA model: {lora_model_name}")
        logger.info(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")
    
    num_classes = 25
    feature_dim = 32
    prototypeseg = PrototypeSegmentation(num_classes, feature_dim)
    
    # Initialize lists for tracking training metrics
    # Restore from checkpoint if resuming
    if resume_checkpoint and 'train_losses' in resume_checkpoint:
        train_losses = resume_checkpoint['train_losses']
        train_proto_losses = resume_checkpoint['train_proto_losses']
        train_lrs = resume_checkpoint['train_lrs']
        epochs_list = resume_checkpoint['epochs_list']
        print(f"Restored training metrics: {len(epochs_list)} epochs")
    else:
        train_losses = []
        train_proto_losses = []
        train_lrs = []
        epochs_list = []
    
    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)
        train_loss = 0.0  
        proto_loss = 0.0 
       
    
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
                output, m_feat = model(sample, multimask_output=True)
                logits = output
                
                loss_orig = loss_fn(logits, lbl)
                protoloss = prototypeseg.compute_loss(m_feat, lbl) * 256 * 256
                
                # [수정] 전체 Loss를 accumulation_steps로 나누어야 정확한 그라디언트 평균이 계산됩니다.
                total_loss_unscaled = loss_orig + protoloss
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
            
            
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f} Proto Loss: {proto_loss / (iter+1):.8f}")
        
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
            writer.add_scalar('train/lr', avg_lr, epoch)
            
            # Plot and save graphs
            plot_training_curves(save_dir, epochs_list, train_losses, train_proto_losses, train_lrs)
            
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
                'train_losses': train_losses,
                'train_proto_losses': train_proto_losses,
                'train_lrs': train_lrs,
                'epochs_list': epochs_list,
            }, last_ckp_path)
            
        torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU} Loss: {train_loss :.8f} Proto Loss: {proto_loss :.8f}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
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
