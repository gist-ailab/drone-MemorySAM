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
from semseg.models.sam2.sam2.build_sam import build_sam2 as build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg_bkup import LoRA_Sam
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import LoRA_Sam_P
import matplotlib
import math
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

class PrototypeSegmentation:
    def __init__(self, num_classes, feature_dim, device):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        # 하드코딩된 'cuda' 대신 device를 사용합니다.
        self.global_prototypes = torch.zeros((num_classes, feature_dim), requires_grad=False).to(device) 

    def update_global_prototypes(self, current_prototypes):
        self.global_prototypes.data = 0.8 * self.global_prototypes.data + 0.2 * current_prototypes.data

    def compute_loss(self, features, labels):
        batch_prototypes = self.calculate_batch_prototypes(features, labels)
        self.update_global_prototypes(batch_prototypes)
        prototype_loss = self.prototype_loss(batch_prototypes)
        return prototype_loss

    def calculate_batch_prototypes(self, features, labels):
        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device)
        count = torch.zeros(self.num_classes, device=features.device)
        
        labels = labels.to(features.device).unsqueeze(1)
        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(1)

        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels_resized = labels_resized.view(-1)

        for i in range(self.num_classes):
            mask = (labels_resized == i)
            if mask.sum() > 0:
                batch_prototypes[i] = features[mask].mean(dim=0)
                count[i] = mask.sum()
        
        count = count.clamp(min=1)
        return batch_prototypes / count.unsqueeze(1)

    def prototype_loss(self, batch_prototypes):
        return F.mse_loss(batch_prototypes, self.global_prototypes)

def plot_training_curves(save_dir, epochs, losses, proto_losses, lrs):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(epochs, losses, 'b-', label='Train Loss')
    axes[0].set_title('Training Loss')
    axes[1].plot(epochs, proto_losses, 'r-', label='Proto Loss')
    axes[1].set_title('Prototype Loss')
    axes[2].plot(epochs, lrs, 'g-', label='Learning Rate')
    axes[2].set_title('Learning Rate')
    axes[2].set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png')
    plt.close()

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 4 # 디버깅 시에는 적은 숫자가 유리할 수 있습니다.
    
    # DDP 환경 변수 안전하게 가져오기
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    is_main_process = rank == 0 # 메인 프로세스 여부
    
    device = torch.device(cfg['DEVICE'] if torch.cuda.is_available() else 'cpu')
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg_yaml = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES

    # 모델 설정 (경로 확인 필요)
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    checkpoint = "/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM/semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2 = build_sam2("sam2_hiera_b+.yaml", checkpoint, hydra_overrides_extra=[
        "++model.pred_obj_scores=false", "++model.fixed_no_obj_ptr=false", "++model.pred_obj_scores_mlp=false"
    ])
    model = LoRA_Sam(sam2, 4).to(device)

    if train_cfg['DDP'] and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
        sampler = DistributedSampler(trainset, world_size, rank, shuffle=True)
    else:
        sampler = RandomSampler(trainset)

    # 파라미터 셋팅
    for param in model.parameters():
        param.requires_grad = False
    # 학습이 필요한 부분만 True (LoRA 및 특정 Head)
    if hasattr(model, 'module'): # DDP
        target_model = model.module
    else:
        target_model = model

    for param in target_model.sam.sam_mask_decoder.parameters(): param.requires_grad = True
    for param in target_model.sam.memory_attention.parameters(): param.requires_grad = True
    for param in target_model.sam.memory_encoder.parameters(): param.requires_grad = True

    # Accumulation 설정
    purposed_batch_size = 16
    accumulation_steps = max(1, math.ceil(purposed_batch_size / (train_cfg['BATCH_SIZE'] * world_size)))
    iters_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * world_size)

    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs + 1) * (iters_per_epoch // accumulation_steps + 1)), 
                              sched_cfg['POWER'], (iters_per_epoch // accumulation_steps) * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    
    if is_main_process:
        writer = SummaryWriter(str(save_dir))
        logger.info('================== training config =====================')
        logger.info(cfg)
    
    prototypeseg = PrototypeSegmentation(25, 32, device)
    train_losses, train_proto_losses, train_lrs, epochs_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP'] and world_size > 1: sampler.set_epoch(epoch)
        
        train_loss, proto_loss = 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(enumerate(DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, sampler=sampler)), 
                    total=iters_per_epoch, disable=not is_main_process)
        
        for iter, (sample, lbl) in pbar:
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)
            
            with autocast(enabled=train_cfg['AMP']):
                output, m_feat = model(sample, multimask_output=True)
                loss_orig = loss_fn(output, lbl)
                protoloss = prototypeseg.compute_loss(m_feat, lbl) * 65536 # 256*256
                total_loss = (loss_orig + protoloss) / accumulation_steps

            scaler.scale(total_loss).backward()
            
            if (iter + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # 로깅용 값 업데이트
            train_loss += (loss_orig.item() + protoloss.item())
            proto_loss += protoloss.item()
            
            if is_main_process:
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] LR: {curr_lr:.8f} Loss: {train_loss/(iter+1):.4f}")

        if is_main_process:
            avg_loss = train_loss / (iters_per_epoch + 1)
            avg_proto = proto_loss / (iters_per_epoch + 1)
            train_losses.append(avg_loss)
            train_proto_losses.append(avg_proto)
            train_lrs.append(optimizer.param_groups[0]['lr'])
            epochs_list.append(epoch + 1)
            
            writer.add_scalar('train/loss', avg_loss, epoch)
            plot_training_curves(save_dir, epochs_list, train_losses, train_proto_losses, train_lrs)

        # Evaluation
        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0) or (epoch+1) == epochs:
            acc, macc, _, _, ious, miou = evaluate(model, DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers), device)
            if is_main_process:
                writer.add_scalar('val/mIoU', miou, epoch)
                if miou > best_mIoU:
                    best_mIoU, best_epoch = miou, epoch + 1
                    save_path = save_dir / "best_model.pth"
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
                logger.info(f"Epoch {epoch+1} mIoU: {miou:.4f} Best: {best_mIoU:.4f}")

    if is_main_process: writer.close()
    cleanup_ddp() if train_cfg['DDP'] and world_size > 1 else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM/configs/bengio_deliver_rgbdel_sam.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    
    # DDP 설정 시도, 실패 시(Single GPU) 일반 설정 유지
    try:
        gpu = setup_ddp()
    except:
        gpu = 0
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'

    save_dir = Path(cfg['SAVE_DIR'], f"{cfg['DATASET']['NAME']}_{cfg['MODEL']['BACKBONE']}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    
    main(cfg, gpu, save_dir)