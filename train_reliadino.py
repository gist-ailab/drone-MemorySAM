"""P34-ReliaDINO training script — lean fork of train_sam2_lora_paper.py.

Kept from the SAM2 trainer (so existing tools/conventions keep working):
  - argparse --cfg + yaml config, SAVE_DIR/<DATASET>_<BACKBONE>_<modals> layout
  - dataset/augmentation creation (semseg.datasets / augmentations_mm)
  - OhemCrossEntropy via semseg.losses.get_loss
  - DDP (torchrun) + AMP (bf16/fp16) + gradient accumulation to eff. batch 16
  - eval val (+test for DELIVER) every TRAIN.EVAL_INTERVAL epochs with
    per-class IoU print
  - top-k checkpoint save in the same {'model_state_dict', ...} dict format and
    `epoch{E}_{miou}_top{K}_checkpoint.pth` naming ('' = val, 'test_' = test)
  - last_checkpoint.pth every epoch + AUTO_RESUME

Dropped: SAM2/build_sam2 imports, prototype loss, quality-gate vis, night-val
(MULTIAQUA-specific — add back when P34 goes to MULTIAQUA), matplotlib curves.

Launch (B200):
  conda activate MMSS_SAM
  torchrun --standalone --nproc_per_node=4 train_reliadino.py \
      --cfg configs/b200-deliver_rgbdel_P34_reliadino.yaml
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tabulate import tabulate
from torch import distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.datasets import *                                    # noqa: F401,F403
from semseg.losses import get_loss
from semseg.metrics import Metrics
from semseg.models.reliadino import build_reliadino
from semseg.optimizers import get_optimizer
from semseg.schedulers import get_scheduler
from semseg.utils.utils import (cleanup_ddp, fix_seeds, get_logger, print_iou,
                                setup_cudnn, setup_ddp)


@torch.no_grad()
def evaluate(model, dataloader, device, dist_sync=False):
    """val_mm_sam.evaluate equivalent, re-declared here to avoid its SAM2 imports.

    dist_sync=True: dataloader는 rank별 shard(DistributedSampler)이고, confusion
    hist를 all_reduce해 모든 rank가 전체셋 지표를 동일하게 반환한다(전 rank가
    같은 횟수의 collective 호출 = 대칭). ⚠️ DDP에서는 반드시 UNWRAPPED module을
    넘길 것 — wrapper forward는 buffer broadcast collective를 rank0에서만 enqueue
    해 peers의 barrier와 어긋나 NCCL desync/hang을 만든다(2026-07-12 크래시 원인).
    DistributedSampler 패딩 중복은 ≤ world_size-1장(무시 가능, 최종치는 val.py로)."""
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    show_bar = not (dist_sync and dist.get_rank() != 0)
    for images, labels in tqdm(dataloader, desc='eval', leave=False, disable=not show_bar):
        images = [x.to(device, non_blocking=True) for x in images]
        labels = labels.to(device, non_blocking=True)
        output, _ = model(images, True)
        metrics.update(output.softmax(dim=1), labels)
    if dist_sync:
        dist.all_reduce(metrics.hist)
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    model.train()
    return acc, macc, f1, mf1, ious, miou


def _update_topk_checkpoints(topk_list, new_miou, new_epoch, save_dir, prefix,
                             ckpt_dict, k=5):
    """Same naming/rotation as train_sam2_lora_paper._update_topk_checkpoints."""
    topk_list = list(topk_list) + [(new_miou, new_epoch)]
    topk_list.sort(key=lambda x: x[0], reverse=True)
    if len(topk_list) > k:
        for miou, ep in topk_list[k:]:
            for f in save_dir.glob(f"{prefix}epoch{ep}_{miou}_top*_checkpoint.pth"):
                f.unlink(missing_ok=True)
        topk_list = topk_list[:k]
    for rank, (miou, ep) in enumerate(topk_list, 1):
        target = save_dir / f"{prefix}epoch{ep}_{miou}_top{rank}_checkpoint.pth"
        if (miou, ep) == (new_miou, new_epoch):
            torch.save(ckpt_dict, target)
        else:
            for old_f in save_dir.glob(f"{prefix}epoch{ep}_{miou}_top*_checkpoint.pth"):
                if old_f != target:
                    old_f.rename(target)
                    break
    return topk_list


def main(cfg, gpu, save_dir, logger):
    start = time.time()
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    ddp_enable = train_cfg['DDP']
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_rank0 = (not ddp_enable) or (dist.get_rank() == 0)
    num_workers = 8

    # ── data ────────────────────────────────────────────────────────────────
    traintransform = get_train_augmentation(
        train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'], dataset_cfg=dataset_cfg)
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    testset = None
    if dataset_cfg.get('NAME') != 'MULTIAQUA':
        try:
            testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', valtransform, dataset_cfg['MODALS'])
        except Exception as e:
            print(f"[INFO] Test set not available: {e}")
    class_names = trainset.CLASSES
    num_classes = trainset.n_classes

    # ── model ───────────────────────────────────────────────────────────────
    model = build_reliadino(cfg, num_classes)
    if train_cfg.get('GRADIENT_CHECKPOINT', False):
        model.set_grad_checkpointing(True)
        print("Encoder gradient checkpointing enabled")
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_rank0:
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info(f"[ReliaDINO] backbone={model.encoder.backbone_name} "
                    f"patch={model.encoder.patch} dim={model.encoder.embed_dim}")
        logger.info(f"[ReliaDINO] params total={n_total/1e6:.1f}M trainable={n_train/1e6:.1f}M "
                    f"({100.0*n_train/max(n_total,1):.2f}%)")

    # ── resume ──────────────────────────────────────────────────────────────
    resume_checkpoint = None
    resume_path = model_cfg.get('RESUME_PATH', '')
    if model_cfg.get('RESUME_ENABLE', False) and resume_path and os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(resume_checkpoint['model_state_dict'], strict=False)
        print(f"Resumed weights from {resume_path} (epoch {resume_checkpoint.get('epoch', 0)})")

    # ── optim / sched / loaders / amp ───────────────────────────────────────
    purposed_batch_size = 16
    accumulation_steps = math.ceil(purposed_batch_size / (train_cfg['BATCH_SIZE'] * world_size))
    updates_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * world_size * accumulation_steps)
    iters_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * world_size)

    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    lambda_cal = (model_cfg.get('CALIBRATION', {}) or {}).get('LAMBDA', 0.1)
    lambda_aux_ce = (model_cfg.get('FUSION', {}) or {}).get('AUX_CE_WEIGHT', 0.5)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    # [P35/T1 seam] LoRA up-projection(b_q/b_v) Frobenius norm cap — ep140 진단에서
    # blocks.1 depth-q ‖dW‖ 606(ep40 대비 36×) 폭주 관찰(리뷰 리스크). 기본 0=off.
    lora_norm_cap = float(train_cfg.get('LORA_NORM_CAP', 0) or 0)
    _lora_up_params = [p for n, p in model.named_parameters()
                       if n.endswith(('.b_q', '.b_v'))] if lora_norm_cap > 0 else []
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer,
                              int((epochs + 1) * updates_per_epoch), sched_cfg['POWER'],
                              updates_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    if ddp_enable:
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)

    start_epoch = 0
    best_mIoU, best_epoch = 0.0, 0
    best_test_mIoU, best_test_epoch = 0.0, 0
    top_day_ckpts, top_test_ckpts = [], []
    if resume_checkpoint:
        start_epoch = resume_checkpoint.get('epoch', 0)
        best_mIoU = resume_checkpoint.get('best_miou', 0.0)
        best_epoch = resume_checkpoint.get('best_epoch', 0)
        best_test_mIoU = resume_checkpoint.get('best_test_miou', 0.0)
        top_day_ckpts = resume_checkpoint.get('top_day_ckpts', [])
        top_test_ckpts = resume_checkpoint.get('top_test_ckpts', [])
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

    _loader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
    if num_workers > 0:
        _loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'],
                             drop_last=True, sampler=sampler, **_loader_kwargs)
    _eval_kwargs = {'num_workers': min(num_workers, 4), 'pin_memory': True}
    # DDP: eval도 rank별 shard로 분산 (기존: rank0 단독 3902장 ≈30분 동안 peers가
    # barrier 대기 → NCCL desync/timeout으로 사망). shuffle=False, hist는 evaluate()
    # 안에서 all_reduce.
    if ddp_enable:
        _val_sampler = DistributedSampler(valset, dist.get_world_size(),
                                          dist.get_rank(), shuffle=False)
        _test_sampler = (DistributedSampler(testset, dist.get_world_size(),
                                            dist.get_rank(), shuffle=False)
                         if testset is not None else None)
    else:
        _val_sampler = _test_sampler = None
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'],
                           sampler=_val_sampler, **_eval_kwargs)
    testloader = DataLoader(testset, batch_size=eval_cfg['BATCH_SIZE'],
                            sampler=_test_sampler, **_eval_kwargs) \
        if testset is not None else None

    _amp_dtype_str = str(train_cfg.get('AMP_DTYPE', 'float16')).lower()
    AMP_DTYPE = torch.bfloat16 if _amp_dtype_str in ('bf16', 'bfloat16') else torch.float16
    scaler = GradScaler(enabled=(train_cfg['AMP'] and AMP_DTYPE == torch.float16))
    if resume_checkpoint and 'scaler_state_dict' in resume_checkpoint and scaler.is_enabled():
        if resume_checkpoint['scaler_state_dict']:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])

    writer = SummaryWriter(str(save_dir)) if is_rank0 else None
    wandb_enabled = False
    if is_rank0 and HAS_WANDB:
        wandb_cfg = cfg.get('WANDB', {}) or {}
        wandb_enabled = (wandb_cfg.get('ENABLE', True)
                         and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'))
        if wandb_enabled:
            if not os.environ.get('WANDB_API_KEY'):
                _key_file = Path(__file__).resolve().parent / '.wandb_key'
                if _key_file.is_file() and _key_file.read_text().strip():
                    os.environ['WANDB_API_KEY'] = _key_file.read_text().strip()
            try:
                wandb.init(project=wandb_cfg.get('PROJECT', 'MemorySAM'),
                           name=wandb_cfg.get('NAME', None) or cfg.get('_CFG_NAME', save_dir.name),
                           dir=str(save_dir),
                           tags=[f"model:ReliaDINO", f"backbone:{model_cfg['BACKBONE']}",
                                 f"dataset:{dataset_cfg['NAME']}",
                                 f"modals:{''.join(m[0] for m in dataset_cfg['MODALS'])}",
                                 f"lora_r:{model_cfg.get('LORA_R', 8)}"],
                           config=cfg)
            except Exception as e:
                print(f"[wandb] init failed ({e}); continuing without wandb")
                wandb_enabled = False

    modals = dataset_cfg['MODALS']
    _core = model.module if hasattr(model, 'module') else model

    # ── train loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        _core._current_epoch = epoch
        if ddp_enable:
            sampler.set_epoch(epoch)
        train_loss = cal_accum = aux_accum = gate_ent_accum = router_accum = 0.0
        cefr_accum = 0.0
        auroc_rows, gate_rows, router_rows, cefr_rows = [], [], [], []

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch [{epoch+1}/{epochs}]", disable=not is_rank0)
        it = 0
        for it, (sample, lbl) in pbar:
            sample = [x.to(device, non_blocking=True) for x in sample]
            lbl = lbl.to(device, non_blocking=True)
            with autocast(enabled=train_cfg['AMP'], dtype=AMP_DTYPE):
                logits, m_feat, aux = model(sample, True, gt_mask=lbl)
                loss_seg = loss_fn(logits, lbl)
                _zero = logits.new_zeros(())
                cal_loss = aux.get('rbma_cal_loss', _zero)
                aux_ce = aux.get('aux_ce', _zero)
                gate_ent = aux.get('gate_entropy', _zero)
                router_reg = aux.get('router_reg', _zero)   # [P36] pre-scaled in fusion
                cefr_reg = aux.get('cefr_reg', _zero)       # [P37a] pre-scaled (decisive+hinge)
                total = (loss_seg + lambda_cal * cal_loss
                         + lambda_aux_ce * aux_ce + gate_ent + router_reg
                         + cefr_reg)
                loss = total / accumulation_steps
            scaler.scale(loss).backward()
            if (it + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if lora_norm_cap > 0:
                    # per-modality slice별 cap (b[m] (attn_dim,r)) — 방향 보존 renorm
                    with torch.no_grad():
                        for p in _lora_up_params:
                            nrm = p.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-12)
                            factor = (lora_norm_cap / nrm).clamp(max=1.0)
                            p.mul_(factor.view(-1, *([1] * (p.dim() - 1))))

            train_loss += total.item()
            cal_accum += float(cal_loss)
            aux_accum += float(aux_ce)
            gate_ent_accum += float(gate_ent)
            router_accum += float(router_reg)
            cefr_accum += float(cefr_reg)
            if _core.fusion._last_rel_auroc is not None:
                auroc_rows.append(_core.fusion._last_rel_auroc)
            if _core.fusion._last_gate_mean is not None:
                gate_rows.append(_core.fusion._last_gate_mean.cpu().tolist())
            if getattr(_core.fusion, '_last_router_mean', None) is not None:
                router_rows.append(_core.fusion._last_router_mean.tolist())
            _cefr = getattr(_core.fusion, 'cefr', None)     # [P37a]
            if _cefr is not None and _cefr._last_w_mean is not None:
                cefr_rows.append(_cefr._last_w_mean.tolist())
            if is_rank0:
                pbar.set_description(
                    f"Epoch [{epoch+1}/{epochs}] Loss {train_loss/(it+1):.4f} "
                    f"cal {cal_accum/(it+1):.4f} auxCE {aux_accum/(it+1):.4f}")

        train_loss /= (it + 1)
        avg_lr = scheduler.get_lr()
        avg_lr = float(sum(avg_lr) / len(avg_lr))

        if is_rank0:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/cal_loss', cal_accum / (it + 1), epoch)
            writer.add_scalar('train/aux_ce', aux_accum / (it + 1), epoch)
            writer.add_scalar('train/lr', avg_lr, epoch)
            log_extra = {}
            if auroc_rows:
                auroc_ep = np.nanmean(np.array(auroc_rows, dtype=np.float64), axis=0)
                for i, name in enumerate(modals[:len(auroc_ep)]):
                    writer.add_scalar(f'p34/rel_auroc_{name}', auroc_ep[i], epoch)
                    log_extra[f'p34/rel_auroc_{name}'] = float(auroc_ep[i])
                logger.info(f"[P34] rel AUROC " +
                            " ".join(f"{n}:{a:.3f}" for n, a in zip(modals, auroc_ep)))
            if gate_rows:
                gbar = np.array(gate_rows, dtype=np.float64).mean(axis=0)
                for i, name in enumerate(modals[:len(gbar)]):
                    writer.add_scalar(f'p34/gate_w_{name}', gbar[i], epoch)
                    log_extra[f'p34/gate_w_{name}'] = float(gbar[i])
                logger.info(f"[P34] gate w̄ " +
                            " ".join(f"{n}:{w:.3f}" for n, w in zip(modals, gbar)))
            if router_rows:
                rbar = np.array(router_rows, dtype=np.float64).mean(axis=0)
                alpha = float(_core.fusion.router_alpha.detach())
                for i, name in enumerate(modals[:len(rbar)]):
                    writer.add_scalar(f'p36/router_w_{name}', rbar[i], epoch)
                    log_extra[f'p36/router_w_{name}'] = float(rbar[i])
                writer.add_scalar('p36/router_alpha', alpha, epoch)
                log_extra['p36/router_alpha'] = alpha
                logger.info(f"[P36] router w̄ " +
                            " ".join(f"{n}:{w:.3f}" for n, w in zip(modals, rbar)) +
                            f" alpha:{alpha:.4f}")
            if cefr_rows:                                   # [P37a] CEFR monitoring
                cbar = np.array(cefr_rows, dtype=np.float64).mean(axis=0)
                sigma_a = float(getattr(_core.fusion.cefr, '_last_sigma_a', 0.0) or 0.0)
                for i, name in enumerate(modals[:len(cbar)]):
                    writer.add_scalar(f'p37/cefr_w_{name}', cbar[i], epoch)
                    log_extra[f'p37/cefr_w_{name}'] = float(cbar[i])
                writer.add_scalar('p37/cefr_sigma_a', sigma_a, epoch)
                log_extra['p37/cefr_sigma_a'] = sigma_a
                log_extra['train/cefr_reg'] = cefr_accum / (it + 1)
                logger.info(f"[P37] cefr w̄ " +
                            " ".join(f"{n}:{w:.3f}" for n, w in zip(modals, cbar)) +
                            f" sigma_a:{sigma_a:.4f}")
            if wandb_enabled:
                wandb.log({'epoch': epoch + 1, 'train/total_loss': train_loss,
                           'train/cal_loss': cal_accum / (it + 1),
                           'train/aux_ce': aux_accum / (it + 1),
                           'train/gate_entropy': gate_ent_accum / (it + 1),
                           'train/router_reg': router_accum / (it + 1),
                           'train/lr': avg_lr, **log_extra}, step=epoch)

            # last checkpoint every epoch (same dict format as SAM2 trainer)
            def _ckpt(extra=None):
                d = {
                    'epoch': epoch + 1,
                    'model_state_dict': (model.module.state_dict() if ddp_enable
                                         else model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if train_cfg['AMP'] else None,
                    'loss': train_loss,
                    'best_miou': best_mIoU,
                    'best_epoch': best_epoch,
                    'best_test_miou': best_test_mIoU,
                    'top_day_ckpts': top_day_ckpts,
                    'top_test_ckpts': top_test_ckpts,
                }
                if extra:
                    d.update(extra)
                return d
            torch.save(_ckpt(), save_dir / 'last_checkpoint.pth')

        # ── eval every EVAL_INTERVAL epochs (val + test, per-class IoU) ──────
        do_eval = (((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0
                    and (epoch + 1) > train_cfg['EVAL_START'])
                   or (epoch + 1) == epochs)
        if do_eval:
            # 전 rank가 자기 shard를 평가(hist all_reduce로 전체 지표 동일) — 기존
            # rank0-단독-eval + peers-barrier 구조는 DDP wrapper의 buffer-broadcast
            # collective가 barrier와 어긋나 NCCL desync SIGABRT를 유발했음(07-12 ×2).
            torch.cuda.empty_cache()
            _eval_model = model.module if ddp_enable else model
            acc, macc, f1, mf1, ious, miou = evaluate(
                _eval_model, valloader, device, dist_sync=ddp_enable)
            if is_rank0:
                writer.add_scalar('val/mIoU', miou, epoch)
                iou_str = " | ".join(f"{c}: {v:.2f}" for c, v in zip(class_names, ious))
                worst_day = top_day_ckpts[-1][0] if len(top_day_ckpts) >= 5 else -1.0
                if miou > worst_day:
                    top_day_ckpts = _update_topk_checkpoints(
                        top_day_ckpts, miou, epoch + 1, save_dir, prefix='',
                        ckpt_dict=_ckpt({'best_miou': miou}), k=5)
                    if miou > best_mIoU:
                        best_mIoU, best_epoch = miou, epoch + 1
                        logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"[Val] epoch:{epoch+1}  mIoU: {miou:.4f}  "
                            f"Best: {best_mIoU:.4f} (ep{best_epoch})\n     IoU: {iou_str}")
                if wandb_enabled:
                    wlog = {'epoch': epoch + 1, 'val/mIoU': miou, 'val/best_mIoU': best_mIoU}
                    wlog.update({f'val_iou/{c}': v for c, v in zip(class_names, ious)})
                    wandb.log(wlog, step=epoch)

            if testloader is not None:
                t_acc, t_macc, t_f1, t_mf1, t_ious, t_miou = evaluate(
                    _eval_model, testloader, device, dist_sync=ddp_enable)
            if testloader is not None and is_rank0:
                writer.add_scalar('test/mIoU', t_miou, epoch)
                t_iou_str = " | ".join(f"{c}: {v:.2f}" for c, v in zip(class_names, t_ious))
                worst_test = top_test_ckpts[-1][0] if len(top_test_ckpts) >= 5 else -1.0
                if t_miou > worst_test:
                    top_test_ckpts = _update_topk_checkpoints(
                        top_test_ckpts, t_miou, epoch + 1, save_dir, prefix='test_',
                        ckpt_dict=_ckpt({'best_test_miou': t_miou}), k=5)
                    if t_miou > best_test_mIoU:
                        best_test_mIoU, best_test_epoch = t_miou, epoch + 1
                logger.info(f"[Test] epoch:{epoch+1}  mIoU: {t_miou:.4f}  "
                            f"Best: {best_test_mIoU:.4f} (ep{best_test_epoch})"
                            f"\n      IoU: {t_iou_str}")
                if wandb_enabled:
                    wlog = {'epoch': epoch + 1, 'test/mIoU': t_miou,
                            'test/best_mIoU': best_test_mIoU}
                    wlog.update({f'test_iou/{c}': v for c, v in zip(class_names, t_ious)})
                    wandb.log(wlog, step=epoch)
        if ddp_enable:
            dist.barrier()

    if is_rank0:
        writer.close()
        if wandb_enabled:
            wandb.finish()
        end = time.gmtime(time.time() - start)
        logger.info(tabulate([
            ['Best Val mIoU', f"{best_mIoU:.2f}  (epoch {best_epoch})"],
            ['Best Test mIoU', f"{best_test_mIoU:.2f}  (epoch {best_test_epoch})"
             if best_test_mIoU > 0 else "N/A"],
            ['Total Training Time', time.strftime("%H:%M:%S", end)],
        ], numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='configs/b200-deliver_rgbdel_P34_reliadino.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg['_CFG_NAME'] = Path(args.cfg).stem

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join(m[0] for m in cfg['DATASET']['MODALS'])
    exp_name = '_'.join([cfg['DATASET']['NAME'], cfg['MODEL']['BACKBONE'], modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)

    # AUTO_RESUME: same semantics as train_sam2_lora_paper.py
    if cfg['MODEL'].get('AUTO_RESUME', False) and not (
            cfg['MODEL'].get('RESUME_ENABLE', False) and cfg['MODEL'].get('RESUME_PATH', '')):
        auto_ckpt = save_dir / 'last_checkpoint.pth'
        if auto_ckpt.is_file():
            cfg['MODEL']['RESUME_ENABLE'] = True
            cfg['MODEL']['RESUME_PATH'] = str(auto_ckpt)
            print(f"[AUTO_RESUME] resuming from {auto_ckpt}")

    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir, logger)
    cleanup_ddp()
