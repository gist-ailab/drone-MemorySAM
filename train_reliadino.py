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
import contextlib
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
from semseg.losses import OhemCrossEntropy, get_loss
from semseg.metrics import Metrics
from semseg.models.reliadino import build_reliadino
from semseg.models.reliadino import p46 as P46
from semseg.models.reliadino import p47 as P47
from semseg.models.reliadino.mmpareto import MMPareto
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


def _atomic_save(obj, path):
    """[ISSUE-030 fix] torch.save는 대상 경로에 직접 쓴다 — 저장 도중 사망(preempt/
    OOM/SIGKILL)하면 파일이 잘린 채 남아 그 이름을 신뢰하는 코드(AUTO_RESUME 등)가
    깨진다. 같은 디렉터리의 임시 파일에 먼저 쓰고 os.replace로 교체한다(동일
    파일시스템 내 rename은 원자적 — 중간 상태가 존재하지 않는다)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
            _atomic_save(ckpt_dict, target)
        else:
            for old_f in save_dir.glob(f"{prefix}epoch{ep}_{miou}_top*_checkpoint.pth"):
                if old_f != target:
                    old_f.rename(target)
                    break
    return topk_list


def _p49_llrd_groups(model, lr: float, decay: float,
                     backbone_prefix: str = 'encoder.backbone'):
    """[P49-A1] layer-wise LR decay 파라미터 그룹.

    A1이 백본을 통째로 푼다(≈300M trainable). 전 층에 같은 LR을 주면 저층의
    사전학습 표현이 초반 warmup에서 지워진다 — DINOv3/BEiT/ViT-Adapter 계열의
    표준 처방이 layer-wise LR decay다. 깊이 i 층의 LR = lr · decay^(L+1−i).

    층 인덱스는 **파라미터 이름**에서 읽는다:
      `<prefix>.blocks.<i>.…`            -> i + 1
      `<prefix>.` 의 나머지(patch_embed / cls_token / pos_embed / reg_token / rope …)
                                          -> 0   (가장 깊은 감쇠)
      백본 밖(보조 인코더 / injector / FPN / head / γ)   -> L + 1 (감쇠 없음, lr 그대로)
    즉 신규 모듈은 항상 full lr 이고 백본만 깎인다. `p.dim() == 1` (norm/bias/γ)은
    weight decay 0 — semseg.optimizers.get_optimizer 와 같은 규약이다.

    이 함수는 P49 + RGB_FT + OPTIMIZER.LLRD 가 모두 켜졌을 때만 호출된다.
    """
    core = model.module if hasattr(model, 'module') else model
    bb = core
    for part in backbone_prefix.split('.'):
        bb = getattr(bb, part, None)
        if bb is None:
            raise RuntimeError(f"[P49-LLRD] '{backbone_prefix}' 를 찾지 못했다 — "
                               f"P49ViTEncoder 레이아웃이 바뀌었는지 확인하라")
    n_layers = len(bb.blocks)
    top = n_layers + 1

    def layer_id(name: str) -> int:
        if not name.startswith(backbone_prefix + '.'):
            return top
        rest = name[len(backbone_prefix) + 1:]
        if rest.startswith('blocks.'):
            try:
                return int(rest.split('.')[1]) + 1
            except (IndexError, ValueError):
                return top
        return 0

    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        key = (lid, p.dim() == 1)
        buckets.setdefault(key, []).append(p)
    groups = []
    for (lid, no_decay), params in sorted(buckets.items()):
        g = {'params': params, 'lr': lr * (decay ** (top - lid))}
        if no_decay:
            g['weight_decay'] = 0.0
        groups.append(g)
    return groups


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
    # MUSES-only knob (P47-MUB D-1): swap the 'projected_to_rgb' subdir for a
    # densified-projection variant (e.g. 'projected_to_rgb_dgf'). Other dataset
    # classes (DELIVER, MULTIAQUA) don't accept this kwarg, so only pass it
    # when NAME == 'MUSES' — DELIVER's __init__ has no proj_dir param and
    # would TypeError on an unexpected kwarg.
    ds_kwargs = {}
    if dataset_cfg.get('NAME') == 'MUSES':
        ds_kwargs['proj_dir'] = dataset_cfg.get('PROJ_DIR', 'projected_to_rgb')
        # MUSES-only knob (조건-전문가 oracle 프로브): train/val 을 하나의 조건
        # 셀로 제한한다. 'fog_night' 같은 조합 또는 'fog'/'night' 단일 축.
        # 미지정(기본)이면 기존과 완전히 동일 — 전 조건 학습.
        if dataset_cfg.get('CASE'):
            ds_kwargs['case'] = dataset_cfg['CASE']
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'], **ds_kwargs)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'], **ds_kwargs)
    testset = None
    if dataset_cfg.get('NAME') != 'MULTIAQUA':
        try:
            testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', valtransform, dataset_cfg['MODALS'], **ds_kwargs)
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
        _ld = model.load_state_dict(resume_checkpoint['model_state_dict'], strict=False)
        print(f"Resumed weights from {resume_path} (epoch {resume_checkpoint.get('epoch', 0)}) "
              f"missing={len(_ld.missing_keys)} unexpected={len(_ld.unexpected_keys)}")
        if _ld.missing_keys:
            print(f"  missing[:8]={_ld.missing_keys[:8]}")
        if _ld.unexpected_keys:
            print(f"  unexpected[:8]={_ld.unexpected_keys[:8]}")
        # FINETUNE_INIT: 가중치만 가져오고 optimizer/scheduler/epoch 카운터는 초기화한다.
        # 수렴한 ckpt에서 짧게 미세조정할 때(조건-전문가 프로브) 필요 — 그냥 RESUME 하면
        # start_epoch 과 거의 소진된 LR 스케줄까지 복원돼 전문화가 일어나지 않는다.
        # ⚠️ AUTO_RESUME 과 같이 켜지 말 것 (크래시 후 epoch 0 으로 되돌아가 무한 재시작).
        if model_cfg.get('FINETUNE_INIT', False):
            resume_checkpoint = None
            print("[FINETUNE_INIT] weights only — optimizer/scheduler/epoch reset to fresh")

    # ── optim / sched / loaders / amp ───────────────────────────────────────
    purposed_batch_size = 16
    accumulation_steps = math.ceil(purposed_batch_size / (train_cfg['BATCH_SIZE'] * world_size))
    updates_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * world_size * accumulation_steps)
    iters_per_epoch = len(trainset) // (train_cfg['BATCH_SIZE'] * world_size)

    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    # [P49-A5] OHEM CE 옵션. `LOSS.OHEM` 이 없으면 위 get_loss 결과 그대로 =
    # 기존 경로 완전 무변경. (`LOSS.NAME: OhemCrossEntropy` 로도 켤 수 있지만
    # 그 경로는 thresh/min_kept 를 노출하지 않아 표준값 고정이 불가능하다.)
    if loss_cfg.get('OHEM', False):
        loss_fn = OhemCrossEntropy(trainset.ignore_label, None,
                                   thresh=float(loss_cfg.get('OHEM_THRESH', 0.7)),
                                   min_kept=int(loss_cfg.get('OHEM_MIN_KEPT', 100000)))
        if is_rank0:
            logger.info(f"[LOSS] OHEM CE on — thresh={loss_cfg.get('OHEM_THRESH', 0.7)} "
                        f"min_kept={loss_cfg.get('OHEM_MIN_KEPT', 100000)}")
    lambda_cal = (model_cfg.get('CALIBRATION', {}) or {}).get('LAMBDA', 0.1)
    lambda_aux_ce = (model_cfg.get('FUSION', {}) or {}).get('AUX_CE_WEIGHT', 0.5)
    # [P37b] class-token aux CE weight (only produced when CLASS_TOKEN.ENABLE)
    lambda_ctd = (model_cfg.get('CLASS_TOKEN', {}) or {}).get('AUX_CE_W', 0.4)
    # [P49-A1] layer-wise LR decay. P49 + RGB_FT + 유효한 LLRD 셋이 전부 켜졌을
    # 때만 다른 경로를 탄다 — 그 외에는 아래 get_optimizer 한 줄로 기존과 동일.
    _p49_cfg = (model_cfg.get('P49', {}) or {})
    _llrd = float(optim_cfg.get('LLRD', 0) or 0)
    if _p49_cfg.get('ENABLE', False) and _p49_cfg.get('RGB_FT', True) and 0.0 < _llrd < 1.0:
        _groups = _p49_llrd_groups(model, lr, _llrd)
        if str(optim_cfg['NAME']).lower() == 'adamw':
            optimizer = torch.optim.AdamW(_groups, lr, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=optim_cfg['WEIGHT_DECAY'])
        else:
            optimizer = torch.optim.SGD(_groups, lr, momentum=0.9,
                                        weight_decay=optim_cfg['WEIGHT_DECAY'])
        if is_rank0:
            _lrs = sorted({g['lr'] for g in _groups})
            logger.info(f"[P49-A1] LLRD on — decay={_llrd} groups={len(_groups)} "
                        f"lr range [{_lrs[0]:.3e}, {_lrs[-1]:.3e}]")
    else:
        optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    # [P35/T1 seam] LoRA up-projection(b_q/b_v) Frobenius norm cap — ep140 진단에서
    # blocks.1 depth-q ‖dW‖ 606(ep40 대비 36×) 폭주 관찰(리뷰 리스크). 기본 0=off.
    lora_norm_cap = float(train_cfg.get('LORA_NORM_CAP', 0) or 0)
    _lora_up_params = [p for n, p in model.named_parameters()
                       if n.endswith(('.b_q', '.b_v'))] if lora_norm_cap > 0 else []
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer,
                              int((epochs + 1) * updates_per_epoch), sched_cfg['POWER'],
                              updates_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    # ── [P46-CTR] class-transfer recovery: C-1 RCS 샘플러 ────────────────────
    # 설계 = decisions/2026-07-28-p46-classtransfer-recovery-proposal.md
    # 진단 대응: rare/thin 클래스(RailTrack·Wall·Water·Bridge…)가 늦게·덜
    # 학습돼 OOD-test에서 무너진다 → 그 클래스를 담은 이미지를 **주 CE/M2F
    # 손실이 더 자주 보게** 만든다 (DAFormer 2111.14887 RCS).
    _p46_cfg = (model_cfg.get('P46', {}) or {})
    _c1 = (_p46_cfg.get('C1_RCS', {}) or {})
    _c2 = (_p46_cfg.get('C2_MCC', {}) or {})
    _c3 = (_p46_cfg.get('C3_PROTO', {}) or {})
    p46_class_ema = None
    rcs_sampler = None
    # C-2 / C-3 2-view는 iteration당 forward를 2회 돈다 → DDP 생성 인자에 영향
    # (아래 broadcast_buffers 주석 참조). 그래서 DDP wrap **전에** 결정한다.
    p46_mcc = bool(_c2.get('ENABLE', False))
    p46_xview = bool(_c3.get('ENABLE', False) and _c3.get('CROSS_VIEW', True))
    p46_branch = p46_mcc or p46_xview
    if _c1.get('ENABLE', False):
        p46_class_ema = P46.ClassLossEMA(num_classes, momentum=_c1.get('EMA_M', 0.99))
        _cache_dir = _c1.get('CACHE_DIR', '') or str(Path(__file__).resolve().parent / '.cache' / 'p46')
        # 빈도 스캔은 전 train 라벨을 1회 읽는다 → rank0만 계산하고 나머지는
        # 캐시가 생길 때까지 barrier에서 대기 (동시 스캔·중복 IO 방지).
        if (not ddp_enable) or dist.get_rank() == 0:
            _pix, _cfiles = P46.compute_class_stats(
                trainset, num_classes, _cache_dir,
                min_pixels=_c1.get('MIN_PIXELS', 1), num_workers=num_workers)
        if ddp_enable:
            dist.barrier()
            if dist.get_rank() != 0:
                _pix, _cfiles = P46.compute_class_stats(
                    trainset, num_classes, _cache_dir,
                    min_pixels=_c1.get('MIN_PIXELS', 1), num_workers=num_workers,
                    verbose=False)
        _base_p = P46.rcs_base_prob(_pix, temperature=_c1.get('TEMP', 0.01),
                                    mode=str(_c1.get('MODE', 'daformer')).lower())
        rcs_sampler = P46.RareClassSampler(
            _cfiles, _base_p, num_samples=len(trainset) // world_size,
            rank=(dist.get_rank() if ddp_enable else 0), world_size=world_size,
            seed=_c1.get('SEED', 0), loss_ema=p46_class_ema,
            blend_w=_c1.get('LOSS_BLEND_W', 1.0), refresh=_c1.get('REFRESH', 32))
        if is_rank0:
            _f = _pix / max(_pix.sum(), 1)
            _order = np.argsort(_f)[:5]
            logger.info(f"[P46-C1] RCS on — mode={_c1.get('MODE', 'daformer')} "
                        f"T={_c1.get('TEMP', 0.01)} blend_w={_c1.get('LOSS_BLEND_W', 1.0)} "
                        f"samples/rank={len(trainset)//world_size}")
            logger.info("[P46-C1] rarest classes: " + ", ".join(
                f"{class_names[c]} f={_f[c]:.2e} P={_base_p[c]:.4f} "
                f"imgs={len(_cfiles[c])}" for c in _order))

    if ddp_enable:
        sampler = (rcs_sampler if rcs_sampler is not None else
                   DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True))
        # 🔴 [P46] broadcast_buffers: DDP는 **매 forward 시작마다** rank0 버퍼를
        # 전 rank에 in-place 복사한다. P46 보조 branch가 켜지면 iteration당
        # forward가 2회라, 2번째 forward의 버퍼 브로드캐스트가 1번째 forward의
        # 그래프가 저장해 둔 버퍼(M2F empty_weight 등)를 in-place로 갈아엎어
        # backward가 "variable ... modified by an inplace operation"으로 죽는다
        # (합성 스모크 tools/smoke_p46.py --ddp 로 실측·재현됨).
        # → 보조 branch가 있을 때만 끈다. P39.1의 버퍼는 empty_weight(상수)뿐이라
        #   끄더라도 rank 간 값이 애초에 동일해 의미 변화가 없고, P46 prototype
        #   bank는 rank-로컬 EMA가 된다(rank마다 타깃이 미세하게 다르나 iid
        #   표본이라 무해 — DDP가 gradient를 평균한다).
        model = DDP(model, device_ids=[gpu], output_device=gpu,
                    find_unused_parameters=True,
                    broadcast_buffers=not p46_branch)
    else:
        sampler = rcs_sampler if rcs_sampler is not None else RandomSampler(trainset)

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
        if p46_class_ema is not None:                       # [P46-C1]
            p46_class_ema.load_state_dict(
                resume_checkpoint.get('p46_class_loss_ema', None))

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
    # [P49] P49AIR 에는 융합 트렁크(`fusion`)가 없다 — 아래 RBMA/gate/router 로깅은
    # 이 핸들이 None 이면 통째로 꺼진다. ReliaDINO 계보에서는 `_core.fusion` 과
    # 동일한 객체라 동작이 바뀌지 않는다.
    _fusion = getattr(_core, 'fusion', None)
    _p49_core = _core if hasattr(_core, 'gamma_log') else None

    # ── [P46-CTR] C-2 (Masked-Context Consistency) + C-3 2-view 보조 branch ──
    # 보조 branch = 주 forward와 **같은 배치**를 (선택적으로) 스타일 변주 →
    # 패치 마스킹한 뒤 한 번 더 통과시키는 forward. 여기서
    #   C-2: 마스킹 영역에서 EMA-teacher(원본 입력) pseudo-label에 consistency
    #   C-3: 같은 prototype bank로 당김 = "스타일이 달라도 같은 클래스는 같은 prototype"
    # 을 얻는다. branch를 하나로 합쳐 forward 비용이 토글 수와 무관하게 ≤2×로 묶인다.
    # (p46_mcc / p46_xview / p46_branch는 DDP wrap 전에 이미 결정돼 있다.)
    p46_teacher = None
    if p46_branch:
        # 🔴 replay 안전성: 보조 branch는 주 forward의 P39 path-dropout 추첨만
        # 재생한다. 입력을 확률적으로 바꾸는 다른 모듈이 켜져 있으면 두 forward의
        # 파라미터 사용 집합이 갈려 DDP reducer가 죽는다(model.__init__ 주석).
        _unsafe = [n for n, on in (
            ('MODAL_DROPOUT', _core.modal_dropout), ('P42.MASK_IMG', _core.p42_mask_img),
            ('P44.LOCAL_MASK', _core.p44_local_mask), ('P40.RCA', _core.rca_enable),
            ('P45.FOGSTYLE', _core.p45_fogstyle)) if on]
        if _unsafe:
            raise RuntimeError(
                f"[P46] C2_MCC/C3.CROSS_VIEW는 확률적 입력 모듈과 함께 쓸 수 없다 "
                f"(replay 불가 → DDP unused-param 불일치): {_unsafe}. 해당 모듈을 끄거나 "
                f"C2_MCC/CROSS_VIEW를 꺼라.")
        _core._p46_replay_path = False
    if p46_mcc:
        p46_teacher = P46.EMATeacher(_core, alpha=_c2.get('EMA_ALPHA', 0.999))
        if is_rank0:
            logger.info(f"[P46-C2] MIC-style masked consistency on — "
                        f"ratio={_c2.get('MASK_RATIO', 0.5)} patch={_c2.get('PATCH', 64)} "
                        f"lambda={_c2.get('LAMBDA', 1.0)} conf={_c2.get('CONF_THRESH', 0.75)} "
                        f"teacher(EMA {p46_teacher.n_ema} params / shared-frozen "
                        f"{p46_teacher.n_shared})")
    if is_rank0 and _c3.get('ENABLE', False):
        logger.info(f"[P46-C3] prototype consistency on — "
                    f"lambda={_c3.get('LAMBDA', 0.1)} tau={_c3.get('TEMPERATURE', 0.1)} "
                    f"ema={_c3.get('EMA', 0.999)} cross_view={p46_xview} "
                    f"feature={_c3.get('FEATURE', 'mfeat')}")
    p46_mcc_w = float(_c2.get('LAMBDA', 1.0))
    p46_mcc_ratio = float(_c2.get('MASK_RATIO', 0.5))
    p46_mcc_patch = int(_c2.get('PATCH', 64))
    p46_mcc_conf = float(_c2.get('CONF_THRESH', 0.75))
    p46_mcc_mode = str(_c2.get('LOSS', 'ce')).lower()
    p46_mcc_warm = int(_c2.get('WARMUP_EP', 5))
    p46_mcc_modals = (None if str(_c2.get('MODALS', 'all')).lower() == 'all'
                      else [modals.index('img')])
    p46_xview_w = float(_c3.get('CROSS_VIEW_W', 1.0))
    p46_c3_warm = int(_c3.get('WARMUP_EP', 5))
    # 보조 branch는 **활성 항이 하나라도 생기는 epoch**부터 돈다 (한 토글의
    # warmup이 다른 토글을 막지 않도록; 각 항은 아래에서 자기 warmup으로 다시 게이팅).
    p46_branch_warm = min([w for w, on in ((p46_mcc_warm, p46_mcc),
                                           (p46_c3_warm, p46_xview)) if on] or [0])
    p46_ema_interval = int(_c1.get('EMA_INTERVAL', 1))
    # ── [P46] 메모리 계측 (`P46_MEM_LOG=<N>` = N iteration마다 기록, 0=off) ────
    # 지연성 OOM 진단용. 두 가지를 **구분**해서 보기 위한 것이다:
    #   (a) 에폭 **안**에서 alloc이 단조증가 → 스텝 간 미해제(진짜 누수)
    #   (b) warmup 경계(epoch == WARMUP_EP)에서 peak가 **계단**으로 뛴다 → 누수가
    #       아니라 보조 branch+teacher가 그 epoch부터 처음 도는 구조적 증가
    # 2026-07-29 ep6-iter0 OOM은 (b)였다 — C2/C3 WARMUP_EP=5, epoch는 0-index라
    # 로그상 ep6(=epoch 5) iter0이 보조 branch가 **최초로** 도는 지점이다.
    p46_mem_log = int(os.environ.get('P46_MEM_LOG', '0'))

    # ── [P47-2] Uni-modal Balance (구 D-2) ──────────────────────────────────
    # 손실(per-modal uni-modal CE)은 model이 만들어 aux['p47_2_uni']로 내려보낸다
    # (pre-scaled). 여기서는 **로깅**과, 선택 토글인 **OGM-GE gradient 변조**만
    # 결선한다 — 후자는 optimizer step 결선이라 모델 안에 있을 수 없다.
    _p47_2 = (model_cfg.get('P47_2', {}) or {})
    _ogm_cfg = (_p47_2.get('OGM_GE', {}) or {})
    p47_2_on = bool(_p47_2.get('ENABLE', False))
    p47_2_ogm = None
    if p47_2_on and is_rank0:
        _act = [modals[i] for i in _core.p47_2.active]
        logger.info(f"[P47-2] uni-modal balance on — lambda_u={_core.p47_2.lambda_u} "
                    f"modals={_act} head={_p47_2.get('HEAD', 'linear')} "
                    f"reduce={_core.p47_2.reduce} gt_div={_core.p47_2.gt_div} "
                    f"warmup_ep={_core.p47_2.warmup_ep} "
                    f"params={sum(p.numel() for p in _core.p47_2.parameters())/1e3:.1f}K")
    if _ogm_cfg.get('ENABLE', False):
        if not p47_2_on:
            raise RuntimeError("[P47-2] OGM_GE는 P47_2.ENABLE 없이는 쓸 수 없다 "
                               "(per-modal 점수의 출처가 uni-modal aux head다)")
        if ((model_cfg.get('P44', {}) or {}).get('MMPARETO', {}) or {}).get('ENABLE', False):
            # 둘 다 optimizer step 직전에 p.grad를 재작성한다 → 결합 의미가 미정의.
            raise RuntimeError("[P47-2] OGM_GE와 P44.MMPARETO는 동시에 켤 수 없다 "
                               "(둘 다 gradient를 재작성한다).")
        p47_2_ogm = P47.OGMGE(_core, num_modalities=len(modals),
                              alpha=float(_ogm_cfg.get('ALPHA', 0.5)),
                              ema=float(_ogm_cfg.get('EMA', 0.9)),
                              min_k=float(_ogm_cfg.get('MIN_K', 0.1)),
                              ge_noise=float(_ogm_cfg.get('GE_NOISE', 0.0)))
        if is_rank0:
            logger.info(f"[P47-2] OGM-GE on — alpha={p47_2_ogm.alpha} "
                        f"ema={p47_2_ogm.ema} min_k={p47_2_ogm.min_k} "
                        f"ge_noise={p47_2_ogm.ge_noise} "
                        f"modulated LoRA tensors={len(p47_2_ogm.params)}")

    # ── [P44-B1] MMPareto gradient 통합 ─────────────────────────────────────
    # OFF(기본)면 mmpareto is None → 아래 학습 루프의 optimizer 경로는 기존과
    # 완전히 동일하다(단일 backward + scaler.step). ON이면 micro-step마다 주
    # 손실/모달-aux 손실을 따로 미분해 Pareto 결합한다. 설계·DDP/AMP 계약은
    # semseg/models/reliadino/mmpareto.py 상단 참조.
    _mmp_cfg = ((model_cfg.get('P44', {}) or {}).get('MMPARETO', {}) or {})
    mmpareto = None
    if _mmp_cfg.get('ENABLE', False):
        mmpareto = MMPareto(_core.named_parameters(), num_modalities=len(modals),
                            modal_names=modals,
                            interval=_mmp_cfg.get('INTERVAL', 1),
                            magnitude=_mmp_cfg.get('MAGNITUDE', 'sum_norm'))
        if is_rank0:
            logger.info(f"[P44-B1] MMPareto on — groups="
                        f"{[g['name'] for g in mmpareto.groups]} "
                        f"interval={mmpareto.interval} magnitude={mmpareto.magnitude} "
                        f"params={len(mmpareto.params)}")
    global_update = 0

    # ── train loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        _core._current_epoch = epoch
        if ddp_enable or rcs_sampler is not None:
            sampler.set_epoch(epoch)      # [P46-C1] RCS는 DDP 여부와 무관하게 필요
        if p46_teacher is not None:
            p46_teacher.set_epoch(epoch)
        if p46_mem_log and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # peak를 epoch 단위로 비교
        train_loss = cal_accum = aux_accum = gate_ent_accum = router_accum = 0.0
        cefr_accum = ctd_accum = m2f_accum = rce_accum = vic_accum = rca_accum = 0.0
        fcr_accum = 0.0   # [P41-F1]
        p43_accum = 0.0   # [P43-T1] mask-cls 주손실
        p42_mask_sum = 0.0; p42_tot = 0   # [P42-M1/D] 실현 마스킹률
        # [P44/P45] 손실항 + 실현 마스킹률 + MMPareto 진단
        mkl_accum = rc_accum = hard_accum = sty_accum = 0.0
        p44_mask_sum = 0.0; p44_tot = 0
        # [P46-CTR] C-2 consistency / C-3 prototype 손실 + pseudo-label 통과율
        mcc_accum = proto_accum = xview_accum = 0.0
        mcc_rate_sum = 0.0; mcc_rate_n = 0
        # [P47-2] uni-modal balance: 손실 + 모달별 CE/정확도 + OGM 계수
        uni_accum = 0.0
        uni_ce_sum = np.zeros(len(modals)); uni_acc_sum = np.zeros(len(modals))
        uni_n = np.zeros(len(modals))
        ogm_k = None
        pareto_stats = []
        window_pareto = False
        rca_picked = rca_seen = 0
        auroc_rows, gate_rows, router_rows, cefr_rows = [], [], [], []

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch [{epoch+1}/{epochs}]", disable=not is_rank0)
        # 감사 2026-07-21: epoch 경계의 accumulation 잔여 gradient가 다음
        # epoch 첫 update로 유출되던 것 차단 (eff-batch 16 계약 준수).
        optimizer.zero_grad(set_to_none=True)
        it = 0
        for it, (sample, lbl) in pbar:
            sample = [x.to(device, non_blocking=True) for x in sample]
            lbl = lbl.to(device, non_blocking=True)
            if it % accumulation_steps == 0:
                # 결정은 optimizer-step 윈도 단위 (윈도 중간에 경로가 바뀌면
                # 누적된 gradient의 의미가 섞인다).
                window_pareto = (mmpareto is not None
                                 and mmpareto.active(global_update))
            # [P44-B1] DDP: pareto 윈도에서는 autograd.grad가 DDP reducer를
            # 우회하므로 no_sync로 hook을 무장 해제하고(안 그러면 다음
            # iteration에서 "Expected to have finished reduction" 사망) 결합
            # 직전에 직접 all_reduce 한다.
            _sync_ctx = (model.no_sync() if (window_pareto and ddp_enable)
                         else contextlib.nullcontext())
            with _sync_ctx:
                with autocast(enabled=train_cfg['AMP'], dtype=AMP_DTYPE):
                    logits, m_feat, aux = model(sample, True, gt_mask=lbl)
                    loss_seg = loss_fn(logits, lbl)
                    _zero = logits.new_zeros(())
                    cal_loss = aux.get('rbma_cal_loss', _zero)
                    aux_ce = aux.get('aux_ce', _zero)
                    gate_ent = aux.get('gate_entropy', _zero)
                    router_reg = aux.get('router_reg', _zero)   # [P36] pre-scaled in fusion
                    cefr_reg = aux.get('cefr_reg', _zero)       # [P37a] pre-scaled (decisive+hinge)
                    ctd_ce = aux.get('ctd_ce', _zero)           # [P37b] class-token aux CE
                    m2f_loss = aux.get('m2f_loss', _zero)       # [P38] pre-scaled (LOSS_W in model)
                    router_ce = aux.get('router_ce', _zero)     # [P39-V5] pre-scaled (ROUTER_CE_W)
                    vicreg = aux.get('vicreg', _zero)           # [P39.1-R2] pre-scaled
                    rca_ce = aux.get('rca_readout', _zero)      # [P40-C3] pre-scaled
                    fcr = aux.get('fcr', _zero)                 # [P41-F1] pre-scaled (λ in model)
                    p43_mask = aux.get('p43_mask_loss', _zero)  # [P43-T1] pre-scaled (λ(t) in model)
                    p44_mkl = aux.get('p44_mutual_kl', _zero)   # [P44-B2] pre-scaled (fusion)
                    p44_rc = aux.get('p44_rel_corr', _zero)     # [P44-B2] pre-scaled (fusion)
                    p44_hard = aux.get('p44_hard_aux', _zero)   # [P44-M3] pre-scaled (model)
                    p45_sty = aux.get('p45_fogstyle', _zero)    # [P45-F1] pre-scaled (model)
                    p46_proto = aux.get('p46_proto', _zero)     # [P46-C3] pre-scaled (LAMBDA in model)
                    p47_uni = aux.get('p47_2_uni', _zero)       # [P47-2] pre-scaled (LAMBDA_U in model)
                    total = (loss_seg + lambda_cal * cal_loss
                             + lambda_aux_ce * aux_ce + gate_ent + router_reg
                             + cefr_reg + lambda_ctd * ctd_ce + m2f_loss + router_ce
                             + vicreg + rca_ce + fcr + p43_mask
                             + p44_mkl + p44_rc + p44_hard + p45_sty + p46_proto
                             + p47_uni)

                    # ── [P46-C2/C3] 보조 branch (스타일 2-view → 패치 마스킹) ──
                    # ⚠️ DDP: 같은 iteration의 2번째 forward. 두 forward의
                    # 파라미터 사용 집합이 같아야 reducer가 살아남으므로
                    #  (a) gt_mask를 **똑같이** 넘기고(내부 aux 손실 결선 동일 —
                    #      반환값은 버린다), (b) path-dropout 추첨을 재생한다.
                    p46_cons = _zero
                    p46_xv = _zero
                    _do_mcc = p46_mcc and epoch >= p46_mcc_warm
                    _do_xv = p46_xview and epoch >= p46_c3_warm
                    if p46_branch and epoch >= p46_branch_warm and (_do_mcc or _do_xv):
                        _bx = list(sample)
                        if _do_xv:
                            _ii = modals.index('img') if 'img' in modals else 0
                            _bx[_ii] = P46.style_jitter_normalized(_bx[_ii])
                        _bm = None
                        if _do_mcc:
                            _bm = P46.patch_mask(_bx[0].shape[0], _bx[0].shape[-2],
                                                 _bx[0].shape[-1], p46_mcc_ratio,
                                                 p46_mcc_patch, _bx[0].device)
                            _bx = P46.apply_patch_mask(_bx, _bm, p46_mcc_modals)
                            # 🔴 메모리: teacher forward를 보조 student forward
                            # **앞으로** 옮겼다. teacher는 4×ViT-L 한 벌을 통째로
                            # 도는데, 뒤에 두면 그 작업메모리가 "주 그래프 + 보조
                            # 그래프"가 **둘 다 살아 있는** 위에 얹혀 peak를 그만큼
                            # 더 올린다. 앞에 두면 주 그래프 하나 위에만 얹힌다.
                            # 값 불변: teacher는 eval+no_grad라 난수를 전혀 쓰지
                            # 않고(경로 dropout·모달 dropout·마스킹 전부 training
                            # 게이트), 파라미터 갱신은 optimizer step 뒤에만 있다.
                            with torch.no_grad():
                                _tlogits = p46_teacher(sample)
                        _core._p46_replay_path = True
                        try:
                            _blogits, _, _baux = model(_bx, True, gt_mask=lbl)
                        finally:
                            _core._p46_replay_path = False
                        # 🔴 메모리: 보조 branch의 aux 손실 중 total에 들어가는 건
                        # p46_proto 하나뿐이다. 나머지(m2f_loss·vicreg·aux_ce·
                        # router_reg/ce)는 backward가 **도달하지 않는** 서브그래프라
                        # backward가 saved tensor를 해제해 주지 않는다. `_baux`가
                        # 살아 있는 한 그 그래프가 통째로 남고, 파이썬 지역변수는
                        # 다음 iteration이 재대입할 때까지 살아 있으므로 결국
                        # "직전 스텝의 미해제 M2F/VICReg 그래프 + 이번 스텝의 주
                        # 그래프 + 이번 스텝의 보조 그래프"가 동시에 존재했다.
                        # 쓰는 항만 꺼내고 즉시 끊는다.
                        _bproto = _baux.get('p46_proto', _zero)
                        del _baux
                        if _do_mcc:
                            p46_cons_raw, p46_mcc_rate = P46.masked_consistency_loss(
                                _blogits, _tlogits, _bm, conf_thresh=p46_mcc_conf,
                                mode=p46_mcc_mode)
                            p46_cons = p46_mcc_w * p46_cons_raw
                            mcc_rate_sum += p46_mcc_rate; mcc_rate_n += 1
                            del p46_cons_raw, _tlogits
                        if _do_xv:
                            # 보조 branch의 prototype 항 = 다른 스타일 view를 **같은**
                            # bank로 당기는 도메인불변 제약 (bank 갱신은 주 forward만).
                            p46_xv = p46_xview_w * _bproto
                        total = total + p46_cons + p46_xv
                        # 손실 텐서가 필요한 그래프를 이미 잡고 있다 → 출력 텐서
                        # 참조는 여기서 끊는다((B,K,768,768) fp32 ≈ 56MiB/장).
                        del _blogits, _bproto, _bx, _bm
                    loss = total / accumulation_steps
                if window_pareto:
                    # per-modal 브랜치 목표(deep-sup aux CE + peer 증류 + hard-pixel
                    # + fog-style)와 주 목표를 분리해 각각 미분한다. 이 분할이
                    # MMPareto의 "unimodal vs multimodal" 목표쌍에 대응한다.
                    _l_aux = (lambda_aux_ce * aux_ce + p44_mkl + p44_rc
                              + p44_hard + p45_sty) / accumulation_steps
                    _l_main = loss - _l_aux
                    # ⚠️ aux 항이 전부 상수(_zero)면 grad 그래프가 없다 →
                    # autograd.grad가 "does not require grad"로 죽는다. 그 경우
                    # aux gradient는 정의상 0이고 결합은 단순 합으로 환원된다.
                    _has_aux = bool(_l_aux.requires_grad)
                    _gm = torch.autograd.grad(scaler.scale(_l_main), mmpareto.params,
                                              retain_graph=_has_aux,
                                              allow_unused=True)
                    _ga = (torch.autograd.grad(scaler.scale(_l_aux), mmpareto.params,
                                               allow_unused=True)
                           if _has_aux else [None] * len(mmpareto.params))
                    mmpareto.accumulate(
                        _gm, _ga,
                        inv_scale=(1.0 / scaler.get_scale() if scaler.is_enabled() else 1.0))
                    del _gm, _ga
                else:
                    scaler.scale(loss).backward()
            if p47_2_ogm is not None:
                # [P47-2] micro-step마다 per-modal uni-modal 정확도를 적재
                # (BS1에서 스텝별 값이 요동치므로 step 경계에서 평균 + EMA한다).
                p47_2_ogm.observe(_core.p47_2.last_acc)
            if (it + 1) % accumulation_steps == 0:
                if p47_2_ogm is not None:
                    # 🔴 순서: backward가 끝나(=DDP all-reduce 완료) p.grad가 확정된
                    # **뒤**, optimizer.step() **전**. AMP GradScaler의 스케일은
                    # 상수배와 교환 가능하므로 unscale_ 없이 안전하다.
                    ogm_k = p47_2_ogm.apply_()
                if window_pareto:
                    _st = mmpareto.combine()      # allreduce → Pareto 결합 → p.grad
                    pareto_stats.append(_st)
                    if scaler.is_enabled():
                        # fp16: unscale_/step을 안 거쳐 GradScaler의 inf 기록이
                        # 없다 → 직접 검사하고 new_scale을 명시해야 한다.
                        _ok = all(bool(torch.isfinite(p.grad).all())
                                  for p in mmpareto.params)
                        if _ok:
                            optimizer.step()
                        scaler.update(scaler.get_scale() * (
                            1.0 if _ok else scaler.get_backoff_factor()))
                    else:
                        optimizer.step()
                    mmpareto.reset()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1
                if p46_teacher is not None:
                    # [P46-C2] optimizer step 직후에만 갱신 — micro-step마다 돌리면
                    # 같은 파라미터 상태를 여러 번 평균해 실효 EMA 감쇠가 왜곡된다.
                    p46_teacher.update(global_update)
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
            ctd_accum += float(ctd_ce)
            m2f_accum += float(m2f_loss)
            rce_accum += float(router_ce)
            vic_accum += float(vicreg)
            rca_accum += float(rca_ce)
            fcr_accum += float(fcr)   # [P41-F1]
            p43_accum += float(p43_mask)   # [P43-T1]
            mkl_accum += float(p44_mkl)     # [P44-B2]
            rc_accum += float(p44_rc)       # [P44-B2]
            hard_accum += float(p44_hard)   # [P44-M3]
            sty_accum += float(p45_sty)     # [P45-F1]
            mcc_accum += float(p46_cons)    # [P46-C2] pre-scaled
            proto_accum += float(p46_proto)  # [P46-C3] pre-scaled (주 view)
            xview_accum += float(p46_xv)    # [P46-C3] pre-scaled (2-view)
            uni_accum += float(p47_uni)     # [P47-2] pre-scaled (LAMBDA_U in model)
            if p47_2_on:
                for _mi, (_c, _a) in enumerate(zip(_core.p47_2.last_ce,
                                                   _core.p47_2.last_acc)):
                    if _c is None:
                        continue
                    uni_ce_sum[_mi] += _c; uni_acc_sum[_mi] += _a; uni_n[_mi] += 1
            if p46_class_ema is not None and (it % p46_ema_interval == 0):
                # [P46-C1] 런타임 per-class 난이도 = 내부신호. 샘플러가 다음
                # refresh 때 이 EMA를 읽어 고-loss 클래스를 추가 up-weight 한다.
                p46_class_ema.update_from_logits(logits, lbl,
                                                 ignore_label=trainset.ignore_label)
            _pm = getattr(_core, '_last_p42_mask', None)   # [P42-M1/D]
            if _pm is not None:
                p42_mask_sum += float(_pm.sum()); p42_tot += int(_pm.numel())
            _p44m = getattr(_core, '_last_p44_mask', None)   # [P44-B3] 실현 픽셀 마스킹률
            if _p44m is not None:
                p44_mask_sum += float(_p44m.sum()); p44_tot += int(_p44m.numel())
            elif getattr(_core, 'p44_local_mask', False):
                p44_tot += int(lbl.shape[0] * lbl.shape[-1] * lbl.shape[-2])
            if getattr(_core, '_rca_pick', None) is not None:
                rca_picked += int(_core._rca_pick.sum())
            rca_seen += lbl.shape[0]
            if _fusion is not None and _fusion._last_rel_auroc is not None:
                auroc_rows.append(_fusion._last_rel_auroc)
            if _fusion is not None and _fusion._last_gate_mean is not None:
                gate_rows.append(_fusion._last_gate_mean.cpu().tolist())
            if getattr(_fusion, '_last_router_mean', None) is not None:
                router_rows.append(_fusion._last_router_mean.tolist())
            _cefr = getattr(_fusion, 'cefr', None)          # [P37a]
            if _cefr is not None and _cefr._last_w_mean is not None:
                cefr_rows.append(_cefr._last_w_mean.tolist())
            if is_rank0:
                pbar.set_description(
                    f"Epoch [{epoch+1}/{epochs}] Loss {train_loss/(it+1):.4f} "
                    f"cal {cal_accum/(it+1):.4f} auxCE {aux_accum/(it+1):.4f}")
            if (p46_mem_log and is_rank0 and torch.cuda.is_available()
                    and it % p46_mem_log == 0):
                logger.info(
                    f"[P46-MEM] ep{epoch+1} it{it} "
                    f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                    f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                    f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB "
                    f"branch={int(bool(_do_mcc or _do_xv))}")
            # 🔴 [P46] 메모리: 루프 지역변수는 **다음 iteration이 재대입할 때까지**
            # 살아 있다 → 직전 스텝의 logits/aux/total이 다음 스텝의 forward가
            # peak를 찍는 내내 메모리를 붙들고 있었다((B,K,768,768) fp32 ≈ 56MiB,
            # aux dict는 backward가 이미 saved tensor를 푼 그래프 노드들).
            # 여기서 명시적으로 끊는다 — 아래 통계는 전부 위에서 float로 뽑아 뒀다.
            del logits, m_feat, aux, total, loss
            del p46_proto, p46_cons, p46_xv, p47_uni

        train_loss /= (it + 1)
        avg_lr = scheduler.get_lr()
        avg_lr = float(sum(avg_lr) / len(avg_lr))

        if is_rank0:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/cal_loss', cal_accum / (it + 1), epoch)
            writer.add_scalar('train/aux_ce', aux_accum / (it + 1), epoch)
            writer.add_scalar('train/lr', avg_lr, epoch)
            # 감사 2026-07-21: wandb 전용이던 항들을 tb에도 (오프라인 서버에서
            # gate 붕괴/router reg 궤적이 소실되던 문제)
            writer.add_scalar('train/gate_entropy', gate_ent_accum / (it + 1), epoch)
            writer.add_scalar('train/router_reg', router_accum / (it + 1), epoch)
            writer.add_scalar('train/cefr_reg', cefr_accum / (it + 1), epoch)
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
                alpha = float(_fusion.router_alpha.detach())
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
                sigma_a = float(getattr(_fusion.cefr, '_last_sigma_a', 0.0) or 0.0)
                for i, name in enumerate(modals[:len(cbar)]):
                    writer.add_scalar(f'p37/cefr_w_{name}', cbar[i], epoch)
                    log_extra[f'p37/cefr_w_{name}'] = float(cbar[i])
                writer.add_scalar('p37/cefr_sigma_a', sigma_a, epoch)
                log_extra['p37/cefr_sigma_a'] = sigma_a
                log_extra['train/cefr_reg'] = cefr_accum / (it + 1)
                logger.info(f"[P37] cefr w̄ " +
                            " ".join(f"{n}:{w:.3f}" for n, w in zip(modals, cbar)) +
                            f" sigma_a:{sigma_a:.4f}")
            if getattr(_core, 'classtoken', None) is not None:
                # [P37b] class-token residual scale beta (zero-init) + aux CE
                ctl_beta = float(_core.classtoken.beta.detach())
                writer.add_scalar('p37/ctl_beta', ctl_beta, epoch)
                writer.add_scalar('train/ctd_ce', ctd_accum / (it + 1), epoch)
                log_extra['p37/ctl_beta'] = ctl_beta
                log_extra['train/ctd_ce'] = ctd_accum / (it + 1)
                logger.info(f"[P37] ctl beta:{ctl_beta:.4f} "
                            f"ctd_ce:{ctd_accum / (it + 1):.4f}")
            if getattr(_core, 'm2f', None) is not None:
                # [P38] query-branch residual scale beta (zero-init) + mask-cls loss
                m2f_beta = float(_core.m2f.beta.detach())
                writer.add_scalar('p38/m2f_beta', m2f_beta, epoch)
                writer.add_scalar('train/m2f_loss', m2f_accum / (it + 1), epoch)
                log_extra['p38/m2f_beta'] = m2f_beta
                log_extra['train/m2f_loss'] = m2f_accum / (it + 1)
                logger.info(f"[P38] m2f beta:{m2f_beta:.4f} "
                            f"m2f_loss:{m2f_accum / (it + 1):.4f}")
            if getattr(_core, 'arb_lambda', None) is not None:
                # [P39] per-class arbitration Λ (softplus space) + router CE
                lam = torch.nn.functional.softplus(_core.arb_lambda.detach())
                writer.add_scalar('p39/arb_lambda_mean', float(lam.mean()), epoch)
                writer.add_scalar('p39/arb_lambda_max', float(lam.max()), epoch)
                writer.add_scalar('train/router_ce', rce_accum / (it + 1), epoch)
                log_extra['p39/arb_lambda_mean'] = float(lam.mean())
                log_extra['p39/arb_lambda_max'] = float(lam.max())
                log_extra['train/router_ce'] = rce_accum / (it + 1)
                logger.info(f"[P39] arb λ mean:{float(lam.mean()):.3f} "
                            f"max:{float(lam.max()):.3f} "
                            f"router_ce:{rce_accum / (it + 1):.4f}")
            if getattr(_core, 'p41_fcr', False):
                writer.add_scalar('train/fcr', fcr_accum / (it + 1), epoch)
                log_extra['train/fcr'] = fcr_accum / (it + 1)
            if getattr(_core, 'p43', None) is not None:
                # [P43-T1] mask-cls 주손실 + λ(t). λ가 목표치까지 올라갔는데
                # 손실이 안 내려가면 = 쿼리 분기가 안 배우는 것(ep30 게이트 ③).
                _p43_lam = _core._p43_lambda_now()
                writer.add_scalar('train/p43_mask_loss', p43_accum / (it + 1), epoch)
                writer.add_scalar('p43/lambda', _p43_lam, epoch)
                log_extra['train/p43_mask_loss'] = p43_accum / (it + 1)
                log_extra['p43/lambda'] = _p43_lam
                logger.info(f"[P43] mask_loss:{p43_accum / (it + 1):.4f} "
                            f"lambda:{_p43_lam:.3f}")
            if getattr(_core, 'p42_mask_img', False):   # [P42-M1/D] 실현 마스킹률 (k=0 무음 탐지)
                _mr = p42_mask_sum / max(p42_tot, 1)
                writer.add_scalar('train/p42_mask_rate', _mr, epoch)
                log_extra['train/p42_mask_rate'] = _mr
                print(f"[P42] ep{epoch} mask_rate={_mr:.3f} (target {getattr(_core,'p42_mask_frac',0):.2f})", flush=True)
            if getattr(_core, 'p44_local_mask', False):   # [P44-B3] 실현 마스킹률
                _mr44 = p44_mask_sum / max(p44_tot, 1)
                writer.add_scalar('train/p44_mask_rate', _mr44, epoch)
                log_extra['train/p44_mask_rate'] = _mr44
                logger.info(f"[P44] ep{epoch} mask_rate={_mr44:.4f} "
                            f"(mode={_core.p44_mask_mode}, "
                            f"target_frac {getattr(_core, 'p44_mask_frac', 0):.2f})")
            if getattr(_fusion, 'p44_mutual_kl', False) or \
                    getattr(_fusion, 'p44_rel_corr', False):
                writer.add_scalar('train/p44_mutual_kl', mkl_accum / (it + 1), epoch)
                writer.add_scalar('train/p44_rel_corr', rc_accum / (it + 1), epoch)
                log_extra['train/p44_mutual_kl'] = mkl_accum / (it + 1)
                log_extra['train/p44_rel_corr'] = rc_accum / (it + 1)
                logger.info(f"[P44-B2] mutual_kl:{mkl_accum / (it + 1):.4f} "
                            f"rel_corr:{rc_accum / (it + 1):.4f}")
            if getattr(_core, 'p44_hard_pixel_aux', False):
                writer.add_scalar('train/p44_hard_aux', hard_accum / (it + 1), epoch)
                log_extra['train/p44_hard_aux'] = hard_accum / (it + 1)
            if getattr(_core, 'p45_fogstyle', False):
                writer.add_scalar('train/p45_fogstyle', sty_accum / (it + 1), epoch)
                log_extra['train/p45_fogstyle'] = sty_accum / (it + 1)
            # ── [P46-CTR] 토글별 즉검 지표 (ep30 게이트에서 no-op 조기 검출) ──
            if p46_class_ema is not None:
                # 샘플러가 실제로 rare 클래스를 뽑았는지 = C-1이 무음 no-op인지 판정.
                _hist = rcs_sampler.last_class_hist.astype(np.float64)
                _hist = _hist / max(_hist.sum(), 1.0)
                _top = np.argsort(_hist)[::-1][:5]
                writer.add_scalar('p46/rcs_class_entropy',
                                  float(-(_hist[_hist > 0] * np.log(_hist[_hist > 0])).sum()), epoch)
                log_extra['p46/rcs_class_entropy'] = float(
                    -(_hist[_hist > 0] * np.log(_hist[_hist > 0])).sum())
                logger.info("[P46-C1] sampled-class top5: " + ", ".join(
                    f"{class_names[c]}:{_hist[c]:.3f}" for c in _top))
                logger.info("[P46-C1] class-loss EMA top5: " + ", ".join(
                    f"{class_names[c]}:{p46_class_ema.val[c]:.3f}"
                    for c in np.argsort(p46_class_ema.val)[::-1][:5]))
            if p46_mcc:
                _rate = mcc_rate_sum / max(mcc_rate_n, 1)
                writer.add_scalar('train/p46_mcc', mcc_accum / (it + 1), epoch)
                writer.add_scalar('p46/mcc_pseudo_rate', _rate, epoch)
                log_extra['train/p46_mcc'] = mcc_accum / (it + 1)
                log_extra['p46/mcc_pseudo_rate'] = _rate
                # pseudo_rate가 계속 0이면 teacher가 conf_thresh를 못 넘는 것 =
                # C-2가 무음 no-op. CONF_THRESH를 낮추거나 WARMUP_EP를 확인하라.
                logger.info(f"[P46-C2] mcc:{mcc_accum / (it + 1):.4f} "
                            f"pseudo_rate:{_rate:.3f}")
            if getattr(_core, 'p46_proto', None) is not None:
                writer.add_scalar('train/p46_proto', proto_accum / (it + 1), epoch)
                writer.add_scalar('p46/proto_coverage',
                                  float(_core.p46_proto._last_cov), epoch)
                log_extra['train/p46_proto'] = proto_accum / (it + 1)
                log_extra['p46/proto_coverage'] = float(_core.p46_proto._last_cov)
                if p46_xview:
                    writer.add_scalar('train/p46_proto_xview',
                                      xview_accum / (it + 1), epoch)
                    log_extra['train/p46_proto_xview'] = xview_accum / (it + 1)
                logger.info(f"[P46-C3] proto:{proto_accum / (it + 1):.4f} "
                            f"xview:{xview_accum / (it + 1):.4f} "
                            f"bank_cov:{float(_core.p46_proto._last_cov):.2f}")
            if p47_2_on:
                # [P47-2] 게이트 진단: per-modal acc가 **모달별로 갈라지는지**.
                # 전부 붙어 있으면 uni-modal 압력이 안 걸린 것(λ_u 상향 검토),
                # img만 홀로 치솟으면 여전히 RGB 지배(=modality laziness 잔존).
                _uni = uni_accum / (it + 1)
                writer.add_scalar('train/p47_2_uni', _uni, epoch)
                log_extra['train/p47_2_uni'] = _uni
                _n = np.maximum(uni_n, 1.0)
                _ce = uni_ce_sum / _n
                _acc = uni_acc_sum / _n
                for i, name in enumerate(modals):
                    if uni_n[i] == 0:
                        continue
                    writer.add_scalar(f'p47/uni_ce_{name}', _ce[i], epoch)
                    writer.add_scalar(f'p47/uni_acc_{name}', _acc[i], epoch)
                    log_extra[f'p47/uni_ce_{name}'] = float(_ce[i])
                    log_extra[f'p47/uni_acc_{name}'] = float(_acc[i])
                _seen = [i for i in range(len(modals)) if uni_n[i] > 0]
                logger.info(
                    f"[P47-2] uni_aux:{_uni:.4f} per-modal ce:" +
                    " ".join(f"{modals[i]}:{_ce[i]:.3f}" for i in _seen) +
                    " acc:" +
                    " ".join(f"{modals[i]}:{_acc[i]:.3f}" for i in _seen) +
                    (" | ogm k:" + " ".join(f"{modals[i]}:{ogm_k[i]:.3f}"
                                            for i in range(len(modals)))
                     if ogm_k is not None else ""))
                if ogm_k is not None:
                    for i, name in enumerate(modals):
                        writer.add_scalar(f'p47/ogm_k_{name}', ogm_k[i], epoch)
                        log_extra[f'p47/ogm_k_{name}'] = float(ogm_k[i])
            if pareto_stats:
                # [P44-B1] 게이트② 진단: modal-aux gradient와 주 gradient의 내적
                # 부호. lidar 그룹의 cos가 음수→양수로 전환하는지가 사전등록 지표.
                _keys = pareto_stats[0].keys()
                _agg = {k: float(np.mean([s[k] for s in pareto_stats])) for k in _keys}
                for k, v in _agg.items():
                    writer.add_scalar(f'p44/{k}', v, epoch)
                    log_extra[f'p44/{k}'] = v
                logger.info("[P44-B1] " + " ".join(
                    f"{k}:{v:+.3f}" for k, v in _agg.items() if k.startswith('cos_')))
            if getattr(_core, 'p391_vicreg', False):
                # [P39.1] VICReg loss + trunk gate γ (gated_mlp mode)
                writer.add_scalar('train/vicreg', vic_accum / (it + 1), epoch)
                log_extra['train/vicreg'] = vic_accum / (it + 1)
                msg = f"[P39.1] vicreg:{vic_accum / (it + 1):.4f}"
                if getattr(_core, 'trunk_gamma', None) is not None:
                    g = torch.tanh(_core.trunk_gamma.detach())
                    for i, name in enumerate(modals[:g.numel()]):
                        writer.add_scalar(f'p391/trunk_gamma_{name}', float(g[i]), epoch)
                        log_extra[f'p391/trunk_gamma_{name}'] = float(g[i])
                    msg += " γ " + " ".join(f"{n}:{float(v):.3f}"
                                            for n, v in zip(modals, g))
                logger.info(msg)
            if _p49_core is not None:
                # ── [P49-AIR] ep30 게이트① : γ 노름이 성장하는가 ──────────────
                # γ≈0 정체면 보조 정보가 모델에 **한 번도 들어오지 않은** 것이다
                # (키1 재발) → 즉시 중단 판정. 제안 §4 사전등록 지표.
                _g = _p49_core.gamma_log()
                for k, v in _g.items():
                    writer.add_scalar(k, v, epoch)
                    log_extra[k] = v
                if getattr(_p49_core, 'vicreg', False):
                    writer.add_scalar('train/vicreg', vic_accum / (it + 1), epoch)
                    log_extra['train/vicreg'] = vic_accum / (it + 1)
                if _g:
                    _names = getattr(_p49_core, 'aux_names', [])
                    logger.info(
                        "[P49] |γ| mean:{:.4f} | ".format(_g.get('p49/gamma_mean', 0.0))
                        + " ".join(
                            f"{n}:" + "/".join(
                                f"{_g.get(f'p49/gamma_b{b}_{n}', 0.0):.3f}"
                                for b in range(getattr(_p49_core, 'num_blocks', 0)))
                            + f"(pyr {_g.get(f'p49/gammapyr_{n}', 0.0):.3f})"
                            for n in _names)
                        + f" | vicreg:{vic_accum / (it + 1):.4f}")
            if getattr(_core, 'rca_enable', False):
                # [P40] RCA 감쇠 채택률 + readout CE
                rate = rca_picked / max(rca_seen, 1)
                writer.add_scalar('p40/rca_pick_rate', rate, epoch)
                writer.add_scalar('train/rca_readout', rca_accum / (it + 1), epoch)
                log_extra['p40/rca_pick_rate'] = rate
                log_extra['train/rca_readout'] = rca_accum / (it + 1)
                logger.info(f"[P40] rca pick_rate:{rate:.3f} "
                            f"readout_ce:{rca_accum / (it + 1):.4f}")
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
                if p46_class_ema is not None:
                    # [P46-C1] resume 시 난이도 EMA를 되살린다 (없으면 재개 직후
                    # 샘플러가 base 분포로 되돌아가 커리큘럼이 끊긴다).
                    d['p46_class_loss_ema'] = p46_class_ema.state_dict()
                if extra:
                    d.update(extra)
                return d
            _atomic_save(_ckpt(), save_dir / 'last_checkpoint.pth')

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
                # [P39.1] per-modality effective-rank monitor (RankMe) on the
                # analysis hook's last eval batch — ep30 게이트(lidar rank≥15)를
                # 학습 로그에서 바로 판정하기 위함. 비용: svdvals 1회/모달.
                _pm = getattr(_core, '_last_per_modal_feats', None)
                if _pm:
                    for _i, _f in enumerate(_pm):
                        z = _f.flatten(2).transpose(1, 2).reshape(-1, _f.shape[1]).float()
                        if z.shape[0] > 4096:
                            z = z[torch.randperm(z.shape[0], device=z.device)[:4096]]
                        try:
                            s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
                            p = (s / s.sum().clamp(min=1e-12)).clamp(min=1e-12)
                            erank = float(torch.exp(-(p * p.log()).sum()))
                            _mn = modals[_i] if _i < len(modals) else str(_i)
                            writer.add_scalar(f'p391/rank_{_mn}', erank, epoch)
                            logger.info(f"[P39.1] eff.rank {_mn}: {erank:.1f}")
                        except Exception:
                            pass
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
            # [P46] eval은 학습과 블록 크기 프로파일이 달라 캐싱 얼로케이터를
            # 파편화시킨 채 끝난다. 곧바로 돌아가는 학습 스텝이 P46 보조 branch
            # 때문에 훨씬 큰 연속 블록을 요구하므로, 여기서 한 번 비워 준다.
            # (프래그멘테이션 완화일 뿐 근본책은 아니다 — eval **전**에도 이미
            #  empty_cache가 있고, 이건 그 짝이다.)
            torch.cuda.empty_cache()
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

    # TRAIN.SEED — 진짜 시드 노브. 미지정이면 기존값 3407 그대로라 기존 런과 바이트 동일.
    # ⚠️ 이걸 넣기 전까지 'seed2/seed3' 런은 시드가 실제로 달라진 적이 없다(3407 고정).
    # MODEL.C3.SEED 는 C1 이 꺼져 있으면 inert 라서 시드 역할을 하지 못했다.
    # 그 런들의 편차는 시드 분산이 아니라 GPU 비결정성만 반영한다 = 참 시드 분산의 하한.
    _seed = cfg.get('TRAIN', {}).get('SEED', 3407)
    fix_seeds(_seed)
    print(f"[SEED] fix_seeds({_seed})")
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join(m[0] for m in cfg['DATASET']['MODALS'])
    exp_name = '_'.join([cfg['DATASET']['NAME'], cfg['MODEL']['BACKBONE'], modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)

    # FINETUNE_INIT 는 epoch 카운터를 0 으로 되돌리므로 AUTO_RESUME 과 같이 켜면
    # 크래시 때마다 처음부터 다시 시작하는 무한 루프가 된다. 기동 전에 막는다.
    if cfg['MODEL'].get('FINETUNE_INIT', False) and cfg['MODEL'].get('AUTO_RESUME', False):
        raise ValueError("MODEL.FINETUNE_INIT 과 MODEL.AUTO_RESUME 은 동시에 켤 수 없다 "
                         "(FINETUNE_INIT 이 epoch 을 0 으로 되돌려 무한 재시작).")

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
