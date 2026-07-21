"""
Object Detection Training Script for MemorySAM.

MemorySAM backbone (P9/P22) + FCOS detection head 학습.

Usage:
    python train_det.py --cfg configs/det/det_P9_base.yaml

Config 구조:
    DATASET:
        ANNOTATION_TRAIN: /path/to/train.json
        ANNOTATION_VAL: /path/to/val.json
        MODALITIES:
            img: { ROOT: /path/to/rgb }
            lidar: { ROOT: /path/to/lidar }
            thermal: { ROOT: /path/to/thermal }
        MODALS: ['img', 'lidar', 'thermal']
        IMG_SIZE: [1024, 1024]
    MODEL:
        SEG_MODEL: LoRA_Sam_P9
        SEG_CHECKPOINT: /path/to/seg_checkpoint.pth
        N_CLASSES: 2
        FREEZE_BACKBONE: true
        HIDDEN_DIM: 256
        N_CONVS: 4
    TRAIN:
        EPOCHS: 50
        BATCH_SIZE: 4
        LR: 0.001
        WEIGHT_DECAY: 0.0001
        LR_SCHEDULER: cosine
        WARMUP_EPOCHS: 5
        SAVE_INTERVAL: 5
    EVAL:
        NMS_THRESH: 0.5
        SCORE_THRESH: 0.05
"""

import os
import torch
import argparse
import yaml
import time
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def setup_ddp():
    """Initialise torch.distributed from torchrun env vars.

    Returns (ddp_enabled, rank, world_size, local_rank). Falls back to single-GPU
    when not launched under torchrun.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Long timeout: eval is now sharded across ranks (see gather_predictions),
        # but rank0 still runs COCOeval + checkpoint save alone while the others
        # wait at dist.barrier(). The default 10-min NCCL watchdog could abort them
        # there (SIGABRT); 2h is a generous upper bound on that tail.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def unwrap(model):
    """Return the underlying module whether or not it is DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model


try:
    import wandb
except ImportError:
    wandb = None


_VIZ_COLORS = [
    (220, 20, 60), (0, 128, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128),
    (0, 200, 200), (255, 0, 255), (128, 128, 0), (0, 128, 128), (255, 99, 71),
    (30, 144, 255), (50, 205, 50),
]


def draw_boxes_pil(img_np, boxes, scores, cls_ids, class_names=None,
                   gt_boxes=None, score_thresh=0.3):
    """Render predicted boxes (colored, with score) + GT boxes (white) on an image.

    img_np: (H,W,3) uint8. boxes/gt_boxes in the same pixel coords as img_np.
    Returns a PIL.Image for wandb.Image / tensorboard logging.
    """
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    if gt_boxes is not None:
        for b in gt_boxes.tolist():
            draw.rectangle(b, outline=(255, 255, 255), width=1)
    for j in range(boxes.shape[0]):
        s = float(scores[j])
        if s < score_thresh:
            continue
        x1, y1, x2, y2 = boxes[j].tolist()
        c = int(cls_ids[j])
        color = _VIZ_COLORS[c % len(_VIZ_COLORS)]
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 1, y1 + 1), f"{name}:{s:.2f}", fill=color)
    return pil

# SAM2 imports are LAZY (moved into build_seg_model's non-ReliaDINO branch) so that
# P34-Det (ReliaDINO/DINOv3) can run in envs lacking SAM2's dep tree (hydra/iopath/
# icecream/...). P29/P30 paths import them on demand.
from objdet.datasets.multimodal_det import MultiModalDetDataset, rescale_boxes_to_orig
from objdet.models.det_model import MemorySAMDetector, MemorySAMDetectorP30, ReliaDINODetector
from objdet.metrics import evaluate_coco, format_predictions_coco


def parse_args():
    parser = argparse.ArgumentParser(description='MemorySAM Detection Training')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
    return parser.parse_args()


def build_seg_model(cfg: dict, device: torch.device, n_classes: int = 10) -> torch.nn.Module:
    """Build and load pretrained segmentation model."""
    model_name = cfg['MODEL']['SEG_MODEL']
    checkpoint_path = cfg['MODEL'].get('SEG_CHECKPOINT', None)
    modals = cfg['DATASET']['MODALS']

    # ── P34: ReliaDINO (DINOv3, no SAM2) — separate build path ────────────
    if model_name == 'ReliaDINO':
        from semseg.models.reliadino import build_reliadino
        seg_model = build_reliadino(cfg, n_classes).to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = seg_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded ReliaDINO seg checkpoint: {checkpoint_path} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return seg_model

    # Lazy SAM2 imports (only reached for P29/P30 SAM2 backbones).
    from semseg.models.sam2.sam2.build_sam import build_sam2
    import semseg.models.sam2.sam2.sam_lora_image_encoder_seg as _segmod

    sam2_checkpoint = cfg['MODEL'].get('SAM2_CHECKPOINT',
        'semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt')
    model_cfg_path = cfg['MODEL'].get('SAM2_CONFIG', 'sam2_hiera_b+.yaml')

    sam2_model = build_sam2(model_cfg_path, sam2_checkpoint, device=device)

    # Dynamically get model class from the SAM2 seg module.
    model_cls = getattr(_segmod, model_name, None)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_name}")

    # NOTE: 2nd positional arg of the LoRA_Sam constructors is the LoRA rank `r`,
    # NOT the modality count. Pass rank/modalities explicitly.
    seg_model = model_cls(
        sam2_model,
        cfg['MODEL'].get('LORA_R', 4),
        num_modalities=len(modals),
        amf_mode=cfg['MODEL'].get('AMF_MODE', 'uniform'),
        **cfg['MODEL'].get('BACKBONE_KWARGS', {}),   # P31: rbma_calibrate, unfreeze_last_n_blocks, ...
    )

    # Optional warm-start from a segmentation checkpoint (LoRA/decoder weights).
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = seg_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded seg checkpoint: {checkpoint_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif checkpoint_path:
        print(f"[warn] seg checkpoint not found: {checkpoint_path} — using SAM2 pretrained init.")

    return seg_model


def build_dataset(cfg: dict, split: str = 'train'):
    """Build detection dataset."""
    ds_cfg = cfg['DATASET']
    ann_key = f'ANNOTATION_{split.upper()}'
    annotation_path = ds_cfg[ann_key]
    img_size = tuple(ds_cfg.get('IMG_SIZE', [1024, 1024]))

    # letterbox/stretch + drop-empty + augmentation (train only)
    resize_mode = ds_cfg.get('RESIZE_MODE', 'stretch')
    drop_empty = bool(ds_cfg.get('DROP_EMPTY', False)) and split == 'train'
    transform = None
    aug_cfg = ds_cfg.get('AUG', {})
    if split == 'train' and aug_cfg.get('ENABLE', False):
        from objdet.augmentations_det import DetAugmentation
        transform = DetAugmentation(
            hflip_prob=aug_cfg.get('HFLIP_PROB', 0.5),
            brightness_range=tuple(aug_cfg.get('BRIGHTNESS_RANGE', [0.6, 1.4])),
            contrast_range=tuple(aug_cfg.get('CONTRAST_RANGE', [0.7, 1.3])),
            crop_prob=aug_cfg.get('CROP_PROB', 0.5),
            crop_scale=tuple(aug_cfg.get('CROP_SCALE', [0.7, 1.0])),
        )
    common = dict(img_size=img_size, min_area=ds_cfg.get('MIN_AREA', 0.0),
                  resize_mode=resize_mode, drop_empty=drop_empty, transform=transform)

    # Mode A: per-image modalities map under a single DATASET ROOT (preferred).
    if 'MODALITY_KEYS' in ds_cfg:
        modality_keys = dict(ds_cfg['MODALITY_KEYS'])
        modals = ds_cfg.get('MODALS', list(modality_keys.keys()))
        return MultiModalDetDataset(
            annotation_path=annotation_path,
            root=ds_cfg['ROOT'],
            modality_keys=modality_keys,
            modals=modals,
            require_all_modalities=ds_cfg.get('REQUIRE_ALL_MODALITIES', True),
            **common,
        )

    # Mode B: legacy parallel per-modality ROOT dirs sharing file_name.
    modality_roots = {name: mc['ROOT'] for name, mc in ds_cfg['MODALITIES'].items()}
    modals = ds_cfg.get('MODALS', list(modality_roots.keys()))
    return MultiModalDetDataset(
        annotation_path=annotation_path,
        modality_roots=modality_roots,
        modals=modals,
        require_all_modalities=ds_cfg.get('REQUIRE_ALL_MODALITIES', False),
        **common,
    )


def train_one_epoch(
    model: MemorySAMDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    use_amp: bool = True,
    accum_steps: int = 1,
    is_main: bool = True,
    ddp: bool = False,
):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_ctr = 0.0
    n_iters = 0
    nan_skips = 0
    grad_clip = float(os.environ.get('DET_GRAD_CLIP', '10.0'))  # env-overridable; DETR heads want ~0.1 (P37a sets it)

    optimizer.zero_grad()
    n_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f'Epoch [{epoch}]') if is_main else dataloader
    for batch_idx, batch in enumerate(pbar):
        # Prepare sample
        modals = [k for k in batch.keys()
                  if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device) for m in modals}
        gt_bboxes = [b.to(device) for b in batch['bboxes']]
        gt_labels = [l.to(device) for l in batch['labels']]

        is_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == n_batches)

        with autocast(enabled=use_amp):
            losses = model(sample, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

        loss = losses['loss_total']
        # NaN/Inf guard: skip a bad batch so one spike can't corrupt weights permanently.
        #
        # 🔴 The skip decision MUST be collective. If only the affected rank takes
        # `continue`, it never runs backward, so its DDP reducer never posts the
        # gradient all-reduce for this step — every other rank blocks in backward
        # until the NCCL watchdog fires (2h here, see setup_ddp), then the job dies.
        # all_reduce(MAX) makes every rank see the same verdict and skip together.
        bad = (~torch.isfinite(loss)).to(dtype=torch.float32)
        if ddp:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bool(bad.item()):
            # Drop whatever this accumulation window had collected: it may already
            # be polluted, and every rank drops the same thing.
            optimizer.zero_grad(set_to_none=True)
            nan_skips += 1
            if is_main and nan_skips <= 20:
                print(f'[nan-guard] non-finite loss at epoch {epoch} batch {batch_idx}; skipped (total {nan_skips})')
            continue
        # Gradient accumulation (effective batch = BATCH_SIZE * world * accum_steps).
        scaled = loss / accum_steps
        # Under DDP, skip the gradient all-reduce until the accumulation boundary.
        sync_ctx = model.no_sync() if (ddp and not is_step) else nullcontext()
        with sync_ctx:
            if use_amp:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        if is_step:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # No GradScaler here, so nothing else screens the gradients.
                # clip_grad_norm_ defaults to error_if_nonfinite=False, i.e. it
                # happily passes NaN/Inf straight through to optimizer.step(),
                # which poisons the weights permanently — every later batch then
                # produces a non-finite loss and the run locks into 100% skip
                # (see det_P30_v2.yaml "ep20 100% skip"). The AMP branch is
                # covered by scaler.unscale_; this branch was not.
                # Grads are already all-reduced by DDP at this point, so the
                # norm — and therefore this verdict — is identical on every rank.
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=grad_clip)
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    nan_skips += 1
                    if is_main and nan_skips <= 20:
                        print(f'[nan-guard] non-finite grad norm at epoch {epoch} '
                              f'batch {batch_idx}; step skipped (total {nan_skips})')
            optimizer.zero_grad()

        # FCOSLoss may return plain floats (0.0) for reg/ctr when a batch has no
        # positive samples — handle both tensor and float when logging.
        def _f(x):
            return x.item() if torch.is_tensor(x) else float(x)

        total_loss += _f(losses['loss_total'])
        total_cls += _f(losses['loss_cls'])
        total_reg += _f(losses['loss_reg'])
        total_ctr += _f(losses['loss_ctr'])
        n_iters += 1

        if is_main:
            pbar.set_postfix({
                'loss': f'{_f(losses["loss_total"]):.4f}',
                'cls': f'{_f(losses["loss_cls"]):.4f}',
                'reg': f'{_f(losses["loss_reg"]):.4f}',
                'n_pos': losses['n_pos'],
            })

    # Skip accounting. Without this a run that NaNs on every batch reports
    # avg_loss = 0/max(0,1) = 0.0000 and looks perfectly healthy while burning
    # GPU on nothing — the deadlock the collective guard removed used to at
    # least kill the job loudly. Surface the rate, and refuse to continue when
    # an entire epoch produced no update.
    skip_rate = nan_skips / max(n_batches, 1)
    if is_main and nan_skips:
        print(f'[nan-guard] epoch {epoch}: skipped {nan_skips}/{n_batches} '
              f'batches ({skip_rate:.1%})')
    if n_iters == 0:
        raise RuntimeError(
            f'[nan-guard] epoch {epoch}: every batch was skipped as non-finite '
            f'({nan_skips}/{n_batches}). Training cannot progress — aborting '
            f'instead of silently reporting loss=0. Check LR / grad clip '
            f'(DET_GRAD_CLIP={grad_clip}) / AMP settings.')

    avg_loss = total_loss / max(n_iters, 1)
    if writer is not None:
        global_step = epoch * len(dataloader)
        writer.add_scalar('train/loss_total', avg_loss, global_step)
        writer.add_scalar('train/nan_skip_rate', skip_rate, global_step)
        writer.add_scalar('train/loss_cls', total_cls / max(n_iters, 1), global_step)
        writer.add_scalar('train/loss_reg', total_reg / max(n_iters, 1), global_step)
        writer.add_scalar('train/loss_ctr', total_ctr / max(n_iters, 1), global_step)

    return avg_loss


@torch.no_grad()
def evaluate(
    model: MemorySAMDetector,
    dataloader: DataLoader,
    device: torch.device,
    annotation_path: str,
    idx_to_cat_id: dict,
    viz_count: int = 0,
    class_names=None,
    viz_score_thresh: float = 0.3,
    resize_mode: str = 'stretch',
    return_raw: bool = False,
    progress: bool = True,
):
    """Run evaluation; return (COCO metrics, list[(PIL.Image, caption)]).

    When viz_count>0, the first viz_count images get a pred(+GT)-box overlay
    (in the 1024 model-input space) for wandb/tensorboard logging.

    With return_raw=True the raw COCO-format prediction list is returned instead
    of the metrics dict, so a caller can shard the val set across DDP ranks and
    run COCOeval **once** on the gathered predictions (distributed eval).
    """
    model.eval()
    all_predictions = []
    viz_images = []

    for batch in tqdm(dataloader, desc='Evaluating', disable=not progress):
        modals = [k for k in batch.keys()
                  if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
        sample = {m: batch[m].to(device) for m in modals}

        results = model(sample)

        orig_sizes = batch['orig_size']  # (B, 2)
        img_size = sample[modals[0]].shape[-2:]  # (H, W)
        rgb_key = 'img' if 'img' in sample else modals[0]

        for i, det in enumerate(results['detections']):
            # Validation visualization (model-input 1024 space, before COCO rescale).
            # Wrapped so a rendering hiccup can never crash a long training run.
            if len(viz_images) < viz_count:
                try:
                    img_np = (sample[rgb_key][i].detach().cpu().clamp(0, 1)
                              .permute(1, 2, 0).numpy() * 255).astype('uint8')
                    gt = batch['bboxes'][i] if 'bboxes' in batch else None
                    pil = draw_boxes_pil(
                        img_np, det['boxes'].cpu(), det['scores'].cpu(),
                        det['class_ids'].cpu(), class_names=class_names,
                        gt_boxes=gt, score_thresh=viz_score_thresh,
                    )
                    n_kept = int((det['scores'] >= viz_score_thresh).sum())
                    viz_images.append((pil, f"{batch['file_name'][i]} | pred={n_kept}"))
                except Exception as e:
                    print(f"[viz][warn] skipped a visualization: {e}")

            if det['boxes'].shape[0] == 0:
                continue

            # Scale boxes back to original image size (matches dataset resize_mode)
            orig_h, orig_w = orig_sizes[i].tolist()
            boxes = rescale_boxes_to_orig(
                det['boxes'].cpu(), orig_h, orig_w, img_size[0], img_size[1], resize_mode)

            preds = format_predictions_coco(
                boxes.cpu(), det['scores'].cpu(), det['class_ids'].cpu(),
                batch['image_id'][i], idx_to_cat_id,
            )
            all_predictions.extend(preds)

    if return_raw:
        # Caller gathers shards across ranks and runs COCOeval once.
        return all_predictions, viz_images

    if not all_predictions:
        print("No predictions — returning zero AP.")
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}, viz_images

    metrics = evaluate_coco(annotation_path, all_predictions)
    return metrics, viz_images


def cap_preds_per_img_cat(preds: list, max_det: int = 100) -> list:
    """Keep only the top-`max_det` detections per (image_id, category_id).

    AP-neutral by construction: COCOeval sorts detections by score and slices
    `dt[0:maxDet]` per (image, category) with maxDet=100, so anything past the
    100th is never scored. Dropping it here bounds the DDP gather payload,
    which is otherwise unbounded — the FCOS head keeps every box above
    score_thresh=0.05 with no post-NMS cap, so early epochs (diffuse logits)
    emit ~1000 boxes/image and the pickled payload reaches 100 MB+ per rank.
    """
    buckets = defaultdict(list)
    for p in preds:
        buckets[(p['image_id'], p['category_id'])].append(p)
    out = []
    for v in buckets.values():
        if len(v) > max_det:
            v.sort(key=lambda d: -d['score'])
            v = v[:max_det]
        out.extend(v)
    return out


def gather_predictions(local_preds: list, ddp: bool, world_size: int,
                       max_det: int = 100) -> list:
    """All-gather per-rank COCO prediction lists into one flat list.

    Shards are built with a strided, non-padded index split (see val_loader
    below), so every image is owned by exactly one rank — no duplicate
    image_id can reach COCOeval.

    The payload is capped (see cap_preds_per_img_cat) *before* the collective:
    NCCL's all_gather_object stages the pickled bytes in a
    `world_size × max_size` GPU buffer, so an uncapped list can add ~1 GB VRAM
    per rank on top of the training allocation.
    """
    local_preds = cap_preds_per_img_cat(local_preds, max_det)
    if not ddp or world_size == 1:
        return local_preds
    bucket = [None] * world_size
    dist.all_gather_object(bucket, local_preds)
    return [p for shard in bucket if shard for p in shard]


def main():
    args = parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # ── DDP setup (torchrun) ──
    ddp, rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    if is_main:
        print(f"[ddp] enabled={ddp} world_size={world_size} device={device}")

    # Output directory + writer (rank 0 only)
    save_dir = Path(cfg.get('OUTPUT_DIR', 'outputs/det')) / Path(args.cfg).stem
    writer = None
    wandb_run = None
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(save_dir / 'tb_logs'))
        with open(save_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)
        if cfg.get('WANDB', {}).get('ENABLE', False) and wandb is not None:
            try:
                wandb_run = wandb.init(
                    project=cfg['WANDB'].get('PROJECT', 'p29-det'),
                    entity=cfg['WANDB'].get('ENTITY', None),
                    name=save_dir.name,
                    config=cfg,
                    dir=str(save_dir),
                )
                print(f"[wandb] logging to project '{wandb_run.project}' "
                      f"run '{wandb_run.name}' (mode={wandb_run.settings.mode})")
            except Exception as e:
                # No login / offline failure → keep training, fall back to tensorboard.
                wandb_run = None
                print(f"[wandb][warn] init failed ({e}); continuing without wandb. "
                      f"Run `wandb login` (or set WANDB_API_KEY / WANDB_MODE=offline).")

    # Datasets (build first so n_classes can be derived from the COCO categories)
    train_dataset = build_dataset(cfg, 'train')
    # [dist-eval] every rank builds the val set; each evaluates a disjoint shard.
    val_dataset = build_dataset(cfg, 'val')

    n_classes = cfg['MODEL'].get('N_CLASSES', train_dataset.n_classes)
    if n_classes != train_dataset.n_classes:
        print(f"[warn] cfg N_CLASSES={n_classes} != dataset n_classes="
              f"{train_dataset.n_classes}; using dataset value.")
        n_classes = train_dataset.n_classes

    # Build model
    print("Building segmentation backbone...")
    seg_model = build_seg_model(cfg, device, n_classes)

    det_name = cfg['MODEL'].get('DET_MODEL', 'MemorySAMDetector')
    common = dict(
        seg_model=seg_model,
        modals=cfg['DATASET']['MODALS'],
        n_classes=n_classes,
        fpn_in_channels=cfg['MODEL'].get('FPN_CHANNELS', [32, 64, 256]),
        fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16]),
        freeze_backbone=cfg['MODEL'].get('FREEZE_BACKBONE', False),
        train_memory=cfg['MODEL'].get('TRAIN_MEMORY', True),
        n_convs=cfg['MODEL'].get('N_CONVS', 4),
        hidden_dim=cfg['MODEL'].get('HIDDEN_DIM', 256),
        assigner=cfg['MODEL'].get('ASSIGNER', 'fcos'),
        atss_topk=cfg['MODEL'].get('ATSS_TOPK', 9),
        atss_scale=cfg['MODEL'].get('ATSS_SCALE', 8.0),
    )
    if det_name == 'ReliaDINOM2FDetector':
        # P38: M2F query head IS the detector (no RF-DETR / extract_det_pyramid).
        from objdet.models.det_model import ReliaDINOM2FDetector
        model = ReliaDINOM2FDetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            num_select=cfg['MODEL'].get('NUM_SELECT', 100),
            freeze_backbone=cfg['MODEL'].get('FREEZE_BACKBONE', False),
        ).to(device)
    elif det_name == 'ReliaDINORFDETRDetector':
        # P37: same ReliaDINO backbone, RF-DETR NMS-free head (COCO-initialised).
        from objdet.models.det_model import ReliaDINORFDETRDetector
        model = ReliaDINORFDETRDetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            fpn_dim=cfg['MODEL'].get('FPN_DIM', 256),
            fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16, 32]),
            det_levels=cfg['MODEL'].get('DET_LEVELS', [2]),
            freeze_backbone=cfg['MODEL'].get('FREEZE_BACKBONE', False),
            num_queries=cfg['MODEL'].get('NUM_QUERIES', 300),
            group_detr=cfg['MODEL'].get('GROUP_DETR', 13),
            dec_layers=cfg['MODEL'].get('DEC_LAYERS', 4),
            dec_n_points=cfg['MODEL'].get('DEC_N_POINTS', 2),
            coco_ckpt=cfg['MODEL'].get('COCO_CKPT', None),
            num_select=cfg['MODEL'].get('NUM_SELECT', 300),
        ).to(device)
    elif det_name == 'ReliaDINODetector':
        # P34: ReliaDINO fuses internally + exposes a 256-ch 4-level pyramid;
        # no per-modality fusion / neck / memory_attention.
        model = ReliaDINODetector(
            seg_model=seg_model,
            modals=cfg['DATASET']['MODALS'],
            n_classes=n_classes,
            fpn_dim=cfg['MODEL'].get('FPN_DIM', 256),
            fpn_strides=cfg['MODEL'].get('FPN_STRIDES', [4, 8, 16, 32]),
            freeze_backbone=cfg['MODEL'].get('FREEZE_BACKBONE', False),
            n_convs=cfg['MODEL'].get('N_CONVS', 4),
            hidden_dim=cfg['MODEL'].get('HIDDEN_DIM', 256),
            assigner=cfg['MODEL'].get('ASSIGNER', 'fcos'),
            atss_topk=cfg['MODEL'].get('ATSS_TOPK', 9),
            atss_scale=cfg['MODEL'].get('ATSS_SCALE', 8.0),
        ).to(device)
    elif det_name == 'MemorySAMDetectorP30':
        model = MemorySAMDetectorP30(
            **common,
            img_size=tuple(cfg['DATASET'].get('IMG_SIZE', [1024, 1024]))[0],
            router_anchor_lambda=cfg['MODEL'].get('ROUTER_ANCHOR_LAMBDA', 1.0),
            router_reg_lambda=cfg['MODEL'].get('ROUTER_REG_LAMBDA', 0.0),
            num_queries=cfg['MODEL'].get('NUM_QUERIES', 100),
            query_dim=cfg['MODEL'].get('QUERY_DIM', 256),
            query_layers=cfg['MODEL'].get('QUERY_LAYERS', 4),
            query_heads=cfg['MODEL'].get('QUERY_HEADS', 8),
            use_fcos_aux=cfg['MODEL'].get('USE_FCOS_AUX', True),
            w_query=cfg['MODEL'].get('W_QUERY', 1.0),
            w_fcos=cfg['MODEL'].get('W_FCOS', 1.0),
            primary_head=cfg['MODEL'].get('PRIMARY_HEAD', 'query'),
            use_calibrated_reliability=cfg['MODEL'].get('USE_CALIBRATED_RELIABILITY', False),
            router_reg_mode=cfg['MODEL'].get('ROUTER_REG_MODE', 'diversity'),
            use_query_aux=cfg['MODEL'].get('USE_QUERY_AUX', False),
        ).to(device)
    else:
        model = MemorySAMDetector(
            **common,
            modality_fuse=cfg['MODEL'].get('MODALITY_FUSE', 'mean'),
        ).to(device)

    # VRAM workaround for 24GB GPUs (e.g. jarvis RTX 4090): gradient-checkpoint the
    # SAM2 Hiera trunk so per-modality encoder activations are recomputed in backward
    # instead of stored. The P27 forward already routes through torch.utils.checkpoint
    # when trunk.gradient_checkpointing is True and the model is training. Keeps the
    # Hiera-B+ backbone (no downsizing) at ~30% extra compute. Use BATCH_SIZE=1 +
    # GRAD_ACCUM_STEPS for a larger effective batch.
    if cfg['MODEL'].get('GRAD_CHECKPOINT', False) and not cfg['MODEL'].get('FREEZE_BACKBONE', False):
        try:
            if hasattr(model.seg_model, 'set_grad_checkpointing'):   # P34 ReliaDINO (DINOv3 ViT)
                model.seg_model.set_grad_checkpointing(True)
                if is_main:
                    print("[mem] ReliaDINO ViT gradient checkpointing: ON")
            else:
                model.seg_model.sam.image_encoder.trunk.gradient_checkpointing = True
                if is_main:
                    print("[mem] SAM2 trunk gradient checkpointing: ON")
        except AttributeError as e:
            if is_main:
                print(f"[mem][warn] could not enable trunk gradient checkpointing: {e}")

    # Count parameters
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Optimizer over the raw model's trainable params (build before DDP wrap).
    base_lr = cfg['TRAIN']['LR']
    pre_mult = cfg['TRAIN'].get('PRETRAINED_HEAD_LR_MULT', None)
    if pre_mult is not None and hasattr(model, 'pretrained_head_params'):
        # A COCO-initialised head carries a prior worth keeping: run it at a fraction
        # of the base LR so the from-scratch parts can move without washing it out.
        pre_ids = {id(p) for p in model.pretrained_head_params()}
        pre = [p for p in model.get_trainable_params() if id(p) in pre_ids]
        rest = [p for p in model.get_trainable_params() if id(p) not in pre_ids]
        param_groups = [{'params': rest, 'lr': base_lr},
                        {'params': pre, 'lr': base_lr * pre_mult}]
        if is_main:
            print(f"  LR groups: {len(rest)} tensors @ {base_lr:g} | "
                  f"{len(pre)} pretrained-head tensors @ {base_lr * pre_mult:g}")
    else:
        param_groups = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=cfg['TRAIN'].get('WEIGHT_DECAY', 1e-4),
    )

    # Resume (load into the raw model before DDP wrapping)
    start_epoch = 0
    best_ap = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_detector_state_dict(ckpt['detector_state_dict'])
        # config가 의도한 LR을 resume이 덮어쓰기 전에 보관
        _want_lrs = [g['lr'] for g in optimizer.param_groups]
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # ── resume LR override (2026-07-16) ────────────────────────────────
        # load_state_dict는 구 런의 param_groups를 복원하고, 거기엔 이전
        # LambdaLR이 심은 'initial_lr'(구 LR)이 남아 있다. 새 LambdaLR은
        # group.setdefault('initial_lr', group['lr']) 이라 **기존 키를 덮어쓰지
        # 않으므로** scheduler.base_lrs가 구 LR로 고정되고 TRAIN.LR이 조용히
        # 무시된다. 에러가 안 나서 로그로도 안 보인다. -> config LR을 명시 재적용.
        for _g, _lr in zip(optimizer.param_groups, _want_lrs):
            _g['lr'] = _lr
            _g.pop('initial_lr', None)   # 새 scheduler가 config LR로 재시드하게
        start_epoch = ckpt['epoch'] + 1
        best_ap = ckpt.get('best_ap', 0.0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best AP: {best_ap:.4f}")

    # ── DDP wrap ──
    # find_unused_parameters: per_modal_decoders/SQG are trainable but receive no
    # gradient from the detection loss (RBMA bias is computed under no_grad and the
    # seg-fusion path is discarded), so DDP must tolerate unused params.
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['TRAIN'].get('NUM_WORKERS', 4),
        collate_fn=MultiModalDetDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    # [dist-eval] Strided, NON-padded shard: rank r owns indices r, r+W, r+2W…
    # DistributedSampler is deliberately NOT used — it pads the tail by repeating
    # samples so every rank gets an equal count, which would feed duplicate
    # image_ids into COCOeval and silently skew AP. A strided split is exactly
    # disjoint; uneven shard sizes are fine because all_gather_object handles
    # variable-length lists.
    val_indices = list(range(len(val_dataset)))[rank::world_size] if ddp \
        else list(range(len(val_dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg['TRAIN'].get('VAL_BATCH_SIZE', cfg['TRAIN']['BATCH_SIZE']),
        shuffle=False,
        num_workers=cfg['TRAIN'].get('NUM_WORKERS', 4),
        collate_fn=MultiModalDetDataset.collate_fn,
        pin_memory=True,
    )
    if is_main:
        print(f"[dist-eval] val {len(val_dataset)} imgs split across {world_size} "
              f"rank(s); rank0 shard = {len(val_indices)}")

    # Scheduler
    epochs = cfg['TRAIN']['EPOCHS']
    warmup_epochs = cfg['TRAIN'].get('WARMUP_EPOCHS', 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # idx_to_cat_id for COCO eval
    idx_to_cat_id = {v: k for k, v in train_dataset.cat_id_to_idx.items()}

    if args.eval_only:
        # [dist-eval] every rank evaluates its shard; rank0 scores the union.
        local_preds, _ = evaluate(
            unwrap(model), val_loader, device,
            cfg['DATASET']['ANNOTATION_VAL'], idx_to_cat_id,
            resize_mode=cfg['DATASET'].get('RESIZE_MODE', 'stretch'),
            return_raw=True, progress=is_main,
        )
        preds = gather_predictions(local_preds, ddp, world_size)
        if is_main:
            metrics = (evaluate_coco(cfg['DATASET']['ANNOTATION_VAL'], preds)
                       if preds else {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0})
            print(f"Eval results: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}")
        if ddp:
            dist.destroy_process_group()
        return

    # Training loop
    for epoch in range(start_epoch, epochs):
        # [P37a] CEFR's two-pass blend reads seg_model._current_epoch
        # (lambda2 warmup). No-op for non-CEFR models.
        _sm = getattr(unwrap(model), 'seg_model', None)
        if _sm is not None and hasattr(_sm, '_current_epoch'):
            _sm._current_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, writer, use_amp=cfg['TRAIN'].get('AMP', True),
            accum_steps=cfg['TRAIN'].get('GRAD_ACCUM_STEPS', 1),
            is_main=is_main, ddp=ddp,
        )
        scheduler.step()

        if is_main:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            if wandb_run is not None:
                wandb_run.log({'train/loss': avg_loss,
                               'lr': scheduler.get_last_lr()[0], 'epoch': epoch})

        # [dist-eval] all ranks evaluate their shard; rank0 scores + checkpoints.
        save_interval = cfg['TRAIN'].get('SAVE_INTERVAL', 5)
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            viz_count = (cfg.get('WANDB', {}).get('VIZ_COUNT', 8)
                         if (is_main and wandb_run is not None) else 0)
            local_preds, viz = evaluate(
                unwrap(model), val_loader, device,
                cfg['DATASET']['ANNOTATION_VAL'], idx_to_cat_id,
                viz_count=viz_count, class_names=train_dataset.class_names,
                viz_score_thresh=cfg.get('WANDB', {}).get('VIZ_SCORE_THRESH', 0.3),
                resize_mode=cfg['DATASET'].get('RESIZE_MODE', 'stretch'),
                return_raw=True, progress=is_main,
            )
            preds = gather_predictions(local_preds, ddp, world_size)
            if is_main:
                metrics = (evaluate_coco(cfg['DATASET']['ANNOTATION_VAL'], preds)
                           if preds else {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0})
                print(f"  Val AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, AP75={metrics['AP75']:.4f}")
                writer.add_scalar('val/AP', metrics['AP'], epoch)
                writer.add_scalar('val/AP50', metrics['AP50'], epoch)
                # wandb: metrics + a fixed val-image sample each epoch. NOTE: with
                # sharded eval rank0 owns indices 0, W, 2W…, so the panel shows
                # those (not 0..N-1). Stable across epochs, but changes if
                # world_size changes — don't compare panels across run widths.
                if wandb_run is not None:
                    log = {'val/AP': metrics['AP'], 'val/AP50': metrics['AP50'],
                           'val/AP75': metrics['AP75'], 'epoch': epoch}
                    if viz:
                        log['val/examples'] = [wandb.Image(im, caption=c) for im, c in viz]
                    wandb_run.log(log)

                # Full model state preserves fine-tuned backbone (LoRA/memory/RBMA);
                # detector_state_dict kept for lightweight head-only reuse.
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': unwrap(model).state_dict(),
                    'detector_state_dict': unwrap(model).detector_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_ap': best_ap,
                    'metrics': metrics,
                    'config': cfg,
                }
                if metrics['AP'] > best_ap:
                    best_ap = metrics['AP']
                    ckpt['best_ap'] = best_ap
                    torch.save(ckpt, save_dir / 'best_checkpoint.pth')
                    print(f"  New best AP: {best_ap:.4f}")
                torch.save(ckpt, save_dir / f'epoch{epoch}_checkpoint.pth')
            # every rank ran evaluate() (which calls model.eval()) — restore train mode
            model.train()
            if ddp:
                dist.barrier()

    if is_main:
        if writer is not None:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Training complete. Best AP: {best_ap:.4f}")
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
