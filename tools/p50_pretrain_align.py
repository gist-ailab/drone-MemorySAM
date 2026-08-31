#!/usr/bin/env python
"""[P50-MAP] 정렬 사전학습 (arm 1 = MultiMAE 식 cross-modal masked reconstruction).

정본 설계 = `.claude_logs/decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md`

무엇을 하는가
  frozen DINOv3-L + 모달별 LoRA + 융합/트렁크/FPN 을 **pseudo-모달 대량 데이터**
  (tools/p50_gen_pseudomodal.py 산출)로 정렬 사전학습한다. 모달 전체를 합친 토큰
  풀에서 가시 예산을 Dirichlet 로 나눠 임의 모달의 패치를 가리고, 남은 모달로부터
  가려진 모달을 복원시킨다 → 트렁크가 모달 간 정보를 실제로 실어 나르게 된다.

무엇을 하지 않는가 (🔴 단일-모델 원칙 / 추론 그래프 무변경)
  - 모델 클래스를 새로 만들지 않는다. `build_reliadino` 로 파인튠과 **같은 모델**을
    짓고 그 부품(encoder/fusion/trunk/fpn)을 그대로 쓴다.
  - recon 헤드는 사전학습 전용이며 산출 state_dict 에서 제외된다(폐기).
  - 산출 = LoRA + 융합 + 트렁크 + FPN state_dict. 파인튠은
    `MODEL.PRETRAINED_ADAPTERS: <path>` 한 줄로 읽는다.

실행:
    # 단일 GPU
    python tools/p50_pretrain_align.py \
        --cfg configs/deliver/deliver_rgbdel_P46_c3only_p50map.yaml \
        --data /path/to/ImageNeXt_p50 --out ckpts/p50map_probe.pth --epochs 30

    # DDP (4090 ×4)
    torchrun --standalone --nproc_per_node=4 tools/p50_pretrain_align.py \
        --cfg <cfg> --data <dir> --out <ckpt> --epochs 30 --bs 4

    # [P50-EXT] MCubeS 팔 — pseudo aolp/dolp/nir 코퍼스로 같은 팔을 돌린다.
    # cfg 의 DATASET.MODALS 는 모델 빌드에도 쓰이므로 --pretrain-modals 지정 시
    # 데이터·모델 양쪽이 이 순서/이름으로 갈아끼워진다('rgb' 는 모델명 'img' 의 별칭).
    python tools/p50_pretrain_align.py --cfg <mcubes_cfg.yaml> \
        --data /path/to/ImageNeXt_p50ext --out ckpts/p50map_mcubes.pth \
        --pretrain-modals rgb,aolp,dolp,nir
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                      # noqa: E402
from semseg.models.reliadino.p50 import (DEFAULT_ADAPTER_GROUPS,         # noqa: E402
                                         ReconHead, filter_adapter_state_dict,
                                         masked_recon_loss,
                                         sample_modal_token_masks,
                                         token_mask_to_pixel_mask)

SCRIPT_VERSION = '1.2.0'
MODAL_DIR = {'img': 'rgb', 'depth': 'depth', 'lidar': 'lidar', 'event': 'event',
             'nir': 'nir'}
# [P50-EXT] uint8 PNG 양자화 원자 파일 모달 — mcubes.py 관행대로 로드 시 3채널로
# stack 한다(aolp = [sin,cos,sin], dolp = 3ch 복제). 디스크는 round 양자화된
# uint8 PNG(해상도 1/255)라 역-affine(dequant_u8)로 float 복원한 뒤 [0,1]/[-1,1]
# 값역 그대로 비정규화 통과 — /255·z-score 정규화를 하지 **않는다**(mcubes 로더도
# aolp/dolp 를 비정규화 통과).
QUANT_MODALS = {'aolp': ('aolp_sin', 'aolp_cos'), 'dolp': ('dolp',)}
# --pretrain-modals 별칭: ReliaDINO 는 'img' 를 이름으로 특수취급(_img_idx)한다.
PRETRAIN_MODAL_ALIASES = {'rgb': 'img'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dequant_u8(u: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """uint8 PNG 픽셀 → float 역-affine: v = vmin + u/255·(vmax−vmin).

    생성기(tools/p50_gen_pseudomodal.py) 양자화 u8 = round((v−vmin)/(vmax−vmin)·255)
    의 역변환이다 — 라운드트립 오차 ≤ (vmax−vmin)/2/255."""
    return (vmin + u.astype(np.float32) / 255.0 * (vmax - vmin)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 데이터
# ═══════════════════════════════════════════════════════════════════════════
class PseudoModalDataset(Dataset):
    """p50_gen_pseudomodal.py 산출 디렉토리를 읽는다.

    정규화는 `semseg/augmentations_mm.py::Normalize` 와 **같은 규약**이다 —
    img 는 /255 후 ImageNet z-score, 비-RGB 는 /255 (DATASET.NORM_ALL_MODALS 가
    켜져 있으면 비-RGB 도 ImageNet z-score). 사전학습과 파인튠의 입력 분포가
    어긋나면 정렬이 전이되지 않는다.
    """

    def __init__(self, root: str, modals: Sequence[str], img_size: int,
                 norm_all_modals: bool = False, train: bool = True,
                 limit: int = 0):
        self.root = Path(root)
        self.modals = list(modals)
        self.img_size = int(img_size)
        self.norm_all = bool(norm_all_modals)
        self.train = train
        for m in self.modals:
            if m not in MODAL_DIR and m not in QUANT_MODALS:
                raise ValueError(f"P50 사전학습이 모르는 모달 '{m}' "
                                 f"(지원: {sorted(list(MODAL_DIR) + list(QUANT_MODALS))})")
            for d in self._modal_dirs(m):
                if not d.is_dir():
                    raise FileNotFoundError(f"모달 디렉토리 없음: {d}")
        idx = self.root / 'index.txt'
        if idx.is_file():
            stems = [s for s in idx.read_text().splitlines() if s.strip()]
        else:
            stems = sorted(p.stem for p in (self.root / 'rgb').glob('*.png'))
        # 선택 모달 전부(원자 파일 포함)가 있는 stem 만 (생성 중단 잔여물 방어)
        self.stems = [s for s in stems
                      if all(p.is_file() for m in self.modals
                             for p in self._modal_files(m, s))]
        if limit > 0:
            self.stems = self.stems[:limit]
        if not self.stems:
            raise RuntimeError(f"사용 가능한 샘플이 0개다: {root}")
        self.mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD)[:, None, None]

    def __len__(self):
        return len(self.stems)

    def _modal_dirs(self, m: str) -> List[Path]:
        if m in QUANT_MODALS:
            return [self.root / d for d in QUANT_MODALS[m]]
        return [self.root / MODAL_DIR[m]]

    def _modal_files(self, m: str, stem: str) -> List[Path]:
        if m in QUANT_MODALS:
            return [self.root / d / f"{stem}.png" for d in QUANT_MODALS[m]]
        return [self.root / MODAL_DIR[m] / f"{stem}.png"]

    def _read_gray(self, subdir: str, stem: str) -> np.ndarray:
        """uint8 단채널 PNG 원자 파일 → (H,W) uint8."""
        return np.asarray(Image.open(self.root / subdir / f"{stem}.png"))

    def _read(self, m: str, stem: str) -> torch.Tensor:
        if m == 'aolp':            # mcubes.py 관행: sin/cos 원자 2파일 → [sin,cos,sin]
            s = dequant_u8(self._read_gray('aolp_sin', stem), -1.0, 1.0)
            c = dequant_u8(self._read_gray('aolp_cos', stem), -1.0, 1.0)
            a = np.stack([s, c, s], axis=2)                  # H×W×3
            return torch.from_numpy(a).permute(2, 0, 1)      # (3,H,W) float32
        if m == 'dolp':            # 3채널 복제 (mcubes.py 관행)
            d = dequant_u8(self._read_gray('dolp', stem), 0.0, 1.0)
            a = np.stack([d, d, d], axis=2)
            return torch.from_numpy(a).permute(2, 0, 1)
        p = self.root / MODAL_DIR[m] / f"{stem}.png"
        # nir(단채널 저장)도 convert('RGB') 로 3채널 복제된다 — mcubes 관행과 동일
        a = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)   # copy(쓰기 가능)
        return torch.from_numpy(a).permute(2, 0, 1)          # (3,H,W) uint8

    def __getitem__(self, i: int) -> torch.Tensor:
        stem = self.stems[i]
        imgs = [self._read(m, stem) for m in self.modals]
        h, w = imgs[0].shape[-2:]
        s = self.img_size
        if h < s or w < s:
            ups = []
            for t in imgs:
                u = F.interpolate(t[None].float(), size=(max(h, s), max(w, s)),
                                  mode='nearest')[0]
                # uint8 입력만 uint8 로 되돌린다 — aolp/dolp(역양자화 float)는 값역 보존
                ups.append(u.to(torch.uint8) if t.dtype == torch.uint8 else u)
            imgs = ups
            h, w = imgs[0].shape[-2:]
        if self.train:
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
        else:
            top, left = (h - s) // 2, (w - s) // 2
        imgs = [t[:, top:top + s, left:left + s] for t in imgs]
        if self.train and random.random() < 0.5:
            imgs = [torch.flip(t, dims=[-1]) for t in imgs]
            # event 프록시는 부호 있는 x-그래디언트에서 만들어졌다. 좌우 반전은
            # d/dx 의 부호를 뒤집으므로 ± 극성 채널도 함께 swap 해야 물리적으로
            # 일관된다 (안 하면 모델이 반전 여부를 극성으로 알아채는 지름길이 생긴다).
            # aolp/dolp/nir 는 공간 반전만 한다 — mcubes.py 로더의 flip 관행과 동일.
            if 'event' in self.modals:
                j = self.modals.index('event')
                imgs[j] = imgs[j][[1, 0, 2]]
        out = []
        for m, t in zip(self.modals, imgs):
            if m in QUANT_MODALS:
                out.append(t.float())          # 역양자화 float([0,1]/[-1,1]) — 비정규화 통과
                continue
            x = t.float() / 255.0
            if m == 'img' or self.norm_all:
                x = (x - self.mean) / self.std
            out.append(x)
        return torch.stack(out, 0)                            # (M,3,S,S)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 모델 래퍼
# ═══════════════════════════════════════════════════════════════════════════
class P50AlignNet(nn.Module):
    """파인튠 모델(build_reliadino)의 encoder/fusion/trunk/fpn 을 그대로 쓰고,
    stride-4 피라미드 위에 모달별 경량 recon 헤드만 얹는다."""

    def __init__(self, base: nn.Module, fpn_dim: int, mask_ratio: float,
                 dirichlet_alpha: float, recon_hidden: Optional[int] = None):
        super().__init__()
        self.base = base
        self.modals = list(base.modalities)
        self.mask_ratio = float(mask_ratio)
        self.alpha = float(dirichlet_alpha)
        self.recon = nn.ModuleList([
            ReconHead(fpn_dim, out_ch=3, upscale=4, hidden=recon_hidden)
            for _ in self.modals])

    def forward(self, x: torch.Tensor):
        """x: (B, M, 3, H, W) — 정규화된 입력. returns (loss, per_modal_losses, stats)"""
        b, m, _, h, w = x.shape
        patch = self.base.encoder.patch
        assert h % patch == 0 and w % patch == 0, \
            f"IMAGE_SIZE {h}x{w} 가 patch {patch} 의 배수가 아니다"
        gh, gw = h // patch, w // patch

        visible = sample_modal_token_masks(b, m, gh * gw, self.mask_ratio,
                                           self.alpha, device=x.device)
        pmask = token_mask_to_pixel_mask(visible, gh, gw, (h, w))   # (M,B,1,H,W)

        inputs = [x[:, i] for i in range(m)]
        masked = [inputs[i] * pmask[i] for i in range(m)]

        feats = [self.base.encoder(masked[i], i) for i in range(m)]
        fused, _aux = self.base.fusion(feats, None, img_mask=None,
                                       img_idx=self.base._img_idx,
                                       presence=None, epoch=0)
        fused = self.base._apply_trunk_exp(fused, feats)
        pyramid = self.base.fpn(fused)

        per_modal, total = [], 0.0
        for i in range(m):
            pred = self.recon[i](pyramid[0])
            li = masked_recon_loss(pred, inputs[i], pmask[i])
            per_modal.append(li)
            total = total + li
        loss = total / float(m)
        stats = {'visible_frac': float(visible.float().mean()),
                 'per_modal': [float(v.detach()) for v in per_modal]}
        return loss, per_modal, stats


def freeze_outside_groups(base: nn.Module, groups: Sequence[str]) -> dict:
    """사전학습 대상(=산출 대상) 밖의 파라미터는 requires_grad=False.

    head/M2F/P43/classtoken 등 과제별 헤드는 pseudo-라벨이 없어 학습할 수 없고,
    학습해 봐야 저장되지도 않는다. 켜 둔 채로 두면 optimizer state 와 DDP
    unused-parameter 비용만 낸다."""
    from semseg.models.reliadino.p50 import ADAPTER_GROUPS
    preds = [ADAPTER_GROUPS[g] for g in groups]
    kept, frozen = 0, 0
    for n, p in base.named_parameters():
        if any(pr(n) for pr in preds):
            if p.requires_grad:
                kept += p.numel()
            else:                       # frozen 백본과 겹치는 키는 없어야 정상
                frozen += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return {'trainable_in_groups': kept, 'frozen': frozen}


# ═══════════════════════════════════════════════════════════════════════════
# 3. 학습 루프
# ═══════════════════════════════════════════════════════════════════════════
def lr_at(step: int, total: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def save_adapters(base: nn.Module, out: str, groups: Sequence[str], meta: dict):
    sd = filter_adapter_state_dict(base.state_dict(), groups)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'p50_map': True, 'model_state_dict': sd,
                'groups': list(groups), 'p50_meta': meta}, out)
    return len(sd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='[P50-MAP] 정렬 사전학습')
    ap.add_argument('--cfg', required=True, help='파인튠과 동일한 MODEL 절을 가진 yaml')
    ap.add_argument('--data', required=True, help='p50_gen_pseudomodal.py 출력 루트')
    ap.add_argument('--out', required=True, help='산출 어댑터 ckpt 경로(.pth)')
    ap.add_argument('--pretrain-modals', type=str, default='',
                    help="모달 강제 지정(쉼표 목록). 기본 '' = cfg DATASET.MODALS 그대로"
                         "(종전 동작). [P50-EXT] MCubeS 팔: 'rgb,aolp,dolp,nir' — rgb 는 "
                         "모델명 'img' 의 별칭, aolp/dolp 는 uint8 PNG 원자 파일(역양자화 "
                         "float 복원, mcubes 관행 stack), nir 는 PNG 3채널 복제. 지정 시 "
                         "모델 빌드의 DATASET.MODALS 도 같은 순서·이름으로 갈아끼운다")
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=4, help='rank 당 배치')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup-steps', type=int, default=500)
    ap.add_argument('--min-lr-ratio', type=float, default=0.01)
    ap.add_argument('--mask-ratio', type=float, default=0.75)
    ap.add_argument('--dirichlet-alpha', type=float, default=1.0)
    ap.add_argument('--img-size', type=int, default=0, help='0 = cfg TRAIN.IMAGE_SIZE')
    ap.add_argument('--num-classes', type=int, default=0,
                    help='0 = cfg DATASET.NUM_CLASSES 또는 25(DELIVER). '
                         '🔴 파인튠 데이터셋의 클래스 수와 같아야 fusion 키 shape 이 맞는다')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=0, help='>0 이면 그 스텝에서 종료(스모크)')
    ap.add_argument('--limit', type=int, default=0, help='>0 이면 데이터셋 앞 N개만')
    ap.add_argument('--recon-hidden', type=int, default=0)
    ap.add_argument('--groups', type=str, default=','.join(DEFAULT_ADAPTER_GROUPS))
    ap.add_argument('--amp', type=str, default='bfloat16',
                    choices=['off', 'bfloat16', 'float16'])
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--log-interval', type=int, default=20)
    ap.add_argument('--save-interval', type=int, default=1, help='에폭 단위')
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args(argv)

    # ── DDP ────────────────────────────────────────────────────────────────
    ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        # gloo 폴백은 **CPU 스모크 전용**이다 (본 학습은 항상 nccl).
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl' if cuda_ok else 'gloo', init_method='env://')
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}') if cuda_ok else torch.device('cpu')
    else:
        rank, world = 0, 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    ds_cfg = cfg['DATASET']
    modals = list(ds_cfg['MODALS'])
    if args.pretrain_modals:
        # [P50-EXT] 모달 강제 지정 — 데이터 로드와 모델 빌드(MODALS) 양쪽에 반영.
        # 순서가 어긋나면 모달 i 번 인코더가 다른 모달을 보게 된다.
        given = [t.strip() for t in args.pretrain_modals.split(',') if t.strip()]
        if not given:
            raise ValueError(f"--pretrain-modals 파싱 결과가 비었다: '{args.pretrain_modals}'")
        modals = [PRETRAIN_MODAL_ALIASES.get(t, t) for t in given]
        ds_cfg = dict(ds_cfg, MODALS=modals)
    img_size = args.img_size or int(cfg['TRAIN']['IMAGE_SIZE'][0])
    num_classes = args.num_classes or int(ds_cfg.get('NUM_CLASSES', 25))
    groups = [g.strip() for g in args.groups.split(',') if g.strip()]

    # ── 데이터 ─────────────────────────────────────────────────────────────
    dset = PseudoModalDataset(args.data, modals, img_size,
                              norm_all_modals=bool(ds_cfg.get('NORM_ALL_MODALS', False)),
                              train=True, limit=args.limit)
    sampler = DistributedSampler(dset, shuffle=True) if ddp else None
    loader = DataLoader(dset, batch_size=args.bs, shuffle=(sampler is None),
                        sampler=sampler, num_workers=args.workers,
                        pin_memory=(device.type == 'cuda'), drop_last=True,
                        persistent_workers=(args.workers > 0))

    # ── 모델 ───────────────────────────────────────────────────────────────
    cfg_for_build = dict(cfg)
    cfg_for_build['TRAIN'] = dict(cfg['TRAIN'], IMAGE_SIZE=[img_size, img_size])
    cfg_for_build['DATASET'] = ds_cfg        # --pretrain-modals 반영(기본값은 무변경)
    base = build_reliadino(cfg_for_build, num_classes)
    fpn_dim = int(cfg['MODEL'].get('FPN_DIM', 256))
    fz = freeze_outside_groups(base, groups)
    model = P50AlignNet(base, fpn_dim, args.mask_ratio, args.dirichlet_alpha,
                        recon_hidden=(args.recon_hidden or None)).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_recon = sum(p.numel() for p in model.recon.parameters())
    if is_main:
        print(f"[P50-MAP] cfg={args.cfg} modals={modals} img_size={img_size} "
              f"num_classes={num_classes} groups={groups}")
        print(f"[P50-MAP] backbone={base.encoder.backbone_name} "
              f"patch={base.encoder.patch} dim={base.encoder.embed_dim}")
        print(f"[P50-MAP] trainable={n_train/1e6:.2f}M "
              f"(recon head {n_recon/1e6:.2f}M — 사전학습 후 폐기) "
              f"frozen={fz['frozen']/1e6:.1f}M | samples={len(dset)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,
                              betas=(0.9, 0.95))
    amp_dtype = {'off': None, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}[args.amp]
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    start_epoch, gstep = 0, 0
    resume_path = args.resume or (args.out + '.resume.pth')
    if args.resume or os.path.isfile(resume_path):
        if os.path.isfile(resume_path):
            ck = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(ck['full_state_dict'], strict=False)
            optim.load_state_dict(ck['optimizer_state_dict'])
            start_epoch, gstep = ck['epoch'] + 1, ck['gstep']
            if is_main:
                print(f"[P50-MAP] resumed from {resume_path} (epoch {ck['epoch']})")
        elif args.resume:
            raise FileNotFoundError(f"--resume 파일 없음: {args.resume}")

    if ddp:
        # router 등 이 손실에 관여하지 않는 가지가 있어 unused param 이 생긴다.
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=([device.index] if device.type == 'cuda' else None),
            find_unused_parameters=True)
    core = model.module if ddp else model

    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * args.epochs
    meta_base = {
        'script': 'tools/p50_pretrain_align.py', 'version': SCRIPT_VERSION,
        'arm': 'multimae_cross_modal_masked_reconstruction',
        'cfg': args.cfg, 'data': args.data, 'modals': modals,
        'pretrain_modals_arg': args.pretrain_modals,
        'img_size': img_size, 'num_classes': num_classes, 'groups': groups,
        'mask_ratio': args.mask_ratio, 'dirichlet_alpha': args.dirichlet_alpha,
        'lr': args.lr, 'bs_per_rank': args.bs, 'world_size': world,
        'epochs': args.epochs, 'samples': len(dset),
        'backbone': base.encoder.backbone_name,
    }
    data_meta = Path(args.data) / 'meta.json'
    if data_meta.is_file():
        try:
            dm = json.loads(data_meta.read_text())
            meta_base['data_depth_backend'] = dm.get('depth_backend')
            meta_base['data_num_complete'] = dm.get('num_complete')
        except Exception:
            pass

    # ── 루프 ───────────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        run_loss, seen = 0.0, 0
        for it, batch in enumerate(loader):
            lr_now = lr_at(gstep, total_steps, args.lr, args.warmup_steps,
                           args.min_lr_ratio)
            for g in optim.param_groups:
                g['lr'] = lr_now
            batch = batch.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=(amp_dtype is not None and device.type == 'cuda')):
                loss, per_modal, stats = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"[P50-MAP] loss 가 유한하지 않다 (step {gstep}) — "
                                   f"조용히 넘기지 않는다. per_modal={stats['per_modal']}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            run_loss += float(loss.detach())
            seen += 1
            gstep += 1
            if is_main and (gstep % max(args.log_interval, 1) == 0 or gstep == 1):
                pm = ' '.join(f"{m}={v:.4f}" for m, v in zip(modals, stats['per_modal']))
                print(f"[P50-MAP] ep{epoch} it{it+1}/{steps_per_epoch} "
                      f"step{gstep}/{total_steps} loss={run_loss/max(seen,1):.4f} "
                      f"({pm}) vis={stats['visible_frac']:.3f} lr={lr_now:.2e} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
            if args.max_steps and gstep >= args.max_steps:
                stop = True
                break

        if is_main and ((epoch + 1) % max(args.save_interval, 1) == 0 or stop
                        or epoch == args.epochs - 1):
            meta = dict(meta_base, epoch=epoch, gstep=gstep,
                        train_loss=run_loss / max(seen, 1),
                        saved=time.strftime('%Y-%m-%dT%H:%M:%S'))
            n = save_adapters(core.base, args.out, groups, meta)
            torch.save({'full_state_dict': core.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch, 'gstep': gstep},
                       args.out + '.resume.pth')
            print(f"[P50-MAP] saved {n} adapter tensors -> {args.out} "
                  f"(loss={run_loss/max(seen,1):.4f})")
        if stop:
            break

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(f"[P50-MAP] done in {(time.time()-t0)/60:.1f}min — "
              f"파인튠에서 MODEL.PRETRAINED_ADAPTERS: {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
