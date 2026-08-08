#!/usr/bin/env python3
"""
tools/probe_backbone_scaling.py — ProbeA2: DINOv3 backbone-scaling probe on MUSES (RGB only).

의뢰서: .claude_logs/decisions/2026-08-08-probea2-backbone-scaling-request.md

Question (의뢰서 ①): frozen DINOv3 representation quality as a function of backbone
capacity, measured under ONE fixed light seg-head protocol.
  - 상한: mIoU(H+) - mIoU(L)   (게이트 G-A2-상한: >= +1.5 / < +0.5 / 중간대역)
  - 하한: mIoU(L)  - mIoU(S+)  (게이트 G-A2-하한: <= 3.0)

Protocol — 2 stages (의뢰서 ② "2단 분리"):
  1. cache : forward every selected MUSES RGB image once through the FROZEN backbone
             (bf16 autocast, no_grad, no activations kept) and write the stride-16
             patch-token grid to a local-SSD memmap. Labels are cached ONCE and are
             SHARED by all backbones (identical image subset -> identical supervision).
  2. probe : train a light seg head on the cached features. The head spec is
             IDENTICAL for all backbones (only the input channel count differs,
             which is forced by the backbone). Identical budget: same epochs, same
             LR set, same schedule, same batch, same seed, same eval.

🔴 비교 유효성 (의뢰서 ⑥):
  - 백본 간 Δ만 유효하다. 절대치를 본 모델 수치·리더보드와 비교하지 말 것 (head가 다름).
  - 이 프로브의 L 수치는 ProbeA1(DELIVER, 25-class, +11.6)과 **직접 비교 불가**다:
    ProbeA1의 산출물 디렉토리(NAS analysis_logs/ProbeA1_dinov3_20260712/)는 비어 있고,
    프로토콜 코드(tools/probe_frozen_backbone.py)는 DELIVER/25-class 전용이다. 여기서는
    의뢰서 ⑤ 폴백대로 head 사양을 새로 고정하되 4종에 완전히 동일하게 적용한다.
    (참고: --feat-norm none --head linear 이면 ProbeA1의 head와 같은 conv1x1 이다.)
  - mIoU는 MUSES letterbox 정사각 프레임 위 EVAL_RES 해상도에서 계산한다 —
    MUSES 공식 벤치마크 프로토콜이 아니다 (tools/eval_muses_official.py 와 다른 수치).

🔴 RANDOM INIT 금지 (의뢰서 ③): 백본 pretrained 로드가 실패하면 즉시 예외로 죽는다.
   semseg/models/reliadino/encoder.py 의 "warn -> RANDOM INIT 후 진행" 폴백은 여기서
   의도적으로 재사용하지 않는다 (조용한 random init = 프로브 전체 무효).
   HF_HUB_OFFLINE=1 이면 시작 전에 거부한다.

⚠️ timm: DINOv3 계열은 timm >= 1.0.x 에서만 인식된다 (시스템 기본 timm 0.4.12 는 0개).
   jarvis 에서는 `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:.` 를 반드시 앞에 붙일 것.

Usage (jarvis, GPU1):
  export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:.
  CACHE=/SSDb/jemo_maeng/cache/probea2 ; OUT=/SSDb/jemo_maeng/probea2_out
  # 0) 검수 게이트 (의뢰서 ③) — 백본 4종 로드 + RANDOM INIT 0건 + feature shape + 손실 하강
  python tools/probe_backbone_scaling.py --stage smoke --gpu 1 --cache-dir $CACHE.smoke --out-dir $OUT
  # 1) feature 추출 (백본별)
  python tools/probe_backbone_scaling.py --stage cache --backbone l  --gpu 1 --cache-dir $CACHE
  # 2) head 학습 + 측정 (백본별)
  python tools/probe_backbone_scaling.py --stage probe --backbone l  --gpu 1 --cache-dir $CACHE --out-dir $OUT
  # 3) 4종 집계 + 게이트 적용
  python tools/probe_backbone_scaling.py --stage report --out-dir $OUT

Local plumbing self-test (no GPU / no timm / no dataset needed):
  python tools/probe_backbone_scaling.py --stage selftest --cache-dir /tmp/probea2_selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ------------------------------------------------------------------ constants
NUM_CLASSES = 19                 # MUSES = Cityscapes 19 trainIds
IGNORE_LABEL = 255
DEFAULT_IMG_SIZE = 1024          # patch16 -> 64x64 token grid (all 4 backbones)
DEFAULT_EVAL_RES = 512           # common eval resolution for ALL backbones
WEATHER = ('clear', 'fog', 'rain', 'snow')
TIME_OF_DAY = ('day', 'night')

# 실측 (2026-08-09, timm 1.0.24): 이름/파라미터/embed_dim 은 모두 확인된 값.
# cache_batch = 24GB(4090) 기준 추론 배치 기본값 (필요시 --batch 로 override).
BACKBONES = {
    'sp': dict(timm='vit_small_plus_patch16_dinov3', dim=384,  params_m=28.7,  cache_batch=8,
               label='DINOv3-S+'),
    'b':  dict(timm='vit_base_patch16_dinov3',       dim=768,  params_m=85.6,  cache_batch=4,
               label='DINOv3-B'),
    'l':  dict(timm='vit_large_patch16_dinov3',      dim=1024, params_m=303.1, cache_batch=4,
               label='DINOv3-L'),
    'hp': dict(timm='vit_huge_plus_patch16_dinov3',  dim=1280, params_m=840.5, cache_batch=2,
               label='DINOv3-H+'),
    # 의뢰서 ⑥ 중간대역(+0.5~1.5)일 때만 추가 측정. 24GB OOM 시 폴백 = H+까지로 종료.
    '7b': dict(timm='vit_7b_patch16_dinov3',         dim=4096, params_m=6716.0, cache_batch=1,
               label='DINOv3-7B'),
}
DEFAULT_SET = ['sp', 'b', 'l', 'hp']

# timm 이 pretrained 로드에 실패하고도 조용히 진행할 때 남기는 문자열들.
_RANDOM_INIT_PATTERNS = (
    'no pretrained weights exist',
    'random initialization',
    'randomly initialized',
    'unable to load pretrained',
    'pretrained weights not found',
    'failed to load pretrained',
)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ fs guards
def mount_fstype(path: Path) -> str:
    """Longest-prefix mountpoint fstype for `path` ('' if undeterminable)."""
    try:
        mounts = []
        for line in Path('/proc/mounts').read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mounts.append((parts[1], parts[2]))
    except OSError:
        return ''
    p = str(path.resolve())
    best = ('', '')
    for mp, fs in mounts:
        if (p == mp or p.startswith(mp.rstrip('/') + '/')) and len(mp) > len(best[0]):
            best = (mp, fs)
    return best[1]


def assert_local_cache_dir(cache: Path):
    """의뢰서 ②: feature 캐시는 로컬 SSD. sshfs/NAS 에 쓰지 말 것."""
    cache.mkdir(parents=True, exist_ok=True)
    fs = mount_fstype(cache)
    bad_fs = fs.startswith('fuse') or fs in ('nfs', 'nfs4', 'cifs', 'smb3', 'sshfs')
    bad_path = any(s in str(cache.resolve()) for s in ('/drone_nas', '/nas_jm', '/NHNHOME'))
    if bad_fs or bad_path:
        raise RuntimeError(
            f"cache-dir '{cache}' looks remote (fstype='{fs}') — 의뢰서 ② 는 로컬 SSD 를 요구한다 "
            f"(sshfs/NAS 에 feature 캐시 금지). 로컬 SSD 경로를 --cache-dir 로 지정하라.")
    log(f"[fs] cache-dir {cache} (fstype='{fs or 'unknown'}') ok")


def require_free_space(path: Path, need_bytes: int, what: str):
    free = shutil.disk_usage(path).free
    if free < need_bytes * 1.10:
        raise RuntimeError(
            f"not enough space for {what}: need ~{need_bytes/2**30:.1f} GiB (+10% margin), "
            f"free {free/2**30:.1f} GiB at {path}")
    log(f"[fs] {what}: need ~{need_bytes/2**30:.1f} GiB, free {free/2**30:.1f} GiB — ok")


def available_ram_bytes() -> int:
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError, AttributeError):
        return 0


# ------------------------------------------------------------------ backbone
def build_backbone(key: str, img_size: int, device, batch_hint: int = None):
    """Frozen DINOv3 encoder. Returns (encode_fn, info dict).

    Raises on ANY pretrained-load problem — no random-init fallback (의뢰서 ③).
    """
    import torch
    import torch.nn.functional as F
    import timm
    from timm.models import get_pretrained_cfg

    if os.environ.get('HF_HUB_OFFLINE', '') in ('1', 'true', 'True', 'TRUE', 'yes', 'on'):
        raise RuntimeError(
            "HF_HUB_OFFLINE is set — fresh DINOv3 weights cannot be fetched and timm would "
            "silently RANDOM-INIT. Unset it (의뢰서 ③ / hpca100 사고).")

    spec = BACKBONES[key]
    name = spec['timm']

    # (1) this timm must know the model AND have weights registered for it.
    if not hasattr(timm, '__version__') or int(timm.__version__.split('.')[0]) < 1:
        raise RuntimeError(
            f"timm {getattr(timm, '__version__', '?')} does not know DINOv3 — need timm>=1.0 "
            f"(jarvis: PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:.)")
    cfg = get_pretrained_cfg(name, allow_unregistered=True)
    if cfg is None:
        raise RuntimeError(f"timm {timm.__version__} has no pretrained cfg for '{name}'")
    if not (cfg.hf_hub_id or cfg.url or cfg.file):
        raise RuntimeError(f"'{name}' has NO registered pretrained weights -> would RANDOM INIT")

    # (2) create with pretrained=True while watching for silent random-init messages.
    records: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Catch(level=logging.DEBUG)
    tlog = logging.getLogger('timm')
    tlog.addHandler(handler)
    prev_level = tlog.level
    tlog.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0,
                                          dynamic_img_size=True, img_size=img_size)
                created_with = 'dynamic_img_size=True, img_size=%d' % img_size
            except TypeError:
                model = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
                created_with = 'img_size=%d' % img_size
            records.extend(str(w.message) for w in wlist)
    finally:
        tlog.removeHandler(handler)
        tlog.setLevel(prev_level)

    hits = [r for r in records if any(p in r.lower() for p in _RANDOM_INIT_PATTERNS)]
    if hits:
        raise RuntimeError(f"'{name}': timm reported a RANDOM-INIT / pretrained-load problem: {hits}")

    # (3) positive evidence: the HF weight file must be present in the local cache.
    hf_file = None
    if cfg.hf_hub_id:
        try:
            from huggingface_hub import try_to_load_from_cache
            repo = cfg.hf_hub_id.split('@')[0]
            for fn in (cfg.hf_hub_filename, 'model.safetensors', 'pytorch_model.bin'):
                if not fn:
                    continue
                got = try_to_load_from_cache(repo, fn)
                if isinstance(got, str):
                    hf_file = got
                    break
        except Exception as e:                                   # evidence only, not the gate
            log(f"[backbone/{key}] (hf cache probe failed: {type(e).__name__}: {e})")
    if cfg.hf_hub_id and hf_file is None:
        raise RuntimeError(
            f"'{name}': no HF weight file found in the local cache after pretrained=True "
            f"(repo {cfg.hf_hub_id}) — refusing to continue (would be RANDOM INIT).")

    n_params = sum(p.numel() for p in model.parameters())
    pe = model.patch_embed.proj.weight.detach()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    patch = model.patch_embed.patch_size[0]
    npre = int(getattr(model, 'num_prefix_tokens', 0))
    grid = img_size // patch

    def encode(x):
        """x: (B,3,img_size,img_size) ImageNet-normalized -> (B,C,grid,grid) float16."""
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=(device.type == 'cuda')):
            f = model.forward_features(x)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        if f.dim() == 4:                       # NHWC map (dynamic_img_size on some models)
            f = f.permute(0, 3, 1, 2).contiguous().float()
        else:
            f = f[:, npre:, :]
            n = f.shape[1]
            assert n == grid * grid, f"{name}: token count {n} != grid {grid}x{grid}"
            f = f.transpose(1, 2).reshape(f.shape[0], -1, grid, grid).float()
        return f

    info = dict(
        key=key, label=spec['label'], timm_name=name, timm_version=timm.__version__,
        pretrained_tag=cfg.tag, hf_hub_id=cfg.hf_hub_id, hf_cache_file=hf_file,
        created_with=created_with, params=n_params, params_m=round(n_params / 1e6, 1),
        embed_dim=int(model.embed_dim), patch=int(patch), num_prefix_tokens=npre,
        grid=int(grid), img_size=int(img_size),
        patch_embed_w_std=round(float(pe.std()), 6),
        patch_embed_w_absmean=round(float(pe.abs().mean()), 6),
        batch=int(batch_hint or spec['cache_batch']),
    )
    if abs(info['params_m'] - spec['params_m']) > max(1.0, 0.02 * spec['params_m']):
        log(f"[backbone/{key}] ⚠️ params {info['params_m']}M != expected {spec['params_m']}M")
    assert info['embed_dim'] == spec['dim'], \
        f"{name}: embed_dim {info['embed_dim']} != expected {spec['dim']}"
    log(f"[backbone/{key}] {spec['label']} '{name}' tag={cfg.tag} params={info['params_m']}M "
        f"dim={info['embed_dim']} patch={patch} prefix={npre} grid={grid}x{grid} "
        f"({created_with})")
    log(f"[backbone/{key}] pretrained OK — hf_file={hf_file} "
        f"patch_embed.w std={info['patch_embed_w_std']} absmean={info['patch_embed_w_absmean']} "
        f"| RANDOM-INIT messages: 0")
    return encode, info


# ------------------------------------------------------------------ index
def cond_of(rgb_path: str) -> str:
    """.../frame_camera/{split}/{weather}/{tod}/{stem}.png -> '{weather}_{tod}'."""
    parts = Path(rgb_path).parts
    if len(parts) >= 3 and parts[-3] in WEATHER and parts[-2] in TIME_OF_DAY:
        return f"{parts[-3]}_{parts[-2]}"
    return 'unknown'


def build_or_load_index(cache: Path, root: str, n_train: int, seed: int,
                        img_size: int, eval_res: int) -> dict:
    """Subset definition SHARED by all backbones. Written once, then only verified.

    Anything that would change what the head sees (image subset, order, resolution)
    lives here, so the 4 backbones are guaranteed to be compared on identical data.
    """
    from semseg.datasets.muses import MUSES

    idx_path = cache / 'index.json'
    index = json.loads(idx_path.read_text()) if idx_path.exists() else None
    # protocol mismatch is checked BEFORE touching the dataset: a cache built under a
    # different img_size/eval_res/seed can never be mixed with this run.
    if index is not None:
        for k, v in (('img_size', img_size), ('eval_res', eval_res), ('seed', seed)):
            if index[k] != v:
                raise RuntimeError(f"index.json {k}={index[k]} != requested {v}; "
                                   f"use a fresh --cache-dir (mixing would break comparability)")

    listing = {}
    for split in ('train', 'val'):
        ds = MUSES(root, split=split, transform=None, modals=['img'])
        rel = [os.path.relpath(f, root) for f in ds.files]
        log(f"[index] MUSES {split}: {len(rel)} images found by glob under {root}")
        listing[split] = rel

    if index is not None:
        for split in ('train', 'val'):
            have = set(listing[split])
            missing = [r for r in index[split]['rel'] if r not in have]
            if missing:
                raise RuntimeError(f"index.json references {len(missing)} {split} files that are "
                                   f"not under {root} (first: {missing[0]})")
        log(f"[index] reuse {idx_path} — train={len(index['train']['rel'])} "
            f"val={len(index['val']['rel'])}")
        return index

    rng = np.random.default_rng(seed)
    index = {'img_size': img_size, 'eval_res': eval_res, 'seed': seed, 'root_at_build': root,
             'num_classes': NUM_CLASSES}
    for split in ('train', 'val'):
        rel = listing[split]
        if split == 'train' and 0 < n_train < len(rel):
            pick = sorted(rng.choice(len(rel), size=n_train, replace=False).tolist())
        else:
            pick = list(range(len(rel)))            # val: 전체 (의뢰서 ②)
        sel = [rel[i] for i in pick]
        index[split] = {'n_total': len(rel), 'n_used': len(sel), 'rel': sel,
                        'cond': [cond_of(r) for r in sel]}
        log(f"[index] {split}: use {len(sel)}/{len(rel)} "
            f"({'random subset, seed %d' % seed if len(sel) < len(rel) else 'ALL'})")
    idx_path.write_text(json.dumps(index, indent=1))
    log(f"[index] wrote {idx_path}")
    return index


# ------------------------------------------------------------------ cache stage
class _SubsetDS:
    """MUSES restricted to the index's file list, in the index's exact order."""

    def __init__(self, root, split, transform, rel_list):
        from semseg.datasets.muses import MUSES
        self.ds = MUSES(root, split=split, transform=transform, modals=['img'])
        pos = {os.path.relpath(f, root): i for i, f in enumerate(self.ds.files)}
        self.idxs = [pos[r] for r in rel_list]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        sample, label = self.ds[self.idxs[i]]
        return sample[0], label


def _split_done(cache: Path, key: str, split: str) -> Path:
    return cache / key / f"{split}.done.json"


def run_cache(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from semseg.augmentations_mm import get_val_augmentation

    cache = Path(args.cache_dir)
    assert_local_cache_dir(cache)
    index = build_or_load_index(cache, args.dataset_root, args.n_train, args.seed,
                                args.img_size, args.eval_res)
    key = args.backbone
    spec = BACKBONES[key]
    grid = args.img_size // 16
    est = sum(index[s]['n_used'] for s in ('train', 'val')) * spec['dim'] * grid * grid * 2
    require_free_space(cache, est, f"{spec['label']} features ({spec['dim']}x{grid}x{grid} fp16)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encode, info = build_backbone(key, args.img_size, device, args.batch)
    assert info['grid'] == grid, f"grid {info['grid']} != {grid} (patch != 16?)"
    (cache / key).mkdir(parents=True, exist_ok=True)
    tf = get_val_augmentation((args.img_size, args.img_size))

    t0 = time.time()
    for split in ('train', 'val'):
        done = _split_done(cache, key, split)
        if done.exists() and not args.overwrite:
            log(f"[cache/{key}] {split}: already done ({json.loads(done.read_text())['n']} imgs) — skip")
            continue
        rel = index[split]['rel']
        ds = _SubsetDS(args.dataset_root, split, tf, rel)
        dl = DataLoader(ds, batch_size=info['batch'], num_workers=args.workers,
                        shuffle=False, pin_memory=True)
        n = len(ds)
        fpath = cache / key / f"{split}.f16.npy"
        feats = np.lib.format.open_memmap(fpath, mode='w+', dtype=np.float16,
                                          shape=(n, spec['dim'], grid, grid))
        lpath = cache / 'labels' / f"{split}.u8.npy"
        lab_done = (cache / 'labels' / f"{split}.done.json")
        need_labels = args.overwrite_labels or not lab_done.exists()
        if need_labels:
            lpath.parent.mkdir(parents=True, exist_ok=True)
            labels = np.lib.format.open_memmap(lpath, mode='w+', dtype=np.uint8,
                                               shape=(n, args.eval_res, args.eval_res))
        i = 0
        for x, y in dl:
            b = x.shape[0]
            assert x.shape[1:] == (3, args.img_size, args.img_size), \
                f"unexpected image shape {tuple(x.shape)} (expected 3x{args.img_size}²)"
            f = encode(x.to(device, non_blocking=True))
            assert torch.isfinite(f).all(), f"non-finite features at {split}[{i}:{i+b}]"
            assert f.shape[1:] == (spec['dim'], grid, grid), f"feat shape {tuple(f.shape)}"
            feats[i:i + b] = f.to(torch.float16).cpu().numpy()
            if need_labels:
                l = F.interpolate(y.unsqueeze(1).float(), size=(args.eval_res, args.eval_res),
                                  mode='nearest').squeeze(1).to(torch.uint8).numpy()
                labels[i:i + b] = l
            i += b
            if (i // max(info['batch'], 1)) % 25 == 0 or i == n:
                log(f"[cache/{key}] {split} {i}/{n} ({time.time()-t0:.0f}s)")
        assert i == n, f"wrote {i} of {n}"
        feats.flush()
        del feats
        if need_labels:
            labels.flush()
            del labels
            lab_done.write_text(json.dumps({'n': n, 'eval_res': args.eval_res,
                                            'ignore_label': IGNORE_LABEL}, indent=1))
        done.write_text(json.dumps({'n': n, 'split': split, 'dim': spec['dim'], 'grid': grid,
                                    'dtype': 'float16', 'backbone': info}, indent=1))
        log(f"[cache/{key}] {split} DONE n={n} feat={spec['dim']}x{grid}x{grid} "
            f"-> {fpath} ({fpath.stat().st_size/2**30:.2f} GiB)")
    log(f"[cache/{key}] ALL DONE in {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------ head
def make_head(in_ch: int, kind: str, hidden: int, feat_norm: str):
    """The ONE head spec, shared by all backbones (only in_ch differs).

    feat_norm='ln' applies a non-learnable channel LayerNorm to the cached features
    before the head. It is applied identically to all backbones and exists so that a
    single LR set is not biased by per-backbone feature scale. --feat-norm none +
    --head linear reproduces the plain conv1x1 linear probe (ProbeA1 head spec).
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_norm = feat_norm
            self.in_ch = in_ch
            if kind == 'linear':
                self.net = nn.Conv2d(in_ch, NUM_CLASSES, 1)
            elif kind == 'conv2':
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                    nn.GroupNorm(32, hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, NUM_CLASSES, 1),
                )
            else:
                raise ValueError(kind)

        def forward(self, x):
            if self.feat_norm == 'ln':
                x = F.layer_norm(x.permute(0, 2, 3, 1), (self.in_ch,)).permute(0, 3, 1, 2)
            return self.net(x)

    return Head()


def _load_feats(cache: Path, key: str, split: str, preload: str):
    fpath = cache / key / f"{split}.f16.npy"
    done = _split_done(cache, key, split)
    if not done.exists():
        raise FileNotFoundError(f"{split} features for '{key}' are not cached "
                                f"(missing {done}) — run --stage cache first")
    arr = np.load(fpath, mmap_mode='r')
    nbytes = arr.size * arr.itemsize
    use_ram = preload == 'ram' or (preload == 'auto' and nbytes < 0.5 * available_ram_bytes())
    if use_ram:
        arr = np.ascontiguousarray(arr)
    log(f"[probe/{key}] {split} feats {arr.shape} {'RAM' if use_ram else 'mmap'} "
        f"({nbytes/2**30:.2f} GiB)")
    return arr, json.loads(done.read_text())


def _confusion(conf, pred, gt):
    import torch
    valid = gt != IGNORE_LABEL
    idx = gt[valid].long() * NUM_CLASSES + pred[valid].long()
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def _iou(conf):
    conf = conf.double()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    iou = np.where(union.cpu().numpy() > 0,
                   (inter / union.clamp(min=1)).cpu().numpy(), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        miou = float(np.nanmean(iou)) if np.isfinite(iou).any() else float('nan')
    return iou, miou


def _evaluate(heads, feats, labels, conds, device, eval_res, batch):
    """Evaluate every head in one pass over the val features."""
    import torch
    import torch.nn.functional as F
    confs = [torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
             for _ in heads]
    uconds = sorted(set(conds))
    cconfs = [{c: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
               for c in uconds} for _ in heads]
    for h in heads:
        h.eval()
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            x = torch.from_numpy(np.array(feats[i:i + batch])).to(device).float()
            y = torch.from_numpy(np.array(labels[i:i + batch])).to(device).long()
            for hi, h in enumerate(heads):
                logits = F.interpolate(h(x), size=(eval_res, eval_res),
                                       mode='bilinear', align_corners=False)
                pred = logits.argmax(1)
                _confusion(confs[hi], pred, y)
                for b in range(x.shape[0]):
                    _confusion(cconfs[hi][conds[i + b]], pred[b], y[b])
    out = []
    for hi in range(len(heads)):
        iou, miou = _iou(confs[hi])
        per_cond = {}
        for c in uconds:
            if cconfs[hi][c].sum() > 0:
                _, cm = _iou(cconfs[hi][c])
                per_cond[c] = round(cm * 100, 2)
        out.append((iou, miou, per_cond))
    return out


def run_probe(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from semseg.datasets.muses import MUSES

    cache = Path(args.cache_dir)
    index = json.loads((cache / 'index.json').read_text())
    eval_res = index['eval_res']
    key = args.backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_f, tr_meta = _load_feats(cache, key, 'train', args.preload)
    va_f, _ = _load_feats(cache, key, 'val', args.preload)
    tr_l = np.load(cache / 'labels' / 'train.u8.npy', mmap_mode='r')
    va_l = np.ascontiguousarray(np.load(cache / 'labels' / 'val.u8.npy', mmap_mode='r'))
    assert len(tr_f) == len(tr_l) == index['train']['n_used']
    assert len(va_f) == len(va_l) == index['val']['n_used']
    C, g = tr_f.shape[1], tr_f.shape[2]
    conds = index['val']['cond']

    lrs = [float(s) for s in args.lr.split(',') if s]
    heads = [make_head(C, args.head, args.hidden, args.feat_norm).to(device) for _ in lrs]
    n_head_params = sum(p.numel() for p in heads[0].parameters())
    opts = [torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=args.weight_decay)
            for h, lr in zip(heads, lrs)]
    iters = math.ceil(len(tr_f) / args.batch)
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs * iters)
              for o in opts]
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    log(f"[probe/{key}] head={args.head} feat_norm={args.feat_norm} in_ch={C} grid={g} "
        f"params={n_head_params/1e3:.1f}k | lrs={lrs} epochs={args.epochs} batch={args.batch} "
        f"train={len(tr_f)} val={len(va_f)}")

    # 동일 예산: 모든 LR head 가 같은 배치·같은 순열을 본다 (데이터 패스 1회 공유).
    gen = torch.Generator().manual_seed(args.seed)
    hist = [[] for _ in lrs]
    best = [{'miou': -1.0, 'epoch': -1, 'state': None} for _ in lrs]
    t0 = time.time()
    ep_loss0 = None
    for ep in range(args.epochs):
        for h in heads:
            h.train()
        perm = torch.randperm(len(tr_f), generator=gen).tolist()
        tot = [0.0] * len(lrs)
        nb = 0
        for i in range(0, len(perm), args.batch):
            b = sorted(perm[i:i + args.batch])          # sorted -> friendlier mmap reads
            x = torch.from_numpy(np.array(tr_f[b])).to(device).float()
            y512 = torch.from_numpy(np.array(tr_l[b])).to(device)
            y = F.interpolate(y512.unsqueeze(1).float(), size=(g, g),
                              mode='nearest').squeeze(1).long()
            for hi, (h, o, s) in enumerate(zip(heads, opts, scheds)):
                loss = crit(h(x), y)
                o.zero_grad(set_to_none=True)
                loss.backward()
                o.step()
                s.step()
                tot[hi] += float(loss)
            nb += 1
        losses = [t / max(nb, 1) for t in tot]
        if ep_loss0 is None:
            ep_loss0 = losses
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            res = _evaluate(heads, va_f, va_l, conds, device, eval_res, args.eval_batch)
            msg = []
            for hi, lr in enumerate(lrs):
                miou = res[hi][1]
                hist[hi].append({'epoch': ep, 'loss': round(losses[hi], 4),
                                 'val_miou': round(miou * 100, 2)})
                if miou > best[hi]['miou']:
                    best[hi] = {'miou': miou, 'epoch': ep,
                                'state': {k: v.detach().clone() for k, v in heads[hi].state_dict().items()}}
                msg.append(f"lr{lr:g}: loss={losses[hi]:.4f} val={miou*100:.2f}")
            log(f"[probe/{key}] ep{ep} " + " | ".join(msg) + f"  ({time.time()-t0:.0f}s)")

    final = _evaluate(heads, va_f, va_l, conds, device, eval_res, args.eval_batch)
    per_lr = []
    for hi, lr in enumerate(lrs):
        iou_f, miou_f, cond_f = final[hi]
        heads[hi].load_state_dict(best[hi]['state'])
        iou_b, miou_b, cond_b = _evaluate([heads[hi]], va_f, va_l, conds, device,
                                          eval_res, args.eval_batch)[0]
        per_lr.append({
            'lr': lr,
            'final': {'miou': round(miou_f * 100, 2), 'per_cond': cond_f,
                      'per_class': {MUSES.CLASSES[c]: (None if np.isnan(iou_f[c])
                                                       else round(float(iou_f[c]) * 100, 2))
                                    for c in range(NUM_CLASSES)}},
            'best': {'miou': round(miou_b * 100, 2), 'epoch': best[hi]['epoch'],
                     'per_cond': cond_b},
            'first_epoch_loss': round(ep_loss0[hi], 4),
            'last_epoch_loss': round(hist[hi][-1]['loss'], 4) if hist[hi] else None,
            'history': hist[hi],
        })

    headline_lr = max(per_lr, key=lambda r: r['best']['miou'])
    report = {
        'probe': 'ProbeA2-backbone-scaling',
        'dataset': 'MUSES', 'modality': 'img (RGB only)', 'num_classes': NUM_CLASSES,
        'backbone': tr_meta['backbone'],
        'protocol': {
            'head': args.head, 'hidden': args.hidden if args.head == 'conv2' else None,
            'feat_norm': args.feat_norm, 'head_params': n_head_params,
            'epochs': args.epochs, 'batch': args.batch, 'lrs': lrs,
            'weight_decay': args.weight_decay, 'sched': 'cosine', 'seed': args.seed,
            'n_train': len(tr_f), 'n_val': len(va_f),
            'img_size': index['img_size'], 'eval_res': eval_res,
            'feat': f"{C}x{g}x{g}",
            'note': 'val mIoU on MUSES letterboxed square @eval_res — NOT the official MUSES '
                    'benchmark protocol; only backbone-to-backbone deltas are meaningful.',
        },
        'headline': {'lr': headline_lr['lr'],
                     'val_miou_best': headline_lr['best']['miou'],
                     'val_miou_final': headline_lr['final']['miou']},
        'per_lr': per_lr,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    of = out / f"probea2_{key}.json"
    of.write_text(json.dumps(report, indent=1))
    log(f"[probe/{key}] DONE headline val mIoU best={report['headline']['val_miou_best']:.2f} "
        f"final={report['headline']['val_miou_final']:.2f} (lr={headline_lr['lr']:g}) -> {of}")


# ------------------------------------------------------------------ report
def run_report(args):
    out = Path(args.out_dir)
    rows = []
    for key in args.backbones.split(','):
        p = out / f"probea2_{key}.json"
        if not p.exists():
            log(f"[report] missing {p} — skip")
            continue
        r = json.loads(p.read_text())
        rows.append((key, r))
    if not rows:
        raise SystemExit(f"[report] no probea2_*.json under {out}")

    order = [k for k in BACKBONES if any(k == kk for kk, _ in rows)]
    by = dict(rows)
    log("")
    log("ProbeA2 — DINOv3 backbone scaling (MUSES val, RGB only, frozen backbone + shared head)")
    log(f"{'backbone':<12}{'params(M)':>10}{'dim':>6}{'val mIoU(best)':>16}{'val mIoU(final)':>17}{'lr':>8}")
    for k in order:
        r = by[k]
        bb = r['backbone']
        log(f"{bb['label'][:11]:<12}{bb['params_m']:>10}{bb['embed_dim']:>6}"
            f"{r['headline']['val_miou_best']:>16.2f}{r['headline']['val_miou_final']:>17.2f}"
            f"{r['headline']['lr']:>8g}")

    def m(k, which='val_miou_best'):
        return by[k]['headline'][which] if k in by else None

    log("")
    for which in ('val_miou_best', 'val_miou_final'):
        hp, l, sp = m('hp', which), m('l', which), m('sp', which)
        log(f"[{which}]")
        if hp is not None and l is not None:
            d = hp - l
            verdict = ('상한 양성 → 백본 승급 본설계 착수' if d >= 1.5 else
                       '표현력 축 소진 → 원장 H6 기록(백본 승급 제외)' if d < 0.5 else
                       '중간대역 → 7B 추가 측정 후 재판정')
            log(f"  G-A2-상한: mIoU(H+) - mIoU(L) = {hp:.2f} - {l:.2f} = {d:+.2f}  → {verdict}")
        else:
            log("  G-A2-상한: H+ 또는 L 결과 없음 — 판정 불가")
        if l is not None and sp is not None:
            d = l - sp
            verdict = ('용량 정합 본런(S+ 전체 레시피)을 논문 슬롯으로 확정' if d <= 3.0 else
                       '본런 없이 "방법 기여 = 대형 백본 전제"로 논문 스코프 명시')
            log(f"  G-A2-하한: mIoU(L) - mIoU(S+) = {l:.2f} - {sp:.2f} = {d:+.2f}  → {verdict}")
        else:
            log("  G-A2-하한: L 또는 S+ 결과 없음 — 판정 불가")
    log("")
    log("⚠️ 절대치를 본 모델·리더보드와 비교 금지 (head가 다름). 백본 간 Δ만 유효. "
        "ProbeA1(+11.6, DELIVER)과도 직접 비교 불가.")
    summary = {k: {'label': by[k]['backbone']['label'],
                   'params_m': by[k]['backbone']['params_m'],
                   'embed_dim': by[k]['backbone']['embed_dim'],
                   **by[k]['headline']} for k in order}
    sp_ = out / 'probea2_summary.json'
    sp_.write_text(json.dumps(summary, indent=1))
    log(f"[report] wrote {sp_}")


# ------------------------------------------------------------------ smoke (의뢰서 ③ 검수 게이트)
def run_smoke(args):
    """기동 전 검수 게이트: 백본 4종 로드 + RANDOM INIT 0건 + feature shape + 손실 하강."""
    import torch

    keys = args.backbones.split(',')
    cache = Path(args.cache_dir)
    assert_local_cache_dir(cache)
    results, failures = {}, []

    # --- gate 1+3: load every backbone, real feature shape on a real MUSES image ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns = {'train': args.smoke_train, 'val': args.smoke_val}
    for key in keys:
        try:
            sub = argparse.Namespace(**vars(args))
            sub.backbone = key
            sub.n_train = ns['train']
            sub.overwrite = True
            sub.overwrite_labels = False
            sub.workers = min(args.workers, 4)
            # smoke uses a tiny index: train=smoke_train, val truncated to smoke_val
            _smoke_index(cache, args, ns)
            run_cache(sub)
            meta = json.loads(_split_done(cache, key, 'val').read_text())
            results[key] = {'loaded': True, 'feat': f"{meta['dim']}x{meta['grid']}x{meta['grid']}",
                            'backbone': meta['backbone']}
        except Exception as e:
            failures.append(f"{key}: {type(e).__name__}: {e}")
            results[key] = {'loaded': False, 'error': f"{type(e).__name__}: {e}"}
            log(f"[smoke/{key}] ❌ {type(e).__name__}: {e}")

    # --- gate 4: head training loss must go down (per backbone that loaded) ---
    for key in keys:
        if not results[key]['loaded']:
            continue
        try:
            sub = argparse.Namespace(**vars(args))
            sub.backbone = key
            sub.epochs = max(3, args.smoke_epochs)
            sub.eval_every = 1
            sub.lr = '1e-2'          # smoke only asks "does it train?", not "how well"
            sub.batch = min(args.batch or 4, 4)
            sub.out_dir = str(Path(args.out_dir) / 'smoke')
            sub.preload = 'ram'
            run_probe(sub)
            r = json.loads((Path(sub.out_dir) / f"probea2_{key}.json").read_text())['per_lr'][0]
            d = r['first_epoch_loss'] - r['last_epoch_loss']
            ok = d > 0
            results[key]['loss_first'] = r['first_epoch_loss']
            results[key]['loss_last'] = r['last_epoch_loss']
            results[key]['loss_decreasing'] = ok
            if not ok:
                failures.append(f"{key}: head loss did not decrease "
                                f"({r['first_epoch_loss']} -> {r['last_epoch_loss']})")
        except Exception as e:
            failures.append(f"{key}: probe {type(e).__name__}: {e}")
            results[key]['probe_error'] = f"{type(e).__name__}: {e}"
            log(f"[smoke/{key}] ❌ probe {type(e).__name__}: {e}")

    log("")
    log("=========== ProbeA2 검수 게이트 (의뢰서 ③) ===========")
    log(f"{'backbone':<6}{'load':>6}{'RANDOM INIT':>13}{'feat shape':>16}{'loss 1st':>10}{'loss last':>11}{'down':>7}")
    for key in keys:
        r = results[key]
        bb = r.get('backbone', {})
        log(f"{key:<6}{'OK' if r['loaded'] else 'FAIL':>6}{'0건' if r['loaded'] else '-':>13}"
            f"{r.get('feat', '-'):>16}{str(r.get('loss_first', '-')):>10}"
            f"{str(r.get('loss_last', '-')):>11}{str(r.get('loss_decreasing', '-')):>7}")
        if bb:
            log(f"       {bb['label']} '{bb['timm_name']}' tag={bb['pretrained_tag']} "
                f"params={bb['params_m']}M dim={bb['embed_dim']} "
                f"hf_file={bb['hf_cache_file']}")
    out = Path(args.out_dir) / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'smoke_report.json').write_text(json.dumps(results, indent=1))
    if failures:
        log("")
        for f in failures:
            log(f"❌ {f}")
        log(f"=========== GATE FAILED ({len(failures)}) ===========")
        raise SystemExit(1)
    log(f"=========== GATE PASSED (4/4 항목) ===========")
    log(f"[smoke] wrote {out/'smoke_report.json'}")


def _smoke_index(cache: Path, args, ns):
    """Tiny shared index for the smoke run (train=smoke_train, val=smoke_val)."""
    idx = cache / 'index.json'
    if idx.exists():
        return
    full = build_or_load_index(cache, args.dataset_root, ns['train'], args.seed,
                              args.img_size, args.eval_res)
    rng = np.random.default_rng(args.seed)
    v = full['val']
    if ns['val'] < v['n_used']:
        pick = sorted(rng.choice(v['n_used'], size=ns['val'], replace=False).tolist())
        full['val'] = {'n_total': v['n_total'], 'n_used': len(pick),
                       'rel': [v['rel'][i] for i in pick], 'cond': [v['cond'][i] for i in pick]}
    full['smoke'] = True
    idx.write_text(json.dumps(full, indent=1))
    log(f"[smoke] tiny index: train={full['train']['n_used']} val={full['val']['n_used']}")


# ------------------------------------------------------------------ selftest (no GPU/timm/data)
def run_selftest(args):
    """Plumbing self-test with a synthetic 'backbone' and synthetic labels.

    Verifies: memmap cache write/read, index reuse guard, label sharing, head training
    (loss down + mIoU up on a learnable synthetic task), evaluation/report math.
    Does NOT touch timm or the dataset — it can run anywhere.
    """
    import torch

    cache = Path(args.cache_dir)
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    C, g, res = 32, 8, 32
    ns = {'train': 24, 'val': 12}
    conds = ['clear_day', 'fog_night']
    index = {'img_size': 128, 'eval_res': res, 'seed': 0, 'num_classes': NUM_CLASSES}
    for split, n in ns.items():
        index[split] = {'n_total': n, 'n_used': n,
                        'rel': [f"{split}/{i}.png" for i in range(n)],
                        'cond': [conds[i % len(conds)] for i in range(n)]}
    (cache / 'index.json').write_text(json.dumps(index, indent=1))

    # synthetic linearly-separable task: label = argmax over a fixed random projection
    W = rng.normal(size=(NUM_CLASSES, C)).astype(np.float32)
    key = 'selftest'
    (cache / key).mkdir(parents=True, exist_ok=True)
    (cache / 'labels').mkdir(parents=True, exist_ok=True)
    for split, n in ns.items():
        f = rng.normal(size=(n, C, g, g)).astype(np.float32)
        cls = (W @ f.reshape(n, C, -1)).argmax(1).reshape(n, g, g).astype(np.uint8)
        lab = np.repeat(np.repeat(cls, res // g, 1), res // g, 2)
        lab[:, 0, :] = IGNORE_LABEL                                # exercise the ignore path
        fm = np.lib.format.open_memmap(cache / key / f"{split}.f16.npy", mode='w+',
                                       dtype=np.float16, shape=(n, C, g, g))
        fm[:] = f.astype(np.float16)
        fm.flush()
        del fm
        lm = np.lib.format.open_memmap(cache / 'labels' / f"{split}.u8.npy", mode='w+',
                                       dtype=np.uint8, shape=(n, res, res))
        lm[:] = lab
        lm.flush()
        del lm
        _split_done(cache, key, split).write_text(json.dumps(
            {'n': n, 'split': split, 'dim': C, 'grid': g, 'dtype': 'float16',
             'backbone': {'key': key, 'label': 'SYNTHETIC', 'timm_name': 'none',
                          'params_m': 0.0, 'embed_dim': C, 'pretrained_tag': None,
                          'hf_cache_file': None}}, indent=1))
        (cache / 'labels' / f"{split}.done.json").write_text(json.dumps({'n': n}, indent=1))

    ok = True
    for head in ('linear', 'conv2'):
        sub = argparse.Namespace(**vars(args))
        sub.backbone = key
        sub.head = head
        sub.epochs = 12
        sub.eval_every = 4
        sub.batch = 8
        sub.eval_batch = 8
        sub.lr = '3e-3,1e-2'
        sub.preload = 'ram'
        sub.out_dir = str(Path(args.out_dir) / 'selftest' / head)
        run_probe(sub)
        r = json.loads((Path(sub.out_dir) / f"probea2_{key}.json").read_text())
        for pl in r['per_lr']:
            down = pl['first_epoch_loss'] > pl['last_epoch_loss']
            log(f"[selftest/{head}] lr={pl['lr']:g} loss {pl['first_epoch_loss']} -> "
                f"{pl['last_epoch_loss']} ({'down' if down else 'NOT DOWN'}) "
                f"val mIoU final={pl['final']['miou']} best={pl['best']['miou']} "
                f"per_cond={pl['final']['per_cond']}")
            ok &= down
        ok &= r['headline']['val_miou_best'] > 5.0        # synthetic task must be learnable
        assert set(r['per_lr'][0]['final']['per_cond']) == set(conds), "per-cond breakdown broken"

    # --- report stage: gate arithmetic (의뢰서 ⑥) on synthetic headline numbers ---
    rout = Path(args.out_dir) / 'selftest' / 'report'
    rout.mkdir(parents=True, exist_ok=True)
    fake = dict(sp=(58.10, 57.40), b=(61.00, 60.20), l=(63.40, 62.90), hp=(65.20, 64.60))
    meta = dict(sp=('DINOv3-S+', 28.7, 384), b=('DINOv3-B', 85.6, 768),
                l=('DINOv3-L', 303.1, 1024), hp=('DINOv3-H+', 840.5, 1280))
    for k, (bst, fin) in fake.items():
        lab, pm, dim = meta[k]
        (rout / f"probea2_{k}.json").write_text(json.dumps({
            'backbone': {'label': lab + ' (SYNTHETIC selftest numbers)', 'params_m': pm,
                         'embed_dim': dim},
            'headline': {'lr': 1e-3, 'val_miou_best': bst, 'val_miou_final': fin}}))
    rep = argparse.Namespace(**vars(args))
    rep.out_dir = str(rout)
    rep.backbones = 'sp,b,l,hp'
    run_report(rep)
    summ = json.loads((rout / 'probea2_summary.json').read_text())
    up = summ['hp']['val_miou_best'] - summ['l']['val_miou_best']      # 65.20-63.40 = +1.80
    lo = summ['l']['val_miou_best'] - summ['sp']['val_miou_best']      # 63.40-58.10 = +5.30
    ok &= abs(up - 1.80) < 1e-6 and abs(lo - 5.30) < 1e-6 and up >= 1.5 and lo > 3.0
    log(f"[selftest/report] gate arithmetic Δ상한={up:+.2f} Δ하한={lo:+.2f} "
        f"({'ok' if ok else 'WRONG'})")

    # index reuse guard must reject a changed protocol
    try:
        build_or_load_index(cache, '/nonexistent', 1, 999, 128, res)
        log("[selftest] ❌ index guard did not fire")
        ok = False
    except RuntimeError as e:
        log(f"[selftest] index guard ok ({str(e)[:60]}...)")
    except Exception as e:                        # dataset missing -> guard untested but fine
        log(f"[selftest] index guard untested (no dataset: {type(e).__name__})")

    log(f"[selftest] {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True,
                    choices=['cache', 'probe', 'report', 'smoke', 'selftest'])
    ap.add_argument('--backbone', choices=list(BACKBONES), help='cache/probe: which backbone')
    ap.add_argument('--backbones', default=','.join(DEFAULT_SET),
                    help='smoke/report: comma list (default S+,B,L,H+)')
    ap.add_argument('--gpu', default='0',
                    help='sets CUDA_VISIBLE_DEVICES only if it is not already exported '
                         '(an existing export wins — 빈 GPU 배치 규약을 런처가 이미 정했을 때 존중)')
    ap.add_argument('--dataset-root', default='/SSDb/jemo_maeng/dset/MUSES')
    ap.add_argument('--cache-dir', default='/mnt/SSD2/probea2_cache',
                    help='LOCAL SSD only (의뢰서 ②). jarvis: /SSDb/jemo_maeng/cache/probea2')
    ap.add_argument('--out-dir', default='./outputs/probea2')
    ap.add_argument('--seed', type=int, default=0)
    # cache
    ap.add_argument('--n-train', type=int, default=3000,
                    help='train subset size (의뢰서 ②: 2~4천). 0 = all')
    ap.add_argument('--img-size', type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument('--eval-res', type=int, default=DEFAULT_EVAL_RES)
    ap.add_argument('--batch', type=int, default=None, help='cache fw batch (default per backbone)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--overwrite-labels', action='store_true')
    # probe
    ap.add_argument('--head', default='linear', choices=['linear', 'conv2'],
                    help="THE shared head spec. 'linear'=conv1x1 (ProbeA1 사양), "
                         "'conv2'=conv3x3->GN->GELU->conv1x1")
    ap.add_argument('--hidden', type=int, default=256, help='conv2 head hidden channels')
    ap.add_argument('--feat-norm', default='ln', choices=['ln', 'none'],
                    help="non-learnable channel LayerNorm on cached features (all backbones alike)")
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', default='1e-3,3e-3,1e-2',
                    help='comma list; one head per LR, all trained on the SAME batches')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--eval-batch', type=int, default=8)
    ap.add_argument('--preload', default='auto', choices=['auto', 'ram', 'mmap'])
    # smoke
    ap.add_argument('--smoke-train', type=int, default=16)
    ap.add_argument('--smoke-val', type=int, default=8)
    ap.add_argument('--smoke-epochs', type=int, default=5)
    args = ap.parse_args()

    if args.stage in ('cache', 'probe') and not args.backbone:
        ap.error(f"--stage {args.stage} requires --backbone")
    if args.stage != 'selftest':
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    if args.stage == 'probe' and args.batch is None:
        args.batch = 16
    elif args.stage == 'selftest':
        args.batch = args.batch or 8

    {'cache': run_cache, 'probe': run_probe, 'report': run_report,
     'smoke': run_smoke, 'selftest': run_selftest}[args.stage](args)


if __name__ == '__main__':
    main()
