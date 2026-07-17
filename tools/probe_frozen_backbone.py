#!/usr/bin/env python3
"""
tools/probe_frozen_backbone.py — A-1 frozen-backbone feasibility probe on DELIVER.

Question: can a modern frozen dense backbone (DINOv3 / DINOv2) represent the classes
that are dead in our frozen-SAM2-Hiera pipeline (Bridge/Water/Wall ~0 test IoU),
under an identical light linear-probe protocol?

Protocol (identical across backbones):
  1. cache : forward every selected DELIVER image through the FROZEN encoder once,
             save fp16 patch-feature grids to disk (per backbone x modality x split).
             Labels are cached once (nearest-downsampled to EVAL_RES uint8), shared.
  2. probe : train a linear head (1x1 conv -> 25 classes) on cached train features,
             CE ignore 255, labels nearest-downsampled to the token grid;
             early-stop on val mIoU; final eval on val + test by bilinear-upsampling
             logits to EVAL_RES (same for ALL backbones).

Backbones:
  dinov3 : timm vit_large_patch16_dinov3.lvd1689m  (patch16 @1024 -> 64x64x1024)
  dinov2 : timm vit_large_patch14_reg4_dinov2.lvd142m (input resized 1008 -> 72x72x1024)
  sam2   : repo build_sam2 hiera_b+ trunk, concat[stride16(448), up(stride32(896))] -> 64x64x1344
           (>= DINO feature dim, so linear-probe capacity does not favor DINO)

Usage (on hinton):
  PYTHONPATH set automatically from repo layout. Example:
  python tools/probe_frozen_backbone.py --stage cache --backbone dinov3 \
      --dataset-root /SSDd/jemo_maeng/dset/DELIVER --cache-dir /home/jemo_maeng/probe_cache --gpu 0
  python tools/probe_frozen_backbone.py --stage probe --backbone dinov3 --modality img \
      --cache-dir /home/jemo_maeng/probe_cache --out-dir /home/jemo_maeng/probe_a1
"""
import argparse, os, sys, json, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'semseg' / 'models' / 'sam2'))

CASES = ['cloud', 'fog', 'night', 'rain', 'sun']
ALL_MODALS = ['img', 'depth', 'event', 'lidar']
PER_CASE_DEFAULT = {'train': 300, 'val': 100, 'test': 100}
EVAL_RES = 512           # common eval resolution for ALL backbones
NUM_CLASSES = 25
HEADLINE = {'Wall': 10, 'Bridge': 14, 'Water': 20}
CLASSES = ["Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk",
           "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack",
           "GroundRail", "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler",
           "Bus", "Truck"]

BACKBONE_SPECS = {
    'dinov3': dict(timm_name='vit_large_patch16_dinov3.lvd1689m', in_size=1024),
    'dinov2': dict(timm_name='vit_large_patch14_reg4_dinov2.lvd142m', in_size=1008),
    'sam2':   dict(timm_name=None, in_size=1024),
}


# ---------------------------------------------------------------- encoders
def build_encoder(backbone, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if backbone in ('dinov3', 'dinov2'):
        import timm
        spec = BACKBONE_SPECS[backbone]
        model = timm.create_model(spec['timm_name'], pretrained=True, num_classes=0)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        npre = getattr(model, 'num_prefix_tokens', 1)
        in_size = spec['in_size']

        def encode(x):  # x: (B,3,H,W) normalized
            if x.shape[-1] != in_size:
                x = F.interpolate(x, size=(in_size, in_size), mode='bilinear', align_corners=False)
            f = model.forward_features(x)
            if isinstance(f, (list, tuple)):
                f = f[-1]
            f = f[:, npre:, :]                       # (B,N,C)
            g = int(math.isqrt(f.shape[1]))
            assert g * g == f.shape[1], f"non-square token grid: {f.shape}"
            return f.transpose(1, 2).reshape(x.shape[0], -1, g, g)
        return encode

    elif backbone == 'sam2':
        from sam2.build_sam import build_sam2
        ckpt = str(REPO / 'semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt')
        sam2 = build_sam2(
            'sam2_hiera_b+.yaml', ckpt,
            device=str(device),
            hydra_overrides_extra=[
                "++model.pred_obj_scores=false",
                "++model.fixed_no_obj_ptr=false",
                "++model.pred_obj_scores_mlp=false",
            ])
        trunk = sam2.image_encoder.trunk
        trunk.eval()
        for p in trunk.parameters():
            p.requires_grad_(False)

        def encode(x):  # (B,3,1024,1024) -> (B,1344,64,64)
            outs = trunk(x)                          # strides 4,8,16,32
            f16, f32 = outs[2], outs[3]
            f32u = F.interpolate(f32.float(), size=f16.shape[-2:], mode='bilinear', align_corners=False)
            return torch.cat([f16.float(), f32u], 1)
        return encode

    raise ValueError(backbone)


# ---------------------------------------------------------------- cache stage
def select_indices(n_total, n_want):
    if n_total <= n_want:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, n_want).round().astype(int).tolist()))


def run_cache(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from semseg.datasets.deliver import DELIVER
    from semseg.augmentations_mm import get_val_augmentation

    device = torch.device('cuda')
    modals = [m for m in args.modalities.split(',') if m]
    encode = build_encoder(args.backbone, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    cache = Path(args.cache_dir)
    tf = get_val_augmentation((args.img_size, args.img_size))

    class CacheDS(Dataset):
        def __init__(self, ds, idxs, case):
            self.ds, self.idxs, self.case = ds, idxs, case

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            j = self.idxs[i]
            sample, label, _meta = self.ds[j]
            return torch.stack(sample, 0), label, f"{self.case}_{j:05d}"

    per_case = {'train': args.per_case_train, 'val': args.per_case_val, 'test': args.per_case_test}
    splits = [s for s in args.splits.split(',') if s]
    meta_all = {}
    t0 = time.time()
    for split in splits:
        n_done = 0
        for case in CASES:
            ds = DELIVER(args.dataset_root, split=split, transform=tf,
                         modals=ALL_MODALS, case=case, return_meta=True)
            idxs = select_indices(len(ds), per_case[split])
            meta_all[f"{split}/{case}"] = {'n_total': len(ds), 'n_used': len(idxs),
                                           'stems': [Path(ds.files[j]).stem for j in idxs]}
            cds = CacheDS(ds, idxs, case)
            dl = DataLoader(cds, batch_size=args.batch, num_workers=args.workers, shuffle=False)
            lbl_dir = cache / 'labels' / split
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for m in modals:
                (cache / args.backbone / m / split).mkdir(parents=True, exist_ok=True)
            for xs, label, keys in dl:
                # labels (shared, write once)
                if not (lbl_dir / f"{keys[-1]}.npy").exists():
                    l512 = F.interpolate(label.unsqueeze(1).float(), size=(EVAL_RES, EVAL_RES),
                                         mode='nearest').squeeze(1).to(torch.uint8).numpy()
                    for k, arr in zip(keys, l512):
                        np.save(lbl_dir / f"{k}.npy", arr)
                for mi, m in enumerate(ALL_MODALS):
                    if m not in modals:
                        continue
                    if (cache / args.backbone / m / split / f"{keys[-1]}.npy").exists() and not args.overwrite:
                        continue
                    x = xs[:, mi].to(device, non_blocking=True)
                    if m != 'img':                    # dataset only /255 for non-img
                        x = (x - mean) / std
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                        f = encode(x)
                    f = f.detach().to(torch.float16).cpu().numpy()
                    for k, arr in zip(keys, f):
                        np.save(cache / args.backbone / m / split / f"{k}.npy", arr)
                n_done += len(keys)
            print(f"[cache/{args.backbone}] {split}/{case}: {len(idxs)} imgs x {len(modals)} modals "
                  f"done (cum {n_done} fw, {time.time()-t0:.0f}s)", flush=True)
    (cache / f"meta_{args.backbone}.json").write_text(json.dumps(meta_all, indent=1))
    print(f"[cache/{args.backbone}] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- probe stage
def load_split(cache, backbone, modality, split):
    import torch
    fdir = Path(cache) / backbone / modality / split
    ldir = Path(cache) / 'labels' / split
    keys = sorted(p.stem for p in fdir.glob('*.npy'))
    assert keys, f"no cached features in {fdir}"
    f0 = np.load(fdir / f"{keys[0]}.npy")
    C, g = f0.shape[0], f0.shape[1]
    feats = torch.empty(len(keys), C, g, g, dtype=torch.float16)
    labels = torch.empty(len(keys), EVAL_RES, EVAL_RES, dtype=torch.uint8)
    for i, k in enumerate(keys):
        feats[i] = torch.from_numpy(np.load(fdir / f"{k}.npy"))
        labels[i] = torch.from_numpy(np.load(ldir / f"{k}.npy"))
    return feats, labels, keys


def confusion_update(conf, pred, gt):
    import torch
    valid = gt != 255
    idx = gt[valid].long() * NUM_CLASSES + pred[valid].long()
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_conf(conf):
    conf = conf.double()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.full_like(inter, float('nan')))
    miou = float(np.nanmean(iou.cpu().numpy()))
    return iou.cpu().numpy(), miou


def evaluate(head, feats, labels, keys, device, batch=16):
    import torch
    import torch.nn.functional as F
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    conf_case = {c: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device) for c in CASES}
    head.eval()
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            x = feats[i:i + batch].to(device).float()
            y = labels[i:i + batch].to(device)
            logits = head(x)
            logits = F.interpolate(logits, size=(EVAL_RES, EVAL_RES), mode='bilinear', align_corners=False)
            pred = logits.argmax(1)
            confusion_update(conf, pred, y)
            for b, k in enumerate(keys[i:i + batch]):
                case = k.split('_')[0]
                confusion_update(conf_case[case], pred[b], y[b])
    iou, miou = iou_from_conf(conf)
    per_case = {}
    for c in CASES:
        if conf_case[c].sum() > 0:
            ciou, cmiou = iou_from_conf(conf_case[c])
            per_case[c] = {'miou': round(cmiou * 100, 2),
                           **{name: round(float(ciou[ci]) * 100, 2) for name, ci in HEADLINE.items()}}
    return iou, miou, per_case


def run_probe(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device('cuda')
    torch.manual_seed(0)
    t0 = time.time()
    tr_f, tr_l, tr_k = load_split(args.cache_dir, args.backbone, args.modality, 'train')
    va_f, va_l, va_k = load_split(args.cache_dir, args.backbone, args.modality, 'val')
    te_f, te_l, te_k = load_split(args.cache_dir, args.backbone, args.modality, 'test')
    C, g = tr_f.shape[1], tr_f.shape[2]
    print(f"[probe/{args.backbone}/{args.modality}] train={len(tr_f)} val={len(va_f)} test={len(te_f)} "
          f"feat={C}x{g}x{g} loaded in {time.time()-t0:.0f}s", flush=True)

    head = nn.Conv2d(C, NUM_CLASSES, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    iters_per_ep = math.ceil(len(tr_f) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * iters_per_ep)
    crit = nn.CrossEntropyLoss(ignore_index=255)

    best = {'miou': -1.0, 'state': None, 'epoch': -1}
    patience, bad = args.patience, 0
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr_f))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            x = tr_f[b].to(device).float()
            y512 = tr_l[b].to(device)
            y = F.interpolate(y512.unsqueeze(1).float(), size=(g, g), mode='nearest').squeeze(1).long()
            loss = crit(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            _, vmiou, _ = evaluate(head, va_f, va_l, va_k, device)
            flag = ''
            if vmiou > best['miou']:
                best = {'miou': vmiou, 'state': {k: v.clone() for k, v in head.state_dict().items()},
                        'epoch': ep}
                bad = 0
                flag = ' *'
            else:
                bad += 1
            print(f"[probe/{args.backbone}/{args.modality}] ep{ep} loss={tot/nb:.4f} "
                  f"val_mIoU={vmiou*100:.2f}{flag}", flush=True)
            if bad >= patience:
                print(f"[probe] early stop at ep{ep} (best ep{best['epoch']})", flush=True)
                break

    head.load_state_dict(best['state'])
    va_iou, va_miou, _ = evaluate(head, va_f, va_l, va_k, device)
    te_iou, te_miou, te_case = evaluate(head, te_f, te_l, te_k, device)

    def pc(iou):
        return {CLASSES[c]: (None if math.isnan(float(iou[c])) else round(float(iou[c]) * 100, 2))
                for c in range(NUM_CLASSES)}

    report = {
        'backbone': args.backbone,
        'backbone_id': BACKBONE_SPECS[args.backbone]['timm_name'] or 'sam2.1_hiera_base_plus trunk s16+up(s32)',
        'modality': args.modality,
        'feat': f"{C}x{g}x{g}", 'eval_res': EVAL_RES,
        'protocol': {'head': 'linear conv1x1', 'lr': args.lr, 'batch': args.batch,
                     'epochs_max': args.epochs, 'best_epoch': best['epoch'],
                     'n_train': len(tr_f), 'n_val': len(va_f), 'n_test': len(te_f)},
        'val': {'miou': round(va_miou * 100, 2), 'per_class': pc(va_iou)},
        'test': {'miou': round(te_miou * 100, 2), 'per_class': pc(te_iou), 'per_case': te_case},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    of = out / f"probe_{args.backbone}_{args.modality}.json"
    of.write_text(json.dumps(report, indent=1))
    hl_v = {n: report['val']['per_class'][n] for n in HEADLINE}
    hl_t = {n: report['test']['per_class'][n] for n in HEADLINE}
    print(f"[probe/{args.backbone}/{args.modality}] DONE val mIoU={va_miou*100:.2f} "
          f"test mIoU={te_miou*100:.2f}", flush=True)
    print(f"  HEADLINE val : {hl_v}", flush=True)
    print(f"  HEADLINE test: {hl_t}", flush=True)
    print(f"  wrote {of}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['cache', 'probe'])
    ap.add_argument('--backbone', required=True, choices=list(BACKBONE_SPECS))
    ap.add_argument('--dataset-root', default='/SSDd/jemo_maeng/dset/DELIVER')
    ap.add_argument('--cache-dir', default='/home/jemo_maeng/probe_cache')
    ap.add_argument('--out-dir', default='/home/jemo_maeng/probe_a1')
    ap.add_argument('--gpu', default='0')
    # cache
    ap.add_argument('--modalities', default='img,depth,event,lidar')
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--per-case-train', type=int, default=PER_CASE_DEFAULT['train'])
    ap.add_argument('--per-case-val', type=int, default=PER_CASE_DEFAULT['val'])
    ap.add_argument('--per-case-test', type=int, default=PER_CASE_DEFAULT['test'])
    ap.add_argument('--img-size', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    # probe
    ap.add_argument('--modality', default='img')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.stage == 'cache':
        run_cache(args)
    else:
        args.batch = 16 if args.batch == 4 else args.batch
        run_probe(args)


if __name__ == '__main__':
    main()
