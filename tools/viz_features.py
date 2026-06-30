#!/usr/bin/env python3
"""
tools/viz_features.py — RBMA feature / fusion diagnostic viewer (P26/P27/P28/P29/P30...).

For each selected image, render ONE panel showing, per modality and for the fusion:
  R1  inputs (per modality)            | GT | Final Pred | Error(vs GT)
  R2  per-modality ENCODER feature PCA | Fused feature PCA (memory-attention fused m_feat)
  R3  per-modality RELIABILITY (1-H)   = RBMA "where is this modality trusted"
  R4  per-modality DECODER argmax pred = "which modality carries which class"
  R5  per-modality UAMM fusion weight  (spatial softmax weight in the fused feature)

All signals come from one eval forward via model._last_* diagnostics + the (m_output, m_feat)
return — NO model-code change. Reliability = 1 - H(softmax(per-modal logits))/log(C), the same
normalized-entropy reliability RBMA uses for the memory-attention bias.

Generic across models: it only needs the model's own config (MODEL block) + a checkpoint, so
P29/P30 work by swapping --cfg/--model_path. Selection: --case <condition>, --indices, or
--contains <ClassA,ClassB> (pick images whose GT contains those classes — e.g. Water/Bridge for
dead classes, RailTrack for domain-sensitive).

Example:
  python tools/viz_features.py --cfg configs/b200-deliver_rgbdel_P28_physaug.yaml \
    --model_path ckpt_P28/test_epoch178_55.27_top1_checkpoint.pth \
    --dataset-root /ailab_mat2/dataset/DELIVER --case sun --contains RailTrack \
    --num 2 --gpu 0 --out-dir ~/viz_P28
"""
import argparse, os, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V  # reuse load_model / create_dataset / get_val_augmentation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]); IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denorm(t):
    a = t.detach().float().cpu().numpy()           # (C,H,W)
    a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 3:
        a = a * IMAGENET_STD + IMAGENET_MEAN
    else:
        a = a[..., :1]
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return np.clip(a, 0, 1)

def pca_rgb(feat):                                  # feat (C,H,W) -> (H,W,3) in [0,1]
    C, H, W = feat.shape
    x = feat.detach().float().reshape(C, -1).t()    # (HW, C)
    x = x - x.mean(0, keepdim=True)
    try:
        _, _, Vt = torch.pca_lowrank(x, q=3)
        y = x @ Vt[:, :3]
    except Exception:
        y = x[:, :3]
    y = (y - y.min(0).values) / (y.max(0).values - y.min(0).values + 1e-8)
    return y.reshape(H, W, 3).cpu().numpy()

def reliability(logits):                            # (C,H,W) logits -> (H,W) 1 - H/logC in [0,1]
    p = F.softmax(logits.detach().float(), dim=0)
    H = -(p * (p + 1e-8).log()).sum(0)
    return (1.0 - H / math.log(p.shape[0])).cpu().numpy()

def to_hw(arr):                                     # squeeze (1,H,W)/(H,W) numpy -> (H,W)
    a = np.asarray(arr)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None, help="override DATASET.ROOT")
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--case', default=None, help="DELIVER condition: cloud|fog|night|rain|sun")
    ap.add_argument('--indices', default=None, help="comma-separated dataset indices")
    ap.add_argument('--contains', default=None, help="comma class names; pick imgs whose GT has them")
    ap.add_argument('--num', type=int, default=3)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    out = Path(args.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root: ds_cfg['ROOT'] = args.dataset_root
    if args.case is not None: ds_cfg['CASE'] = args.case
    if isinstance(ds_cfg.get('PHYSAUG'), dict): ds_cfg['PHYSAUG']['ENABLE'] = False

    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    image_size = cfg['EVAL']['IMAGE_SIZE'] if args.split == 'val' else cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(image_size, dataset_cfg=ds_cfg)
    split = 'val' if args.split == 'val' else 'test'
    dataset, has_gt = V.create_dataset(ds_cfg, split, transform, args.split, macvi=False, eval_day=False)
    classes = list(dataset.CLASSES); palette = dataset.PALETTE
    ds_cls = V._get_dataset_class(dataset)
    modals = ds_cfg.get('MODALS')

    # ---- select indices
    if args.indices:
        sel = [int(x) for x in args.indices.split(',')]
    elif args.contains:
        want = [classes.index(c.strip()) for c in args.contains.split(',') if c.strip() in classes]
        sel = []
        for i in range(len(dataset)):
            lbl = dataset[i][1]
            ids = set(np.unique(np.asarray(lbl)))
            if any(w in ids for w in want):
                sel.append(i)
            if len(sel) >= args.num: break
    else:
        sel = list(range(min(args.num, len(dataset))))
    sel = sel[:args.num]
    print(f"[viz] {len(dataset)} imgs in split={split} case={args.case}; selected {sel}", flush=True)

    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model

    for idx in sel:
        images, label, meta = dataset[idx]
        imgs = [im.unsqueeze(0).to(device) for im in images]
        with torch.no_grad():
            m_output, m_feat = model(imgs, multimask_output=True)
        m = len(imgs)
        per_feat = getattr(core, '_last_per_modal_feats', None)
        per_out = getattr(core, '_last_per_modal_outputs', None)
        uamm_sp = getattr(core, '_last_uamm_spatial', None)
        final_pred = m_output[0].argmax(0).cpu().numpy().astype(np.uint8)
        gt = np.asarray(label).astype(np.int32)
        ignore = getattr(dataset, 'ignore_label', 255)
        # resize pred to gt size for error map
        ph, pw = final_pred.shape; gh, gw = gt.shape
        if (ph, pw) != (gh, gw):
            final_pred_r = np.asarray(
                torch.nn.functional.interpolate(
                    torch.tensor(final_pred)[None, None].float(), size=(gh, gw), mode='nearest')[0, 0]).astype(np.uint8)
        else:
            final_pred_r = final_pred
        err = ((final_pred_r != gt) & (gt != ignore)).astype(np.float32)

        ncol = m + 3
        nrow = 5
        fig, ax = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
        for r in range(nrow):
            for c in range(ncol):
                ax[r, c].axis('off')
        def show(r, c, img, title, cmap=None):
            ax[r, c].imshow(img, cmap=cmap); ax[r, c].set_title(title, fontsize=8); ax[r, c].axis('off')
        # R1 inputs + GT/Pred/Error
        for i in range(m):
            show(0, i, denorm(images[i]).squeeze(), f"in:{modals[i]}", cmap='gray' if denorm(images[i]).shape[-1]==1 else None)
        show(0, m,   ds_cls.decode_segmap(gt.astype(np.uint8), palette), "GT")
        show(0, m+1, ds_cls.decode_segmap(final_pred_r, palette), "Pred")
        show(0, m+2, err, "Error", cmap='Reds')
        # R2 feature PCA per modal + fused
        if per_feat is not None:
            for i in range(m): show(1, i, pca_rgb(per_feat[i][0]), f"featPCA:{modals[i]}")
        show(1, m, pca_rgb(m_feat[0]), "FUSED featPCA")
        # R3 reliability 1-H per modal
        if per_out is not None:
            for i in range(m): show(2, i, reliability(per_out[i][0]), f"reliab:{modals[i]}", cmap='viridis')
        # R4 per-modal argmax pred
        if per_out is not None:
            for i in range(m):
                pm = per_out[i][0].argmax(0).cpu().numpy().astype(np.uint8)
                show(3, i, ds_cls.decode_segmap(pm, palette), f"pred:{modals[i]}")
        # R5 UAMM fusion weight per modal
        if uamm_sp is not None:
            for i in range(m): show(4, i, to_hw(uamm_sp[i][0]), f"UAMM w:{modals[i]}", cmap='magma')
        stem = meta.get('stem', f"idx{idx}") if isinstance(meta, dict) else f"idx{idx}"
        fig.suptitle(f"{Path(args.model_path).stem} | {args.case or split} | {stem}", fontsize=10)
        fig.tight_layout()
        fp = out / f"panel_{args.case or split}_{stem}.png"
        fig.savefig(fp, dpi=110, bbox_inches='tight'); plt.close(fig)
        print(f"[viz] saved {fp}", flush=True)
    print(f"[viz] done -> {out}")

if __name__ == '__main__':
    main()
