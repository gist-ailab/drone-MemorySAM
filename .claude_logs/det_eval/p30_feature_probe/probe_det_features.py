#!/usr/bin/env python3
"""P30-Det feature probe (v2): per-modality encoding features + PCA, RBMA on/off,
memory-attention before/after, router fusion weights — for MANY images.

Outputs (--out_dir):
  panel_<id>.png       per-image combined visualization
  raw/<id>.npz         RAW feature values for programmatic re-analysis
                       (mem_{img,lidar,thermal}, fused_p5, rel_{...}, router_w_L{0,1,2},
                        det_boxes/scores/labels, gt_labels)  [float16]
  probe_stats.json     compact numerical stats per image + aggregate
  summary.png          aggregate plots across all probed images

Usage: python probe_det_features.py --cfg C --det_checkpoint CK --out_dir D
       [--n 24] [--classes 8,6,7,9] [--no_raw]
"""
import os, json, argparse
import numpy as np
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import objdet.tools.diag_det as dd
from tools.viz_features import pca_rgb, reliability

MODALS = ['img', 'lidar', 'thermal']


def feat_stats(t):
    x = t.float().flatten(1); xc = x - x.mean(1, keepdim=True)
    try:
        s = torch.linalg.svdvals(xc.T @ xc); evr = (s / s.sum()).cpu().numpy()
        top3 = [float(v) for v in evr[:3]]
    except Exception:
        top3 = [float('nan')] * 3
    return {'l2_mean': float(t.float().pow(2).mean().sqrt()),
            'abs_mean': float(t.float().abs().mean()),
            'pca_evr_top3': top3,
            'active_frac': float((t.float().abs() > 1e-3).float().mean())}


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float(F.cosine_similarity(a[None], b[None]).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True); ap.add_argument('--det_checkpoint', required=True)
    ap.add_argument('--out_dir', default='out_probe'); ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--classes', default=''); ap.add_argument('--no_raw', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rawdir = os.path.join(args.out_dir, 'raw'); os.makedirs(rawdir, exist_ok=True)
    import yaml; cfg = yaml.safe_load(open(args.cfg))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = dd.build_dataset(cfg, 'val')
    model = dd.build_model(cfg, dev, ds.n_classes)
    ck = torch.load(args.det_checkpoint, map_location=dev, weights_only=False)
    miss, unexp = model.load_state_dict(ck['model_state_dict'], strict=False)
    print(f"[ckpt] ep={ck.get('epoch')} missing={len(miss)} unexpected={len(unexp)}")
    model.eval(); seg = model.seg_model
    names = ds.class_names
    has_router = hasattr(model, 'routers'); has_lam = hasattr(seg, 'lambda_bias')

    # ---- image selection: spread across val UNION small-object images ----
    prefer = set(int(x) for x in args.classes.split(',') if x != '')
    small = []
    if prefer:
        for i in range(0, len(ds), 3):                     # sample every 3rd to find small-obj imgs
            s = ds[i]
            if len(s['labels']) and set(int(l) for l in s['labels']) & prefer:
                small.append(i)
            if len(small) >= args.n // 2:
                break
    step = max(1, len(ds) // (args.n))
    spread = list(range(0, len(ds), step))
    idxs = sorted(set(small + spread))[:args.n]
    print("probe image indices:", idxs)

    mem_io = []
    def ma_hook(mod, a, kw, out):
        b = kw.get('curr', a[0] if len(a) else None)
        if isinstance(b, (list, tuple)) and len(b): b = b[-1]
        mem_io.append((b, out))
    h = seg.sam.memory_attention.register_forward_hook(ma_hook, with_kwargs=True)
    router_w = []
    rhooks = []
    if has_router:
        def mk(lvl):
            def rh(m, i, o): router_w.append((lvl, (o[0] if isinstance(o, (tuple, list)) else o).detach()))
            return rh
        rhooks = [model.routers[l].register_forward_hook(mk(l)) for l in range(len(model.routers))]

    agg = []
    for i in idxs:
        s = ds[i]; sample = {m: s[m].unsqueeze(0).to(dev) for m in MODALS}
        fid = str(s['image_id']); mem_io.clear(); router_w.clear()
        with torch.no_grad():
            feats = seg.extract_det_features([sample['img'], sample['lidar'], sample['thermal']])
            fused, reg = model.extract_fused_fpn(sample)
            out = model(sample); det = out['detections'][0]
            lam = float(seg.lambda_bias.detach().mean()) if has_lam else 0.0
            mem_on = [t.detach().clone() for t in feats['mem']]
            if has_lam:
                old = seg.lambda_bias.detach().clone(); seg.lambda_bias.zero_()
                mem_off = [t.detach().clone() for t in seg.extract_det_features(
                    [sample['img'], sample['lidar'], sample['thermal']])['mem']]
                seg.lambda_bias.copy_(old)
            else:
                mem_off = mem_on

        rec = {'image_id': fid, 'file': s.get('file_name'), 'lambda_bias': lam,
               'n_det': int(len(det['scores'])), 'router_reg': float(reg),
               'gt_labels': [names[int(l)] for l in s['labels']]}
        for mi, m in enumerate(MODALS):
            rec[f'{m}_fpn0'] = feat_stats(feats['fpn0'][mi][0])
            rec[f'{m}_mem'] = feat_stats(feats['mem'][mi][0])
            rec[f'{m}_rbma_cos'] = cos(mem_on[mi][0], mem_off[mi][0])
            rec[f'{m}_rel_mean'] = float(reliability(feats['output'][mi][0]).mean())
            rec[f'{m}_memattn_cos'] = cos(mem_io[mi][0], mem_io[mi][1]) if mi < len(mem_io) and mem_io[mi][0] is not None else float('nan')
        rw = {}
        for lvl, w in router_w[:3]:
            wm = w.float().mean(dim=list(range(1, w.dim())))
            rw[f'level{lvl}'] = [float(x) for x in wm.cpu()]
        rec['router_weights'] = rw
        agg.append(rec)

        # ---- RAW dump for programmatic comparison ----
        if not args.no_raw:
            npz = {'image_id': fid}
            for mi, m in enumerate(MODALS):
                npz[f'mem_{m}'] = mem_on[mi][0].half().cpu().numpy()          # (256,64,64)
                npz[f'rel_{m}'] = reliability(feats['output'][mi][0]).astype('float16')  # (H,W)
            npz['fused_p5'] = fused[-1][0].half().cpu().numpy()               # (256,64,64)
            for k, v in rw.items(): npz[f'router_{k}'] = np.array(v, 'float32')
            npz['det_boxes'] = det['boxes'].cpu().numpy().astype('float32')
            npz['det_scores'] = det['scores'].cpu().numpy().astype('float32')
            npz['det_labels'] = det['class_ids'].cpu().numpy().astype('int32') if 'class_ids' in det else np.array([])
            npz['gt_labels'] = np.array(rec['gt_labels'])
            np.savez_compressed(os.path.join(rawdir, f'{fid}.npz'), **npz)

        # ---- panel ----
        ncol = len(MODALS) + 1; nrow = 4
        fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
        def show(r, c, im, t, cmap=None):
            a = ax[r][c]; a.imshow(im, cmap=cmap); a.set_title(t, fontsize=8); a.axis('off')
        for mi, m in enumerate(MODALS):
            img = s[m].permute(1, 2, 0).cpu().numpy()
            show(0, mi, img if m == 'img' else img[..., 0], f"in:{m}", None if m == 'img' else 'gray')
            show(1, mi, pca_rgb(feats['fpn0'][mi][0]), f"{m} fpn0 PCA")
            show(2, mi, pca_rgb(mem_on[mi][0]), f"{m} mem PCA")
            show(3, mi, reliability(feats['output'][mi][0]), f"{m} reliability", 'viridis')
        show(0, len(MODALS), pca_rgb(fused[-1][0]), "FUSED P5 PCA")
        show(1, len(MODALS), (mem_on[2]-mem_off[2])[0].abs().mean(0).cpu().numpy(), "RBMA |Δmem| thm", 'magma')
        a = ax[2][len(MODALS)]; a.bar(MODALS, rw.get('level2', [0, 0, 0])); a.set_ylim(0, 1); a.set_title("router w P5", fontsize=8); a.tick_params(labelsize=6)
        a = ax[3][len(MODALS)]; a.imshow(s['img'].permute(1, 2, 0).cpu().numpy())
        a.set_title(f"det n={rec['n_det']}  GT=red pred=green", fontsize=7); a.axis('off')
        # GT boxes (red dashed) — same 1024 letterboxed space as predictions
        gt = s['bboxes'].cpu().numpy() if len(s['bboxes']) else np.zeros((0, 4))
        for g in gt:
            x1, y1, x2, y2 = g; a.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', lw=1.4, linestyle='--'))
        sc = det['scores'].cpu().numpy(); bx = det['boxes'].cpu().numpy()
        for j in np.argsort(-sc)[:20]:
            if sc[j] < 0.3: continue
            x1, y1, x2, y2 = bx[j]; a.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', lw=1))
        fig.suptitle(f"img {fid}  λ={lam:.3f}  GT={rec['gt_labels'][:6]}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"panel_{fid}.png"), dpi=110); plt.close(fig)
        print(f"  img {fid}: det={rec['n_det']} routerP5={[round(x,3) for x in rw.get('level2',[0,0,0])]} "
              f"rbma_cos={rec['thermal_rbma_cos']:.3f} memattn_cos={rec['img_memattn_cos']:.3f}", flush=True)
        # free memory between images (avoid cross-image accumulation / fragmentation OOM)
        del feats, fused, out, det, mem_on, mem_off
        mem_io.clear(); router_w.clear()
        if dev.type == 'cuda': torch.cuda.empty_cache()

    h.remove(); [r.remove() for r in rhooks]
    json.dump(agg, open(os.path.join(args.out_dir, 'probe_stats.json'), 'w'), indent=1)

    # ---- aggregate summary figure ----
    import statistics as stt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    rP5 = np.array([[r['router_weights'].get('level2', [0, 0, 0])[k] for k in range(3)] for r in agg])
    ax[0][0].boxplot([rP5[:, 0], rP5[:, 1], rP5[:, 2]], labels=MODALS); ax[0][0].set_title("router P5 weight (fusion)"); ax[0][0].set_ylabel("weight")
    memrank = {m: [r[f'{m}_mem']['pca_evr_top3'][0] for r in agg] for m in MODALS}
    ax[0][1].boxplot([memrank[m] for m in MODALS], labels=MODALS); ax[0][1].set_title("mem PCA top-1 EVR (↑=degenerate)")
    for k, lab, col in [('_rbma_cos', 'RBMA on/off', 'r'), ('_memattn_cos', 'mem-attn before/after', 'b')]:
        vals = [stt.mean([r[f'{m}{k}'] for r in agg if r[f'{m}{k}'] == r[f'{m}{k}']]) for m in MODALS]
        ax[0][2].plot(MODALS, vals, 'o-', color=col, label=lab)
    ax[0][2].set_title("cosine (1=no change, 0=orthogonal)"); ax[0][2].legend(fontsize=8); ax[0][2].set_ylim(-0.05, 1.05)
    fnorm = {m: [r[f'{m}_fpn0']['l2_mean'] for r in agg] for m in MODALS}
    ax[1][0].boxplot([fnorm[m] for m in MODALS], labels=MODALS); ax[1][0].set_title("fpn0 L2 norm")
    ax[1][1].hist([r['n_det'] for r in agg], bins=15); ax[1][1].set_title("#detections / image")
    rel = {m: [r[f'{m}_rel_mean'] for r in agg] for m in MODALS}
    ax[1][2].boxplot([rel[m] for m in MODALS], labels=MODALS); ax[1][2].set_title("per-modal reliability mean")
    fig.suptitle(f"P30-Det ep{ck.get('epoch')} feature summary over {len(agg)} images", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'summary.png'), dpi=120); plt.close(fig)
    print("saved:", args.out_dir, "| panels+raw+summary+stats  images:", len(agg))


if __name__ == '__main__':
    main()
