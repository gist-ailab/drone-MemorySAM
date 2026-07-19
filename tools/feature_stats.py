#!/usr/bin/env python3
"""
tools/feature_stats.py — [분석항목 2] 모달리티별 추출 피쳐의 전체-테스트셋 수치 통계
— model-agnostic, P31/32/33/34+ 재사용. (per-image 정성 패널은 tools/viz_features.py)

Aggregates, over the FULL test set (or --max-imgs cap) per condition, numeric statistics
of each modality's encoder feature (`_last_per_modal_feats`, fpn[0] 32ch) and the fused
feature (m_feat, model return[1]):

  A. feature magnitude   : mean ||f||, per-channel mean|act| → DEAD CHANNELS
                           (channel whose dataset-wide activation ≈ 0 → 용량 낭비)
  B. effective rank      : participation ratio of the feature covariance eigenvalues
                           (낮으면 그 모달 피쳐가 저차원으로 붕괴 = 정보 부족)
  C. cross-modal CKA     : linear CKA between modal features at matched pixels
                           (1에 가까우면 모달들이 같은 걸 보는 것 = 상보성 없음)
  D. fused-feature stats : same A/B for m_feat + per-modal cos(m_feat, f_i)
                           (융합이 어느 모달 쪽으로 기울었는지)

Optional --viz: 2D PCA scatter of sampled feature vectors colored by modality
(모달 피쳐 공간이 분리돼 있는지 한 장으로).

Usage:
  python tools/feature_stats.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --conditions cloud,fog,night,rain,sun \
    --max-imgs -1 --samples-per-img 256 --gpu 0 --out <prefix> [--viz]

Output: <out>.json + <out>.md [+ <out>_pca.png]
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V


def linear_cka(X, Y):
    """X, Y: (N, D) — unbiased-ish linear CKA."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    nx = np.linalg.norm(X.T @ X, 'fro')
    ny = np.linalg.norm(Y.T @ Y, 'fro')
    return float(hsic / max(nx * ny, 1e-12))


def participation_ratio(X):
    """Effective rank of (N, D) samples: (Σλ)²/Σλ² of covariance eigvals."""
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(Xc.T) + 1e-12 * np.eye(X.shape[1]))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='cloud,fog,night,rain,sun')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--max-imgs', type=int, default=-1, help='-1 = full test set')
    ap.add_argument('--samples-per-img', type=int, default=256)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--viz', action='store_true')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root:
        ds_cfg['ROOT'] = args.dataset_root
    if isinstance(ds_cfg.get('PHYSAUG'), dict):
        ds_cfg['PHYSAUG']['ENABLE'] = False
    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(isz, dataset_cfg=ds_cfg)
    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model
    modals = ds_cfg.get('MODALS', [])
    rng = np.random.default_rng(0)

    report = {'model': Path(args.model_path).stem, 'modals': modals, 'conditions': {}}
    viz_bank = {}
    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, args.split, transform, args.split, macvi=False, eval_day=False)
        n = len(dataset) if args.max_imgs < 0 else min(args.max_imgs, len(dataset))
        M = None
        Cch = None
        norm_sum = None          # (M+1,) mean ||f|| accumulator (last = fused)
        chan_absmean = None      # (M+1, C) dataset-wide per-channel mean|act|
        samples = None           # list of per-modal sample banks for rank/CKA
        fused_cos_sum = None     # (M,) cos(m_feat, f_i)
        for idx in range(n):
            images, label, _ = dataset[idx]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            with torch.no_grad():
                m_out = model(imgs, multimask_output=True)
            m_feat = m_out[1][0].float()                    # (C, h, w)
            feats = [t[0].to(device).float() for t in core._last_per_modal_feats]
            if M is None:
                M = len(feats); Cch = feats[0].shape[0]
                norm_sum = np.zeros(M + 1)
                # 채널 수는 feature마다 다를 수 있음 (예: ReliaDINO per-modal 1024 vs fused 256)
                chan_absmean = [np.zeros(f.shape[0]) for f in feats] + [np.zeros(m_feat.shape[0])]
                samples = [[] for _ in range(M + 1)]
                fused_cos_sum = np.zeros(M)
            h, w = feats[0].shape[-2:]
            pos = rng.integers(0, h * w, size=args.samples_per_img)
            for i, f in enumerate(feats + [m_feat]):
                if f.shape[-2:] != (h, w):
                    f = F.interpolate(f[None], size=(h, w), mode='bilinear', align_corners=False)[0]
                norm_sum[i] += f.norm().item() / (h * w) ** 0.5
                chan_absmean[i] += f.abs().mean(dim=(1, 2)).cpu().numpy()
                fl = f.reshape(f.shape[0], -1)[:, pos].T.cpu().numpy()   # (S, C)
                if len(samples[i]) * args.samples_per_img < 20000:       # cap bank
                    samples[i].append(fl)
                if i < M:
                    mf = m_feat.reshape(m_feat.shape[0], -1)[:, pos].T
                    ff = torch.as_tensor(fl, device=device)
                    if mf.shape[1] == ff.shape[1]:
                        fused_cos_sum[i] += F.cosine_similarity(mf, ff, dim=1).mean().item()
        banks = [np.concatenate(sl, 0) if sl else np.zeros((1, len(chan_absmean[i])))
                 for i, sl in enumerate(samples)]
        names = [modals[i] if i < len(modals) else f'mod{i}' for i in range(M)] + ['FUSED']
        per = {}
        for i, name in enumerate(names):
            ch = chan_absmean[i] / n
            per[name] = {
                'mean_feat_norm': round(norm_sum[i] / n, 4),
                'dead_channels': int((ch < 0.01 * max(ch.max(), 1e-8)).sum()),
                'total_channels': int(len(ch)),
                'effective_rank': round(participation_ratio(banks[i]), 2),
            }
            if i < M:
                per[name]['cos_with_fused'] = round(fused_cos_sum[i] / n, 4)
        cka = {}
        for i in range(M):
            for j in range(i + 1, M):
                k = f"{names[i]}~{names[j]}"
                nmin = min(len(banks[i]), len(banks[j]))
                cka[k] = round(linear_cka(banks[i][:nmin], banks[j][:nmin]), 4)
        report['conditions'][cond] = {'n': n, 'per_feature': per, 'cross_modal_cka': cka}
        viz_bank[cond] = banks
        print(f"[feature_stats] {cond}: n={n} " + '  '.join(
            f"{k}:rank={v['effective_rank']},dead={v['dead_channels']}" for k, v in per.items()), flush=True)

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Feature statistics — `{report['model']}` (full-testset per-modal numeric)",
             "- 읽는 법: `effective_rank`↓=피쳐 붕괴, `dead_channels`↑=용량 낭비, "
             "`cka`↑(→1)=모달 간 중복(상보성 없음), `cos_with_fused`=융합 기여 방향.", ""]
    for cond, cd in report['conditions'].items():
        lines += [f"## {cond} (n={cd['n']})",
                  "| feature | mean‖f‖ | dead ch | eff.rank | cos(fused) |", "|---|---|---|---|---|"]
        for k, v in cd['per_feature'].items():
            lines.append(f"| {k} | {v['mean_feat_norm']} | {v['dead_channels']}/{v['total_channels']} "
                         f"| {v['effective_rank']} | {v.get('cos_with_fused', '—')} |")
        lines += ["", "cross-modal CKA: " + ', '.join(f"{k}={v}" for k, v in cd['cross_modal_cka'].items()), ""]
    Path(args.out + '.md').write_text('\n'.join(lines))

    if args.viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        conds = list(viz_bank.keys())
        fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4.5), squeeze=False)
        for ax, cond in zip(axes[0], conds):
            banks = viz_bank[cond]
            allv = np.concatenate([b[:2000] for b in banks[:-1]], 0)
            mu = allv.mean(0); Xc = allv - mu
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            for i, b in enumerate(banks[:-1]):
                p = (b[:2000] - mu) @ Vt[:2].T
                nm = modals[i] if i < len(modals) else f'mod{i}'
                ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.3, label=nm)
            ax.set_title(f'{cond} — per-modal feature PCA'); ax.legend(markerscale=4)
        fig.tight_layout(); fig.savefig(args.out + '_pca.png', dpi=120); plt.close(fig)
        print(f"[feature_stats] wrote {args.out}_pca.png")
    print(f"[feature_stats] wrote {args.out}.json / .md")


if __name__ == '__main__':
    main()
