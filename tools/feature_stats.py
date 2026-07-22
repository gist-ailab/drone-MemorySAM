#!/usr/bin/env python3
"""
tools/feature_stats.py — [분석항목 2] 모달리티별 추출 피쳐의 전체-테스트셋 수치 통계
— model-agnostic, P31/32/33/34+ 재사용. (per-image 정성 패널은 tools/viz_features.py)

Aggregates, over the FULL test set (or --max-imgs cap) per condition, numeric statistics
of each modality's encoder feature (`_last_per_modal_feats` = encoder raw output, full
embed_dim e.g. 1024 — NOT fpn 32ch) and the fused feature (m_feat, model return[1]).
[§0.5] Optional extra taps `_last_fused_postfusion`(T3) / `_last_fused_prehead`(T5)
appear as `FUSED_pf`/`PREHEAD` when the model stashes them (disable with --no-extra-taps):

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


def cov_spectrum(X, max_n=8000):
    """(N,D) 표본 → 공분산 고유값(내림차순, 음수 클립). effective_rank·PCA가 **공용**으로 쓰도록
    한 번만 계산한다. 표본 행렬 SVD(np.linalg.svd of (N,D))는 N=15000·D=1024에서 feature당
    수백 초라 못 쓴다 — cov는 D×D 고유분해라 훨씬 싸다. N이 크면 max_n로 subsample
    (D보다 충분히 크면 스펙트럼 안정; 결정적 seed)."""
    if X.shape[0] > max_n:
        idx = np.random.default_rng(0).choice(X.shape[0], max_n, replace=False)
        X = X[idx]
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(Xc.T) + 1e-12 * np.eye(X.shape[1]))
    return np.clip(ev[::-1], 0.0, None)   # 내림차순


def participation_ratio(ev):
    """Effective rank from covariance eigenvalues: (Σλ)²/Σλ²."""
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def activation_dist(X):
    """[§0.5 activation 분포] X:(N,D) sampled activations →
    sparsity(%|act|<eps of dataset scale), excess kurtosis(평균, 무거운 꼬리=포화/스파이크)."""
    scale = float(np.abs(X).mean()) + 1e-8
    sparsity = float((np.abs(X) < 0.01 * scale).mean())        # 데이터 스케일 대비 ≈0 비율
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    z = (X - mu) / sd
    kurt = float((z ** 4).mean() - 3.0)                        # per-dim 정규화 후 평균 excess kurtosis
    return round(sparsity, 4), round(kurt, 3)


def pca_stats(ev, k=5, var_target=0.90):
    """[§0.5 PCA 정량] 공분산 고유값(내림차순) → top-k 설명분산비 + 내재차원(누적 var_target 도달 성분수).
    cov 고유값 λ_i = SVD 특이값² 비례이므로 설명분산비는 SVD와 동일(단 훨씬 저렴). scatter(viz)와 달리
    '정보의 실제 차원'을 스칼라로."""
    tot = float(ev.sum())
    if tot <= 1e-12:            # 퇴화(all-zero fallback) bank → 내재차원 의미 없음
        return {'expvar_topk': [], 'intrinsic_dim': None}
    ratio = ev / tot
    cum = np.cumsum(ratio)
    intrinsic = int(np.searchsorted(cum, var_target) + 1)      # var_target 도달에 필요한 성분 수
    return {'expvar_topk': [round(float(r), 4) for r in ratio[:k]],
            'intrinsic_dim': intrinsic}


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
    ap.add_argument('--no-extra-taps', action='store_true',
                    help='[§0.5] T3(post-fusion)/T5(pre-head) stash 특성화 비활성 (기본=자동 포함)')
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
        n_extra = 0; extra_names = []   # [§0.5] 첫 이미지에서 확정 (변수 존재 보장; n=0 condition은 기존과 동일 미지원)
        norm_sum = None          # (M+1,) mean ||f|| accumulator (last = fused)
        chan_absmean = None      # (M+1, C) dataset-wide per-channel mean|act|
        samples = None           # list of per-modal sample banks for rank/CKA
        fused_cos_sum = None     # (M,) cos(m_feat, f_i)
        for idx in range(n):
            images, label, _ = dataset[idx]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            with torch.no_grad():
                m_out = model(imgs, multimask_output=True)
            m_feat = m_out[1][0].float()                    # (C, h, w) — decode 피쳐 (기존 'FUSED')
            feats = [t[0].to(device).float() for t in core._last_per_modal_feats]
            # [§0.5] T3 post-fusion / T5 pre-head tap — stash 있을 때만, per-modal 뒤·m_feat 앞에 삽입
            extra, extra_nm = [], []
            if not args.no_extra_taps:
                for attr, nm in (('_last_fused_postfusion', 'FUSED_pf'),
                                 ('_last_fused_prehead', 'PREHEAD')):
                    t = getattr(core, attr, None)
                    if t is not None:
                        extra.append(t[0].to(device).float()); extra_nm.append(nm)
            if M is None:
                M = len(feats); Cch = feats[0].shape[0]
                n_extra = len(extra); extra_names = list(extra_nm)
                norm_sum = np.zeros(M + n_extra + 1)
                # 채널 수는 feature마다 다를 수 있음 (per-modal 1024 vs fused 256)
                chan_absmean = ([np.zeros(f.shape[0]) for f in feats]
                                + [np.zeros(e.shape[0]) for e in extra]
                                + [np.zeros(m_feat.shape[0])])
                samples = [[] for _ in range(M + n_extra + 1)]
                fused_cos_sum = np.zeros(M)
            if len(extra) != n_extra:   # stash가 이미지마다 달라지면(비정상) 정렬 붕괴 방지
                raise RuntimeError(f"extra-tap count changed {n_extra}->{len(extra)} at img {idx}")
            h, w = feats[0].shape[-2:]
            pos = rng.integers(0, h * w, size=args.samples_per_img)
            for i, f in enumerate(feats + extra + [m_feat]):
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
        names = ([modals[i] if i < len(modals) else f'mod{i}' for i in range(M)]
                 + extra_names + ['FUSED'])
        per = {}
        for i, name in enumerate(names):
            ch = chan_absmean[i] / n
            spars, kurt = activation_dist(banks[i])
            ev = cov_spectrum(banks[i])          # rank·PCA 공용 스펙트럼(한 번만)
            pcs = pca_stats(ev)
            per[name] = {
                'mean_feat_norm': round(norm_sum[i] / n, 4),
                'dead_channels': int((ch < 0.01 * max(ch.max(), 1e-8)).sum()),
                'total_channels': int(len(ch)),
                'effective_rank': round(participation_ratio(ev), 2),
                'sparsity': spars,                        # [§0.5] 데이터 스케일 대비 ≈0 비율
                'kurtosis': kurt,                         # [§0.5] 포화/스파이크(무거운 꼬리)
                'intrinsic_dim90': pcs['intrinsic_dim'],  # [§0.5] 누적분산 90% 성분수
                'pca_expvar_top5': pcs['expvar_topk'],
            }
            if i < M:
                per[name]['cos_with_fused'] = round(fused_cos_sum[i] / n, 4)
        cka = {}
        for i in range(M):
            for j in range(i + 1, M):
                k = f"{names[i]}~{names[j]}"
                nmin = min(len(banks[i]), len(banks[j]))
                cka[k] = round(linear_cka(banks[i][:nmin], banks[j][:nmin]), 4)
        # [§0.5 stage CKA] per-modal→PREHEAD(전 스택이 모달 정보를 얼마나 바꿨나) +
        # FUSED_pf→PREHEAD(제안 모듈이 fused를 바꿨나; ≈1이면 피쳐-레벨 no-op). 매칭 픽셀·매칭 표본이라
        # 차원 달라도(1024 vs 256) linear CKA 유효.
        nidx = {nm: k for k, nm in enumerate(names)}
        def _stage_cka(a, b):
            if a not in nidx or b not in nidx:
                return None
            ia, ib = nidx[a], nidx[b]
            nmin = min(len(banks[ia]), len(banks[ib]))
            if nmin < 2:
                return None
            return round(linear_cka(banks[ia][:nmin], banks[ib][:nmin]), 4)
        stage = {}
        for i in range(M):
            v = _stage_cka(names[i], 'PREHEAD')
            if v is not None:
                stage[f"{names[i]}~PREHEAD"] = v
        for a, b in (('FUSED_pf', 'PREHEAD'), ('FUSED_pf', 'FUSED')):
            v = _stage_cka(a, b)
            if v is not None:
                stage[f"{a}~{b}"] = v
        report['conditions'][cond] = {'n': n, 'per_feature': per,
                                      'cross_modal_cka': cka, 'stage_cka': stage}
        viz_bank[cond] = banks
        print(f"[feature_stats] {cond}: n={n} " + '  '.join(
            f"{k}:rank={v['effective_rank']},dead={v['dead_channels']}" for k, v in per.items()), flush=True)

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Feature statistics — `{report['model']}` (full-testset per-modal numeric)",
             "- 읽는 법: `eff.rank`↓·`idim90`↓=피쳐 저차원 붕괴, `dead`↑·`sparsity`↑=용량 낭비, "
             "`kurt`↑=포화/스파이크, `cka`↑(→1)=모달 간 중복(상보성 없음), `cos(fused)`=융합 기여 방향.",
             "- tap: per-modal=T0(encoder raw), `FUSED_pf`=T3(fusion 직후), `PREHEAD`=T5(head 직전), `FUSED`=decode 피쳐.",
             "- `stage_cka`(→PREHEAD)↓=그 스테이지가 피쳐를 많이 바꿈; `FUSED_pf~PREHEAD`≈1 = 제안 모듈이 피쳐-레벨 no-op.", ""]
    for cond, cd in report['conditions'].items():
        lines += [f"## {cond} (n={cd['n']})",
                  "| feature | mean‖f‖ | dead ch | eff.rank | sparsity | kurt | idim90 | cos(fused) |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, v in cd['per_feature'].items():
            lines.append(f"| {k} | {v['mean_feat_norm']} | {v['dead_channels']}/{v['total_channels']} "
                         f"| {v['effective_rank']} | {v.get('sparsity', '—')} | {v.get('kurtosis', '—')} "
                         f"| {v.get('intrinsic_dim90', '—')} | {v.get('cos_with_fused', '—')} |")
        lines += ["", "cross-modal CKA: " + ', '.join(f"{k}={v}" for k, v in cd['cross_modal_cka'].items())]
        if cd.get('stage_cka'):
            lines += ["stage CKA: " + ', '.join(f"{k}={v}" for k, v in cd['stage_cka'].items())]
        lines += [""]
    Path(args.out + '.md').write_text('\n'.join(lines))

    if args.viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        conds = list(viz_bank.keys())
        fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4.5), squeeze=False)
        for ax, cond in zip(axes[0], conds):
            banks = viz_bank[cond]
            # per-modal(T0)만 PCA scatter — extra tap(FUSED_pf/PREHEAD)·FUSED는 차원이 달라
            # (256 vs 1024) 같은 공간에 못 섞는다. banks[:M] = per-modal only.
            per_modal = banks[:M]
            allv = np.concatenate([b[:2000] for b in per_modal], 0)
            mu = allv.mean(0); Xc = allv - mu
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            for i, b in enumerate(per_modal):
                p = (b[:2000] - mu) @ Vt[:2].T
                nm = modals[i] if i < len(modals) else f'mod{i}'
                ax.scatter(p[:, 0], p[:, 1], s=2, alpha=0.3, label=nm)
            ax.set_title(f'{cond} — per-modal feature PCA'); ax.legend(markerscale=4)
        fig.tight_layout(); fig.savefig(args.out + '_pca.png', dpi=120); plt.close(fig)
        print(f"[feature_stats] wrote {args.out}_pca.png")
    print(f"[feature_stats] wrote {args.out}.json / .md")


if __name__ == '__main__':
    main()
