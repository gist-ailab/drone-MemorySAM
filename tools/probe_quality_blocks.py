#!/usr/bin/env python3
"""[P54-Q1b] 블록별 품질 헤드 프로브 — 어느 DINOv3 블록이 광학 열화를 인지하나.

최종 블록 토큰 위 품질 헤드(tools/probe_quality_head.py, "Q1b-2")는 결측/가림 모달은
거의 완벽히(AUROC ~1.0) 잡았지만 광학 RGB 열화(블러 0.67·감마 0.71·색이동 0.73·
가우시안 노이즈 0.71)는 못 잡았다. 가설: 초기·중기 블록이 저수준 통계(밝기·블러·질감)를
더 보존하므로, 블록 k 토큰을 읽는 품질 헤드가 광학 열화를 더 잘 인지하고 낮/밤도 더 잘
가른다. 본 모델은 동결, 작은 헤드만 학습한다.

- 요청 블록마다 `model.encoder.backbone.blocks[k-1]` 에 hook → 한 forward 로 전 블록·전
  모달 출력을 잡아 인코더 tap 경로와 동일하게 (B,C,h,w) 변환(prefix 토큰 제거).
- 융합 직전 feats(기존 Q1 입력)도 `fusion_in` 참조 항목으로 포함.
- 특징원마다 QualityHead 1개를 같은 열화 배치(같은 Degrader·시드)로 동시 학습.
- `--daynight`: DELIVER 조건(cloud/fog/night/rain/sun)을 경로에서 복원, RGB clean 블록
  토큰 mean-pool 특징으로 로지스틱 회귀(밤 vs 나머지 AUROC·5-way 정확도).

  python tools/probe_quality_blocks.py --cfg <cfg> --ckpt <ckpt> --blocks 2 4 6 12 18 24
  python tools/probe_quality_blocks.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets.degrade import Degrader                                    # noqa: E402
from semseg.models.reliadino.quality_head import QualityHead                    # noqa: E402
from tools.probe_quality_head import (                                          # noqa: E402
    rank_auroc, per_operator_stats, build_loader, load_frozen_model,
    train_step, FusionInputHook)

CONDS = ('cloud', 'fog', 'night', 'rain', 'sun')     # DELIVER 5조건(day/night 진단용)


# ===========================================================================
# 순수 헬퍼 — 테스트가 직접 부른다
# ===========================================================================
def tokens_to_map(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """블록 출력 (B,N,C)/(B,h,w,C) → (B,C,h,w). prefix(cls/register) 토큰 제거.

    인코더 `FrozenViTEncoder._to_map`(encoder.py:617)과 동일: 뒤 h*w 토큰만 남긴다.
    """
    if t.dim() == 4:
        return t.permute(0, 3, 1, 2).contiguous()
    t = t[:, t.shape[1] - h * w:]                    # prefix(cls/reg) 토큰 제거
    return t.transpose(1, 2).reshape(t.shape[0], -1, h, w)


def map_captures_to_modalities(caps, M):
    """순차 forward 로 잡힌 블록 출력 리스트를 모달 순서로 매핑한다.

    비-CMLC 경로는 `encoder(x[i], i)` 를 i=0..M-1 순차 호출 → hook 이 forward 당 M 번
    발화, 그 순서가 모달 순서. 개수가 M 과 다르면(CMLC 배치·추가 forward) 실패시킨다.
    """
    assert len(caps) == M, f"블록 캡처 {len(caps)}개 != 모달 수 M={M} (순차 forward 가정 위반)"
    return list(caps)


def _standardize(Xtr, Xva):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xva - mu) / sd


def fit_logreg(Xtr, ytr, Xva, n_classes, steps=400, lr=0.1, wd=1e-3):
    """torch 로 구현한 다항 로지스틱 회귀. 학습 통계로 표준화 후 val 로짓 반환."""
    Xtr, Xva = _standardize(Xtr.float(), Xva.float())
    W = torch.zeros(Xtr.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(Xtr @ W + b, ytr.long()).backward()
        opt.step()
    with torch.no_grad():
        return Xva @ W + b


def logreg_auroc(Xtr, ytr, Xva, yva, steps=400):
    """이진 로지스틱 회귀 후 val AUROC(로짓 차이를 점수로)."""
    logits = fit_logreg(Xtr, ytr, Xva, 2, steps=steps)
    score = (logits[:, 1] - logits[:, 0]).numpy()
    return rank_auroc(score, np.asarray(yva).astype(np.int64))


def cond_of(path: str):
    """DELIVER 경로에서 조건(cloud/fog/night/rain/sun)을 복원. 못 찾으면 None."""
    p = path.lower()
    for c in CONDS:
        if f'/{c}/' in p or f'\\{c}\\' in p:
            return c
    for c in CONDS:                                  # 폴백: 부분 문자열
        if c in p:
            return c
    return None


# ===========================================================================
# 특징원 통계 — probe_quality_head.evaluate 의 누적 로직을 특징원별로 공유
# ===========================================================================
def _new_acc(M):
    return dict(tok_scores=[[] for _ in range(M)], tok_labels=[[] for _ in range(M)],
                tok_op=[[] for _ in range(M)], sev_pred=[], sev_true=[],
                sev_mod=[], sev_op=[])


def _accumulate(acc, pred, labels, M, device):
    eta_t = pred['eta_token'].float()
    mask16 = labels['mask16'].float().to(device)
    if eta_t.shape[-2:] != mask16.shape[-2:]:
        B = eta_t.shape[0]
        eta_t = F.interpolate(eta_t.reshape(B * M, 1, *eta_t.shape[-2:]),
                              size=mask16.shape[-2:], mode='bilinear',
                              align_corners=False).reshape(B, M, *mask16.shape[-2:])
    op = labels['op'].to(device)
    B = eta_t.shape[0]
    for m in range(M):
        et_flat = eta_t[:, m].reshape(B, -1)
        op_patch = op[:, m].unsqueeze(1).expand(-1, et_flat.shape[1])
        acc['tok_scores'][m].append(et_flat.reshape(-1).cpu().numpy())
        acc['tok_labels'][m].append(mask16[:, m].reshape(-1).cpu().numpy())
        acc['tok_op'][m].append(op_patch.reshape(-1).cpu().numpy())
    presence = labels['presence'].to(device)
    severity = labels['severity'].to(device)
    eta_s = pred['eta_scalar'].float()
    pm = presence > 0.5
    for m in range(M):
        sel = pm[:, m]
        if sel.any():
            acc['sev_pred'].append(eta_s[sel, m].cpu().numpy())
            acc['sev_true'].append(severity[sel, m].cpu().numpy())
            acc['sev_mod'].append(np.full(int(sel.sum()), m))
            acc['sev_op'].append(op[sel, m].cpu().numpy())


def _finalize(acc, modal_names):
    M = len(modal_names)

    def _auroc(mods):
        s = np.concatenate([np.concatenate(acc['tok_scores'][m]) for m in mods])
        l = np.concatenate([np.concatenate(acc['tok_labels'][m]) for m in mods])
        return rank_auroc(s, l)

    auroc_per_modal = {modal_names[m]: _auroc([m]) for m in range(M)}
    sp = np.concatenate(acc['sev_pred']) if acc['sev_pred'] else np.zeros(0)
    st = np.concatenate(acc['sev_true']) if acc['sev_true'] else np.zeros(0)
    sm = np.concatenate(acc['sev_mod']) if acc['sev_mod'] else np.zeros(0)
    mae_per_modal = {modal_names[m]: (float(np.mean(np.abs(sp[sm == m] - st[sm == m])))
                                      if (sm == m).any() else float('nan'))
                     for m in range(M)}
    return dict(
        auroc=_auroc(range(M)),
        auroc_per_modal=auroc_per_modal,
        severity_mae=float(np.mean(np.abs(sp - st))) if sp.size else float('nan'),
        severity_mae_per_modal=mae_per_modal,
        op_table=per_operator_stats(acc['tok_scores'], acc['tok_labels'], acc['tok_op'],
                                    acc['sev_pred'], acc['sev_true'], acc['sev_mod'],
                                    acc['sev_op'], modal_names))


# ===========================================================================
# 멀티 특징원 캡처 — 한 forward 로 전 블록 + 융합입력을 잡는다
# ===========================================================================
class BlockTapHooks:
    """요청 블록마다 forward-hook. 발화마다(=모달마다) 출력을 리스트로 쌓는다."""

    def __init__(self, model, blocks):
        self.blocks = list(blocks)
        self.buf = {k: [] for k in self.blocks}
        self.handles = []
        for k in self.blocks:
            self.handles.append(
                model.encoder.backbone.blocks[k - 1].register_forward_hook(self._mk(k)))

    def _mk(self, k):
        def hook(_m, _i, out):
            self.buf[k].append(out[0] if isinstance(out, tuple) else out)
        return hook

    def clear(self):
        for k in self.blocks:
            self.buf[k].clear()

    def remove(self):
        for h in self.handles:
            h.remove()


class MultiSource:
    """융합입력 pre-hook + 블록 hook 을 묶어 한 forward 로 전 특징원을 캡처."""

    def __init__(self, model, blocks):
        self.model = model
        self.M = model.num_modalities
        self.fusion = FusionInputHook(model)
        self.bh = BlockTapHooks(model, blocks)
        self.blocks = list(blocks)

    @torch.no_grad()
    def capture(self, batched_input):
        self.fusion.feats = None
        self.bh.clear()
        self.model(batched_input)
        assert self.fusion.feats is not None, "fusion hook 이 feats 를 못 잡았다"
        fin = list(self.fusion.feats)                # M x (B,C,h,w)
        h, w = fin[0].shape[-2:]
        out = {'fusion_in': fin}
        for k in self.blocks:
            caps = map_captures_to_modalities(self.bh.buf[k], self.M)
            out[f'block{k}'] = [tokens_to_map(t, h, w) for t in caps]
        return out

    def remove(self):
        self.fusion.remove()
        self.bh.remove()


def evaluate_sources(heads, loader, degrader, srcs, modal_names, device, mgr,
                     subset_every=1, limit=0):
    for hd in heads.values():
        hd.eval()
    M = len(modal_names)
    accs = {s: _new_acc(M) for s in srcs}
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        if bi % subset_every != 0:
            continue
        images = [x.to(device) for x in batch[0]]
        deg, labels = degrader(images, modal_names)
        feats_by_src = mgr.capture(deg)
        with torch.no_grad():
            for s in srcs:
                _accumulate(accs[s], heads[s](feats_by_src[s]), labels, M, device)
    return {s: _finalize(accs[s], modal_names) for s in srcs}


# ===========================================================================
# day/night 진단 — RGB clean 블록 토큰 mean-pool → 로지스틱 회귀
# ===========================================================================
def _collect_daynight(mgr, loader, files, img_idx, blocks, device, limit):
    assert len(files) == len(loader.dataset), \
        f"파일 목록 {len(files)} != 데이터셋 {len(loader.dataset)} (순서 정합 실패)"
    feats = {k: [] for k in blocks}
    conds, idx = [], 0
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        images = [x.to(device) for x in batch[0]]
        B = images[0].shape[0]
        cap = mgr.capture(images)                    # clean(열화 없음)
        for k in blocks:
            fm = cap[f'block{k}'][img_idx]           # (B,C,h,w)
            feats[k].append(fm.mean(dim=(-1, -2)).cpu())
        for j in range(B):
            conds.append(cond_of(files[idx + j]))
        idx += B
    return {k: torch.cat(v) for k, v in feats.items()}, conds


def daynight_probe(mgr, cfg, blocks, modal_names, device, batch_size, limit):
    img_idx = modal_names.index('img') if 'img' in modal_names else 0
    tr = build_loader(cfg, 'train', batch_size, train_aug=False)   # shuffle=False
    va = build_loader(cfg, 'val', batch_size, train_aug=False)
    if not hasattr(tr.dataset, 'files') or not hasattr(va.dataset, 'files'):
        print("[daynight] 데이터셋이 .files 를 노출하지 않음 → 진단 생략")
        return None
    Xtr_b, ctr = _collect_daynight(mgr, tr, list(tr.dataset.files), img_idx, blocks, device, limit)
    Xva_b, cva = _collect_daynight(mgr, va, list(va.dataset.files), img_idx, blocks, device, limit)
    ci = {c: i for i, c in enumerate(CONDS)}
    mtr = [i for i, c in enumerate(ctr) if c in ci]
    mva = [i for i, c in enumerate(cva) if c in ci]
    ytr = torch.tensor([ci[ctr[i]] for i in mtr])
    yva = torch.tensor([ci[cva[i]] for i in mva])
    out = {}
    for k in blocks:
        Xtr = Xtr_b[k][mtr]
        Xva = Xva_b[k][mva]
        night_tr = (ytr == ci['night']).long()
        night_va = (yva == ci['night']).long()
        au = logreg_auroc(Xtr, night_tr, Xva, night_va)
        logits = fit_logreg(Xtr, ytr, Xva, len(CONDS))
        acc = float((logits.argmax(1) == yva).float().mean().item())
        out[f'block{k}'] = {'night_auroc': au, 'cond_acc': acc,
                            'n_train': len(mtr), 'n_val': len(mva)}
        print(f"[daynight] block{k}: night_AUROC={au:.3f} 5way_acc={acc:.3f}")
    return out


# ===========================================================================
# 마크다운 표
# ===========================================================================
def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) and v == v else '-'


def build_markdown(res_train, res_held, srcs, modal_names, daynight):
    non_img = [m for m in modal_names if m != 'img']
    cols = (['blur', 'gamma', 'color_shift', 'gauss_noise(H)', 'salt_pepper(H)', 'rgb_missing']
            + [f'{m}_auroc' for m in non_img] + ['sev_mae'])
    lines = ['| source | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for s in srcs:
        it = res_train[s]['op_table'].get('img', {})
        ih = res_held[s]['op_table'].get('img', {})
        row = [_fmt(it.get('gaussian_blur', {}).get('auroc')),
               _fmt(it.get('gamma_gain', {}).get('auroc')),
               _fmt(it.get('color_shift', {}).get('auroc')),
               _fmt(ih.get('gaussian_noise', {}).get('auroc')),
               _fmt(ih.get('salt_pepper', {}).get('auroc')),
               _fmt(it.get('missing', {}).get('auroc'))]
        row += [_fmt(res_train[s]['auroc_per_modal'].get(m)) for m in non_img]
        row.append(_fmt(res_train[s]['severity_mae']))
        lines.append(f"| {s} | " + ' | '.join(row) + ' |')
    md = '\n'.join(lines)
    if daynight:
        md += '\n\n## day/night (RGB clean 블록 특징)\n'
        md += '| source | night_vs_rest_auroc | 5way_cond_acc |\n|---|---|---|\n'
        for s, dn in daynight.items():
            md += f"| {s} | {_fmt(dn['night_auroc'])} | {_fmt(dn['cond_acc'])} |\n"
    return md


def _write_out(out, srcs, modal_names, res_train, res_held, daynight, extra):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(sources=srcs, modal_names=modal_names, train=res_train,
                   heldout=res_held, daynight=daynight, **extra)
    (out / 'block_probe_summary.json').write_text(json.dumps(summary, indent=2))
    md = build_markdown(res_train, res_held, srcs, modal_names, daynight)
    (out / 'block_probe_table.md').write_text(md)
    print(md)
    print(f"\n→ {out / 'block_probe_summary.json'}")
    return summary


# ===========================================================================
# 드라이런 — 무작위 텐서로 멀티헤드 학습 + json 쓰기 경로
# ===========================================================================
def dry_run(out, blocks, modal_names, hidden=32):
    dim, M, B, h, w = 64, len(modal_names), 2, 8, 8
    srcs = ['fusion_in'] + [f'block{k}' for k in blocks]
    heads = {s: QualityHead(dim, M, hidden=hidden) for s in srcs}
    opts = {s: torch.optim.AdamW(heads[s].parameters(), lr=1e-3) for s in srcs}
    print(f"[dry_run] 특징원 {len(srcs)}개 멀티헤드 학습 2 스텝")
    for step in range(2):
        _, labels = Degrader(seed=step)([torch.randn(B, 3, h * 16, w * 16) for _ in range(M)],
                                        modal_names)
        for s in srcs:
            train_step(heads[s], opts[s], [torch.randn(B, dim, h, w) for _ in range(M)], labels)
        print(f"  step{step}: ok")
    dev = torch.device('cpu')
    res = {False: {}, True: {}}
    for hd_flag in (False, True):
        for s in srcs:
            acc = _new_acc(M)
            for st, p in enumerate((1.0, 0.0)):
                _, labels = Degrader(cfg={'p_per_modal': p}, seed=100 + st,
                                     heldout=hd_flag)(
                    [torch.randn(B, 3, h * 16, w * 16) for _ in range(M)], modal_names)
                with torch.no_grad():
                    pred = heads[s]([torch.randn(B, dim, h, w) for _ in range(M)])
                _accumulate(acc, pred, labels, M, dev)
            res[hd_flag][s] = _finalize(acc, modal_names)
    _write_out(out, srcs, modal_names, res[False], res[True], None, {'dry_run': True})
    print("[dry_run] OK")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg')
    ap.add_argument('--ckpt')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--out', default='./block_probe_out')
    ap.add_argument('--subset_every', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch', '--batch_size', dest='batch_size', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--blocks', type=int, nargs='+', default=[2, 4, 6, 12, 18, 24])
    ap.add_argument('--daynight', action='store_true')
    ap.add_argument('--limit_batches', type=int, default=0)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        modal_names = ['img', 'depth', 'event', 'lidar']
        if args.cfg and Path(args.cfg).exists():
            with open(args.cfg) as f:
                modal_names = yaml.safe_load(f)['DATASET']['MODALS']
        dry_run(args.out, args.blocks, modal_names, hidden=args.hidden if args.hidden else 32)
        return

    assert args.cfg and args.ckpt, "--cfg 와 --ckpt 가 필요하다(또는 --dry_run)"
    device = torch.device(args.device)
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    modal_names = cfg['DATASET']['MODALS']
    M = len(modal_names)

    model = load_frozen_model(cfg, args.ckpt, device)
    n_blk = len(model.encoder.backbone.blocks)
    blocks = sorted({min(max(int(k), 1), n_blk) for k in args.blocks})
    print(f"[probe] blocks(1-indexed)={blocks} / n_blk={n_blk} modals={modal_names}")
    mgr = MultiSource(model, blocks)
    srcs = ['fusion_in'] + [f'block{k}' for k in blocks]

    train_loader = build_loader(cfg, 'train', args.batch_size, train_aug=True)
    val_loader = build_loader(cfg, 'val', args.batch_size, train_aug=False)

    # 첫 배치로 특징원별 dim 확정 후 헤드/옵티마이저 생성
    first = next(iter(train_loader))
    imgs0 = [x.to(device) for x in first[0]]
    deg0, _ = Degrader(seed=args.seed, heldout=False)(imgs0, modal_names)
    cap0 = mgr.capture(deg0)
    dims = {s: cap0[s][0].shape[1] for s in srcs}
    print(f"[probe] dims={dims} grid={tuple(cap0['fusion_in'][0].shape[-2:])}")
    heads = {s: QualityHead(dims[s], M, hidden=args.hidden).to(device) for s in srcs}
    opts = {s: torch.optim.AdamW(heads[s].parameters(), lr=1e-3) for s in srcs}

    for ep in range(args.epochs):
        for hd in heads.values():
            hd.train()
        deg_ep = Degrader(seed=args.seed + ep, heldout=False)
        for bi, batch in enumerate(train_loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            images = [x.to(device) for x in batch[0]]
            deg, labels = deg_ep(images, modal_names)
            feats_by_src = mgr.capture(deg)
            for s in srcs:
                train_step(heads[s], opts[s], feats_by_src[s], labels)
        print(f"[ep{ep}] done")

    res_train = evaluate_sources(heads, val_loader, Degrader(seed=args.seed + 999, heldout=False),
                                 srcs, modal_names, device, mgr, args.subset_every, args.limit_batches)
    res_held = evaluate_sources(heads, val_loader, Degrader(seed=args.seed + 1000, heldout=True),
                                srcs, modal_names, device, mgr, args.subset_every, args.limit_batches)

    daynight = None
    if args.daynight:
        daynight = daynight_probe(mgr, cfg, blocks, modal_names, device,
                                  args.batch_size, args.limit_batches)

    _write_out(args.out, srcs, modal_names, res_train, res_held, daynight,
               {'cfg': args.cfg, 'ckpt': args.ckpt, 'epochs': args.epochs,
                'blocks': blocks, 'hidden': args.hidden})
    mgr.remove()


if __name__ == '__main__':
    main()
