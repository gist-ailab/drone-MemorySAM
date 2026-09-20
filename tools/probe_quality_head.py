#!/usr/bin/env python3
"""[P54-Q1] 품질 헤드 실현성 프로브.

제안서 `.claude_logs/decisions/2026-09-20-p54-quality-aware-fusion-proposal.md`
§2-5 1단계·§6 Q1. 동결 E1 모델의 **모달별 LoRA 출력 토큰(융합 직전)** 위에 작은
품질 헤드(QualityHead)를 붙여, 우리가 주입한 합성 열화의 유형·강도·패치 마스크를
라벨로 감독 학습했을 때 열화를 인지할 수 있는지 잰다.

hook 대상 모듈: **`model.fusion`** (semseg/models/reliadino/model.py:1395 에서
`fused, aux = self.fusion(feats, ...)` 로 호출; feats = 융합 직전 모달별 LoRA
토큰 리스트 = ReliabilityGatedFusion.forward 의 입력, fusion.py:672). forward-pre-hook
으로 이 입력 feats(list of (B,C,h,w))를 가로챈다 — model.py/val.py/로더 미수정.

합격 판정(사전 고정, §2-5):
  1. 열화 vs clean 패치 AUROC > 0.9
  2. severity MAE < 0.1
  3. held-out 열화(Gaussian σ + S&P D)에서도 AUROC > 0.8

사용:
  python tools/probe_quality_head.py --cfg <E1 학습 config> --ckpt <E1 top1 ckpt> \
      --epochs 5 --out <dir> [--subset_every K] [--device cuda]
  python tools/probe_quality_head.py --dry_run           # 무작위 텐서 2스텝 스모크
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets.degrade import Degrader                       # noqa: E402
from semseg.models.reliadino.quality_head import QualityHead, quality_loss  # noqa: E402


# ===========================================================================
# 순수 계산 — 스모크·드라이런이 직접 부른다
# ===========================================================================
def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC. labels ∈{0,1}. 한 클래스만 있으면 nan."""
    labels = labels.astype(bool)
    n1 = int(labels.sum())
    n0 = int(labels.size - n1)
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_step(head: QualityHead, opt, feats, labels) -> dict:
    """한 스텝: feats(detached) → QualityHead → quality_loss → step."""
    pred = head([f.detach() for f in feats])
    losses = quality_loss(pred, labels)
    opt.zero_grad(set_to_none=True)
    losses['total'].backward()
    opt.step()
    return {k: float(v) for k, v in losses.items()}


# ===========================================================================
# 데이터·모델 (실제 프로브 경로)
# ===========================================================================
def build_loader(cfg, split, batch_size, train_aug):
    import val as valmod
    from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
    ds_cfg = cfg['DATASET']
    ignore = ds_cfg.get('IGNORE_LABEL', 255)
    if train_aug:
        size = cfg['TRAIN']['IMAGE_SIZE']
        transform = get_train_augmentation(size, seg_fill=ignore, dataset_cfg=ds_cfg)
    else:
        size = cfg.get('EVAL', {}).get('IMAGE_SIZE', cfg['TRAIN']['IMAGE_SIZE'])
        transform = get_val_augmentation(size, dataset_cfg=ds_cfg)
    dataset, _ = valmod.create_dataset(ds_cfg, split, transform, mode='val')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_aug,
                        num_workers=4, drop_last=train_aug,
                        collate_fn=valmod._collate_fn, pin_memory=False)
    return loader


def load_frozen_model(cfg, ckpt, device):
    import val as valmod
    model = valmod.load_model(cfg, ckpt, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class FusionInputHook:
    """model.fusion 의 forward-pre-hook. 입력 feats(list of (B,C,h,w))를 잡는다."""

    def __init__(self, model):
        self.feats = None
        self.handle = model.fusion.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        feats = args[0] if args else kwargs.get('feats')
        self.feats = list(feats)

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def run_encoder(model, batched_input, hook):
    hook.feats = None
    model(batched_input)
    assert hook.feats is not None, "fusion hook 이 feats 를 못 잡았다"
    return hook.feats


# ===========================================================================
# 평가
# ===========================================================================
def evaluate(model, head, loader, degrader, modal_names, device, hook,
             subset_every=1, save_eta_path=None):
    """패치 AUROC·severity MAE·모달별 분해를 계산한다."""
    head.eval()
    M = len(modal_names)
    tok_scores = [[] for _ in range(M)]     # 모달별 패치 점수
    tok_labels = [[] for _ in range(M)]
    sev_pred, sev_true, sev_mod = [], [], []
    eta_scalar_all, eta_token_all = [], []
    seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi % subset_every != 0:
                continue
            images = [x.to(device) for x in batch[0]]
            deg, labels = degrader(images, modal_names)
            feats = run_encoder(model, deg, hook)
            pred = head(feats)
            eta_t = pred['eta_token'].float()                    # (B,M,h,w)
            mask16 = labels['mask16'].float().to(device)
            # eta_token 을 라벨 해상도로 맞춤
            if eta_t.shape[-2:] != mask16.shape[-2:]:
                B = eta_t.shape[0]
                eta_t = torch.nn.functional.interpolate(
                    eta_t.reshape(B * M, 1, *eta_t.shape[-2:]),
                    size=mask16.shape[-2:], mode='bilinear',
                    align_corners=False).reshape(B, M, *mask16.shape[-2:])
            for m in range(M):
                tok_scores[m].append(eta_t[:, m].reshape(-1).cpu().numpy())
                tok_labels[m].append(mask16[:, m].reshape(-1).cpu().numpy())
            presence = labels['presence'].to(device)
            severity = labels['severity'].to(device)
            eta_s = pred['eta_scalar'].float()
            pm = presence > 0.5
            for m in range(M):
                sel = pm[:, m]
                if sel.any():
                    sev_pred.append(eta_s[sel, m].cpu().numpy())
                    sev_true.append(severity[sel, m].cpu().numpy())
                    sev_mod.append(np.full(int(sel.sum()), m))
            eta_scalar_all.append(eta_s.cpu().numpy())
            eta_token_all.append(pred['eta_token'].float().cpu().numpy())
            seen += images[0].shape[0]

    # 집계
    def _auroc(mods):
        s = np.concatenate([np.concatenate(tok_scores[m]) for m in mods])
        l = np.concatenate([np.concatenate(tok_labels[m]) for m in mods])
        return rank_auroc(s, l)

    auroc_all = _auroc(range(M))
    auroc_per_modal = {modal_names[m]: _auroc([m]) for m in range(M)}
    sp = np.concatenate(sev_pred) if sev_pred else np.zeros(0)
    st = np.concatenate(sev_true) if sev_true else np.zeros(0)
    sm = np.concatenate(sev_mod) if sev_mod else np.zeros(0)
    mae_all = float(np.mean(np.abs(sp - st))) if sp.size else float('nan')
    mae_per_modal = {}
    for m in range(M):
        msk = sm == m
        mae_per_modal[modal_names[m]] = (
            float(np.mean(np.abs(sp[msk] - st[msk]))) if msk.any() else float('nan'))

    if save_eta_path is not None:
        np.savez_compressed(
            save_eta_path,
            eta_scalar=np.concatenate(eta_scalar_all, axis=0),
            modal_names=np.array(modal_names))
    return {
        'auroc': auroc_all,
        'auroc_per_modal': auroc_per_modal,
        'severity_mae': mae_all,
        'severity_mae_per_modal': mae_per_modal,
        'n_samples': seen,
    }


# ===========================================================================
# 드라이런 — 무작위 텐서로 학습 루프 2 스텝
# ===========================================================================
def dry_run(modal_names):
    dim = 64                        # 드라이런 전용 축소 dim(속도)
    M = len(modal_names)
    head = QualityHead(dim, M, hidden=32)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    degrader = Degrader(seed=0)
    B, h, w = 2, 8, 8
    print("[dry_run] 학습 루프 2 스텝 (무작위 텐서)")
    for step in range(2):
        # 무작위 입력(정규화 공간 모사)으로 라벨 생성 후 feats 는 무작위
        imgs = [torch.randn(B, 3, h * 16, w * 16) for _ in range(M)]
        _, labels = degrader(imgs, modal_names)
        # 라벨 mask16 을 feats 해상도(h,w)에 맞춰 QualityHead·loss 가 처리
        feats = [torch.randn(B, dim, h, w, requires_grad=False) for _ in range(M)]
        log = train_step(head, opt, feats, labels)
        print(f"  step{step}: " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
    print("[dry_run] OK")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg')
    ap.add_argument('--ckpt')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--out', default='./probe_out')
    ap.add_argument('--subset_every', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        modal_names = ['img', 'depth', 'event', 'lidar']
        if args.cfg and Path(args.cfg).exists():
            with open(args.cfg) as f:
                modal_names = yaml.safe_load(f)['DATASET']['MODALS']
        dry_run(modal_names)
        return

    assert args.cfg and args.ckpt, "--cfg 와 --ckpt 가 필요하다(또는 --dry_run)"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    modal_names = cfg['DATASET']['MODALS']
    M = len(modal_names)

    model = load_frozen_model(cfg, args.ckpt, device)
    hook = FusionInputHook(model)
    train_loader = build_loader(cfg, 'train', args.batch_size, train_aug=True)
    val_loader = build_loader(cfg, 'val', args.batch_size, train_aug=False)

    train_deg = Degrader(seed=args.seed, heldout=False)
    # dim 은 첫 배치의 feats 채널에서 확정(하드코딩 회피)
    first = next(iter(train_loader))
    imgs0 = [x.to(device) for x in first[0]]
    deg0, lab0 = train_deg(imgs0, modal_names)
    feats0 = run_encoder(model, deg0, hook)
    dim = feats0[0].shape[1]
    print(f"[probe] dim={dim} h,w={tuple(feats0[0].shape[-2:])} modals={modal_names}")

    head = QualityHead(dim, M, hidden=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    # 학습
    head.train()
    for ep in range(args.epochs):
        deg_ep = Degrader(seed=args.seed + ep, heldout=False)
        running = {}
        n = 0
        for batch in train_loader:
            images = [x.to(device) for x in batch[0]]
            deg, labels = deg_ep(images, modal_names)
            feats = run_encoder(model, deg, hook)
            log = train_step(head, opt, feats, labels)
            for k, v in log.items():
                running[k] = running.get(k, 0.0) + v
            n += 1
        print(f"[ep{ep}] " + " ".join(f"{k}={running[k]/max(n,1):.4f}" for k in running))

    # 평가 — 학습 열화 / held-out 열화 각각
    res_train = evaluate(model, head, val_loader,
                         Degrader(seed=args.seed + 999, heldout=False),
                         modal_names, device, hook, args.subset_every,
                         save_eta_path=str(out / 'eta_val.npz'))
    res_held = evaluate(model, head, val_loader,
                        Degrader(seed=args.seed + 1000, heldout=True),
                        modal_names, device, hook, args.subset_every)

    # 합격 판정
    pass_auroc = res_train['auroc'] > 0.9
    pass_mae = res_train['severity_mae'] < 0.1
    pass_held = res_held['auroc'] > 0.8
    report = {
        'gates': {
            'train_patch_auroc>0.9': {'value': res_train['auroc'], 'pass': bool(pass_auroc)},
            'severity_mae<0.1': {'value': res_train['severity_mae'], 'pass': bool(pass_mae)},
            'heldout_auroc>0.8': {'value': res_held['auroc'], 'pass': bool(pass_held)},
        },
        'overall_pass': bool(pass_auroc and pass_mae and pass_held),
        'train_degrade': res_train,
        'heldout_degrade': res_held,
        'cfg': args.cfg, 'ckpt': args.ckpt, 'epochs': args.epochs,
        'modal_names': modal_names,
    }
    with open(out / 'probe_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n=== 품질 헤드 프로브 결과 ===")
    print(f"{'gate':32s} {'value':>8s}  판정")
    for k, v in report['gates'].items():
        print(f"{k:32s} {v['value']:8.4f}  {'PASS' if v['pass'] else 'FAIL'}")
    print(f"\n모달별 train AUROC: {res_train['auroc_per_modal']}")
    print(f"모달별 held-out AUROC: {res_held['auroc_per_modal']}")
    print(f"모달별 severity MAE: {res_train['severity_mae_per_modal']}")
    print(f"\n종합: {'PASS' if report['overall_pass'] else 'FAIL'}  → {out/'probe_report.json'}")
    hook.remove()


if __name__ == '__main__':
    main()
