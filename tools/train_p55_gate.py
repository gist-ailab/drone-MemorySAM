#!/usr/bin/env python3
"""[P55] 2단계 게이트 학습기 — 학습된 ckpt(예: Q2)를 로드해 **P55Gate 만** 학습한다.

연구 질문: 품질/열화/조건 라벨 없이 세그 손실만 융합을 통해 흘려보내면 게이트가 상황에
따라 달라지는가(밤·비에 RGB↓·depth/LiDAR↑) 아니면 상수로 붕괴하는가.

- 다른 모든 파라미터는 freeze(requires_grad=False), P55Gate 만 학습한다.
- 학습 데이터 = **실제 train split**(이미 밤·안개·비·해·구름 + 모션블러/노출/지터/저해상 케이스를
  포함). 조건 라벨은 학습에 절대 쓰지 않는다(경로 파싱은 tools/p55_gate_by_condition.py 의
  분석 전용).
- target=none(기본): loss = 최종 로짓의 메인 세그 손실(CE/OHEM). gradient 가 융합을 통해
  게이트로만 흐른다(그 외 전부 freeze). (모델의 다른 aux 손실은 frozen 모듈만 건드리므로
  게이트 학습에 무의미해 더하지 않는다 — 제안서가 허용한 "메인 CE" 경로, 명시.)
- target=loo(비교 팔): no_grad 로 full-fusion 로짓과 각 모달 m 을 0 으로 만든 로짓의 픽셀 CE 를
  게이트 격자로 pool → t_m = sigmoid((ℓ_{-m} − ℓ_full)/τ) → BCE(r, t)×weight 를 더한다.
  인코더는 M 번 재실행하지 않고(캐시 feats), fusion+헤드만 재실행한다.
- anti-collapse(전부 기본 off): --degrade_p(열화 패스, 라벨 미사용) · MODEL.P55.VAR_W(r std floor).

예:
  python tools/train_p55_gate.py --cfg configs/hpca100-...P55gate.yaml \
    --ckpt <Q2 val-best.pth> --epochs 10 --target none --out <analysis_logs>/p55_gate_YYYYMMDD
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ===========================================================================
# 순수 헬퍼 — 스모크가 직접 부른다(데이터·CUDA 불필요)
# ===========================================================================
def freeze_except_gate(model):
    """P55Gate 를 제외한 전 파라미터를 freeze. 반환 = 학습 파라미터 리스트."""
    if getattr(model, 'p55_gate', None) is None:
        raise RuntimeError("[P55] 모델에 p55_gate 가 없다 — MODEL.P55.ENABLE=true config 필요.")
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    for p in model.p55_gate.parameters():
        p.requires_grad_(True)
        train_params.append(p)
    return train_params


@torch.no_grad()
def loo_target(model, feats, gt, tau: float):
    """[P55-loo] leave-one-out 타깃 t_m (B,M,h,w) ∈ (0,1). 인코더 재실행 없음.

    ℓ_full, ℓ_{-m} = model.p55_fused_seg_logits(feats[, zero_idx=m]) (fusion+헤드만).
    픽셀 CE 를 게이트 격자(feats 해상도)로 average-pool → t_m = sigmoid((CE_{-m}−CE_full)/τ).
    """
    Hh, Ww = feats[0].shape[-2:]
    gt_f = gt.unsqueeze(1).float()

    def ce_grid(logits):
        gd = F.interpolate(gt_f, size=logits.shape[-2:], mode='nearest').squeeze(1).long()
        ce = F.cross_entropy(logits.float(), gd, ignore_index=255, reduction='none')  # (B,h',w')
        return F.adaptive_avg_pool2d(ce.unsqueeze(1), (Hh, Ww)).squeeze(1)             # (B,h,w)

    ce_full = ce_grid(model.p55_fused_seg_logits(feats))
    cols = []
    for m in range(len(feats)):
        ce_m = ce_grid(model.p55_fused_seg_logits(feats, zero_idx=m))
        cols.append(torch.sigmoid((ce_m - ce_full) / max(tau, 1e-6)))
    return torch.stack(cols, dim=1)                                                    # (B,M,h,w)


def _bce(r, t):
    """fp32 BCE(r, t) — autocast 안전(quality_head 규약과 동일하게 직접 수식)."""
    r = r.float().clamp(1e-6, 1 - 1e-6)
    t = t.float()
    return -(t * torch.log(r) + (1.0 - t) * torch.log1p(-r)).mean()


def gate_train_step(model, criterion, imgs, gt, *, target='none', tau=0.1,
                    loo_w=1.0):
    """게이트 학습 1 스텝의 손실을 만든다(backward 는 호출자). 반환 (loss, logdict).

    target='loo' 면 loo 타깃을 no_grad 로 먼저 계산(캐시 feats)한 뒤 메인 forward 를 돈다.
    메인 forward 는 게이트 r(grad 有)을 model._p55_r_token_live 로 남긴다."""
    dev = gt.device
    t_target = None
    if target == 'loo':
        with torch.no_grad():
            feats = model._encode_all([x.to(dev) for x in imgs])
            t_target = loo_target(model, feats, gt, tau)
    # 학습(train_reliadino)과 같은 AMP 조건으로 forward 한다. fp32 + 가법 key 마스크에서는
    # SDPA 가 mem-efficient 커널을 골라 backward 가 "LSE is not correctly aligned" 로 죽는다
    # (torch 2.3, hpca100 2026-09-29 실측) → main() 에서 mem-efficient SDP 를 끄고 bf16 autocast 사용.
    _amp = bool(getattr(model, '_p55_amp', False)) and gt.is_cuda
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=_amp):
        logits, _, aux = model(imgs, True, gt_mask=gt)
    logits = logits.float()
    loss = criterion(logits, gt)
    log = {'seg': float(loss.detach())}
    if 'p55_var_reg' in aux:
        loss = loss + aux['p55_var_reg']
        log['var_reg'] = float(aux['p55_var_reg'].detach())
    if target == 'loo':
        r_live = getattr(model, '_p55_r_token_live', None)
        if r_live is None:
            raise RuntimeError("[P55] target=loo 인데 게이트 r 가 노출되지 않았다.")
        # r_live 격자와 t_target 격자가 같아야 한다(둘 다 feats 해상도).
        if r_live.shape[-2:] != t_target.shape[-2:]:
            t_target = F.interpolate(t_target, size=r_live.shape[-2:], mode='bilinear',
                                     align_corners=False)
        l_bce = _bce(r_live, t_target)
        loss = loss + loo_w * l_bce
        log['loo_bce'] = float(l_bce.detach())
        with torch.no_grad():
            r_flat = r_live.float().mean(dim=(-1, -2))            # (B,M)
            t_flat = t_target.float().mean(dim=(-1, -2))
            rc = torch.stack([r_flat.flatten(), t_flat.flatten()])
            corr = torch.corrcoef(rc)[0, 1] if rc.shape[1] > 1 else torch.tensor(float('nan'))
        log['corr_r_t'] = float(corr)
        log['t_mean_per_modal'] = t_flat.mean(dim=0).cpu().tolist()
    if model._last_p55_stats is not None:
        log['r_mean_per_modal'] = model._last_p55_stats['r_mean_per_modal']
    return loss, log


# ===========================================================================
# 모델 로드 (val.load_model 규약: 백본 다운로드 끄고 ckpt 전체 로드)
# ===========================================================================
def load_gate_model(cfg, ckpt_path, device):
    from semseg.models.reliadino import build_reliadino
    ds = cfg['DATASET']
    _default = {'DELIVER': 25, 'MULTIAQUA': 4, 'MUSES': 19}.get(
        str(ds.get('NAME', '')).upper(), 25)
    n_cls = cfg['MODEL'].get('LORA_NUM_CLASSES', ds.get('NUM_CLASSES', _default))
    _cfg = copy.deepcopy(cfg)
    _cfg['MODEL']['PRETRAINED_BACKBONE'] = False
    model = build_reliadino(_cfg, n_cls)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        gate_missing = [k for k in missing if k.startswith('p55_gate.')]
        other_missing = [k for k in missing if not k.startswith('p55_gate.')]
        print(f"[P55] ckpt 로드: gate_missing={len(gate_missing)}(신규 init) "
              f"other_missing={len(other_missing)} unexpected={len(unexpected)}")
        if other_missing:
            print(f"[P55] ⚠️ 비-게이트 missing {sorted(other_missing)[:12]} "
                  "— 백본/세대 불일치 의심")
    return model.to(device)


# ===========================================================================
# 학습 루프 (실제 데이터)
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True, help='초기값 ckpt(예: Q2 val-best).')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--target', choices=['none', 'loo'], default='none')
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--loo_w', type=float, default=1.0)
    ap.add_argument('--degrade_p', type=float, default=0.0,
                    help='>0 이면 매 스텝 입력에 모달 열화(라벨 미사용)를 주입 — anti-collapse.')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--limit_batches', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='./p55_gate_out')
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--no_amp', action='store_true', help='cfg TRAIN.AMP 를 무시하고 fp32 로')
    args = ap.parse_args()

    from semseg.augmentations_mm import get_train_augmentation
    from semseg.datasets import DELIVER, MUSES                      # noqa: F401
    from semseg.losses import get_loss
    from semseg.datasets.degrade import Degrader
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    ds_cfg, tr_cfg = cfg['DATASET'], cfg['TRAIN']
    modal_names = list(ds_cfg['MODALS'])

    model = load_gate_model(cfg, args.ckpt, device)
    if device.type == 'cuda':
        torch.backends.cuda.enable_mem_efficient_sdp(False)   # 위 주석 참조(math/flash 커널만)
    model._p55_amp = bool(tr_cfg.get('AMP', False)) and not args.no_amp
    train_params = freeze_except_gate(model)
    n_gate = sum(p.numel() for p in train_params)
    print(f"[P55] 학습 파라미터(gate only)={n_gate:,} · target={args.target} "
          f"· degrade_p={args.degrade_p}")

    tfm = get_train_augmentation(tr_cfg['IMAGE_SIZE'],
                                 seg_fill=ds_cfg['IGNORE_LABEL'], dataset_cfg=ds_cfg)
    trainset = eval(ds_cfg['NAME'])(ds_cfg['ROOT'], 'train', tfm, modal_names)
    loader = DataLoader(trainset, batch_size=max(1, args.batch), shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
    criterion = get_loss(cfg['LOSS']['NAME'], trainset.ignore_label, None)

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * (args.limit_batches or len(loader)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    degrader = Degrader(cfg={'p_per_modal': args.degrade_p}) if args.degrade_p > 0 else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(args.epochs):
        model.train()
        model._current_epoch = ep
        agg = {}
        n = 0
        for bi, batch in enumerate(loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            imgs = [x.to(device) for x in batch[0]]
            gt = batch[1].to(device)
            if degrader is not None:
                imgs, _ = degrader([x.clone() for x in imgs], modal_names)
            opt.zero_grad(set_to_none=True)
            loss, log = gate_train_step(model, criterion, imgs, gt,
                                        target=args.target, tau=args.tau,
                                        loo_w=args.loo_w)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in log.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        row = {'epoch': ep, 'lr': float(sched.get_last_lr()[0]),
               **{k: (agg[k] / max(n, 1)) for k in agg},
               'r_stats': model._last_p55_stats}
        history.append(row)
        print(f"[P55][ep{ep}] " + " ".join(
            f"{k}={row[k]:.4f}" for k in ('seg', 'loo_bce', 'corr_r_t', 'var_reg')
            if k in row) + f" | r_mean={model._last_p55_stats['r_mean_per_modal']}")

    # 게이트 전용 + val.load_model 호환 전체 ckpt 저장
    gate_sd = {k: v for k, v in model.state_dict().items() if k.startswith('p55_gate.')}
    torch.save(gate_sd, out_dir / 'p55_gate_only.pth')
    torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg,
                'p55_target': args.target, 'epochs': args.epochs},
               out_dir / 'p55_gate_full.pth')
    (out_dir / 'train_history.json').write_text(
        json.dumps({'args': vars(args), 'history': history}, indent=2, ensure_ascii=False))
    print(f"[P55] 저장 완료 → {out_dir}")


if __name__ == '__main__':
    main()
