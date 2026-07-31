"""피아식별(IFF) 정확도 — Allies vs Enemies 크롭 분류, 반복 시험 지원.

인증 항목 (C) 피아식별 정확도용. 검출 mAP와 달리 "찾은 표적의 아군/적군 판정"만
분리해서 잰다. --trials N (기본 5) 로 시드를 바꿔 N회 반복 학습·평가하고
평균±표준편차 + 혼동행렬을 리포트한다.

🔴 train 크롭은 전부 주간이므로(야간 0장) 저조도 증강을 기본 ON 한다
(--no-lowlight-aug 로 끌 수 있음). 야간 test 성능은 이 증강에 의존한다.

  python iff_eval.py --data ~/dset/poongsan_iff_crops --trials 5 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CLASSES = ['Allies', 'Enemies']
GRN, RED, BOLD, RST = '\033[32m', '\033[31m', '\033[1m', '\033[0m'


def build_transforms(crop: int, lowlight_aug: bool):
    """train: 표준 증강 + (필수) 저조도 증강 — train에 야간 프레임이 없기 때문."""
    train_t = [transforms.Resize((crop, crop)), transforms.RandomHorizontalFlip()]
    if lowlight_aug:
        # dark-tail: 실측(train luma 117 vs test 야간 꼬리 p10 48)에 맞춘 어둡게-편향
        train_t += [
            transforms.ColorJitter(brightness=(0.25, 1.0), contrast=(0.5, 1.1), saturation=(0.6, 1.1)),
            transforms.RandomApply([transforms.Lambda(lambda im: transforms.functional.adjust_gamma(im, random.uniform(1.2, 2.2)))], p=0.7),
        ]
    train_t += [transforms.ToTensor()]
    if lowlight_aug:   # 저광량 센서 노이즈
        train_t += [transforms.Lambda(lambda t: (t + torch.randn_like(t) * random.uniform(0.0, 0.06)).clamp(0, 1))]
    train_t += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    test_t = [transforms.Resize((crop, crop)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(train_t), transforms.Compose(test_t)


def run_trial(args, seed: int, dev):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_t, test_t = build_transforms(args.crop, not args.no_lowlight_aug)
    tr = datasets.ImageFolder(f'{args.data}/train', transform=train_t)
    te = datasets.ImageFolder(f'{args.data}/test', transform=test_t)
    assert tr.classes == CLASSES == te.classes, f'class mismatch: {tr.classes} vs {te.classes}'

    # 클래스 불균형 보정 (Allies 4753 vs Enemies 3739)
    cnt = np.bincount([y for _, y in tr.samples], minlength=len(CLASSES))
    w = torch.tensor((cnt.sum() / (len(CLASSES) * cnt)), dtype=torch.float32, device=dev)

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)
    vl = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=6)

    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    for ep in range(args.epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(m(x), y).backward()
            opt.step()
        sched.step()

    # ---- eval: 전체 / 야간 / 주간 + 혼동행렬 ----
    m.eval()
    cm = np.zeros((2, 2), dtype=int)                       # [true][pred]
    cm_night = np.zeros((2, 2), dtype=int)
    cm_day = np.zeros((2, 2), dtype=int)
    idx = 0
    with torch.no_grad():
        for x, y in vl:
            p = m(x.to(dev)).argmax(1).cpu().numpy()
            y = y.numpy()
            for i in range(len(y)):
                path = te.samples[idx + i][0]
                night = path.endswith('_night.jpg')
                cm[y[i], p[i]] += 1
                (cm_night if night else cm_day)[y[i], p[i]] += 1
            idx += len(y)

    def acc(c):
        return float(c.trace() / c.sum()) if c.sum() else float('nan')

    def bal_acc(c):
        rec = [c[i, i] / c[i].sum() if c[i].sum() else float('nan') for i in range(2)]
        return float(np.nanmean(rec)), rec

    a, (ba, rec) = acc(cm), bal_acc(cm)
    return {'seed': seed, 'acc': a, 'balanced_acc': ba,
            'recall_allies': rec[0], 'recall_enemies': rec[1],
            'acc_night': acc(cm_night), 'acc_day': acc(cm_day),
            'cm': cm.tolist(), 'cm_night': cm_night.tolist(), 'cm_day': cm_day.tolist(),
            'n_test': int(cm.sum()), 'n_night': int(cm_night.sum()), 'n_day': int(cm_day.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='build_iff_crops.py 출력 디렉터리')
    ap.add_argument('--trials', type=int, default=5, help='반복 시험 횟수 (인증 (C) = 5회)')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--no-lowlight-aug', action='store_true',
                    help='저조도 증강 끄기 (train에 야간이 없어 기본 ON)')
    ap.add_argument('--out', default='runs/cert_iff')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    print(f'\n{BOLD}╔═══ 피아식별(IFF) 정확도 — Allies vs Enemies ═══╗{RST}')
    print(f'  Data       : {args.data}')
    print(f'  Model      : MobileNetV3-small (ImageNet init), crop {args.crop}px')
    print(f'  Device     : {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})')
    print(f'  Low-light aug : {"OFF" if args.no_lowlight_aug else "ON (train에 야간 0장이라 필수)"}')
    print(f'  {BOLD}반복 시험   : {args.trials}회{RST} (seed 0..{args.trials-1}), 각 {args.epochs} epochs\n')

    rows, t0 = [], time.time()
    for s in range(args.trials):
        st = time.time()
        r = run_trial(args, s, dev)
        rows.append(r)
        print(f'  [trial {s+1}/{args.trials}] acc={r["acc"]:.4f}  bal={r["balanced_acc"]:.4f}  '
              f'night={r["acc_night"]:.4f}  day={r["acc_day"]:.4f}   ({time.time()-st:.0f}s)')

    def ms(k):
        v = [r[k] for r in rows]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    print(f'\n{BOLD}╔══════════ 피아식별 리포트 ({args.trials}회 평균) ══════════╗{RST}')
    print(f'  test crops : {rows[0]["n_test"]}  (night {rows[0]["n_night"]} / day {rows[0]["n_day"]})')
    for k, lab in (('acc', '정확도 (accuracy)'), ('balanced_acc', 'balanced accuracy'),
                   ('recall_allies', '  └ Allies recall'), ('recall_enemies', '  └ Enemies recall'),
                   ('acc_night', '야간 정확도'), ('acc_day', '주간 정확도')):
        mu, sd = ms(k)
        print(f'  {lab:22s} {mu:.4f} ± {sd:.4f}')
    cm = np.mean([r['cm'] for r in rows], axis=0)
    print(f'\n  {BOLD}혼동행렬 (평균){RST}      pred:Allies  pred:Enemies')
    for i, c in enumerate(CLASSES):
        print(f'    true:{c:9s} {cm[i,0]:10.1f} {cm[i,1]:12.1f}')
    mu, sd = ms('acc')
    print(f'\n  ►►  피아식별 정확도 = {mu:.4f} ± {sd:.4f}  ({args.trials}회)  ◄◄')
    print(f'{BOLD}╚════════════════════════════════════════════╝{RST}')

    out = {'trials': rows, 'mean': {k: ms(k)[0] for k in
           ('acc', 'balanced_acc', 'recall_allies', 'recall_enemies', 'acc_night', 'acc_day')},
           'std': {k: ms(k)[1] for k in
           ('acc', 'balanced_acc', 'recall_allies', 'recall_enemies', 'acc_night', 'acc_day')},
           'n_trials': args.trials, 'epochs': args.epochs, 'crop': args.crop,
           'lowlight_aug': not args.no_lowlight_aug, 'elapsed_s': time.time() - t0}
    p = f'{args.out}/iff_report.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  report: {p}\n')


if __name__ == '__main__':
    main()
