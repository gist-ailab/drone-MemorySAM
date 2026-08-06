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


def _pick_display():
    """활성 X 디스플레이를 찾아 DISPLAY 를 맞춘다. 데모박스는 :0 이 아니라 :1 인
    경우가 있어(로그인 세션에 따라) 하드코딩하면 창이 안 뜬다."""
    import glob, os
    if os.environ.get('DISPLAY'):
        return os.environ['DISPLAY']
    for s in sorted(glob.glob('/tmp/.X11-unix/X*')):
        d = ':' + os.path.basename(s)[1:]
        os.environ['DISPLAY'] = d
        return d
    return None


def _display(img, title: str):
    """GT vs 예측 비교를 화면에 띄운다. cv2 가 headless 빌드면 matplotlib 로 폴백."""
    try:
        import cv2
        import numpy as _np
        cv2.imshow(title, cv2.cvtColor(_np.array(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return
    except Exception:
        pass
    try:
        _pick_display()
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        import matplotlib.pyplot as plt
        plt.figure(title, figsize=(16, 9))
        plt.imshow(img); plt.axis('off'); plt.title(title)
        plt.show(block=False); plt.pause(0.1)
    except Exception as e:
        print(f'  (화면 표시 불가 — PNG 로만 저장됩니다: {type(e).__name__})')


OK_BLUE = (54, 140, 245)      # 맞춘 것 = 파랑
BAD_RED = (232, 72, 60)       # 틀린 것 = 빨강


def save_iff_viz(items, viz_dir: str, seed: int, cols: int = 8, rows: int = 6,
                 tile: int = 150, show: bool = False):
    """발표용 피아식별 시각화 — 입력 크롭마다 예측 결과를 색으로 표시.

    맞추면 파란 테두리, 틀리면 빨간 테두리. 각 타일에 GT→예측(신뢰도)을 적어
    "무엇을 무엇으로 봤는지"가 한눈에 보이게 한다. 오답을 앞쪽에 배치해
    한 장으로 성공/실패를 함께 확인할 수 있게 구성.
    """
    from PIL import Image, ImageDraw, ImageFont
    import glob as _glob
    os.makedirs(viz_dir, exist_ok=True)

    # 한글이 있는 폰트를 먼저 찾는다 — 없으면 영문 라벨로 폴백(두부 문자 방지)
    kor = None
    for pat in ('/usr/share/fonts/**/NanumGothic*.ttf', '/usr/share/fonts/**/*Nanum*.ttf',
                '/usr/share/fonts/**/NotoSansCJK*.ttc', '/usr/share/fonts/**/NotoSansCJK*.otf',
                '/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc', '/usr/share/fonts/**/*CJK*.ttc',
                '/usr/share/fonts/**/malgun*.ttf'):
        hit = _glob.glob(pat, recursive=True)
        if hit:
            kor = hit[0]
            break

    def _font(sz):
        for p in (kor, '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
            if p:
                try:
                    return ImageFont.truetype(p, sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    font = _font(13)
    head_font = _font(17)
    KO = kor is not None            # 한글 사용 가능 여부

    wrong = [it for it in items if it[1] != it[2]]
    right = [it for it in items if it[1] == it[2]]
    # 오답 전부(최대 절반) + 정답을 섞어 채운다 — 야간 우선(어려운 조건을 보여준다)
    n = cols * rows
    wrong.sort(key=lambda it: (not it[4], -it[3]))          # 야간 먼저, 확신 높은 오답 먼저
    right.sort(key=lambda it: (not it[4], -it[3]))
    pick = wrong[:n // 2] + right[:n - len(wrong[:n // 2])]

    pad, head, cap = 8, 40, 20
    W = cols * (tile + pad) + pad
    H = head + rows * (tile + pad + cap) + pad
    sheet = Image.new('RGB', (W, H), (18, 20, 24))
    d = ImageDraw.Draw(sheet)
    n_w, n_r = len(wrong), len(right)
    acc = n_r / max(n_r + n_w, 1)
    title = (f'피아식별(IFF) — 정답 {n_r} / 오답 {n_w}   정확도 {acc:.3f}   (seed {seed})'
             if KO else
             f'Friend-or-Foe (IFF) — correct {n_r} / wrong {n_w}   acc {acc:.3f}   (seed {seed})')
    d.text((pad, 6), title, fill=(235, 240, 247), font=head_font)
    # 범례 — 색이 무슨 뜻인지 발표에서 바로 읽히게
    lx = pad
    for col, lab in ((OK_BLUE, '정답 (correct)' if KO else 'correct'),
                     (BAD_RED, '오답 (wrong)' if KO else 'wrong')):
        d.rectangle([lx, 26, lx + 22, 36], outline=col, width=3)
        d.text((lx + 28, 24), lab, fill=col, font=font)
        lx += 150

    for k, (path, t, p, conf, night) in enumerate(pick):
        r, c = divmod(k, cols)
        x0 = pad + c * (tile + pad)
        y0 = head + r * (tile + pad + cap)
        try:
            im = Image.open(path).convert('RGB').resize((tile, tile), Image.BILINEAR)
        except Exception:
            continue
        col = OK_BLUE if t == p else BAD_RED
        sheet.paste(im, (x0, y0))
        d.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline=col, width=4)
        # 야간 배지는 타일 위에 (폰트 이모지 의존 제거)
        if night:
            d.rectangle([x0 + 4, y0 + 4, x0 + 52, y0 + 22], fill=(0, 0, 0))
            d.text((x0 + 8, y0 + 5), 'NIGHT', fill=(255, 214, 120), font=font)
        # 캡션: 정답이면 클래스명만, 오답이면 GT→예측 (무엇을 무엇으로 봤는지)
        tag = (f'{CLASSES[t]}  {conf:.2f}' if t == p
               else f'{CLASSES[t]} → {CLASSES[p]}  {conf:.2f}')
        d.text((x0 + 2, y0 + tile + 3), tag, fill=col, font=font)

    out = f'{viz_dir}/iff_seed{seed}.png'
    sheet.save(out)
    if show:
        _display(sheet, 'IFF  GT vs Pred  (blue=correct, red=wrong)')
    return out


def build_model(dev, n_cls: int = 2):
    """인증 대상 분류기 — MobileNetV3-small (ImageNet init, classifier 2-class)."""
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
    return m.to(dev)


def train_model(args, dev, seed: int = 0):
    """[인증 준비] 분류기를 한 번 학습해 체크포인트로 확정한다.
    인증 시험 때는 이 가중치를 로드해 '평가만' 반복한다(모델 고정)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_t, _ = build_transforms(args.crop, not args.no_lowlight_aug)
    tr = datasets.ImageFolder(f'{args.data}/train', transform=train_t)
    assert tr.classes == CLASSES, f'class mismatch: {tr.classes}'
    cnt = np.bincount([y for _, y in tr.samples], minlength=len(CLASSES))
    w = torch.tensor((cnt.sum() / (len(CLASSES) * cnt)), dtype=torch.float32, device=dev)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)

    m = build_model(dev)
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
    os.makedirs(os.path.dirname(args.ckpt) or '.', exist_ok=True)
    torch.save({'state_dict': m.state_dict(), 'classes': CLASSES, 'crop': args.crop,
                'epochs': args.epochs, 'seed': seed,
                'model': 'MobileNetV3-small (ImageNet-1k init, classifier->2-class)',
                'lowlight_aug': not args.no_lowlight_aug}, args.ckpt)
    print(f'[IFF] 학습 완료 -> 체크포인트 저장: {args.ckpt}')
    return m


def evaluate(args, dev, model, trial: int, viz_dir: str | None = None):
    _, test_t = build_transforms(args.crop, not args.no_lowlight_aug)
    te = datasets.ImageFolder(f'{args.data}/test', transform=test_t)
    assert te.classes == CLASSES, f'class mismatch: {te.classes}'
    vl = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=6)
    m = model

    # ---- eval: 전체 / 야간 / 주간 + 혼동행렬 (+ 발표용 시각화) ----
    m.eval()
    cm = np.zeros((2, 2), dtype=int)                       # [true][pred]
    cm_night = np.zeros((2, 2), dtype=int)
    cm_day = np.zeros((2, 2), dtype=int)
    idx = 0
    viz_items = []                                          # (path, true, pred, conf, night)
    with torch.no_grad():
        for x, y in vl:
            logit = m(x.to(dev))
            prob = torch.softmax(logit, 1)
            p = logit.argmax(1).cpu().numpy()
            conf = prob.max(1).values.cpu().numpy()
            y = y.numpy()
            for i in range(len(y)):
                path = te.samples[idx + i][0]
                night = path.endswith('_night.jpg')
                cm[y[i], p[i]] += 1
                (cm_night if night else cm_day)[y[i], p[i]] += 1
                viz_items.append((path, int(y[i]), int(p[i]), float(conf[i]), night))
            idx += len(y)

    def acc(c):
        return float(c.trace() / c.sum()) if c.sum() else float('nan')

    def bal_acc(c):
        rec = [c[i, i] / c[i].sum() if c[i].sum() else float('nan') for i in range(2)]
        return float(np.nanmean(rec)), rec

    a, (ba, rec) = acc(cm), bal_acc(cm)
    if viz_dir:                       # 발표용: 맞음=파랑 / 틀림=빨강 컨택트시트
        save_iff_viz(viz_items, viz_dir, trial, show=getattr(args, 'show', False))
    return {'trial': trial, 'acc': a, 'balanced_acc': ba,
            'recall_allies': rec[0], 'recall_enemies': rec[1],
            'acc_night': acc(cm_night), 'acc_day': acc(cm_day),
            'cm': cm.tolist(), 'cm_night': cm_night.tolist(), 'cm_day': cm_day.tolist(),
            'n_test': int(cm.sum()), 'n_night': int(cm_night.sum()), 'n_day': int(cm_day.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='build_iff_crops.py 출력 디렉터리')
    ap.add_argument('--mode', choices=['train', 'eval'], default='eval',
                    help="train=학습해 체크포인트 저장(인증 전 1회), eval=가중치 로드해 평가만 반복")
    ap.add_argument('--ckpt', default='weights/iff_mobilenetv3.pt',
                    help='분류기 체크포인트 (모델 고정 — 인증 시 이 가중치로 평가만)')
    ap.add_argument('--trials', type=int, default=2, help='반복 시험 횟수 (피아식별 = 2회)')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--no-lowlight-aug', action='store_true',
                    help='저조도 증강 끄기 (train에 야간이 없어 기본 ON)')
    ap.add_argument('--out', default='runs/cert_iff')
    ap.add_argument('--no-viz', action='store_true', help='발표용 시각화(정답 파랑/오답 빨강) 끄기')
    ap.add_argument('--show', action='store_true',
                    help='평가 중 GT vs 예측 비교를 화면에 표시 ($DISPLAY 필요)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    print(f'\n{BOLD}╔═══ 피아식별(IFF) 정확도 — Allies vs Enemies ═══╗{RST}')
    print(f'  Data       : {args.data}')
    print(f'  Model      : MobileNetV3-small (ImageNet init), crop {args.crop}px')
    print(f'  Device     : {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})')
    print(f'  Low-light aug : {"OFF" if args.no_lowlight_aug else "ON (train에 야간 0장이라 필수)"}')

    t0 = time.time()
    if args.mode == 'train':
        print(f'  {BOLD}모드        : 학습 (인증 전 1회, {args.epochs} epochs){RST}\n')
        train_model(args, dev)
        print(f'  체크포인트: {args.ckpt}  — 인증 시에는 --mode eval 로 이 가중치를 로드해 평가만 반복\n')
        return

    # ---- eval 모드: 고정된 가중치를 로드해 '평가만' 반복 (모델 고정) ----
    if not os.path.exists(args.ckpt):
        raise SystemExit(f'체크포인트가 없습니다: {args.ckpt}\n'
                         f'  먼저 학습하세요:  python {os.path.basename(__file__)} '
                         f'--mode train --data {args.data} --ckpt {args.ckpt}')
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build_model(dev)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    print(f'  Checkpoint  : {args.ckpt}  (학습 seed {ck.get("seed")}, {ck.get("epochs")} ep)')
    print(f'  {BOLD}반복 시험   : {args.trials}회 (모델 고정, 평가만 반복){RST}\n')

    rows = []
    viz_dir = None if args.no_viz else f'{args.out}/viz'
    for s in range(args.trials):
        st = time.time()
        r = evaluate(args, dev, model, s, viz_dir if s == 0 else None)
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
    print(f'\n  ►►  피아식별 정확도 = {mu:.4f} ± {sd:.4f}  ({args.trials}회, 모델 고정)  ◄◄')
    print(f'{BOLD}╚════════════════════════════════════════════╝{RST}')

    out = {'model': 'MobileNetV3-small (torchvision, ImageNet-1k init, classifier→2-class)',
           'trials': rows, 'mean': {k: ms(k)[0] for k in
           ('acc', 'balanced_acc', 'recall_allies', 'recall_enemies', 'acc_night', 'acc_day')},
           'std': {k: ms(k)[1] for k in
           ('acc', 'balanced_acc', 'recall_allies', 'recall_enemies', 'acc_night', 'acc_day')},
           'n_trials': args.trials, 'epochs': args.epochs, 'crop': args.crop,
           'ckpt': args.ckpt, 'eval_only': True,
           'optimizer': f'AdamW lr={args.lr} wd=1e-4, cosine, batch={args.batch}',
           'lowlight_aug': not args.no_lowlight_aug, 'elapsed_s': time.time() - t0}
    p = f'{args.out}/iff_report.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  report: {p}')
    if viz_dir:
        print(f'  시각화: {viz_dir}/iff_seed*.png  (파랑=정답, 빨강=오답)')
    print()


if __name__ == '__main__':
    main()
