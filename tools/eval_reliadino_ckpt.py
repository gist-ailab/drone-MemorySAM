#!/usr/bin/env python3
"""
tools/eval_reliadino_ckpt.py — G0a: ReliaDINO 체크포인트를 학습-eval 프로토콜
"그대로"(동일 transform/배치/metric 경로)로 val/test 평가.

목적: 현 headline test 57.60은 test-선정 ckpt(test_epoch140_top1)에서 나온 수치라
리뷰 불법. val-best(epoch120_68.19_top1)의 test 수치를 같은 프로토콜로 측정해
reviewer-legal headline을 확정한다 (P35 설계 v2 §3 G0a).

주의: 반드시 train_reliadino.py와 같은 경로를 재사용한다 — get_val_augmentation(
EVAL.IMAGE_SIZE), eval(NAME)(ROOT, split, valtransform, MODALS), evaluate() 본체.
cfg는 학습 당시 그대로 사용(임의 override 금지 — 프로토콜 요동 ~2pt가 마진보다 큼).

Usage:
  PYTHONPATH=pylibs_p34:. python tools/eval_reliadino_ckpt.py \
    --cfg configs/b200-deliver_rgbdel_P34_reliadino.yaml \
    --ckpt outputs/.../epoch120_68.19_top1_checkpoint.pth --split both --gpu 2
"""
import argparse, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# CUDA_VISIBLE_DEVICES는 torch import 전에 argv에서 선반영
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import yaml, torch                                                 # noqa: E402
from torch.utils.data import DataLoader                            # noqa: E402
from semseg.augmentations_mm import get_val_augmentation           # noqa: E402
from semseg.datasets import *                                      # noqa: F401,F403,E402
from semseg.models.reliadino.model import build_reliadino          # noqa: E402
from train_reliadino import evaluate                               # noqa: E402  동일 metric 경로


class _DropModalityDataset(torch.utils.data.Dataset):
    """Wraps a MODALS-based dataset, zero-filling one modality tensor per item —
    same semantics as tools/feature_stats.py --drop-modality. Model structure
    unchanged; only the input at `idx` is replaced with zeros."""
    def __init__(self, base, idx):
        self.base, self.idx = base, idx
        self.n_classes, self.CLASSES = base.n_classes, base.CLASSES
        self.ignore_label = getattr(base, 'ignore_label', 255)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        imgs, label = self.base[i]
        imgs = list(imgs)
        imgs[self.idx] = torch.zeros_like(imgs[self.idx])
        return imgs, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='both', choices=['val', 'test', 'both'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch', type=int, default=None,
                    help='기본 = cfg EVAL.BATCH_SIZE (프로토콜 유지; 변경 시 명시)')
    ap.add_argument('--dataset-root', default=None,
                    help='DATASET.ROOT만 서버-로컬 경로로 override (프로토콜 무관)')
    ap.add_argument('--toggles', default='',
                    help='쉼표목록 post-load 모듈 off (module_ablation.py와 동일 시맨틱): '
                         'bias,cons,gate,veto,calib')
    ap.add_argument('--drop-modality', default='',
                    help='DATASET.MODALS 중 하나(예: radar)를 zero-fill해 fwd — '
                         'drop-modality dMIoU 측정용(tools/feature_stats.py --drop-modality와 '
                         '동일 시맨틱). 모델 구조는 그대로, 입력만 0으로 채운다.')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    dataset_cfg, eval_cfg = cfg['DATASET'], cfg['EVAL']
    if args.dataset_root:
        dataset_cfg['ROOT'] = args.dataset_root
    device = torch.device('cuda')

    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)
    ds_name = dataset_cfg['NAME']
    probe = eval(ds_name)(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    num_classes, class_names = probe.n_classes, probe.CLASSES

    model = build_reliadino(cfg, num_classes)
    ck = torch.load(args.ckpt, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    msg = model.load_state_dict(state, strict=False)
    n_missing = len(msg.missing_keys); n_unexp = len(msg.unexpected_keys)
    print(f"[G0a] ckpt={Path(args.ckpt).name} epoch={ck.get('epoch','?')} "
          f"missing={n_missing} unexpected={n_unexp}")
    assert n_missing == 0 and n_unexp == 0, \
        f"state_dict mismatch — 모델/ckpt config 불일치: {msg.missing_keys[:3]} {msg.unexpected_keys[:3]}"
    # G0c: post-load 모듈 toggle (tools/module_ablation.py p34_* 와 동일 시맨틱)
    for t in [x.strip() for x in args.toggles.split(',') if x.strip()]:
        fus = model.fusion
        if t == 'bias':    fus.lambda1.data.zero_()
        elif t == 'cons':  fus.lambda2.data.zero_()
        elif t == 'gate':  fus.gate_enable = False
        elif t == 'veto':  fus.veto_floor = False
        elif t == 'calib': fus.calibrate = False
        else: raise ValueError(f'unknown toggle {t}')
        print(f"[G0c] toggle OFF: {t}")
    model = model.to(device)

    bs = args.batch or eval_cfg['BATCH_SIZE']
    splits = ['val', 'test'] if args.split == 'both' else [args.split]
    drop_idx = None
    if args.drop_modality:
        modals = dataset_cfg['MODALS']
        assert args.drop_modality in modals, \
            f"--drop-modality {args.drop_modality} not in DATASET.MODALS {modals}"
        drop_idx = modals.index(args.drop_modality)
        print(f"[G0a] drop-modality: {args.drop_modality} (index {drop_idx}, zero-filled)")

    for split in splits:
        dset = probe if split == 'val' else eval(ds_name)(
            dataset_cfg['ROOT'], split, valtransform, dataset_cfg['MODALS'])
        if drop_idx is not None:
            dset = _DropModalityDataset(dset, drop_idx)
        loader = DataLoader(dset, batch_size=bs, num_workers=4, pin_memory=True)
        with torch.no_grad():
            try:
                acc, macc, f1, mf1, ious, miou = evaluate(model, loader, device, dist_sync=False)
            except TypeError:   # 구버전 train_reliadino.evaluate (dist_sync 이전)
                acc, macc, f1, mf1, ious, miou = evaluate(model, loader, device)
        print(f"\n[G0a][{split}] n={len(dset)}  mIoU: {miou:.4f}  mAcc: {macc:.4f}  mF1: {mf1:.4f}")
        print("  per-class IoU: " + " | ".join(
            f"{c}: {float(v):.2f}" for c, v in zip(class_names, ious)))


if __name__ == '__main__':
    main()
