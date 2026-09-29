#!/usr/bin/env python3
"""[P56-A] 모달 충돌 열화 시각 점검기 — glare/잘못된 내용이 실제로 그럴듯한지 눈으로 확인.

Degrader(conflict_p=1.0) 로 강제 충돌 표본을 만들고, 표본마다 모달별 before/after/mask 를
격자 PNG 로 저장한다. 표시용 정규화 해제는 이미지별 min-max 로 대충 편다(정확한 mean/std
역변환이 아니라 육안 확인용).

  # 데이터 없이(랜덤 텐서) 파이프라인만 확인
  python tools/viz_conflict_aug.py --dry_run --out /tmp/p56a_viz

  # config 의 데이터셋에서 N 표본
  PYTHONPATH=semseg/models/sam2:. python tools/viz_conflict_aug.py \
      --cfg configs/hpca100-deliver_rgbdel_P46_c3only_seed20260821_screen40_P56A.yaml \
      --n 8 --out /tmp/p56a_viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets.degrade import Degrader, OP_NAMES               # noqa: E402

_CONF_NAMES = ('region_conflict', 'specular_glare')


def _to_disp(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) 텐서 → (H,W,3) [0,1], 이미지별 min-max 로 대충 편다."""
    x = t.detach().float().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x[:3]
    lo = x.amin()
    hi = x.amax()
    x = (x - lo) / (hi - lo + 1e-6)
    return x.permute(1, 2, 0).numpy()


def _load_from_cfg(cfg_path: str, n: int, modals_override=None):
    import yaml
    from semseg.augmentations_mm import get_val_augmentation
    from semseg.datasets.deliver import DELIVER          # noqa: F401
    try:
        from semseg.datasets.muses import MUSES          # noqa: F401
    except Exception:
        pass
    try:
        from semseg.datasets.mcubes import MCubeS         # noqa: F401
    except Exception:
        pass
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    dcfg = cfg['DATASET']
    modals = list(dcfg['MODALS'])
    tr = get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'], dataset_cfg=dcfg)
    ds = eval(dcfg['NAME'])(dcfg['ROOT'], 'val', tr, modals)
    n = min(n, len(ds))
    per_modal = [[] for _ in modals]
    for i in range(n):
        sample, _ = ds[i]
        for mi in range(len(modals)):
            per_modal[mi].append(sample[mi])
    tensors = [torch.stack(col) for col in per_modal]     # 각 (n,3,H,W)
    return tensors, modals


def _random_batch(n: int, modals, s: int = 128):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(n, 3, s, s, generator=g) for _ in modals], modals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None, help='데이터셋 config (미지정 시 --dry_run 필요)')
    ap.add_argument('--dry_run', action='store_true', help='랜덤 텐서로 파이프라인만 확인')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', required=True, help='PNG 저장 디렉터리')
    ap.add_argument('--conflict_modals', nargs='+', default=['img', 'depth'])
    ap.add_argument('--glare_frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    if not a.dry_run and not a.cfg:
        print("[ERR] --cfg 또는 --dry_run 중 하나가 필요하다")
        return 2

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.dry_run:
        modals = ['img', 'depth', 'event', 'lidar']
        tensors, modals = _random_batch(a.n, modals)
    else:
        tensors, modals = _load_from_cfg(a.cfg, a.n, None)

    clean = [t.clone() for t in tensors]
    deg = Degrader(cfg={'conflict_p': 1.0,
                        'conflict_modals': list(a.conflict_modals),
                        'conflict_glare_frac': float(a.glare_frac)},
                   seed=a.seed)
    out, lab = deg([t.clone() for t in tensors], modals)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    B = clean[0].shape[0]
    M = len(modals)
    saved = []
    for b in range(B):
        fig, axes = plt.subplots(M, 3, figsize=(9, 3 * M))
        if M == 1:
            axes = axes[None, :]
        for m in range(M):
            op_idx = int(lab['op'][b, m])
            op_name = OP_NAMES[op_idx] if op_idx < len(OP_NAMES) else str(op_idx)
            tag = f"[{op_name}]" if op_name in _CONF_NAMES else ""
            axes[m, 0].imshow(_to_disp(clean[m][b]))
            axes[m, 0].set_title(f"{modals[m]} clean")
            axes[m, 1].imshow(_to_disp(out[m][b]))
            axes[m, 1].set_title(f"{modals[m]} after {tag}")
            axes[m, 2].imshow(lab['mask'][b, m].detach().float().cpu().numpy(),
                              cmap='gray', vmin=0, vmax=1)
            axes[m, 2].set_title("mask")
            for c in range(3):
                axes[m, c].axis('off')
        fig.tight_layout()
        p = out_dir / f"conflict_sample_{b:02d}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        saved.append(p.name)

    print(f"[viz] saved {len(saved)} PNG → {out_dir}")
    for name in saved:
        print(f"  {name}")
    # op 요약
    ops = lab['op']
    n_conf = int(((ops == OP_NAMES.index('region_conflict'))
                  | (ops == OP_NAMES.index('specular_glare'))).any(dim=1).sum())
    print(f"[viz] 충돌 표본 {n_conf}/{B}  modals={modals}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
