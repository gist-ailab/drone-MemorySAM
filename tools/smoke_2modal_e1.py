"""CPU smoke: confirm 2-modal (RGB+X) builds succeed for N-RGBX-T configs (DRN-261001-02).
Adapted from tools/smoke_taps_e1.py forward/backward call pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from semseg.models.reliadino.model import build_reliadino

NUM_CLASSES = 6
_FAILS = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _cfg(modals):
    return {
        'MODEL': {
            'BACKBONE_TIMM': 'vit_tiny_patch16_224', 'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
            'PRETRAINED_BACKBONE': False, 'LORA_R': 2, 'FPN_DIM': 64,
            'TAPS': {'ENABLE': True, 'LAYERS': [3, 6, 9, 12], 'MODE': 'per_modal'},
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0, 'AUX_HIDDEN': 32,
                       'AUX_CE_WEIGHT': 0.5, 'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False}, 'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': False}, 'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False}, 'M2F': {'ENABLE': False},
            'P39': {'TRUNK_EXP': True, 'ARBITER': False, 'TRUNK_MODE': 'gated_mlp', 'TRUNK_HIDDEN': 64,
                    'VICREG': {'ENABLE': False}},
        },
        'DATASET': {'MODALS': list(modals)},
        'TRAIN': {'IMAGE_SIZE': [64, 64]},
    }


def main():
    try:
        import timm  # noqa: F401
    except Exception as e:
        print(f"[SKIP] timm 부재 — 스모크 건너뜀: {e}")
        return 0

    for modals in (['img', 'depth'], ['img', 'lidar'], ['img', 'event']):
        print(f"\n[smoke] modals={modals}")
        try:
            torch.manual_seed(0)
            cfg = _cfg(modals)
            m = build_reliadino(cfg, NUM_CLASSES).train()
            g = torch.Generator().manual_seed(1234)
            xs = [torch.randn(1, 3, 64, 64, generator=g) for _ in modals]
            gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
            out = m(xs, True, gt)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            check(f"modals={modals}: forward OK", True, f"logits shape={tuple(logits.shape)}")
            logits.float().sum().backward()
            has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for p in m.parameters() if p.requires_grad)
            check(f"modals={modals}: backward OK, grad_flows={has_grad}", has_grad)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"modals={modals}: build+forward+backward", False, str(e))

    print("\n" + ("=" * 60))
    if _FAILS:
        print(f"[2MODAL SMOKE] FAIL {len(_FAILS)}건: {_FAILS}")
        return 1
    print("[2MODAL SMOKE] 전 항목 PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
