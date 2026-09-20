#!/usr/bin/env python3
"""[P54-Q1] 품질 헤드·열화 주입기 CPU 스모크 — 데이터·ckpt 불필요.

검사(전부 assert):
  (a) Degrader p=0 이면 입력과 바이트 동일(+라벨 clean).
  (b) 시드 고정 재현(같은 seed → 동일 출력·라벨).
  (c) 라벨 형상·범위(presence/severity/mask/mask16).
  (d) held-out 함수가 학습 경로에서 호출되지 않음(monkeypatch 로 호출 시 예외).
  (e) QualityHead 출력 형상·범위·grad 흐름.
  (f) fp32 유지(half 입력에도 출력 float32).

실행:  python tools/smoke_quality_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets import degrade as D                      # noqa: E402
from semseg.datasets.degrade import Degrader                  # noqa: E402
from semseg.models.reliadino.quality_head import QualityHead, quality_loss  # noqa: E402

MODALS = ['img', 'depth', 'event', 'lidar']
_FAILS = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _batch(B=2, H=64, W=64):
    return [torch.randn(B, 3, H, W) for _ in range(len(MODALS))]


def test_a_p0_identity():
    x = _batch()
    deg = Degrader(cfg={'p_per_modal': 0.0}, seed=1)
    out, lab = deg(x, MODALS)
    same = all(torch.equal(a, b) for a, b in zip(out, x))
    check("(a) p=0 입력 바이트 동일", same)
    check("(a) p=0 presence 전부 1", bool((lab['presence'] == 1).all()))
    check("(a) p=0 severity 전부 0", bool((lab['severity'] == 0).all()))
    check("(a) p=0 mask 전부 0", bool((lab['mask'] == 0).all()))


def test_b_reproducible():
    x = _batch()
    o1, l1 = Degrader(cfg={'p_per_modal': 1.0}, seed=7)(x, MODALS)
    o2, l2 = Degrader(cfg={'p_per_modal': 1.0}, seed=7)(x, MODALS)
    out_same = all(torch.equal(a, b) for a, b in zip(o1, o2))
    lab_same = all(torch.equal(l1[k], l2[k]) for k in l1)
    check("(b) 시드 고정 출력 재현", out_same)
    check("(b) 시드 고정 라벨 재현", lab_same)


def test_c_label_shapes():
    B, H, W = 3, 64, 48
    x = [torch.randn(B, 3, H, W) for _ in MODALS]
    _, lab = Degrader(cfg={'p_per_modal': 1.0}, seed=3)(x, MODALS)
    M = len(MODALS)
    h, w = H // 16, W // 16
    check("(c) presence shape", tuple(lab['presence'].shape) == (B, M))
    check("(c) severity shape", tuple(lab['severity'].shape) == (B, M))
    check("(c) mask shape", tuple(lab['mask'].shape) == (B, M, H, W))
    check("(c) mask16 shape", tuple(lab['mask16'].shape) == (B, M, h, w))
    check("(c) severity 범위 [0,1]",
          bool((lab['severity'] >= 0).all() and (lab['severity'] <= 1).all()))
    check("(c) presence 이진", bool(((lab['presence'] == 0) | (lab['presence'] == 1)).all()))
    check("(c) mask 이진", bool(((lab['mask'] == 0) | (lab['mask'] == 1)).all()))
    check("(c) mask16 이진", bool(((lab['mask16'] == 0) | (lab['mask16'] == 1)).all()))


def test_d_heldout_not_called():
    orig_g, orig_s = D.gaussian_noise, D.salt_pepper

    def boom(*a, **k):
        raise RuntimeError("held-out 함수가 학습 경로에서 호출됨")

    D.gaussian_noise = boom
    D.salt_pepper = boom
    try:
        ok = True
        try:
            x = _batch()
            for seed in range(30):
                Degrader(cfg={'p_per_modal': 1.0}, seed=seed)(x, MODALS)
        except RuntimeError:
            ok = False
        check("(d) 학습 경로가 held-out 미호출", ok)
        # 반대로 held-out Degrader 는 실제로 호출한다(예외 발생 확인)
        raised = False
        try:
            Degrader(cfg={'p_per_modal': 1.0}, seed=0, heldout=True)(_batch(), MODALS)
        except RuntimeError:
            raised = True
        check("(d) held-out Degrader 는 held-out 함수 호출", raised)
    finally:
        D.gaussian_noise, D.salt_pepper = orig_g, orig_s


def test_e_head_shapes_grad():
    B, dim, h, w = 2, 64, 8, 8
    head = QualityHead(dim, len(MODALS), hidden=32)
    feats = [torch.randn(B, dim, h, w) for _ in MODALS]
    pred = head(feats)
    M = len(MODALS)
    check("(e) eta_scalar shape", tuple(pred['eta_scalar'].shape) == (B, M))
    check("(e) eta_token shape", tuple(pred['eta_token'].shape) == (B, M, h, w))
    check("(e) eta_scalar 범위 [0,1]",
          bool((pred['eta_scalar'] >= 0).all() and (pred['eta_scalar'] <= 1).all()))
    check("(e) eta_token 범위 [0,1]",
          bool((pred['eta_token'] >= 0).all() and (pred['eta_token'] <= 1).all()))
    # grad 흐름: 라벨 만들어 loss backward
    labels = {
        'presence': torch.randint(0, 2, (B, M)).float(),
        'severity': torch.rand(B, M),
        'mask16': torch.randint(0, 2, (B, M, h, w)).float(),
    }
    losses = quality_loss(pred, labels)
    losses['total'].backward()
    gnorm = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    check("(e) grad 흐름 (norm>0)", gnorm > 0, f"gnorm={gnorm:.3e}")
    keys = {'total', 'presence', 'severity', 'mask', 'rank'}
    check("(e) loss 항 분리", set(losses) == keys)


def test_f_fp32():
    B, dim, h, w = 2, 64, 8, 8
    head = QualityHead(dim, len(MODALS), hidden=32)
    feats = [torch.randn(B, dim, h, w).half() for _ in MODALS]   # half 입력
    pred = head(feats)
    check("(f) eta_scalar fp32", pred['eta_scalar'].dtype == torch.float32,
          str(pred['eta_scalar'].dtype))
    check("(f) eta_token fp32", pred['eta_token'].dtype == torch.float32,
          str(pred['eta_token'].dtype))


def main():
    print("== smoke_quality_head ==")
    test_a_p0_identity()
    test_b_reproducible()
    test_c_label_shapes()
    test_d_heldout_not_called()
    test_e_head_shapes_grad()
    test_f_fp32()
    print()
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == '__main__':
    main()
