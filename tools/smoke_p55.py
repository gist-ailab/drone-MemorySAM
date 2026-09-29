#!/usr/bin/env python3
"""[P55] 무감독 다중블록 모달 게이트 CPU 스모크 — 데이터·ckpt 불필요(tiny 모델).

검사(전부 assert):
  (a) MODEL.P55 off ⇒ 키 없는 baseline 과 state_dict 키 + forward 출력 byte-동일.
      fusion.quality_head/p55_gate 미생성.
  (b) MODEL.P55 on tiny: forward 실행 · eta∈[0, 1−r_floor] · 시작 r≈sigmoid(init_bias).
  (c) 게이트 학습 1 스텝 (none/loo 둘 다):
      none — 세그 손실만으로 게이트 파라미터 grad≠0 · 그 외 전 파라미터 grad None.
      loo  — loo 타깃 형상 · 손실 유한.
  (c2) 조건 분석 도구가 합성 레코드로 표/리포트를 만든다.
  (d) leave-one-out 로짓이 모달 0-화 시 full 과 다르다.
  (e) η̂ 오버라이드(all r=1 ⇒ eta=0)가 eta=0 경로 출력과 일치.

실행:  python tools/smoke_p55.py   (timm 부재 시 모델 항목 SKIP, 순수 항목만)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MODALS = ['img', 'depth', 'event', 'lidar']
K = 6
_FAILS = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _base_cfg():
    return {'MODEL': {'BACKBONE_TIMM': 'vit_tiny_patch16_224',
                      'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
                      'PRETRAINED_BACKBONE': False, 'LORA_R': 2, 'FPN_DIM': 64,
                      'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0,
                                 'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
                                 'ATTN_BIAS': {'ENABLE': False}},
                      'CONSISTENCY': {'ENABLE': False},
                      'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
                      'CALIBRATION': {'ENABLE': False}, 'ROUTER': {'ENABLE': True},
                      'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
                      'M2F': {'ENABLE': False},
                      'P39': {'TRUNK_EXP': True, 'ARBITER': False,
                              'TRUNK_MODE': 'gated_mlp', 'TRUNK_HIDDEN': 64,
                              'VICREG': {'ENABLE': False}}},
            'DATASET': {'MODALS': list(MODALS)}, 'TRAIN': {'IMAGE_SIZE': [64, 64]}}


def _p55_cfg(**over):
    cfg = _base_cfg()
    blk = {'ENABLE': True, 'HIDDEN': 16, 'FRACTIONS': [0.1, 0.25, 0.5],
           'R_FLOOR': 0.05, 'INIT_BIAS': 3.0, 'MASK_MODALS': list(MODALS)}
    blk.update(over)
    cfg['MODEL']['P55'] = blk
    return cfg


def _build(cfg):
    from semseg.models.reliadino.model import build_reliadino
    return build_reliadino(cfg, K)


def _seeded(fn, seed=20260929):
    torch.manual_seed(seed)
    return fn()


def _inputs(B=1, S=64):
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


def test_a_off_byte_identical():
    print("\n[a] MODEL.P55 off ⇒ baseline byte-동일")
    m_base = _seeded(lambda: _build(_base_cfg())).eval()          # P55 키 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['P55'] = {'ENABLE': False, 'MASK_MODALS': list(MODALS)}
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    kb, ko = list(m_base.state_dict().keys()), list(m_off.state_dict().keys())
    check("(a) state_dict 키 동일", kb == ko,
          f"diff={set(kb) ^ set(ko)}")
    check("(a) p55_gate 미생성", m_off.p55_gate is None)
    check("(a) quality_head 미생성", m_off.fusion.quality_head is None)
    x = _inputs()
    with torch.no_grad():
        ob, _ = m_base(x, True)
        oo, _ = m_off(x, True)
    check("(a) forward 출력 byte-동일", torch.equal(ob, oo),
          f"max|Δ|={(ob - oo).abs().max().item() if ob.shape == oo.shape else 'shape≠'}")


def test_b_on_forward():
    print("\n[b] MODEL.P55 on ⇒ forward · eta 범위 · 시작 r≈sigmoid(bias)")
    m = _seeded(lambda: _build(_p55_cfg())).eval()
    check("(b) p55_gate 생성", m.p55_gate is not None)
    check("(b) quality_head 미생성(η̂ 는 게이트 공급)", m.fusion.quality_head is None)
    check("(b) gate params ≤ ~1M", sum(p.numel() for p in m.p55_gate.parameters()) <= 1_000_000)
    x = _inputs()
    with torch.no_grad():
        out, _ = m(x, True)
    check("(b) forward shape", tuple(out.shape) == (1, K, 64, 64), str(tuple(out.shape)))
    rt = m._last_p55_r_token
    eta = 1.0 - rt
    floor = m.p55_gate.r_floor
    check("(b) r≈sigmoid(3)≈0.9526 시작", abs(float(rt.mean()) - torch.sigmoid(torch.tensor(3.0)).item()) < 1e-3,
          f"r_mean={float(rt.mean()):.4f}")
    # eta_token 은 게이트 dict 에서 clamp 됨 — 직접 확인
    pred, r_tok, r_sc = m.p55_gate(m.p55_capture.collect_flat(4, 4, len(MODALS)))
    check("(b) eta∈[0,1−r_floor]",
          bool((pred['eta_token'] >= 0).all() and (pred['eta_token'] <= 1 - floor + 1e-6).all()
               and (pred['eta_scalar'] >= 0).all() and (pred['eta_scalar'] <= 1 - floor + 1e-6).all()),
          f"eta_max={float(pred['eta_token'].max()):.4f} floor_cap={1 - floor:.3f}")


def test_c_gate_step():
    print("\n[c] 게이트 학습 1 스텝 — none/loo")
    from tools.train_p55_gate import freeze_except_gate, gate_train_step

    def crit(logits, gt):
        return F.cross_entropy(logits, gt, ignore_index=255)

    # none: 게이트 grad≠0, 그 외 grad None
    m = _seeded(lambda: _build(_p55_cfg())).train()
    train_params = freeze_except_gate(m)
    x = _inputs()
    gt = torch.randint(0, K, (1, 64, 64))
    m.zero_grad(set_to_none=True)
    loss, log = gate_train_step(m, crit, x, gt, target='none')
    loss.backward()
    g_gate = sum(float(p.grad.abs().sum()) for p in train_params if p.grad is not None)
    check("(c-none) 게이트 grad≠0 (세그 손실만)", g_gate > 0, f"gnorm={g_gate:.3e}")
    others_none = all(p.grad is None for n, p in m.named_parameters()
                      if not n.startswith('p55_gate.'))
    check("(c-none) 그 외 전 파라미터 grad None", others_none)
    check("(c-none) 손실 유한", bool(torch.isfinite(loss)), f"loss={float(loss):.4f}")

    # loo: 타깃 형상 · 손실 유한 · 게이트 grad≠0
    m2 = _seeded(lambda: _build(_p55_cfg())).train()
    tp2 = freeze_except_gate(m2)
    m2.zero_grad(set_to_none=True)
    loss2, log2 = gate_train_step(m2, crit, x, gt, target='loo', tau=0.1, loo_w=1.0)
    loss2.backward()
    g2 = sum(float(p.grad.abs().sum()) for p in tp2 if p.grad is not None)
    check("(c-loo) loo_bce 로그 존재", 'loo_bce' in log2, str(list(log2.keys())))
    check("(c-loo) 손실 유한", bool(torch.isfinite(loss2)), f"loss={float(loss2):.4f}")
    check("(c-loo) 게이트 grad≠0", g2 > 0, f"gnorm={g2:.3e}")


def test_c2_condition_tool():
    print("\n[c2] 조건 분석 도구 — 합성 레코드")
    import numpy as np
    from tools.p55_gate_by_condition import compute_condition_report, build_md
    rng = np.random.RandomState(0)
    conds, cases, rows = [], [], []
    for _ in range(8):                       # clean: r_img 높음
        conds.append('sun'); cases.append('none')
        rows.append([rng.uniform(0.85, 0.95)] + list(rng.uniform(0.6, 0.8, 3)))
    for _ in range(8):                       # night: r_img 낮음(상황 적응 신호)
        conds.append('night'); cases.append('none')
        rows.append([rng.uniform(0.2, 0.4)] + list(rng.uniform(0.7, 0.9, 3)))
    r = np.array(rows)
    rep = compute_condition_report(r, conds, cases, MODALS)
    check("(c2) 그룹 2개", len(rep['groups']) == 2, str(list(rep['groups'])))
    check("(c2) night r_img < clean r_img → AUROC>0.5",
          rep['r_img_dark_vs_clean_auroc'] > 0.9, f"auroc={rep['r_img_dark_vs_clean_auroc']:.3f}")
    check("(c2) 붕괴 std>0(정적 아님)", rep['collapse_std_overall'] > 0)
    md = build_md(rep)
    check("(c2) md 표 생성", isinstance(md, str) and 'condition' in md and 'rgb/others' in md)


def test_d_loo_differs():
    print("\n[d] leave-one-out 로짓이 모달 0-화 시 full 과 다르다")
    m = _seeded(lambda: _build(_p55_cfg())).eval()
    x = _inputs()
    feats = m._encode_all(x)
    l_full = m.p55_fused_seg_logits(feats)
    diffs = []
    for mi in range(len(MODALS)):
        l_m = m.p55_fused_seg_logits(feats, zero_idx=mi)
        diffs.append(not torch.allclose(l_full, l_m))
    check("(d) 각 모달 0-화가 로짓을 바꾼다", all(diffs), str(diffs))


def test_e_override():
    print("\n[e] η̂ 오버라이드(all r=1 ⇒ eta=0)가 eta=0 경로와 일치")
    from tools.qaf_permutation_test import zero_override
    m = _seeded(lambda: _build(_p55_cfg())).eval()
    x = _inputs()
    # (1) override=zero → eta 강제 0
    m.fusion._qaf_eta_override = zero_override
    with torch.no_grad():
        out_ovr, _ = m(x, True)
    m.fusion._qaf_eta_override = None
    # (2) 게이트를 eta=0 스텁으로 교체
    real_gate = m.p55_gate

    class _ZeroGate(nn.Module):
        fractions = real_gate.fractions
        r_floor = real_gate.r_floor
        def forward(self, flat):
            B = flat[0].shape[0]; h, w = flat[0].shape[-2:]; Mn = len(MODALS)
            z = flat[0].new_zeros
            return ({'eta_scalar': z(B, Mn), 'eta_token': z(B, Mn, h, w)},
                    flat[0].new_ones(B, Mn, h, w), flat[0].new_ones(B, Mn))
    m.p55_gate = _ZeroGate()
    with torch.no_grad():
        out_zero, _ = m(x, True)
    m.p55_gate = real_gate
    check("(e) override-zero == eta0-gate", torch.allclose(out_ovr, out_zero, atol=1e-6),
          f"max|Δ|={(out_ovr - out_zero).abs().max().item():.2e}")


def main():
    print("== smoke_p55 ==")
    try:
        import timm  # noqa: F401
        has_timm = True
    except Exception as e:
        print(f"[SKIP] timm 부재 — 모델 항목 건너뜀: {e}")
        has_timm = False
    # 순수(모델 불필요) 항목은 timm 없어도 돈다
    test_c2_condition_tool()
    if has_timm:
        test_a_off_byte_identical()
        test_b_on_forward()
        test_c_gate_step()
        test_d_loo_differs()
        test_e_override()
    print()
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
