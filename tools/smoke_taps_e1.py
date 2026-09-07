"""[E1] 다중 레벨 탭 읽기(MODEL.TAPS) 단독 토글 스모크 — CPU, ~1분.

검사 항목:
  (a) TAPS off  ⇒ TAPS 키가 아예 없는 baseline 과 state_dict 키·forward 출력 byte-동일.
  (b) TAPS on(per_modal) ⇒ 출력 shape 가 off 와 동일, 탭 투영(taps_proj_permodal)에
      gradient 도달(backward 후 grad 노름 > 0). mean 모드도 같은 성질 확인.
  (c) P43(M2F_HEAD on) 동작 불변 ⇒ 빌드·forward 성공, mean 경로(p43_lateral 사용,
      taps_proj_permodal None)로 붙는다.

실행:
    python tools/smoke_taps_e1.py
timm(+torch) 이 없으면 SKIP 한다. 실제 pretrained 백본은 받지 않는다
(PRETRAINED_BACKBONE=False, vit_tiny_patch16_224 = 12블록).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_FAILS = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


NUM_CLASSES = 6
MODALS = ['img', 'depth', 'event', 'lidar']


def _base_cfg():
    """B0 계열을 축소한 tiny 학습 config. TAPS 블록은 호출부가 덧붙인다."""
    return {
        'MODEL': {
            'BACKBONE_TIMM': 'vit_tiny_patch16_224',
            'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
            'PRETRAINED_BACKBONE': False, 'LORA_R': 2, 'FPN_DIM': 64,
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0,
                       'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
                       'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False},
            'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': False},
            'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
            'M2F': {'ENABLE': False},
            'P39': {'TRUNK_EXP': True, 'ARBITER': False, 'TRUNK_MODE': 'gated_mlp',
                    'TRUNK_HIDDEN': 64,
                    'VICREG': {'ENABLE': False}},
        },
        'DATASET': {'MODALS': list(MODALS)},
        'TRAIN': {'IMAGE_SIZE': [64, 64]},
    }


def _build(cfg):
    from semseg.models.reliadino.model import build_reliadino
    return build_reliadino(cfg, NUM_CLASSES)


def _inputs(B=1, S=64):
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


def _seeded(fn):
    torch.manual_seed(20260907)
    return fn()


def main():
    try:
        import timm  # noqa: F401
    except Exception as e:
        print(f"[SKIP] timm 부재 — 스모크 건너뜀: {e}")
        return 0

    # ── (a) TAPS off ⇒ baseline(키 없음) 과 byte-동일 ────────────────────────
    print("\n[a] TAPS off ⇒ baseline byte-동일 (state_dict 키 + forward 출력)")
    cfg_base = _base_cfg()                                  # TAPS 키 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['TAPS'] = {'ENABLE': False, 'LAYERS': [3, 6, 9, 12],
                                'MODE': 'per_modal'}
    m_base = _seeded(lambda: _build(cfg_base)).eval()
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    k_base = list(m_base.state_dict().keys())
    k_off = list(m_off.state_dict().keys())
    check("off: state_dict 키가 baseline 과 동일", k_base == k_off,
          f"base={len(k_base)} off={len(k_off)} "
          f"diff={set(k_base) ^ set(k_off)}")
    check("off: taps_proj_permodal 미생성", m_off.taps_proj_permodal is None)
    check("off: p43_lateral 미생성", m_off.p43_lateral is None)
    x = _inputs()
    with torch.no_grad():
        o_base, _ = m_base(x, True)
        o_off, _ = m_off(x, True)
    check("off: forward 출력 byte-동일", torch.equal(o_base, o_off),
          f"max|Δ|={ (o_base - o_off).abs().max().item() if o_base.shape==o_off.shape else 'shape≠'}")

    # ── (b) TAPS on ⇒ 출력 shape 동일 + 탭 투영에 grad 도달 ──────────────────
    print("\n[b] TAPS on ⇒ 출력 shape 동일 + 탭 투영 grad>0")
    for mode, proj_attr in (('per_modal', 'taps_proj_permodal'),
                            ('mean', 'p43_lateral')):
        cfg_on = _base_cfg()
        cfg_on['MODEL']['TAPS'] = {'ENABLE': True, 'LAYERS': [3, 6, 9, 12],
                                   'MODE': mode}
        m_on = _seeded(lambda: _build(cfg_on)).train()
        # 탭 층 해석 확인: 1-indexed [3,6,9,12] → 12블록에서 [2,5,8,11]
        check(f"on({mode}): tap_layers=resolve_taps([3,6,9,12]|n=12)",
              m_on.encoder.tap_layers == [2, 5, 8, 11],
              str(m_on.encoder.tap_layers))
        proj = getattr(m_on, proj_attr)
        check(f"on({mode}): {proj_attr} 생성", proj is not None)
        check(f"on({mode}): taps_mode=={mode}", m_on.taps_mode == mode)
        gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
        out = m_on(x, True, gt)
        logits = out[0]
        check(f"on({mode}): 출력 shape 가 off 와 동일", logits.shape == o_off.shape,
              f"{tuple(logits.shape)} vs {tuple(o_off.shape)}")
        logits.float().sum().backward()
        # 탭 투영 파라미터 grad 노름 합
        gnorm = 0.0
        n_par = 0
        for p in proj.parameters():
            n_par += 1
            if p.grad is not None:
                gnorm += float(p.grad.detach().float().norm())
        check(f"on({mode}): 탭 투영 grad 노름>0 (도달 확인)", gnorm > 0.0,
              f"Σ‖grad‖={gnorm:.4e} over {n_par} params")

    # ── (c) P43(M2F_HEAD on) 동작 불변 ───────────────────────────────────────
    print("\n[c] P43 (M2F_HEAD on) — 빌드·forward 성공 + mean 경로")
    cfg_p43 = _base_cfg()
    cfg_p43['MODEL']['P43'] = {'M2F_HEAD': True, 'LATERAL': True, 'NUM_TAPS': 3,
                               'NUM_QUERIES': 8, 'DEC_LAYERS': 3, 'DIM': 64,
                               'NUM_HEADS': 4, 'NUM_POINTS': 256}
    m_p43 = _seeded(lambda: _build(cfg_p43)).train()
    check("P43: p43_lateral 사용(mean 경로)", m_p43.p43_lateral is not None)
    check("P43: taps_proj_permodal None(per_modal 미개입)",
          m_p43.taps_proj_permodal is None)
    check("P43: taps_mode=='mean'", m_p43.taps_mode == 'mean')
    check("P43: mask-cls 헤드 생성", m_p43.p43 is not None)
    # num_taps=3 → 등간격 [2,5,8] (n=12)
    check("P43: tap_layers=num_taps 등간격 [2,5,8]",
          m_p43.encoder.tap_layers == [2, 5, 8], str(m_p43.encoder.tap_layers))
    gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
    out = m_p43(x, True, gt)
    check("P43: forward 성공(로짓 반환, off 와 같은 shape)",
          out[0].shape == o_off.shape, str(tuple(out[0].shape)))

    print("\n" + ("=" * 60))
    if _FAILS:
        print(f"[E1 SMOKE] FAIL {len(_FAILS)}건: {_FAILS}")
        return 1
    print("[E1 SMOKE] 전 항목 PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
