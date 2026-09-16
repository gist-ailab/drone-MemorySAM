"""[DETAIL_BRANCH] 고해상도 세부 가지 단독 토글 스모크 — CPU, ~1분.

검사 항목:
  (a) ENABLE false ⇒ DETAIL 키가 아예 없는 baseline 과 state_dict 키·forward 출력
      byte-동일 (off 계약).
  (b) ENABLE true  ⇒ 출력 유한, 세부 가지 파라미터와 레벨 게이트 g 에 gradient 도달,
      게이트 init 이 0.1.
  (c) 4모달 768² 더미 입력에서 forward 형태(off 와 동일) 확인 + 세부 가지 파라미터 수 출력.
  (d) TAPS on + DETAIL on 동시 forward 정상(두 경로 독립).
  (e) per_modal 모드 동작(모달별 스템 생성 + grad 도달).

실행:
    python tools/smoke_detail_branch.py
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


def _base_cfg(img=64):
    """B0 계열을 축소한 tiny 학습 config. DETAIL_BRANCH/TAPS 는 호출부가 덧붙인다."""
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
        'TRAIN': {'IMAGE_SIZE': [img, img]},
    }


def _build(cfg):
    from semseg.models.reliadino.model import build_reliadino
    return build_reliadino(cfg, NUM_CLASSES)


def _inputs(B=1, S=64):
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


def _seeded(fn):
    torch.manual_seed(20260917)
    return fn()


def _detail_params(m):
    return sum(p.numel() for p in m.detail.parameters())


def main():
    try:
        import timm  # noqa: F401
    except Exception as e:
        print(f"[SKIP] timm 부재 — 스모크 건너뜀: {e}")
        return 0

    x = _inputs(S=64)

    # ── (a) DETAIL off ⇒ baseline(키 없음) 과 byte-동일 ──────────────────────
    print("\n[a] DETAIL off ⇒ baseline byte-동일 (state_dict 키 + forward 출력)")
    cfg_base = _base_cfg()                                  # DETAIL 키 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['DETAIL_BRANCH'] = {'ENABLE': False, 'STEM_DIM': 32,
                                         'LEVELS': [4, 8], 'MODE': 'shared_stem',
                                         'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    m_base = _seeded(lambda: _build(cfg_base)).eval()
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    k_base = list(m_base.state_dict().keys())
    k_off = list(m_off.state_dict().keys())
    check("off: state_dict 키가 baseline 과 동일", k_base == k_off,
          f"base={len(k_base)} off={len(k_off)} diff={set(k_base) ^ set(k_off)}")
    check("off: self.detail 미생성", m_off.detail is None)
    with torch.no_grad():
        o_base, _ = m_base(x, True)
        o_off, _ = m_off(x, True)
    check("off: forward 출력 byte-동일", torch.equal(o_base, o_off),
          f"max|Δ|={(o_base - o_off).abs().max().item() if o_base.shape == o_off.shape else 'shape≠'}")

    # ── (b) DETAIL on ⇒ 출력 유한 + 세부/게이트 grad 도달 + gate init 0.1 ────
    print("\n[b] DETAIL on(shared_stem) ⇒ 출력 유한 + 세부/게이트 grad>0 + gate init 0.1")
    cfg_on = _base_cfg()
    cfg_on['MODEL']['DETAIL_BRANCH'] = {'ENABLE': True, 'STEM_DIM': 32,
                                        'LEVELS': [4, 8], 'MODE': 'shared_stem',
                                        'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    m_on = _seeded(lambda: _build(cfg_on)).train()
    check("on: self.detail 생성", m_on.detail is not None)
    check("on: gate init == 0.1", torch.allclose(m_on.detail.gate.detach(),
          torch.full_like(m_on.detail.gate, 0.1)),
          f"gate={m_on.detail.gate.detach().tolist()}")
    check("on: strides==[4,8] pyr_index==[0,1]",
          m_on.detail.strides == [4, 8] and m_on.detail.pyr_index == [0, 1],
          f"strides={m_on.detail.strides} pyr_index={m_on.detail.pyr_index}")
    gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
    out = m_on(x, True, gt)
    logits = out[0]
    check("on: 출력 shape 가 off 와 동일", logits.shape == o_off.shape,
          f"{tuple(logits.shape)} vs {tuple(o_off.shape)}")
    check("on: 출력 유한", torch.isfinite(logits).all().item())
    logits.float().sum().backward()
    gnorm = sum(float(p.grad.detach().float().norm())
                for p in m_on.detail.parameters() if p.grad is not None)
    check("on: 세부 가지 파라미터 grad 노름>0 (도달)", gnorm > 0.0, f"Σ‖grad‖={gnorm:.4e}")
    gg = m_on.detail.gate.grad
    check("on: 게이트 g grad 도달(≠None, 노름>0)",
          gg is not None and float(gg.norm()) > 0.0,
          f"‖g.grad‖={float(gg.norm()) if gg is not None else 'None'}")

    # ── (c) 4모달 768² forward 형태 + 세부 파라미터 수 ──────────────────────
    print("\n[c] 4모달 768² forward 형태(off 와 동일) + 세부 파라미터 수")
    cfg_hd = _base_cfg(img=768)
    cfg_hd['MODEL']['DETAIL_BRANCH'] = {'ENABLE': True, 'STEM_DIM': 32,
                                        'LEVELS': [4, 8], 'MODE': 'shared_stem',
                                        'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    m_hd = _seeded(lambda: _build(cfg_hd)).eval()
    x_hd = _inputs(S=768)
    with torch.no_grad():
        o_hd, _ = m_hd(x_hd, True)
    # 모델은 로짓을 입력 해상도로 업샘플해 반환한다(off 도 동일 규약).
    check("hd: 768² 출력 shape (B,K,768,768)",
          tuple(o_hd.shape) == (1, NUM_CLASSES, 768, 768), str(tuple(o_hd.shape)))
    check("hd: 768² 출력 유한", torch.isfinite(o_hd).all().item())
    n_det = _detail_params(m_hd)
    print(f"    [INFO] DetailStem params (shared_stem, STEM_DIM=32, fpn_dim=64) = {n_det:,}")
    # 참고: 실제 학습(fpn_dim=256)의 파라미터 수는 별도 계산 — 아래 estimate 출력.
    m_full = _seeded(lambda: _build_full_dim()).eval()
    n_full = _detail_params(m_full)
    print(f"    [INFO] DetailStem params (shared_stem, STEM_DIM=32, fpn_dim=256) = {n_full:,}")
    check("hd: 세부 파라미터 총량 < 1M (목표)", n_full < 1_000_000, f"{n_full:,}")

    # ── (d) TAPS on + DETAIL on 동시 ────────────────────────────────────────
    print("\n[d] TAPS on + DETAIL on 동시 forward 정상(두 경로 독립)")
    cfg_td = _base_cfg()
    cfg_td['MODEL']['TAPS'] = {'ENABLE': True, 'LAYERS': [3, 6, 9, 12],
                               'MODE': 'per_modal'}
    cfg_td['MODEL']['DETAIL_BRANCH'] = {'ENABLE': True, 'STEM_DIM': 16,
                                        'LEVELS': [4, 8], 'MODE': 'shared_stem',
                                        'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    m_td = _seeded(lambda: _build(cfg_td)).train()
    check("td: taps_proj_permodal + detail 둘 다 생성",
          m_td.taps_proj_permodal is not None and m_td.detail is not None)
    out_td = m_td(x, True, gt)
    check("td: forward 출력 shape 동일 + 유한",
          out_td[0].shape == o_off.shape and torch.isfinite(out_td[0]).all().item(),
          str(tuple(out_td[0].shape)))
    out_td[0].float().sum().backward()
    gd = sum(float(p.grad.detach().float().norm())
             for p in m_td.detail.parameters() if p.grad is not None)
    gtp = sum(float(p.grad.detach().float().norm())
              for p in m_td.taps_proj_permodal.parameters() if p.grad is not None)
    check("td: detail grad>0 AND taps 투영 grad>0 (독립 경로 둘 다 흐름)",
          gd > 0.0 and gtp > 0.0, f"detail={gd:.3e} taps={gtp:.3e}")

    # ── (e) per_modal 모드 ──────────────────────────────────────────────────
    print("\n[e] DETAIL MODE=per_modal ⇒ 모달별 스템 생성 + grad 도달")
    cfg_pm = _base_cfg()
    cfg_pm['MODEL']['DETAIL_BRANCH'] = {'ENABLE': True, 'STEM_DIM': 16,
                                        'LEVELS': [4, 8], 'MODE': 'per_modal',
                                        'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    m_pm = _seeded(lambda: _build(cfg_pm)).train()
    check("pm: stems(모달별) 생성 + shared_stem None",
          m_pm.detail.stems is not None and m_pm.detail.shared_stem is None,
          f"stems={None if m_pm.detail.stems is None else len(m_pm.detail.stems)}")
    check("pm: 모달별 스템 수 == 4", len(m_pm.detail.stems) == len(MODALS))
    out_pm = m_pm(x, True, gt)
    check("pm: forward shape 동일 + 유한",
          out_pm[0].shape == o_off.shape and torch.isfinite(out_pm[0]).all().item())
    out_pm[0].float().sum().backward()
    gpm = sum(float(p.grad.detach().float().norm())
              for p in m_pm.detail.parameters() if p.grad is not None)
    check("pm: per_modal 세부 grad 노름>0", gpm > 0.0, f"Σ‖grad‖={gpm:.4e}")

    print("\n" + ("=" * 60))
    if _FAILS:
        print(f"[DETAIL SMOKE] FAIL {len(_FAILS)}건: {_FAILS}")
        return 1
    print("[DETAIL SMOKE] 전 항목 PASS")
    return 0


def _build_full_dim():
    """fpn_dim=256(실 학습 값)에서 DetailStem 파라미터 수를 재는 용도."""
    cfg = _base_cfg()
    cfg['MODEL']['FPN_DIM'] = 256
    cfg['MODEL']['DETAIL_BRANCH'] = {'ENABLE': True, 'STEM_DIM': 32,
                                     'LEVELS': [4, 8], 'MODE': 'shared_stem',
                                     'GATE_INIT': 0.1, 'INPUT_NORM': 'same'}
    return _build(cfg)


if __name__ == '__main__':
    raise SystemExit(main())
