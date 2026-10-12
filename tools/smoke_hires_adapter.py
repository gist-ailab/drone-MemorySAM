"""[HIRES_ADAPTER] 고해상 경로 단독 토글 스모크 — CPU, ~1분.

검사 항목:
  (a) off 계약: ENABLE false ⇒ HIRES 키가 아예 없는 baseline 과 state_dict 키·값,
      forward 출력 byte-동일(allclose atol 0). off 모델에 hires 키 없음 + 같은 시드
      빌드 2회가 byte-동일(off 경로가 RNG 를 소비하지 않음 = RNG 스트림 불변).
  (b) on: forward 출력 shape 가 off 와 동일, backward 후 SPM·extractor·탭 투영·출력
      투영 파라미터 grad 가 전부 non-zero(이름별 assert).
  (c) MSDeformAttnTorch core 등가: 같은 입력·파라미터로 vendored
      ms_deform_attn_core_pytorch(rfdetr) 와 allclose(import 막히면 표준 core 복제와 비교).
  (d) 입력 256·320 두 크기에서 forward 통과(정렬 경로 포함).
  (e) 어댑터 파라미터 수 출력.

실행:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=semseg/models/sam2:. \\
        /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_hires_adapter.py
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
# tiny 백본은 12블록 — 어댑터가 요구하는 4탭을 블록 3/6/9/12 로 건다.
TAP_LAYERS = [3, 6, 9, 12]


def _base_cfg(img=64):
    """B0 계열을 축소한 tiny 학습 config. HIRES_ADAPTER/TAPS 는 호출부가 덧붙인다."""
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


def _hires_block(dim=64, stem_dim=16, mode='shared_stem'):
    return {'ENABLE': True, 'DIM': dim, 'STEM_DIM': stem_dim, 'MODE': mode,
            'N_HEADS': 8, 'N_POINTS': 4, 'DEFORM_RATIO': 0.5, 'CFFN_RATIO': 0.25,
            'EXTRA_EXTRACTORS': 2}


def _taps_block():
    return {'ENABLE': True, 'LAYERS': list(TAP_LAYERS), 'MODE': 'per_modal'}


def _build(cfg):
    from semseg.models.reliadino.model import build_reliadino
    return build_reliadino(cfg, NUM_CLASSES)


def _inputs(B=1, S=64):
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


def _seeded(fn):
    torch.manual_seed(20261012)
    return fn()


def _hires_params(m):
    return sum(p.numel() for p in m.hires.parameters())


# ── (c) 표준 core 복제(vendored import 막힐 때의 ground-truth) ───────────────
def _canonical_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    import torch.nn.functional as F
    N_, _, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    split = [int(h) * int(w) for h, w in value_spatial_shapes]
    value_list = value.split(split, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    out_list = []
    for lid, (h, w) in enumerate(value_spatial_shapes):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, int(h), int(w))
        g = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out_list.append(F.grid_sample(v, g, mode='bilinear', padding_mode='zeros',
                                      align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    out = (torch.stack(out_list, dim=-2).flatten(-2) * aw).sum(-1).view(N_, M_ * D_, Lq_)
    return out.transpose(1, 2).contiguous()


def _check_core_equiv():
    from semseg.models.reliadino.hires_adapter import _ms_deform_attn_core_pytorch
    torch.manual_seed(7)
    N, M, D, Lq, P = 2, 4, 8, 50, 4
    h, w = 7, 9
    shapes = [(h, w)]
    value = torch.randn(N, h * w, M, D)
    loc = torch.rand(N, Lq, M, 1, P, 2)                      # ∈ [0,1]
    aw = torch.rand(N, Lq, M, 1, P)
    mine = _ms_deform_attn_core_pytorch(value, shapes, loc, aw)

    # 우선 vendored(rfdetr) 와 비교(레이아웃 변환). import 막히면 표준 복제와 비교.
    ref, ref_name = None, None
    try:
        vend = REPO / 'semseg' / 'models' / 'rfdetr_head' / '_vendor'
        if str(vend) not in sys.path:
            sys.path.insert(0, str(vend))
        from rfdetr.models.ops.functions.ms_deform_attn_func import \
            ms_deform_attn_core_pytorch as vendored
        # vendored value 레이아웃: (N, n_heads, head_dim, spatial)
        v_vendored = value.permute(0, 2, 3, 1).contiguous()
        sp = torch.as_tensor(shapes, dtype=torch.long)
        ref = vendored(v_vendored, sp, loc, aw, value_spatial_shapes_hw=shapes)
        ref_name = "rfdetr.ms_deform_attn_core_pytorch (vendored)"
    except Exception as e:                                   # noqa: BLE001
        ref = _canonical_core(value, shapes, loc, aw)
        ref_name = f"표준 core 복제 (vendored import 실패: {type(e).__name__})"

    ok = mine.shape == ref.shape and torch.allclose(mine, ref, atol=1e-5, rtol=1e-4)
    md = (mine - ref).abs().max().item() if mine.shape == ref.shape else float('nan')
    check(f"c: MSDeformAttn core 등가 [{ref_name}]", ok, f"max|Δ|={md:.3e}")


def main():
    try:
        import timm  # noqa: F401
    except Exception as e:
        print(f"[SKIP] timm 부재 — 스모크 건너뜀: {e}")
        return 0

    x = _inputs(S=64)

    # ── (a) off 계약 ────────────────────────────────────────────────────────
    print("\n[a] HIRES off ⇒ baseline(키 없음) byte-동일 + RNG 스트림 불변")
    cfg_base = _base_cfg()                                   # HIRES 키 없음, TAPS 도 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['HIRES_ADAPTER'] = _hires_block()
    cfg_off['MODEL']['HIRES_ADAPTER']['ENABLE'] = False
    m_base = _seeded(lambda: _build(cfg_base)).eval()
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    k_base = list(m_base.state_dict().keys())
    k_off = list(m_off.state_dict().keys())
    check("off: state_dict 키가 baseline 과 동일", k_base == k_off,
          f"base={len(k_base)} off={len(k_off)} diff={set(k_base) ^ set(k_off)}")
    check("off: state_dict 에 hires 키 없음",
          not any('hires' in k for k in k_off))
    check("off: self.hires 미생성", m_off.hires is None)
    sd_b, sd_o = m_base.state_dict(), m_off.state_dict()
    vals_equal = all(torch.equal(sd_b[k], sd_o[k]) for k in k_base)
    check("off: state_dict 값 byte-동일 (RNG 스트림 불변)", vals_equal)
    with torch.no_grad():
        o_base, _ = m_base(x, True)
        o_off, _ = m_off(x, True)
    check("off: forward 출력 byte-동일 (allclose atol 0)",
          o_base.shape == o_off.shape and torch.equal(o_base, o_off),
          f"max|Δ|={(o_base - o_off).abs().max().item() if o_base.shape == o_off.shape else 'shape≠'}")

    # ── (b) on: grad 도달(이름별) ───────────────────────────────────────────
    print("\n[b] HIRES on ⇒ 출력 shape=off, SPM·extractor·탭투영·출력투영 grad>0")
    cfg_on = _base_cfg()
    cfg_on['MODEL']['TAPS'] = _taps_block()
    cfg_on['MODEL']['HIRES_ADAPTER'] = _hires_block()
    m_on = _seeded(lambda: _build(cfg_on)).train()
    check("on: self.hires 생성", m_on.hires is not None)
    gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
    out = m_on(x, True, gt)
    logits = out[0]
    check("on: 출력 shape 가 off 와 동일", logits.shape == o_off.shape,
          f"{tuple(logits.shape)} vs {tuple(o_off.shape)}")
    check("on: 출력 유한", torch.isfinite(logits).all().item())
    logits.float().sum().backward()

    def _grp_norm(substr):
        return sum(float(p.grad.detach().float().norm())
                   for n, p in m_on.hires.named_parameters()
                   if substr in n and p.grad is not None)
    groups = {'spm': 'spm.', 'extractor': 'interactions.',
              'tap_proj': 'tap_proj.', 'out_proj': 'out_proj.'}
    for label, sub in groups.items():
        gn = _grp_norm(sub)
        check(f"on: {label} grad 노름>0 (도달)", gn > 0.0, f"Σ‖grad‖={gn:.4e}")

    # ── (c) MSDeformAttn core 등가 ──────────────────────────────────────────
    print("\n[c] MSDeformAttnTorch core 등가 (vendored 또는 표준 복제)")
    _check_core_equiv()

    # ── (d) 256·320 두 크기 forward ─────────────────────────────────────────
    print("\n[d] 입력 256·320 두 크기 forward 통과(정렬 경로 포함)")
    for S in (256, 320):
        cfg_s = _base_cfg(img=S)
        cfg_s['MODEL']['TAPS'] = _taps_block()
        cfg_s['MODEL']['HIRES_ADAPTER'] = _hires_block()
        m_s = _seeded(lambda: _build(cfg_s)).eval()
        xs = _inputs(S=S)
        with torch.no_grad():
            os_, _ = m_s(xs, True)
        check(f"d: {S}² 출력 shape (B,K,{S},{S})",
              tuple(os_.shape) == (1, NUM_CLASSES, S, S), str(tuple(os_.shape)))
        check(f"d: {S}² 출력 유한", torch.isfinite(os_).all().item())

    # ── (e) 어댑터 파라미터 수 ──────────────────────────────────────────────
    print("\n[e] 어댑터 파라미터 수")
    n_tiny = _hires_params(m_on)
    print(f"    [INFO] HiResAdapter params (tiny: DIM=64, STEM_DIM=16, fpn_dim=64) = {n_tiny:,}")
    m_full = _seeded(lambda: _build_full_dim()).eval()
    n_full = _hires_params(m_full)
    print(f"    [INFO] HiResAdapter params (DIM=512, STEM_DIM=64, fpn_dim=256) = {n_full:,}")

    print("\n" + ("=" * 60))
    if _FAILS:
        print(f"[HIRES SMOKE] FAIL {len(_FAILS)}건: {_FAILS}")
        return 1
    print("[HIRES SMOKE] 전 항목 PASS")
    return 0


def _build_full_dim():
    """실 학습 폭(DIM=512, STEM_DIM=64, fpn_dim=256)에서 어댑터 파라미터 수를 재는 용도."""
    cfg = _base_cfg()
    cfg['MODEL']['FPN_DIM'] = 256
    cfg['MODEL']['TAPS'] = _taps_block()
    cfg['MODEL']['HIRES_ADAPTER'] = _hires_block(dim=512, stem_dim=64)
    return _build(cfg)


if __name__ == '__main__':
    raise SystemExit(main())
