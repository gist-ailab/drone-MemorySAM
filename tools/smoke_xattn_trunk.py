"""[A/B trunk] cross-attention 트렁크 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_xattn_trunk.py

대상 = `MODEL.FUSION.TRUNK: gated_mlp(기본) | xattn` 토글. 통제 A/B이므로
"기본값에서 아무것도 안 변한다"가 첫 번째 검사다.

검사 항목
  A. 기본값 등가성 : TRUNK 키 없음 ↔ TRUNK: gated_mlp 가 같은 seed 에서
                     state_dict 키·값 전부 일치 + eval logits |Δ|max == 0.
                     신규 파라미터 0개, trunk_xattn 미생성 (init RNG 소비 없음).
  B. 키1 gradient  : xattn 팔 1-step fwd+bwd 에서 **모든 모달의 모든 프로젝션**
                     (q/k/v/proj/mlp/LayerScale)에 gradient 가 실제로 도달.
                     + LayerScale init == 1.0, γ init == 0.1 (zero-init 금지).
  C. 출력 shape    : `_apply_trunk_exp` 출력과 최종 logits shape 이 두 팔에서 동일.
                     (트렁크 인터페이스가 바뀌지 않았다는 실증)
  D. eval 결정론   : xattn 팔 eval() 2회 forward 가 bitwise 동일 + 손실 aux 키
                     집합이 두 팔에서 동일(VICReg 등 기존 학습손실 유지).
  E. 파라미터 수   : 총 파라미터 + 트렁크 단독 파라미터의 gated_mlp 대비 증가분.
  F. 가드          : TRUNK:xattn + TRUNK_EXP:false → 조용히 baseline 을 돌리지
                     않고 ValueError. 잘못된 TRUNK 값도 ValueError.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402

K = 25                      # DELIVER 클래스 수
MODALS = ['img', 'depth', 'event', 'lidar']      # P46-c3only 대표 레시피와 동일
SIZE = 128                  # tiny ViT-16 -> 8x8 토큰
BS = 2

# 대표 레시피(configs/deliver/deliver_rgbdel_P46_c3only_xattn_trunk.yaml)의 값
XA_LAYERS, XA_HEADS, XA_MLP_RATIO = 1, 8, 2.0
LS_INIT, GAMMA_INIT = 1.0, 0.1


def base_cfg(trunk=None, trunk_exp=True, xattn=None):
    """P46-c3only(λ0.1) config 의 토글 구성을 tiny 스케일로 옮긴 것.

    trunk=None 이면 FUSION.TRUNK 키 자체를 넣지 않는다 (= 기존 config 전부)."""
    fusion = {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 4.0,
              'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
              'ATTN_BIAS': {'ENABLE': False}}
    if trunk is not None:
        fusion['TRUNK'] = trunk
    if xattn is not None:
        fusion['XATTN'] = xattn
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE_TIMM': 'vit_tiny_patch16_224',
        'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
        'PRETRAINED_BACKBONE': False,
        'LORA_R': 4, 'FPN_DIM': 64,
        'FUSION': fusion,
        'CONSISTENCY': {'ENABLE': False},
        'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
        'CALIBRATION': {'ENABLE': False},
        'ROUTER': {'ENABLE': True, 'HIDDEN': 16},          # P36 — 대표 레시피 유지
        'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
        'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 2, 'DIM': 64,
                'NUM_HEADS': 4, 'POINTS': 256, 'SRC': 'modal',
                'ANCHORED': True, 'POINT_QUOTA': 8, 'LOSS_W': 0.5},
        'P39': {'TRUNK_EXP': trunk_exp, 'ARBITER': True,
                'TRUNK_MODE': 'gated_mlp', 'TRUNK_HIDDEN': 32,
                'VICREG': {'ENABLE': True, 'TOKENS': 128}},
        'P46': {'C3_PROTO': {'ENABLE': True, 'FEATURE': 'mfeat', 'LAMBDA': 0.1,
                             'PIXELS': 256, 'WARMUP_EP': 5}},
        'MODAL_DROPOUT': {'ENABLE': False},
    }
    return {'MODEL': m, 'DATASET': {'MODALS': MODALS},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


XA_CFG = {'LAYERS': XA_LAYERS, 'NUM_HEADS': XA_HEADS,
          'MLP_RATIO': XA_MLP_RATIO, 'LS_INIT': LS_INIT,
          'GAMMA_INIT': GAMMA_INIT}


def make_batch(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g).to(device) for _ in MODALS]
    y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g).to(device)
    y[:, :8, :8] = 255                          # ignore 영역도 섞는다
    return x, y


def build(device, seed=0, **kw):
    torch.manual_seed(seed)
    return build_reliadino(base_cfg(**kw), K).to(device)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def group_grad(model, pred):
    tot = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and pred(n) and p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


# ── A. 기본값 등가성 (TRUNK 키 없음 ↔ gated_mlp) ────────────────────────────
def check_default_identity(device):
    a = build(device, seed=1234)                          # FUSION.TRUNK 키 없음
    b = build(device, seed=1234, trunk='gated_mlp')       # 명시적 기본값
    sa, sb = a.state_dict(), b.state_dict()
    same_keys = list(sa.keys()) == list(sb.keys())
    same_vals = same_keys and all(torch.equal(sa[k], sb[k]) for k in sa)
    new_params = [k for k in sb if 'trunk_xattn' in k]
    no_module = a.trunk_xattn is None and b.trunk_xattn is None
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=7)
    with torch.no_grad():
        la, _ = a(x, True)
        lb, _ = b(x, True)
    dmax = float((la - lb).abs().max())
    ok = same_keys and same_vals and not new_params and no_module and torch.equal(la, lb)
    return ok, dict(same_keys=same_keys, same_vals=same_vals,
                    n_keys=len(sa), new_params=len(new_params),
                    trunk_xattn_is_None=no_module, dmax=dmax)


# ── B. xattn gradient 도달 (전 모달 × 전 프로젝션) + init 값 ────────────────
def check_xattn_grad(device):
    m = build(device, seed=7, trunk='xattn', xattn=XA_CFG)
    assert m.trunk_xattn is not None, 'trunk_xattn 이 만들어지지 않았다'
    # init 검사 — LayerScale 1.0 / γ 0.1 (zero-init 금지, 원장 키1)
    ls_ok = all(
        float(l.ls_attn.min()) == LS_INIT and float(l.ls_attn.max()) == LS_INIT
        and float(l.ls_mlp.min()) == LS_INIT and float(l.ls_mlp.max()) == LS_INIT
        for l in m.trunk_xattn.layers)
    gamma_ok = torch.allclose(m.trunk_gamma.detach(),
                              torch.full_like(m.trunk_gamma, GAMMA_INIT))

    m.train()
    m._current_epoch = 10
    x, y = make_batch(device, seed=3)
    logits, _, aux = m(x, True, gt_mask=y)
    total = F.cross_entropy(logits, y, ignore_index=255)
    for k, v in aux.items():
        if torch.is_tensor(v) and v.dim() == 0:
            total = total + v
    m.zero_grad(set_to_none=True)
    total.backward()

    # 모달별 × 프로젝션별 gradient L2. 하나라도 0이면 그 경로는 사장된 것이다.
    rows, ok = [], ls_ok and bool(gamma_ok)
    for i, name in enumerate(MODALS):
        cell = {}
        for tag in ('q', 'k', 'v', 'proj', 'mlp', 'norm_q', 'norm_kv', 'norm2'):
            g = group_grad(m, lambda n, t=tag, i=i:
                           n.startswith('trunk_xattn.layers.')
                           and f'.{t}.{i}.' in n)
            cell[tag] = g
            ok &= g > 0
        # LayerScale 은 (m, dim) 한 텐서 → 모달 슬라이스로 본다
        for tag in ('ls_attn', 'ls_mlp'):
            g = 0.0
            for n, p in m.named_parameters():
                if n.startswith('trunk_xattn.layers.') and n.endswith('.' + tag) \
                        and p.grad is not None:
                    g += float(p.grad[i].detach().pow(2).sum())
            cell[tag] = g ** 0.5
            ok &= cell[tag] > 0
        g_gamma = float(m.trunk_gamma.grad[i].abs()) if m.trunk_gamma.grad is not None else 0.0
        cell['gamma'] = g_gamma
        ok &= g_gamma > 0
        rows.append((name, cell))
    return ok, rows, ls_ok, bool(gamma_ok), float(total)


# ── C. 출력 shape 동일 (트렁크 인터페이스 불변) ─────────────────────────────
def check_shapes(device):
    a = build(device, seed=11)                              # gated_mlp
    b = build(device, seed=11, trunk='xattn', xattn=XA_CFG)
    dim = a.fusion.num_modalities and a.trunk_exp[0][1].in_channels
    h = SIZE // 16
    fused = torch.randn(BS, dim, h, h, device=device)
    feats = [torch.randn(BS, dim, h, h, device=device) for _ in MODALS]
    with torch.no_grad():
        oa = a._apply_trunk_exp(fused, feats)
        ob = b._apply_trunk_exp(fused, feats)
        ya = b.trunk_xattn.modal_outputs(feats)
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=5)
    with torch.no_grad():
        la, ma = a(x, True)
        lb, mb = b(x, True)
    ok = (oa.shape == ob.shape == fused.shape
          and all(t.shape == feats[0].shape for t in ya)
          and la.shape == lb.shape and ma.shape == mb.shape
          # 같은 입력에서 값이 달라야 실제로 다른 트렁크다 (no-op 아님)
          and not torch.equal(oa, ob))
    return ok, tuple(oa.shape), tuple(ob.shape), tuple(la.shape), tuple(lb.shape), \
        float((oa - ob).abs().max())


# ── D. eval 결정론 + aux 키 집합 동일 ───────────────────────────────────────
def check_determinism_and_aux(device):
    b = build(device, seed=21, trunk='xattn', xattn=XA_CFG)
    b.eval()
    x, y = make_batch(device, seed=9)
    with torch.no_grad():
        l1, _ = b(x, True)
        l2, _ = b(x, True)
    det = torch.equal(l1, l2)

    a = build(device, seed=21)
    keys = {}
    for tag, m in (('gated_mlp', a), ('xattn', b)):
        m.train()
        m._current_epoch = 10
        torch.manual_seed(99)
        _, _, aux = m(x, True, gt_mask=y)
        keys[tag] = sorted(aux.keys())
    same_aux = keys['gated_mlp'] == keys['xattn']
    return det and same_aux, det, keys, float((l1 - l2).abs().max())


# ── E. 파라미터 수 ──────────────────────────────────────────────────────────
def check_params(device):
    a = build(device, seed=31)
    rows = []
    tot_a, trunk_a = n_params(a), n_params(a.trunk_exp) + a.trunk_gamma.numel()
    rows.append(('gated_mlp', tot_a, trunk_a, 1.0, 1.0))
    for lay in (1, 2):
        cfg = dict(XA_CFG, LAYERS=lay)
        b = build(device, seed=31, trunk='xattn', xattn=cfg)
        tot_b = n_params(b)
        trunk_b = n_params(b.trunk_xattn) + b.trunk_gamma.numel()
        rows.append((f'xattn L{lay}', tot_b, trunk_b,
                     trunk_b / trunk_a, tot_b / tot_a))
        del b
    return rows


# ── F. 가드 (조용한 실패 금지) ──────────────────────────────────────────────
def check_guards(device):
    out = {}
    for tag, kw in (('xattn + TRUNK_EXP:false', dict(trunk='xattn', trunk_exp=False,
                                                     xattn=XA_CFG)),
                    ('TRUNK: bogus', dict(trunk='bogus'))):
        try:
            build(device, seed=1, **kw)
            out[tag] = (False, 'no error')
        except ValueError as e:                              # noqa: PERF203
            out[tag] = (True, str(e).split('\n')[0][:70])
        except Exception as e:                               # noqa: BLE001
            out[tag] = (False, f'{type(e).__name__}: {e}')
    return all(v[0] for v in out.values()), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)
    ok = True

    print("=" * 100)
    print("A. 기본값 등가성 — FUSION.TRUNK 키 없음 ↔ TRUNK: gated_mlp (같은 seed)")
    print("=" * 100)
    aok, info = check_default_identity(dev)
    print(f"    state_dict 키 동일={info['same_keys']} ({info['n_keys']}개)  "
          f"값 전부 동일={info['same_vals']}  신규 trunk_xattn 파라미터={info['new_params']}개")
    print(f"    trunk_xattn is None = {info['trunk_xattn_is_None']}   "
          f"eval logits max|Δ| = {info['dmax']:.3e}   {'OK' if aok else 'FAIL'}")
    ok &= aok

    print()
    print("=" * 100)
    print("B. xattn 1-step fwd+bwd — 전 모달 × 전 프로젝션 gradient 도달 (키1)")
    print("=" * 100)
    bok, rows, ls_ok, gamma_ok, loss = check_xattn_grad(dev)
    cols = ['q', 'k', 'v', 'proj', 'mlp', 'norm_q', 'norm_kv', 'norm2',
            'ls_attn', 'ls_mlp', 'gamma']
    print(f"    total loss = {loss:.4f}   LayerScale init=={LS_INIT} → {ls_ok}   "
          f"γ init=={GAMMA_INIT} → {gamma_ok}")
    print("    " + f"{'modal':<7}" + "".join(f"{c:>10}" for c in cols))
    for name, cell in rows:
        print("    " + f"{name:<7}" + "".join(f"{cell[c]:>10.2e}" for c in cols))
    print(f"    전 항목 > 0 = {bok}   {'OK' if bok else 'FAIL'}")
    ok &= bok

    print()
    print("=" * 100)
    print("C. 출력 shape 동일 (_apply_trunk_exp 및 최종 logits)")
    cok, sa, sb, la, lb, d = check_shapes(dev)
    print(f"    trunk out  gated_mlp={sa}  xattn={sb}   max|Δ|={d:.3e} (>0 = 실제로 다른 트렁크)")
    print(f"    logits     gated_mlp={la}  xattn={lb}   {'OK' if cok else 'FAIL'}")
    ok &= cok

    print()
    print("=" * 100)
    print("D. eval 결정론 + 학습 aux 키 집합 동일 (VICReg 등 기존 손실 유지)")
    dok, det, keys, dd = check_determinism_and_aux(dev)
    print(f"    eval 2회 bitwise 동일 = {det} (max|Δ|={dd:.3e})")
    print(f"    aux gated_mlp = {keys['gated_mlp']}")
    print(f"    aux xattn     = {keys['xattn']}   {'OK' if dok else 'FAIL'}")
    ok &= dok

    print()
    print("=" * 100)
    print("E. 파라미터 수 (트렁크 대비 증가분)")
    print(f"    {'arm':<12}{'total':>14}{'trunk only':>14}{'trunk ×':>10}{'total ×':>10}")
    for name, tot, trunk, rt, rtot in check_params(dev):
        print(f"    {name:<12}{tot:>14,}{trunk:>14,}{rt:>10.2f}{rtot:>10.3f}")
    print("    ⚠️ tiny ViT(dim=192) 기준 — 본 레시피(dim=1024, 4모달)에서는 배수가 더 크다")

    print()
    print("=" * 100)
    print("F. 가드 (조용한 실패 금지)")
    fok, guards = check_guards(dev)
    for tag, (good, msg) in guards.items():
        print(f"    {tag:<26} raises ValueError = {good}   {msg}")
    ok &= fok

    print()
    print("=" * 100)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
