"""[N2] MLE-SAM 평균융합 baseline 토글 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_meanfusion.py

대상 = `MODEL.FUSION.TRUNK: mean` 옵션(기존 gated_mlp|xattn 토글에 추가).
동일백본 통제비교이므로 "기본값에서 아무것도 안 변한다"가 첫 번째 검사다.

검사 항목
  A. 기본값 등가성 : TRUNK 키 없음 ↔ TRUNK: gated_mlp 가 같은 seed 에서
                     state_dict 키·값 전부 일치 + eval logits bitwise 동일.
                     trunk_mean 미생성 (init RNG 소비 없음).
  B. mean fwd/bwd  : mean 팔 1-step fwd+bwd — loss 유한, **전 모달 × 전종**
                     LoRA(a_q/b_q/a_v/b_v, 텐서 dim0=모달)에 gradient 실제
                     도달. + 학습 aux 키 집합이 gated_mlp 와 동일(VICReg 등
                     per-modal 손실이 mean 모드에서도 그대로 걸린다 — 스펙 3).
  C. 출력 shape    : `_apply_trunk_exp` 출력 shape 가 두 팔에서 동일 + mean
                     출력이 torch.stack(feats).mean(0) 과 allclose(진짜 산술
                     평균이다) + 게이트 융합 fused 와는 다르다(교체가 실제로
                     일어난다) + 최종 logits shape 동일.
  D. 파라미터 수   : mean trainable 총수 = gated_mlp 총수 − 트렁크 분
                     (trunk_exp + trunk_gamma). trunk_mean 파라미터 == 0.
  E. 가드          : TRUNK:mean + TRUNK_EXP:true → ValueError(조용한 실패
                     금지). 잘못된 TRUNK 값도 ValueError.
"""
from __future__ import annotations

import argparse
import math
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

LORA_KEYS = ('.a_q', '.b_q', '.a_v', '.b_v')     # encoder.MultiModalLoRAQKV


def base_cfg(trunk=None, trunk_exp=True):
    """P46-c3only(λ0.1) config 의 토글 구성을 tiny 스케일로 옮긴 것
    (smoke_xattn_trunk.py 와 동일 — XATTN 서브딕트만 mean 에는 불필요).

    trunk=None 이면 FUSION.TRUNK 키 자체를 넣지 않는다 (= 기존 config 전부)."""
    fusion = {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 4.0,
              'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
              'ATTN_BIAS': {'ENABLE': False}}
    if trunk is not None:
        fusion['TRUNK'] = trunk
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


# ── A. 기본값 등가성 (TRUNK 키 없음 ↔ gated_mlp) ────────────────────────────
def check_default_identity(device):
    a = build(device, seed=1234)                          # FUSION.TRUNK 키 없음
    b = build(device, seed=1234, trunk='gated_mlp')       # 명시적 기본값
    sa, sb = a.state_dict(), b.state_dict()
    same_keys = list(sa.keys()) == list(sb.keys())
    same_vals = same_keys and all(torch.equal(sa[k], sb[k]) for k in sa)
    no_module = a.trunk_mean is None and b.trunk_mean is None
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=7)
    with torch.no_grad():
        la, _ = a(x, True)
        lb, _ = b(x, True)
    dmax = float((la - lb).abs().max())
    ok = same_keys and same_vals and no_module and torch.equal(la, lb)
    return ok, dict(same_keys=same_keys, same_vals=same_vals,
                    n_keys=len(sa), trunk_mean_is_None=no_module, dmax=dmax)


# ── B. mean fwd/bwd — LoRA grad 도달 + aux 키 집합 동일 ────────────────────
def check_mean_grad(device):
    m = build(device, seed=7, trunk='mean', trunk_exp=False)
    assert m.trunk_mean is not None, 'trunk_mean 이 만들어지지 않았다'
    assert m.trunk_exp is None and m.trunk_gamma is None, \
        'mean 팔에서 파라미터 트렁크가 만들어져서는 안 된다'

    m.train()
    m._current_epoch = 10                     # VICReg 항상 + C3 proto(warmup 5) 켬
    x, y = make_batch(device, seed=3)

    def fwd_bwd():
        logits, _, aux = m(x, True, gt_mask=y)
        total = F.cross_entropy(logits, y, ignore_index=255)
        for k, v in aux.items():
            if torch.is_tensor(v) and v.dim() == 0:
                total = total + v
        m.zero_grad(set_to_none=True)
        total.backward()
        return float(total), aux

    def lora_grad_rows():
        """모달(dim0) × 종(4) 별 LoRA gradient L2 — 하나라도 0이면 그 모달의
        encoder 어댑터가 mean 융합 경로에서 사장된 것이다."""
        rows = []
        for i, name in enumerate(MODALS):
            cell = {}
            for key in LORA_KEYS:
                g2 = 0.0
                for n, p in m.named_parameters():
                    if n.endswith(key) and p.grad is not None:
                        g2 += float(p.grad[i].detach().pow(2).sum())
                cell[key] = g2 ** 0.5
            rows.append((name, cell))
        return rows

    loss1, aux = fwd_bwd()
    step1 = lora_grad_rows()
    # b_* 는 zero-init(encoder.MultiModalLoRAQKV)이라 초기 스텝 출력이 0 →
    # a_* 는 이 스텝에 grad 를 못 받고 b_* 만 받는다(기존 팔과 같은 LoRA 성질,
    # mean 때문이 아니다). 한 스텝 밟고 재측정해 전종 도달을 실증한다.
    lora_params = [p for n_, p in m.named_parameters()
                   if any(n_.endswith(k) for k in LORA_KEYS)]
    opt = torch.optim.SGD(lora_params, lr=1e-1)
    opt.step()
    loss2, _ = fwd_bwd()
    step2 = lora_grad_rows()

    ok = math.isfinite(loss1) and math.isfinite(loss2)
    for _, cell in step1:                     # 스텝1: 최소한 b_* 는 받아야 한다
        ok &= cell['.b_q'] > 0 and cell['.b_v'] > 0
    for _, cell in step2:                     # 스텝2: 전종 도달
        ok &= all(cell[k] > 0 for k in LORA_KEYS)

    # aux 키 집합 비교 — mean 모드에서도 VICReg 등 기존 학습손실이 동일하게 걸린다
    g = build(device, seed=7, trunk_exp=True)
    keys = {}
    for tag, mm in (('gated_mlp', g), ('mean', m)):
        mm.train()
        mm._current_epoch = 10
        torch.manual_seed(99)
        _, _, aux_g = mm(x, True, gt_mask=y)
        keys[tag] = sorted(aux_g.keys())
    same_aux = keys['gated_mlp'] == keys['mean']
    has_vicreg = 'vicreg' in keys['mean']
    ok &= same_aux and has_vicreg
    return ok, step1, step2, keys, loss1, loss2


# ── C. 출력 shape 동일 + 산술 평균 정합 (트렁크 인터페이스 불변) ─────────────
def check_shapes(device):
    a = build(device, seed=11)                                        # gated_mlp
    b = build(device, seed=11, trunk='mean', trunk_exp=False)         # mean
    dim = a.trunk_exp[0][1].in_channels
    h = SIZE // 16
    torch.manual_seed(0)
    fused = torch.randn(BS, dim, h, h, device=device)
    feats = [torch.randn(BS, dim, h, h, device=device) for _ in MODALS]
    with torch.no_grad():
        oa = a._apply_trunk_exp(fused, feats)
        ob = b._apply_trunk_exp(fused, feats)
        ref = torch.stack(feats, dim=0).mean(dim=0)
        is_mean = torch.allclose(ob, ref, rtol=0, atol=0)
        replaced = not torch.allclose(ob, fused)
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=5)
    with torch.no_grad():
        la, ma = a(x, True)
        lb, mb = b(x, True)
    ok = (oa.shape == ob.shape == fused.shape
          and la.shape == lb.shape and ma.shape == mb.shape
          and is_mean and replaced)
    return ok, tuple(oa.shape), tuple(ob.shape), tuple(la.shape), \
        tuple(lb.shape), is_mean, replaced


# ── D. 파라미터 수 — mean = gated_mlp − 트렁크 분 ───────────────────────────
def check_params(device):
    a = build(device, seed=31)                                # gated_mlp
    b = build(device, seed=31, trunk='mean', trunk_exp=False)  # mean
    tot_a, tot_b = n_params(a), n_params(b)
    trunk_a = n_params(a.trunk_exp) + a.trunk_gamma.numel()
    trunk_b = n_params(b.trunk_mean)
    diff = tot_a - tot_b
    ok = (trunk_b == 0) and (diff == trunk_a)
    return ok, [('gated_mlp', tot_a, trunk_a),
                ('mean', tot_b, trunk_b)], diff, trunk_a


# ── E. 가드 (조용한 실패 금지) ──────────────────────────────────────────────
def check_guards(device):
    out = {}
    for tag, kw in (('mean + TRUNK_EXP:true',
                     dict(trunk='mean', trunk_exp=True)),
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
    aok, info = check_default_identity(dev)
    print(f"    state_dict 키 동일={info['same_keys']} ({info['n_keys']}개)  "
          f"값 전부 동일={info['same_vals']}  trunk_mean is None={info['trunk_mean_is_None']}")
    print(f"    eval logits max|Δ| = {info['dmax']:.3e}   {'OK' if aok else 'FAIL'}")
    ok &= aok

    print()
    print("=" * 100)
    print("B. mean fwd/bwd — 전 모달 LoRA gradient 도달 + aux 키 집합 동일")
    bok, step1, step2, keys, loss1, loss2 = check_mean_grad(dev)
    cols = [k.lstrip('.') for k in LORA_KEYS]
    print(f"    total loss = {loss1:.4f} → (1 SGD step 후) {loss2:.4f}  "
          f"finite={math.isfinite(loss1) and math.isfinite(loss2)}")
    print("    [스텝1] b_* zero-init 이라 b_* 만 grad 를 받는다(기존 팔과 동일한 LoRA 성질):")
    print("    " + f"{'modal':<7}" + "".join(f"{c:>10}" for c in cols))
    for name, cell in step1:
        print("    " + f"{name:<7}" + "".join(f"{cell[k]:>10.2e}" for k in LORA_KEYS))
    print("    [스텝2] 1스텝 뒤 재측정 — 전종 × 전 모달 도달:")
    print("    " + f"{'modal':<7}" + "".join(f"{c:>10}" for c in cols))
    for name, cell in step2:
        print("    " + f"{name:<7}" + "".join(f"{cell[k]:>10.2e}" for k in LORA_KEYS))
    print(f"    aux gated_mlp = {keys['gated_mlp']}")
    print(f"    aux mean      = {keys['mean']}")
    print(f"    키 집합 동일={keys['gated_mlp'] == keys['mean']}  vicreg 포함="
          f"{'vicreg' in keys['mean']}  전 항목 통과={bok}   {'OK' if bok else 'FAIL'}")
    ok &= bok

    print()
    print("=" * 100)
    print("C. 출력 shape 동일 (_apply_trunk_exp 및 최종 logits) + 산술 평균 정합")
    cok, sa, sb, la, lb, is_mean, replaced = check_shapes(dev)
    print(f"    trunk out  gated_mlp={sa}  mean={sb}")
    print(f"    logits     gated_mlp={la}  mean={lb}")
    print(f"    mean == torch.stack(feats).mean(0) (bitwise) = {is_mean}"
          f"   mean != 게이트 융합 fused = {replaced}")
    print(f"    {'OK' if cok else 'FAIL'}")
    ok &= cok

    print()
    print("=" * 100)
    print("D. 파라미터 수 — mean 이 gated_mlp 보다 트렁크 분만큼 적다")
    dok, rows_p, diff, trunk_a = check_params(dev)
    print(f"    {'arm':<12}{'total':>14}{'trunk only':>14}")
    for name, tot, trunk in rows_p:
        print(f"    {name:<12}{tot:>14,}{trunk:>14,}")
    print(f"    차이 = {diff:,} == gated_mlp 트렁크 분 = {trunk_a:,} → {diff == trunk_a}"
          f"   (mean 트렁크 파라미터 = 0 → {rows_p[1][2] == 0})")
    print(f"    {'OK' if dok else 'FAIL'}")
    ok &= dok

    print()
    print("=" * 100)
    print("E. 가드 (조용한 실패 금지)")
    eok, guards = check_guards(dev)
    for tag, (good, msg) in guards.items():
        print(f"    {tag:<26} raises ValueError = {good}   {msg}")
    ok &= eok

    print()
    print("=" * 100)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
