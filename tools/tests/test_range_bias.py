"""[P56-B] 거리 조건화 교차 attention 단위 테스트 (순수 CPU, 작은 dim).

검증 항목:
  1. off 등가   — RANGE_BIAS 미도입(default)과 ENABLE=false 의 출력·state_dict 동일.
  2. 초기 등가  — ENABLE=true, INIT_LAMBDA=0 이면 켠 출력 == 끈 출력 (atol 1e-6).
  3. 유효 마스크 — valid=False 토큰이 걸린 편향 항이 0.
  4. 기울기     — INIT_LAMBDA=0 에서도 lambda_h 로 grad 가 흐른다(softplus 죽지 않음).
  5. 형상       — (B,H,Nq,Nk) 편향이 두 층에 같은 텐서로 들어간다.

실행: PYTHONPATH=semseg/models/sam2:. python tools/tests/test_range_bias.py
      또는 pytest tools/tests/test_range_bias.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))

from semseg.models.reliadino.fusion import (   # noqa: E402
    CrossModalAttentionLayer, ReliabilityGatedFusion, RangeBias)

DIM, NC, M, HEADS = 16, 5, 3, 4
HW = 3                       # 3x3 토큰 = N 9
torch.manual_seed(0)


def _lean_kwargs(**extra):
    """RBMA·gate·calibrate·router 를 전부 꺼 거리 편향만 남긴 최소 융합."""
    kw = dict(dim=DIM, num_classes=NC, num_modalities=M, num_layers=2,
              num_heads=HEADS, mlp_ratio=2.0, aux_hidden=32,
              attn_bias=False, consistency_bias=False, gate_enable=False,
              calibrate=False, router_enable=False)
    kw.update(extra)
    return kw


def _feats(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, DIM, HW, HW, generator=g) for _ in range(M)]


def _range(B=2, seed=2):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(B, 1, HW, HW, generator=g)          # [0,1) 거리 대용값
    valid = torch.ones(B, 1, HW, HW, dtype=torch.bool)
    return r, valid


def _fused(mod, feats, **fwd):
    mod.eval()
    with torch.no_grad():
        out, _ = mod(feats, None, **fwd)
    return out


def test_off_equivalence():
    off = ReliabilityGatedFusion(**_lean_kwargs())                 # default(미도입)
    dis = ReliabilityGatedFusion(**_lean_kwargs(range_bias_enable=False))
    dis.load_state_dict(off.state_dict())                          # 같은 가중치
    assert set(off.state_dict()) == set(dis.state_dict())
    assert not any(k.startswith('range_bias') for k in off.state_dict())
    feats = _feats()
    assert torch.equal(_fused(off, feats), _fused(dis, feats))
    print('[1] off 등가 OK')


def test_state_dict_keys_when_on():
    on = ReliabilityGatedFusion(**_lean_kwargs(range_bias_enable=True))
    keys = [k for k in on.state_dict() if k.startswith('range_bias')]
    assert set(keys) == {'range_bias.lambda_h', 'range_bias.sigma_h'}, keys
    print('[1b] ENABLE=true state_dict 키 OK:', keys)


def test_init_equivalence():
    off = ReliabilityGatedFusion(**_lean_kwargs())
    on = ReliabilityGatedFusion(**_lean_kwargs(
        range_bias_enable=True, range_bias_init_lambda=0.0))
    on.load_state_dict(off.state_dict(), strict=False)             # 공유 가중치 복사
    feats = _feats()
    r, valid = _range()
    y_off = _fused(off, feats)
    y_on = _fused(on, feats, range_map=r, range_valid=valid)
    diff = (y_off - y_on).abs().max().item()
    assert diff < 1e-6, diff
    print(f'[2] 초기 등가 OK (max|Δ|={diff:.2e})')


def test_valid_mask_zeroes_bias():
    rb = RangeBias(num_heads=HEADS, init_lambda=0.7, per_head=True)
    B, N = 2, HW * HW
    g = torch.Generator().manual_seed(3)
    rho = torch.randn(B, N, generator=g)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 0] = False                                            # 무효 토큰 하나
    bias = rb(rho, valid, key_mult=M - 1, dtype=torch.float32)
    # query 토큰 0(무효)이 걸린 모든 항 = 0
    assert torch.count_nonzero(bias[0, :, 0, :]) == 0
    # key 쪽 무효(토큰 0 이 (m-1) 블록마다 반복)도 0
    for blk in range(M - 1):
        assert torch.count_nonzero(bias[0, :, :, blk * N + 0]) == 0
    # 유효-유효 쌍은 일반적으로 0 이 아님
    assert torch.count_nonzero(bias[1]) > 0
    print('[3] 유효 마스크 OK')


def test_lambda_gradient_flows():
    on = ReliabilityGatedFusion(**_lean_kwargs(
        range_bias_enable=True, range_bias_init_lambda=0.0))
    on.train()
    feats = _feats()
    r, valid = _range()
    fused, _ = on(feats, None, range_map=r, range_valid=valid)
    fused.sum().backward()
    g = on.range_bias.lambda_h.grad
    assert g is not None and float(g.abs().sum()) > 0, g
    print(f'[4] lambda_h grad OK (Σ|grad|={float(g.abs().sum()):.2e})')


def test_pair_bias_shared_shape():
    captured = []
    orig = CrossModalAttentionLayer.forward

    def spy(self, x, kv, key_bias, pair_bias=None):
        captured.append(pair_bias)
        return orig(self, x, kv, key_bias, pair_bias)

    on = ReliabilityGatedFusion(**_lean_kwargs(range_bias_enable=True))
    B, N = 2, HW * HW
    feats = _feats(B=B)
    r, valid = _range(B=B)
    CrossModalAttentionLayer.forward = spy
    try:
        _fused(on, feats, range_map=r, range_valid=valid)
    finally:
        CrossModalAttentionLayer.forward = orig
    # 모달 3개 × 층 2개 = 6 회 호출, 전부 같은 pair_bias 객체
    assert len(captured) == M * 2
    first = captured[0]
    assert first.shape == (B, HEADS, N, N * (M - 1)), first.shape
    for pb in captured:
        assert pb is first                                         # 한 번만 계산·공유
    # PER_HEAD=false → (B,1,Nq,Nk)
    on1 = ReliabilityGatedFusion(**_lean_kwargs(
        range_bias_enable=True, range_bias_per_head=False))
    pb1 = on1.range_bias(torch.randn(B, N), torch.ones(B, N, dtype=torch.bool),
                         key_mult=M - 1, dtype=torch.float32)
    assert pb1.shape == (B, 1, N, N * (M - 1)), pb1.shape
    print('[5] 형상·공유 OK')


TESTS = [test_off_equivalence, test_state_dict_keys_when_on,
         test_init_equivalence, test_valid_mask_zeroes_bias,
         test_lambda_gradient_flows, test_pair_bias_shared_shape]


if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f'\nALL {len(TESTS)} PASSED')
