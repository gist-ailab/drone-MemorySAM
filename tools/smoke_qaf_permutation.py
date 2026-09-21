#!/usr/bin/env python3
"""[P54-QAF G-signal] 치환 검정 CPU 스모크 — 데이터·ckpt·timm 불필요.

검사(전부 assert, 순수 torch + ReliabilityGatedFusion):
  (a) 오버라이드 훅 None ⇒ QAF 정상 forward 와 출력 byte-동일(훅이 정상 경로 무변경).
  (b) η̂=0 훅 ⇒ 가중치 공유 항등: key 마스크 on==off, 가중 평균==산술 평균
      (η̂=0 → log(1−0)=0 무마스크, (1−0)=1 균등가중 = 융합 항등).
  (c) 셔플 훅(B=2)이 실제로 다른 샘플의 η̂ 를 쓴다(마스크 모달 교환, 비마스크 불변).
  (d) build_protocols 3종(clean/missing_<modal>/rmm_<modal>_<r>)이 import·형상 정상.

실행:  python tools/smoke_qaf_permutation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.models.reliadino.fusion import ReliabilityGatedFusion  # noqa: E402
from tools.qaf_permutation_test import (                          # noqa: E402
    ShuffleOverride, zero_override, build_protocols)

MODALS = ['img', 'depth', 'event', 'lidar']
NUM_CLASSES = 6
_FAILS = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


def _fusion(key_mask=True, wmean=True, self_attn=True, dim=32, m=4):
    return ReliabilityGatedFusion(
        dim=dim, num_classes=NUM_CLASSES, num_modalities=m,
        num_layers=1, num_heads=4, mlp_ratio=1.0, aux_hidden=32,
        attn_bias=False, consistency_bias=False, gate_enable=False,
        veto_floor=False, calibrate=False,
        qaf_enable=True, qaf_mask_modals=['depth', 'lidar'],
        qaf_self_attn=self_attn, qaf_key_mask=key_mask, qaf_weighted_mean=wmean,
        qaf_head_hidden=16, qaf_modal_names=MODALS)


def test_a_hook_none_identical():
    print("\n[a] 훅 None ⇒ QAF 정상 forward 와 byte-동일")
    B, dim, h, w = 2, 32, 5, 5
    torch.manual_seed(0)
    fus = _fusion(dim=dim).eval()
    feats = [torch.randn(B, dim, h, w) for _ in MODALS]
    check("(a) 기본값 _qaf_eta_override is None", fus._qaf_eta_override is None)
    with torch.no_grad():
        out_ref, _ = fus(feats)
        fus._qaf_eta_override = None
        out_none, _ = fus(feats)
    check("(a) 훅 None 출력 byte-동일", torch.equal(out_ref, out_none),
          f"max|Δ|={(out_ref - out_none).abs().max().item():.2e}")


def test_b_zero_identity():
    print("\n[b] η̂=0 훅 ⇒ 가중치 공유 항등(마스크 on==off, 가중==산술)")
    B, dim, h, w = 2, 32, 5, 5
    torch.manual_seed(1)
    fus = _fusion(dim=dim).eval()
    feats = [torch.randn(B, dim, h, w) for _ in MODALS]
    with torch.no_grad():
        fus._qaf_eta_override = zero_override
        fus.qaf_key_mask = True
        out_mask_on, _ = fus(feats)
        fus.qaf_key_mask = False
        out_mask_off, _ = fus(feats)
    check("(b) η̂=0 → key 마스크 항등(on==off)",
          torch.allclose(out_mask_on, out_mask_off, atol=1e-6),
          f"max|Δ|={(out_mask_on - out_mask_off).abs().max().item():.2e}")
    with torch.no_grad():
        fus.qaf_key_mask = True
        fus.qaf_weighted_mean = True
        out_wmean, _ = fus(feats)          # η̂=0 → wts 전부 1 → 산술평균
        fus.qaf_weighted_mean = False
        out_mean, _ = fus(feats)           # sum/m
    check("(b) η̂=0 → 가중 평균 == 산술 평균",
          torch.allclose(out_wmean, out_mean, atol=1e-6),
          f"max|Δ|={(out_wmean - out_mean).abs().max().item():.2e}")


def test_c_shuffle_swaps():
    print("\n[c] 셔플 훅(B=2) ⇒ 마스크 모달 η̂ 교환 · 비마스크 불변")
    B, m, h, w = 2, 4, 3, 3
    mask_idx = [1, 3]                       # depth, lidar
    # 샘플별로 구분되는 η̂ 를 심어 교환을 눈으로 확인.
    es = torch.zeros(B, m)
    et = torch.zeros(B, m, h, w)
    for b in range(B):
        for j in range(m):
            es[b, j] = float(b * 10 + j)   # 유일한 값
            et[b, j] = float(b * 10 + j)
    pred = {'eta_scalar': es.clone(), 'eta_token': et.clone(),
            'presence_logit': torch.zeros(B, m)}
    ov = ShuffleOverride(mask_idx, seed=0)
    out = ov(pred)
    # B=2 순환치환 k∈[1,1] → k=1 → perm=(arange+1)%2 = [1,0] = swap.
    swap_ok = True
    for j in mask_idx:
        swap_ok = swap_ok and bool(out['eta_scalar'][0, j] == es[1, j]) \
            and bool(out['eta_scalar'][1, j] == es[0, j]) \
            and bool((out['eta_token'][0, j] == et[1, j]).all())
    check("(c) 마스크 모달 η̂ 가 다른 샘플 것으로 교환", swap_ok,
          f"scalar[:, {mask_idx}]={out['eta_scalar'][:, mask_idx].tolist()}")
    unchg = all(bool((out['eta_scalar'][:, j] == es[:, j]).all())
                for j in range(m) if j not in mask_idx)
    check("(c) 비마스크 모달 η̂ 불변", unchg)
    # 롤링(B==1): 첫 배치는 자기 유지, 둘째 배치는 첫 배치 η̂ 를 씀.
    ov1 = ShuffleOverride(mask_idx, seed=0)
    p1 = {'eta_scalar': torch.full((1, m), 1.0),
          'eta_token': torch.full((1, m, h, w), 1.0),
          'presence_logit': torch.zeros(1, m)}
    o1 = ov1(p1)
    check("(c) B==1 첫 배치 치환 없음(이전 없음)",
          torch.equal(o1['eta_scalar'], p1['eta_scalar']))
    p2 = {'eta_scalar': torch.full((1, m), 2.0),
          'eta_token': torch.full((1, m, h, w), 2.0),
          'presence_logit': torch.zeros(1, m)}
    o2 = ov1(p2)
    roll_ok = all(bool(o2['eta_scalar'][0, j] == 1.0) for j in mask_idx) \
        and all(bool(o2['eta_scalar'][0, j] == 2.0) for j in range(m) if j not in mask_idx)
    check("(c) B==1 롤링: 둘째 배치가 첫 배치 η̂ 사용", roll_ok,
          f"scalar={o2['eta_scalar'].tolist()}")


def test_d_protocols():
    print("\n[d] build_protocols 3종 import · 형상 정상")
    def gen_factory():
        g = torch.Generator()
        g.manual_seed(0)
        return g

    names = ['clean', 'rmm_depth_0.5', 'missing_depth']
    protocols = build_protocols(names, MODALS, gen_factory)
    check("(d) 3 프로토콜 생성", len(protocols) == 3,
          str([p for p, _ in protocols]))
    B, C, H, W = 2, 3, 8, 8
    base = [torch.randn(B, C, H, W) for _ in MODALS]
    depth_i = MODALS.index('depth')
    got = {name: fn(base) for name, fn in protocols}
    # 형상 보존.
    shape_ok = all(all(t.shape == (B, C, H, W) for t in lst) for lst in got.values())
    check("(d) 모든 프로토콜 출력 형상 보존", shape_ok)
    # clean = 원본 그대로.
    check("(d) clean 무변경",
          all(torch.equal(a, b) for a, b in zip(got['clean'], base)))
    # missing_depth = depth 만 0, 나머지 불변.
    md = got['missing_depth']
    check("(d) missing_depth: depth==0 · 나머지 불변",
          bool((md[depth_i] == 0).all())
          and all(torch.equal(md[i], base[i]) for i in range(len(MODALS)) if i != depth_i))
    # rmm_depth_0.5 = depth 변경(일부 0), 나머지 불변.
    rd = got['rmm_depth_0.5']
    check("(d) rmm_depth_0.5: depth 변경 · 나머지 불변",
          (not torch.equal(rd[depth_i], base[depth_i]))
          and all(torch.equal(rd[i], base[i]) for i in range(len(MODALS)) if i != depth_i))
    # 알 수 없는/오류 이름은 시끄럽게 실패.
    err_ok = True
    for bad in ['missing_nonexist', 'rmm_depth', 'wat']:
        try:
            build_protocols([bad], MODALS, gen_factory)
            err_ok = False
        except ValueError:
            pass
    check("(d) 잘못된 프로토콜 이름 ValueError", err_ok)


def main():
    print("== smoke_qaf_permutation ==")
    test_a_hook_none_identical()
    test_b_zero_identity()
    test_c_shuffle_swaps()
    test_d_protocols()
    print()
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
