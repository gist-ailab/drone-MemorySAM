#!/usr/bin/env python3
"""[P56-A] 모달 충돌 열화(conflict) 단위 테스트 — CPU 전용, 데이터/ckpt 불필요.

검증 대상:
  1. conflict_p=0(및 키 부재) ⇒ 출력·라벨·생성기 최종 상태가 기존 기본 cfg 와 bit-동일
     (cfg={} vs cfg={'conflict_p':0.0}, 같은 seed, 여러 번 호출).
  2. conflict_p=1.0, B=2·B=1 ⇒ 표본마다 정확히 한 모달만 clean 과 다르고, 그 모달은
     conflict_modals 중 하나, 나머지 모달은 입력과 bit-동일, presence 전부 1,
     op 인덱스는 충돌 op, mask 면적비 정상 범위, 출력 유한.
  3. specular_glare 는 마스크 영역 평균을 이미지 최대 쪽으로 올린다;
     region_conflict 는 마스크 내부 내용을 바꾸고 alpha==0 픽셀은 그대로 둔다.
  4. 기존 op 들의 OP_NAMES 인덱스 불변(옛 튜플 하드코딩 대조).
  5. heldout=True + conflict_p=1.0 ⇒ 충돌 op 절대 미발생.

실행:  python tools/tests/test_degrade_conflict.py   (모두 통과 시 'ALL PASS')
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets import degrade as D                            # noqa: E402
from semseg.datasets.degrade import (                               # noqa: E402
    Degrader, OP_NAMES, OPS, TRAIN_OPS, HELDOUT_OPS,
    region_conflict, specular_glare)

MODALS = ['img', 'depth', 'event', 'lidar']
_CONF_IDX = (OP_NAMES.index('region_conflict'), OP_NAMES.index('specular_glare'))
_n = 0


def check(cond, msg):
    global _n
    assert cond, f"FAIL: {msg}"
    _n += 1
    print(f"  ok: {msg}")


def _inputs(B, seed=1234, S=64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


# ---------------------------------------------------------------------------
def test_conflict_off_bit_identical():
    print("[1] conflict_p=0 ⇒ 기존과 bit-동일(출력·라벨·생성기 상태)")
    for seed in (0, 7, 123):
        dA = Degrader(cfg={}, seed=seed)                            # 키 부재
        dB = Degrader(cfg={'conflict_p': 0.0}, seed=seed)           # 명시 0.0
        for call in range(4):
            x = _inputs(B=3, seed=1000 + call)
            oa, la = dA([t.clone() for t in x], MODALS)
            ob, lb = dB([t.clone() for t in x], MODALS)
            for m in range(len(MODALS)):
                check(torch.equal(oa[m], ob[m]),
                      f"seed{seed} call{call} 모달{m} 출력 동일")
            for k in la:
                check(torch.equal(la[k], lb[k]),
                      f"seed{seed} call{call} 라벨[{k}] 동일")
        check(torch.equal(dA.g.get_state(), dB.g.get_state()),
              f"seed{seed} 생성기 최종 상태 동일")


def test_conflict_on(B):
    print(f"[2] conflict_p=1.0, B={B} ⇒ 표본당 한 모달만 충돌, 나머지 clean")
    x = _inputs(B=B, seed=42)
    deg = Degrader(cfg={'conflict_p': 1.0,
                        'conflict_modals': ['img', 'depth'],
                        'conflict_glare_frac': 0.5}, seed=3)
    out, lab = deg([t.clone() for t in x], MODALS)
    conf_mods = {MODALS.index('img'), MODALS.index('depth')}
    for b in range(B):
        diff = [m for m in range(len(MODALS)) if not torch.equal(out[m][b], x[m][b])]
        check(len(diff) == 1, f"B{B} b{b} 정확히 한 모달만 변경 (diff={diff})")
        md = diff[0]
        check(md in conf_mods, f"B{B} b{b} 변경 모달이 conflict_modals ({MODALS[md]})")
        for m in range(len(MODALS)):
            if m != md:
                check(torch.equal(out[m][b], x[m][b]),
                      f"B{B} b{b} 비대상 모달{m} 입력과 bit-동일")
        check(float(lab['presence'][b, md]) == 1.0, f"B{B} b{b} presence=1")
        check(int(lab['op'][b, md]) in _CONF_IDX,
              f"B{B} b{b} op 인덱스가 충돌 op ({int(lab['op'][b, md])})")
        # 비대상 모달 op 는 clean(0)
        for m in range(len(MODALS)):
            if m != md:
                check(int(lab['op'][b, m]) == 0, f"B{B} b{b} 비대상 모달{m} op=clean")
        frac = float(lab['mask'][b, md].mean())
        check(0.0 < frac < 0.8, f"B{B} b{b} mask 면적비 정상 ({frac:.3f})")
        check(bool(torch.isfinite(out[md][b]).all()), f"B{B} b{b} 출력 유한")


def test_ops_semantics():
    print("[3] specular_glare / region_conflict 의미 검증")
    g = torch.Generator().manual_seed(11)
    x = torch.randn(3, 96, 96, generator=g)

    # specular_glare: 마스크 영역 평균이 이미지 최대 쪽으로 상승, 전 픽셀 x 이상
    out_s, sev_s, mk_s = specular_glare(x.clone(), g)
    msk = mk_s > 0.5
    check(bool(msk.any()), "glare 마스크 비어있지 않음")
    x_m = x[:, msk].mean()
    o_m = out_s[:, msk].mean()
    check(float(o_m) > float(x_m), f"glare 마스크 평균 상승 ({float(x_m):.3f}→{float(o_m):.3f})")
    check(bool((out_s >= x - 1e-5).all()), "glare 는 전 픽셀을 최대 쪽으로만 이동")
    check(0.0 <= sev_s <= 1.0, f"glare severity∈[0,1] ({sev_s:.3f})")

    # region_conflict: donor 로 마스크 내부 내용 교체, alpha==0(=out==x) 픽셀은 불변
    donor = torch.randn(3, 96, 96, generator=g)
    out_r, sev_r, mk_r = region_conflict(x.clone(), g, donor=donor)
    mskr = mk_r > 0.5
    check(bool(mskr.any()), "region 마스크 비어있지 않음")
    changed = ~torch.isclose(out_r, x).all(dim=0)                   # (H,W) 변경 픽셀
    check(bool(changed[mskr].all()), "마스크 내부 픽셀은 내용이 바뀜")
    unchanged = torch.isclose(out_r, x).all(dim=0)                  # alpha==0 픽셀
    check(bool(unchanged.any()), "alpha==0(불변) 픽셀 존재")
    # 불변 픽셀은 정확히 x 와 같아야 한다(alpha==0 ⇒ out=x)
    check(bool(torch.equal(out_r[:, unchanged], x[:, unchanged])),
          "불변 픽셀은 입력과 정확히 동일")
    check(0.0 <= sev_r <= 1.0, f"region severity∈[0,1] ({sev_r:.3f})")


def test_op_index_stable():
    print("[4] 기존 op 들의 OP_NAMES 인덱스 불변")
    old = ('clean', 'missing', 'patch_drop', 'gaussian_blur', 'gamma_gain',
           'color_shift', 'depth_hole', 'lidar_beamdrop', 'event_lowres',
           'gaussian_noise', 'salt_pepper')
    check(OP_NAMES[:len(old)] == old, "선두 11개 이름·순서 불변")
    check(OP_NAMES == old + ('region_conflict', 'specular_glare'),
          "충돌 op 두 개가 끝에 추가")
    # 충돌 op 는 학습/held-out 집합 어디에도 없다
    for name in ('region_conflict', 'specular_glare'):
        for modal, ops in TRAIN_OPS.items():
            check(name not in ops, f"{name} ∉ TRAIN_OPS[{modal}]")
        check(name not in HELDOUT_OPS, f"{name} ∉ HELDOUT_OPS")
        check(name in OPS, f"{name} ∈ OPS")


def test_heldout_no_conflict():
    print("[5] heldout=True + conflict_p=1.0 ⇒ 충돌 op 미발생")
    x = _inputs(B=3, seed=9)
    deg = Degrader(cfg={'conflict_p': 1.0}, seed=5, heldout=True)
    _, lab = deg([t.clone() for t in x], MODALS)
    ops = lab['op']
    has_conf = bool(((ops == _CONF_IDX[0]) | (ops == _CONF_IDX[1])).any())
    check(not has_conf, "held-out 경로에 충돌 op 없음")


def main():
    test_conflict_off_bit_identical()
    test_conflict_on(B=2)
    test_conflict_on(B=1)
    test_ops_semantics()
    test_op_index_stable()
    test_heldout_no_conflict()
    print(f"\nALL PASS ({_n} checks)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
