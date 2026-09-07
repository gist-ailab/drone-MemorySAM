#!/usr/bin/env python
"""[E2] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

LoRA 타깃 확장(Q/V → Q/K/V + proj + fc1/fc2)의 래퍼·파싱·파라미터 수를
검증한다. GPU·네트워크 불요(CPU + tiny ViT + 합성 입력, 1분 이내).

    python tools/smoke_lora_targets_e2.py

⚠️ model.py 는 아직 MODEL.LORA_TARGETS 를 인코더로 넘기지 않으므로(워커 보고서의
"model.py 후속 diff" 참고), 이 스모크는 build_reliadino 를 거치지 않고
FrozenViTEncoder 를 직접 생성해 lora_targets 경로를 검사한다.

검사 항목
  (a) byte-동일 : lora_targets None(기본) == ['qkv'] 명시 → state_dict 키·값·forward
                  완전 일치 + 어댑터 레이아웃 = arm A(a_q/b_q/a_v/b_v), 확장 텐서 없음
  (b) 전 타깃    : 출력 shape 불변 · 각 타깃 LoRA up-proj 에 grad 도달 ·
                  enabled=False → base 출력 · active_modality 바꾸면 출력 상이 ·
                  modality_ids(all=m) == active_modality=m (P51 등가)
  (c) 파라미터 수: 타깃별 = (슬라이스 수)·M·r·(in+out) 공식과 일치 ·
                  타깃별 rank(E2b: fc1/fc2 r16) · scale(=α/r) 균일 유지
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino.encoder import (FrozenViTEncoder,          # noqa: E402
                                             MultiModalLoRAQKV,
                                             MultiModalLoRALinear)

FAILS: list = []
MODALS = ['img', 'depth', 'event', 'lidar']
M = len(MODALS)
SIZE = 64            # 64/16 = 4x4 tokens


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def build_enc(targets, r=8, alpha=None, mode='per_modal', seed=1234):
    torch.manual_seed(seed)
    return FrozenViTEncoder(
        backbone='vit_tiny_patch16_224', fallback='vit_tiny_patch16_224',
        pretrained=False, img_size=SIZE, num_modalities=M,
        lora_r=r, lora_alpha=alpha, lora_mode=mode, lora_targets=targets)


def sd_hash(mod: nn.Module) -> str:
    import hashlib
    h = hashlib.md5()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        if v.dtype.is_floating_point:
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def adapter_numel(w: nn.Module) -> int:
    return sum(p.numel() for n, p in w.named_parameters() if not n.startswith('base.'))


# ═══════════════════════════════════════════════════════════════════════════
def test_byte_identity():
    print("\n(a) byte-동일 — lora_targets None(기본) == ['qkv']")
    S = 20260907
    e_def = build_enc(None, seed=S)
    e_qkv = build_enc(['qkv'], seed=S)

    check('a1 state_dict 키 집합 일치',
          set(e_def.state_dict()) == set(e_qkv.state_dict()))
    check('a2 state_dict 값까지 완전 일치(같은 seed)',
          sd_hash(e_def) == sd_hash(e_qkv))
    # 확장 텐서가 하나도 없어야 한다(K/proj/fc LoRA 미생성).
    ext = [k for k in e_def.state_dict()
           if any(t in k for t in ('.a_k', '.b_k', '.a_lin', '.b_lin'))]
    check('a3 확장 어댑터 텐서 없음(a_k/b_k/a_lin/b_lin)', not ext, str(ext[:3]))
    w = e_def.backbone.blocks[0].attn.qkv
    check('a4 qkv 래퍼 = MultiModalLoRAQKV, include_k=False',
          isinstance(w, MultiModalLoRAQKV) and not w.include_k)
    suffixes = {n.rsplit('.', 1)[-1] for n, _ in w.named_parameters()
                if not n.startswith('base.')}
    check('a5 어댑터 파라미터 = arm A 레이아웃(a_q/b_q/a_v/b_v)',
          suffixes == {'a_q', 'b_q', 'a_v', 'b_v'}, str(sorted(suffixes)))
    # forward byte-동일.
    x = torch.randn(2, 3, SIZE, SIZE)
    e_def.eval(); e_qkv.eval()
    with torch.no_grad():
        y0 = e_def(x, 0)
        y1 = e_qkv(x, 0)
    check('a6 forward 출력 bit-동일', torch.equal(y0, y1),
          f"max|Δ|={float((y0 - y1).abs().max()):.2e}")


# ═══════════════════════════════════════════════════════════════════════════
def test_all_targets():
    print("\n(b) 전 타깃 — [qkv_full, proj, fc1, fc2]")
    x = torch.randn(2, 3, SIZE, SIZE)

    # 출력 shape = 기본(qkv-only)과 동일해야 한다.
    e_base = build_enc(['qkv'], r=4, alpha=8, seed=5)
    e = build_enc(['qkv_full', 'proj', 'fc1', 'fc2'], r=4, alpha=8, seed=5)
    e_base.eval()
    with torch.no_grad():
        y_ref = e_base(x, 0)
    e.eval()
    with torch.no_grad():
        y = e(x, 0)
    check('b1 출력 shape 불변', tuple(y.shape) == tuple(y_ref.shape),
          f"{tuple(y.shape)} vs {tuple(y_ref.shape)}")

    # 타깃 종류 = 정확히 5개(qkv_full·proj·fc1·fc2) 래퍼가 블록마다.
    tags = sorted({getattr(w, '_e2_target', 'qkv') for w in e.lora_layers})
    check('b2 래퍼 타깃 = {qkv_full, proj, fc1, fc2}',
          set(tags) == {'qkv_full', 'proj', 'fc1', 'fc2'}, str(tags))

    # grad 도달: b_* zero-init 이라 init 국면엔 ∂L/∂b≠0 (∂L/∂a==0). 타깃별 up-proj
    # (b_q/b_k/b_v · b_lin) grad>0 을 확인한다.
    e.train()
    e.zero_grad(set_to_none=True)
    e(x, 1).sum().backward()

    def up_grad_by_target():
        agg: dict = {}
        for w in e.lora_layers:
            t = getattr(w, '_e2_target', 'qkv')
            g = 0.0
            for n, p in w.named_parameters():
                if n.startswith('base.'):
                    continue
                leaf = n.rsplit('.', 1)[-1]
                if leaf.startswith('b') and p.grad is not None:
                    g += float(p.grad.abs().sum())
            agg[t] = agg.get(t, 0.0) + g
        return agg

    grads = up_grad_by_target()
    check('b3 각 타깃 up-proj grad>0',
          all(grads.get(t, 0) > 0 for t in ('qkv_full', 'proj', 'fc1', 'fc2')),
          str({k: round(v, 3) for k, v in grads.items()}))

    # enabled=False → base 출력 · 다시 True → delta 로 출력 변화 · 되돌리면 base 복귀.
    # b_* 를 0 에서 떼어 delta 가 실제로 살아 있게 한다.
    with torch.no_grad():
        for n, p in e.named_parameters():
            if n.rsplit('.', 1)[-1] in ('b_q', 'b_k', 'b_v', 'b_lin'):
                p.add_(torch.randn_like(p) * 0.05)
    e.eval()
    with torch.no_grad():
        e.set_lora_enabled(False); y_off = e(x, 0)
        e.set_lora_enabled(True);  y_on = e(x, 0)
        e.set_lora_enabled(False); y_off2 = e(x, 0)
    check('b4 enabled=True 는 delta 로 base 와 상이',
          not torch.allclose(y_on, y_off, atol=1e-5))
    check('b5 enabled=False 는 base 출력(재현)', torch.equal(y_off, y_off2))

    # active_modality 바꾸면 출력이 달라진다(per-modal 어댑터).
    e.set_lora_enabled(True)
    with torch.no_grad():
        y_m0 = e(x, 0)
        y_m1 = e(x, 1)
    check('b6 active_modality 바꾸면 출력 상이', not torch.allclose(y_m0, y_m1, atol=1e-5))

    # P51 등가: modality_ids(all=m) == active_modality=m (proj 래퍼 하나로 검사).
    proj_w = next(w for w in e.lora_layers if getattr(w, '_e2_target', '') == 'proj')
    xx = torch.randn(3, 7, proj_w.in_features)
    ok_all = True
    proj_w.eval()
    for m in range(M):
        proj_w.modality_ids = None; proj_w.active_modality = m
        ys = proj_w(xx)
        proj_w.modality_ids = torch.full((3,), m, dtype=torch.long)
        yi = proj_w(xx)
        ok_all = ok_all and torch.allclose(ys, yi, atol=1e-5)
    proj_w.modality_ids = None
    check('b7 MultiModalLoRALinear: modality_ids(all=m)==active_modality=m', ok_all)


# ═══════════════════════════════════════════════════════════════════════════
def test_param_counts():
    print("\n(c) 파라미터 수 · 타깃별 rank · scale")
    r = 4
    e = build_enc(['qkv_full', 'proj', 'fc1', 'fc2'], r=r, alpha=2 * r, seed=9)
    nblk = len(e.backbone.blocks)

    def expected(w):
        if isinstance(w, MultiModalLoRAQKV):
            slices = 3 if w.include_k else 2
            return slices * w.num_modalities * w.r * (w.in_features + w.attn_dim)
        return w.num_modalities * w.r * (w.in_features + w.out_features)

    by_target: dict = {}
    ok_each = True
    for w in e.lora_layers:
        t = getattr(w, '_e2_target', 'qkv')
        act, exp = adapter_numel(w), expected(w)
        ok_each = ok_each and (act == exp)
        by_target[t] = by_target.get(t, 0) + act
    check('c1 타깃별 파라미터 수 = 공식 (슬라이스·M·r·(in+out))', ok_each,
          str({k: f"{v:,}" for k, v in by_target.items()}))

    # scale(=α/r) 균일 = 2.0 (alpha=2r).
    scales = {round(float(w.scale), 6) for w in e.lora_layers}
    check('c2 scale 전 타깃 균일 = 2.0', scales == {2.0}, str(scales))

    # 타깃별 rank 지정(E2b): fc1/fc2 는 r16, qkv_full/proj 는 r32. scale 은 여전히 균일.
    eb = build_enc(['qkv_full', 'proj', ['fc1', 16], ['fc2', 16]],
                   r=32, alpha=64, seed=9)
    rank_of = {getattr(w, '_e2_target', 'qkv'): w.r for w in eb.lora_layers}
    check('c3 E2b 타깃별 rank: qkv_full/proj=32, fc1/fc2=16',
          rank_of == {'qkv_full': 32, 'proj': 32, 'fc1': 16, 'fc2': 16}, str(rank_of))
    scb = {round(float(w.scale), 6) for w in eb.lora_layers}
    check('c4 E2b rank 달라도 scale 균일 = 2.0 (α=2·r_t)', scb == {2.0}, str(scb))
    print(f"      [E2 params] blocks={nblk}  by_target(전타깃 r{r})="
          f"{ {k: f'{v:,}' for k, v in by_target.items()} }")


# ═══════════════════════════════════════════════════════════════════════════
def test_guards():
    print("\n(d) 방어 — 잘못된 조합은 조용히 무시하지 말고 에러")
    def raises(fn):
        try:
            fn(); return False
        except (ValueError, NotImplementedError):
            return True
    check('d1 qkv+qkv_full 동시 지정 거부',
          raises(lambda: build_enc(['qkv', 'qkv_full'])))
    check('d2 알 수 없는 타깃 거부', raises(lambda: build_enc(['bogus'])))
    check('d3 shared 모드 + 확장 타깃 거부',
          raises(lambda: build_enc(['qkv_full', 'fc1'], mode='shared')))
    # shared + qkv 단독은 허용(기존 arm B 경로).
    ok = True
    try:
        build_enc(['qkv'], mode='shared')
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"      shared+qkv 빌드 예외: {e!r}")
    check('d4 shared 모드 + qkv 단독은 허용', ok)


def main() -> int:
    print("=" * 72)
    print("[E2] smoke — LoRA 타깃 확장 (qkv_full · proj · fc1 · fc2)")
    print("=" * 72)
    test_byte_identity()
    test_all_targets()
    test_param_counts()
    test_guards()
    print("\n" + "=" * 72)
    if FAILS:
        print(f"❌ {len(FAILS)} FAIL: {FAILS}")
        return 1
    print("✅ ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
