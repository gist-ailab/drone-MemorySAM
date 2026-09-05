#!/usr/bin/env python
"""[E-LORA] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

LoRA 구조 ablation 3-arm (A per_modal / B shared / C shared_residual)의 래퍼·
빌더·파라미터 수를 검증한다. GPU·네트워크 불요(CPU + tiny ViT + 합성 입력).

    python tools/smoke_elora.py

검사 항목
  [1] 래퍼 단위 (timm 불요, 합성 nn.Linear)
      T1 init 등가성 : 세 모드 모두 init 시 delta=0 → y == base(x) (b_* zero-init)
      T2 파라미터 수 : B == A/M · C < A · C 공유항 = 고정(항상 활성)
      T3 grad 도달   : shared/residual 신설 파라미터에 실제로 grad>0
      T4 modality_ids: shared_residual 잔차 gather 경로가 스칼라 경로와 수치 등가
                       (모든 원소 = m 인 ids ↔ active_modality=m)
      T5 모달 분화   : shared 는 모달 무관(동일 출력) · shared_residual 은 모달별 상이
  [2] 빌더/모델 통합 (tiny ViT)
      B1 byte-동일   : LORA_MODE 미지정(off) == 'per_modal' 명시 → state_dict 완전 일치
                       + qkv 래퍼가 손대지 않은 MultiModalLoRAQKV (arm A 불변조건)
      B2 3-arm 빌드  : shared/shared_residual 도 정상 빌드 + forward + backward
                       + [E-LORA] 파라미터 수 로그의 방향(B<A, C<A) 확인
      B3 eval 결정론 : model.eval() 두 forward 가 bit-동일
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino.encoder import (MultiModalLoRAQKV,          # noqa: E402
                                             SharedLoRAQKV)

FAILS: list = []
MODALS = ['img', 'depth', 'event', 'lidar']
M = len(MODALS)


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def n_params(mod: nn.Module) -> int:
    """어댑터 파라미터 수(래퍼가 감싼 base Linear 는 제외)."""
    return sum(p.numel() for n, p in mod.named_parameters() if not n.startswith('base.'))


# ═══════════════════════════════════════════════════════════════════════════
def test_wrappers():
    print("\n[1] 래퍼 단위 (per_modal / shared / shared_residual)")
    torch.manual_seed(0)
    in_f, d = 192, 192
    base = nn.Linear(in_f, 3 * d, bias=True)
    B, N = 2, 5
    x = torch.randn(B, N, in_f)

    a = MultiModalLoRAQKV(base, M, r=16, alpha=None)                 # arm A
    b = SharedLoRAQKV(base, M, shared_r=16, residual_r=0)            # arm B
    c = SharedLoRAQKV(base, M, shared_r=8, residual_r=8)             # arm C

    # T1 — init 등가성: b_* zero-init 이라 어떤 모드도 init 시 base 와 동일해야 한다.
    y0 = base(x)
    for name, w in [('per_modal', a), ('shared', b), ('shared_residual', c)]:
        w.active_modality = 0
        check(f'T1 {name} init delta=0 (y==base)',
              torch.allclose(w(x), y0, atol=1e-6))

    # T2 — 파라미터 수 방향.
    na, nb, nc = n_params(a), n_params(b), n_params(c)
    check('T2a B == A / M (완전공유, 둘 다 r16)', nb * M == na, f"A={na} B={nb} M={M}")
    check('T2b C < A (공유8+잔차8 < per-modal16)', nc < na, f"A={na} C={nc}")
    check('T2c C > B (C 는 잔차항만큼 더 크다)', nc > nb, f"B={nb} C={nc}")
    # C 공유항(항상 활성) = 2*(a_q_s + b_q_s) = 2*shared_r*(in+d).
    shared_only = c.a_q_s.numel() + c.b_q_s.numel() + c.a_v_s.numel() + c.b_v_s.numel()
    resid_only = c.a_q_r.numel() + c.b_q_r.numel() + c.a_v_r.numel() + c.b_v_r.numel()
    check('T2d C = 공유항(고정) + 잔차항', shared_only + resid_only == nc,
          f"shared={shared_only} resid={resid_only} total={nc}")
    check('T2e C 공유항 = 2·r_s·(in+d) 고정, 모달 무관',
          shared_only == 2 * 8 * (in_f + d), f"shared={shared_only}")

    # T3 — grad 도달. b_* zero-init 이라 init 국면엔 ∂L/∂a==0, ∂L/∂b≠0 (arm A 규약).
    #   국면1: init 에서 b_*_s / b_*_r 에 grad. 국면2: b 를 0 에서 떼면 a 에도 grad.
    def grad_reaches(w, suffixes, lift=False):
        if lift:
            with torch.no_grad():
                for n, p in w.named_parameters():
                    if n.startswith('a_') or n.startswith('b_'):
                        p.add_(0.05)
        w.zero_grad(set_to_none=True)
        w.modality_ids = None
        w.active_modality = 1
        w(x).sum().backward()
        out = {}
        for s in suffixes:
            g = [float(p.grad.abs().sum()) for n, p in w.named_parameters()
                 if n == s and p.grad is not None]
            out[s] = g[0] if g else 0.0
        return out

    gb = grad_reaches(b, ['b_q_s', 'b_v_s'])
    check('T3a shared b_q_s/b_v_s grad>0 (init)',
          gb['b_q_s'] > 0 and gb['b_v_s'] > 0, str(gb))
    gc = grad_reaches(c, ['b_q_s', 'b_v_s', 'b_q_r', 'b_v_r'])
    check('T3b shared_residual 공유+잔차 b_* grad>0 (init)',
          all(v > 0 for v in gc.values()), str({k: round(v, 4) for k, v in gc.items()}))
    gc2 = grad_reaches(c, ['a_q_s', 'a_v_s', 'a_q_r', 'a_v_r'], lift=True)
    check('T3c shared_residual 공유+잔차 a_* grad>0 (b≠0 이후)',
          all(v > 0 for v in gc2.values()), str({k: round(v, 4) for k, v in gc2.items()}))

    # T4 — modality_ids 잔차 gather 경로가 스칼라 경로와 수치 등가여야 한다(P51 계약).
    c2 = SharedLoRAQKV(base, M, shared_r=8, residual_r=8)
    with torch.no_grad():                            # b 를 0 에서 떼어 잔차가 실제로 살게
        for n, p in c2.named_parameters():
            if n.startswith('b_'):
                p.add_(torch.randn_like(p) * 0.1)
    c2.eval()
    ok_all = True
    for m in range(M):
        c2.modality_ids = None
        c2.active_modality = m
        y_scalar = c2(x)
        c2.modality_ids = torch.full((B,), m, dtype=torch.long)
        y_ids = c2(x)
        ok_all = ok_all and torch.allclose(y_scalar, y_ids, atol=1e-5)
    c2.modality_ids = None
    check('T4 shared_residual: modality_ids(all=m) == active_modality=m', ok_all)

    # T5 — 모달 분화. shared 는 모달 무관(동일), shared_residual 은 모달별 상이.
    b.modality_ids = None
    b.active_modality = 0; yb0 = b(x)
    b.active_modality = 1; yb1 = b(x)
    check('T5a shared: 모달 바꿔도 출력 동일(모달 무관)', torch.allclose(yb0, yb1, atol=1e-6))
    c2.active_modality = 0; yc0 = c2(x)
    c2.active_modality = 1; yc1 = c2(x)
    check('T5b shared_residual: 모달별 출력 상이(잔차 활성)',
          not torch.allclose(yc0, yc1, atol=1e-4))


# ═══════════════════════════════════════════════════════════════════════════
def tiny_cfg(extra_model: dict | None = None, size: int = 128) -> dict:
    """smoke_p50 의 tiny 판과 동일 레시피(router/M2F/P39 on) — LoRA 모드만 갈아끼운다."""
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE': 'ELORA-tiny',
        'BACKBONE_TIMM': 'vit_tiny_patch16_224',
        'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
        'PRETRAINED_BACKBONE': False,
        'LORA_R': 16, 'LORA_ALPHA': None, 'FPN_DIM': 64,
        'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 2.0,
                   'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5, 'TRUNK': 'gated_mlp',
                   'ATTN_BIAS': {'ENABLE': False}},
        'CONSISTENCY': {'ENABLE': False},
        'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
        'CALIBRATION': {'ENABLE': False},
        'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
        'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 1, 'DIM': 64,
                'NUM_HEADS': 4, 'MLP_RATIO': 2.0, 'POINTS': 64, 'SRC': 'modal',
                'ANCHORED': True, 'POINT_QUOTA': 8},
        'P39': {'TRUNK_EXP': True, 'ARBITER': True, 'TRUNK_MODE': 'gated_mlp',
                'TRUNK_HIDDEN': 32, 'VICREG': {'ENABLE': True, 'TOKENS': 64}},
    }
    if extra_model:
        m.update(extra_model)
    return {
        'MODEL': m,
        'DATASET': {'NAME': 'DELIVER', 'MODALS': MODALS, 'NUM_CLASSES': 25,
                    'IGNORE_LABEL': 255},
        'TRAIN': {'IMAGE_SIZE': [size, size], 'BATCH_SIZE': 1, 'DDP': False},
    }


def _sd_hash(model: nn.Module) -> str:
    import hashlib
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        if v.dtype.is_floating_point:
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _build(cfg):
    from semseg.models.reliadino import build_reliadino
    return build_reliadino(cfg, 25)


def test_integration():
    print("\n[2] 빌더/모델 통합 (tiny ViT)")
    size = 128
    inputs = [torch.randn(1, 3, size, size) for _ in range(M)]

    # B1 — byte-동일: off(키 없음) 와 명시 per_modal 이 완전히 같은 모델이어야 한다.
    torch.manual_seed(1234)
    m_off = _build(tiny_cfg())
    torch.manual_seed(1234)
    m_pm = _build(tiny_cfg({'LORA_MODE': 'per_modal'}))
    check('B1a off == per_modal state_dict 완전 일치', _sd_hash(m_off) == _sd_hash(m_pm))
    wrapper = m_off.encoder.backbone.blocks[0].attn.qkv
    check('B1b off 래퍼 = 손대지 않은 MultiModalLoRAQKV',
          isinstance(wrapper, MultiModalLoRAQKV) and not isinstance(wrapper, SharedLoRAQKV),
          type(wrapper).__name__)
    suffixes = {n.rsplit('.', 1)[-1] for n, _ in wrapper.named_parameters()
                if not n.startswith('base.')}
    check('B1c off 어댑터 파라미터 = arm A 레이아웃(a_q/b_q/a_v/b_v)',
          suffixes == {'a_q', 'b_q', 'a_v', 'b_v'}, str(sorted(suffixes)))

    # B2 — 3-arm 빌드 + forward/backward + 파라미터 수 방향.
    def lora_train_params(model):
        return sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and '.attn.qkv.' in n and '.base.' not in n)

    counts = {}
    for arm, extra in [('A per_modal', {'LORA_MODE': 'per_modal', 'LORA_R': 16}),
                       ('B shared', {'LORA_MODE': 'shared', 'LORA_R': 16}),
                       ('C shared_residual', {'LORA_MODE': 'shared_residual',
                                              'LORA_SHARED_R': 8, 'LORA_RESIDUAL_R': 8})]:
        torch.manual_seed(7)
        model = _build(tiny_cfg(extra))
        counts[arm] = lora_train_params(model)
        model.train()
        out = model(inputs, gt_mask=None)
        logits = out[0]
        loss = logits.float().sum()
        loss.backward()
        # 신설 파라미터에 grad 가 실제로 흐르는지 (arm 별로 존재하는 것만).
        reached = 0
        for n, p in model.named_parameters():
            if '.attn.qkv.' in n and '.base.' not in n and p.requires_grad:
                if p.grad is not None and float(p.grad.abs().sum()) > 0:
                    reached += 1
        check(f'B2 {arm}: forward+backward OK · qkv 어댑터 grad 도달 텐서 {reached}개',
              torch.isfinite(loss) and reached > 0, f"loss={float(loss):.3e}")

    A, Bc, Cc = counts['A per_modal'], counts['B shared'], counts['C shared_residual']
    check('B2d 파라미터 수 방향 B == A/M', Bc * M == A, f"A={A:,} B={Bc:,} M={M}")
    check('B2e 파라미터 수 방향 C < A', Cc < A, f"A={A:,} C={Cc:,}")
    print(f"      [E-LORA params] A(per_modal r16)={A:,}  "
          f"B(shared r16)={Bc:,}  C(shared8+resid8)={Cc:,}")

    # B3 — eval 결정론.
    torch.manual_seed(7)
    model = _build(tiny_cfg({'LORA_MODE': 'shared_residual',
                             'LORA_SHARED_R': 8, 'LORA_RESIDUAL_R': 8}))
    model.eval()
    with torch.no_grad():
        o1 = model(inputs)[0]
        o2 = model(inputs)[0]
    check('B3 shared_residual eval 두 forward bit-동일', torch.equal(o1, o2))


def main() -> int:
    print("=" * 72)
    print("[E-LORA] smoke — LoRA 구조 ablation 3-arm (per_modal / shared / shared_residual)")
    print("=" * 72)
    test_wrappers()
    try:
        test_integration()
    except Exception as e:               # timm 부재/네트워크 등 통합만 스킵되면 표시
        import traceback
        traceback.print_exc()
        check('[2] 통합 스모크 실행', False, repr(e))

    print("\n" + "=" * 72)
    if FAILS:
        print(f"❌ {len(FAILS)} FAIL: {FAILS}")
        return 1
    print("✅ ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
