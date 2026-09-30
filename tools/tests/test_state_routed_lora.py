#!/usr/bin/env python3
"""[P56-C] state-routed shared/per-sensor LoRA 단위 테스트 — CPU 전용, 데이터/ckpt 불필요.

검증 대상(과제 지시 5항):
  1. off 등가  : LORA_MODE=per_modal 경로가 이 변경 전과 동일 — (a) 키 없음 vs 명시
     per_modal state_dict hash 일치 + 전 블록 래퍼가 손대지 않은 MultiModalLoRAQKV
     + p56c 키 부재, (b) git HEAD 의 encoder.py 로 같은 seed 로 FrozenViTEncoder 를
     만들어 state_dict·forward 출력이 bitwise 일치(git 불가 환경선 SKIP).
  2. 극단 등가 : α≡1 → shared 항만, α≡0 → 센서별 항만과 출력 일치(Q/V 슬라이스).
  3. 초기값    : 라우터 초기 α = 0.5 (±1e-6) — 최종 Linear zero-init.
  4. 기울기    : 라우터·공유·센서별 LoRA 전 파라미터에 grad 도달(2단계: init 국면
     b/head[-1], lift 국면 a/norm/embed). 라우터 입력은 detach — 블록 k 출력으로
     grad 가 라우터를 통해 역류하지 않는다.
  5. 모달 전환 : set_modality 후 α 가 None 으로 초기화(이전 모달 α 미세첨).
  부가: PER_TOKEN=false 스칼라 α 형상, modality_ids==스칼라 경로 등가, 로깅 stats.

실행:  PYTHONPATH=semseg/models/sam2:. python tools/tests/test_state_routed_lora.py
       (모두 통과 시 'ALL PASS')  — timm 필요([3] 통합), GPU 불요.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.models.reliadino.encoder import (                    # noqa: E402
    FrozenViTEncoder, MultiModalLoRAQKV, StateRoutedLoRAQKV)
from semseg.models.reliadino.p56c_router import StateRouter      # noqa: E402

MODALS = ['img', 'depth', 'event', 'lidar']
M = len(MODALS)
_n = 0
_fails: list = []


def check(cond, msg):
    global _n
    _n += 1
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        _fails.append(msg)
    return ok


# ── [1] 모듈 단위: StateRoutedLoRAQKV (합성 nn.Linear qkv, timm 불요) ────────
def test_wrapper():
    print("\n[1] StateRoutedLoRAQKV 래퍼 단위 (nn.Linear qkv)")
    torch.manual_seed(0)
    in_f, d = 192, 192
    base = nn.Linear(in_f, 3 * d, bias=True)
    B, N = 2, 5
    x = torch.randn(B, N, in_f)
    w = StateRoutedLoRAQKV(base, M, shared_r=8, residual_r=8)
    # 어댑터를 0에서 떼어 shared·sensor 델타가 실제로 살게 한다(극단 등가 재현).
    with torch.no_grad():
        for n, p in w.named_parameters():
            if n.startswith(('a_', 'b_')):
                p.add_(torch.randn_like(p) * 0.1)

    def deltas(modality):
        w.modality_ids = None
        w.active_modality = modality
        dq_s = torch.nn.functional.linear(
            torch.nn.functional.linear(x, w.a_q_s), w.b_q_s) * w.scale_s
        dq_r = torch.nn.functional.linear(
            torch.nn.functional.linear(x, w.a_q_r[modality]), w.b_q_r[modality]) * w.scale_r
        return dq_s, dq_r

    # T1 — init 등가는 별도 인스턴스(b zero-init)로: α=None(상수 0.5) 경로 포함.
    w0 = StateRoutedLoRAQKV(base, M, shared_r=8, residual_r=8)
    w0.active_modality = 1
    check(torch.equal(w0(x), base(x)), 'T1 init delta=0 (b_* zero-init, α=None→0.5 상수)')

    # T2 — 극단 등가: α≡1 → shared 만, α≡0 → sensor_m 만.
    y_base = base(x)
    dq_s, dq_r = deltas(1)
    w.alpha = torch.ones(B, N, 1)
    y1 = w(x)
    exp1 = y_base.clone()
    exp1[..., :d] += dq_s
    exp1[..., 2 * d:] += (torch.nn.functional.linear(
        torch.nn.functional.linear(x, w.a_v_s), w.b_v_s) * w.scale_s)
    check(torch.allclose(y1, exp1, atol=1e-5), 'T2a α≡1 → shared 항만과 일치')
    w.alpha = torch.zeros(B, N, 1)
    y0 = w(x)
    exp0 = y_base.clone()
    exp0[..., :d] += dq_r
    exp0[..., 2 * d:] += (torch.nn.functional.linear(
        torch.nn.functional.linear(x, w.a_v_r[1]), w.b_v_r[1]) * w.scale_r)
    check(torch.allclose(y0, exp0, atol=1e-5), 'T2b α≡0 → sensor_1 항만과 일치')
    check(torch.allclose(y0[..., d:2 * d], y_base[..., d:2 * d], atol=1e-7),
          'T2c K 슬라이스 불변')

    # T3 — α 텐서 경로 grad: α.requires_grad → 라우터 방향으로 grad 흐름.
    w.alpha = torch.full((B, N, 1), 0.3, requires_grad=True)
    w(x).sum().backward()
    check(w.alpha.grad is not None and float(w.alpha.grad.abs().sum()) > 0,
          'T3 α 텐서에 grad 역류 (라우터 학습 경로)')

    # T4 — modality_ids(배치-모달) 잔차 gather == active_modality 스칼라 경로.
    w.alpha = None
    ok_all = True
    for m in range(M):
        w.modality_ids = None
        w.active_modality = m
        y_scalar = w(x)
        w.modality_ids = torch.full((B,), m, dtype=torch.long)
        ok_all = ok_all and torch.allclose(y_scalar, w(x), atol=1e-5)
    w.modality_ids = None
    check(ok_all, 'T4 modality_ids(all=m) == active_modality=m')


# ── [2] StateRouter 단위 ─────────────────────────────────────────────────────
def test_router():
    print("\n[2] StateRouter 단위")
    torch.manual_seed(1)
    C, H = 1024, 64
    B, N = 2, 7
    r = StateRouter(dim=C, num_modalities=M, hidden=H, per_token=True)
    tokens = torch.randn(B, N, C, requires_grad=True)

    # R1 — 초기 α=0.5 (최종 Linear zero-weight/bias → σ(0)).
    with torch.no_grad():
        a = r(tokens, 0)
    check(a.shape == (B, N, 1) and float((a - 0.5).abs().max()) <= 1e-6,
          f'R1 초기 α=0.5±1e-6 (max|Δ|={float((a - 0.5).abs().max()):.2e}, shape {tuple(a.shape)})')

    # R2 — 파라미터 수 ≈ 7만(dim1024): LN 2C + embed M·C + L1 C·H+H + L2 H+1.
    n = sum(p.numel() for p in r.parameters())
    exp = 2 * C + M * C + (C * H + H) + (H + 1)
    check(n == exp and 60000 <= n <= 90000, f'R2 파라미터 수 {n:,} == 공식 {exp:,} (~7만)')

    # R3 — grad: 라우터 파라미터에는 도달, 입력 토큰에는 미도달(stop-grad).
    a = r(tokens, 1)
    a.sum().backward()
    check(tokens.grad is None, 'R3a 라우터 입력 detach — 블록 k 출력으로 grad 미역류')
    got = [n_ for n_, p in r.named_parameters() if p.grad is not None
           and float(p.grad.abs().sum()) > 0]
    check(set(got) == {'head.2.weight', 'head.2.bias'},
          f'R3b init 국면 grad 도달 = 최종 Linear 만 ({sorted(got)})')
    r2 = StateRouter(dim=C, num_modalities=M, hidden=H)
    with torch.no_grad():                      # 최종 Linear 를 떼서 전 층에 grad 관통
        for p in r2.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    t2 = torch.randn(B, N, C)
    r2(t2, 2).sum().backward()
    got2 = {n_ for n_, p in r2.named_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) > 0}
    check(got2 == {n_ for n_, _ in r2.named_parameters()},
          f'R3c lift 국면 라우터 전 파라미터 grad 도달 ({len(got2)}/{len(list(r2.named_parameters()))})')

    # R4 — PER_TOKEN=false → (B,1,1) 스칼라 α(설계서 C-2).
    r3 = StateRouter(dim=C, num_modalities=M, hidden=H, per_token=False)
    with torch.no_grad():
        a3 = r3(tokens.detach(), 0)
    check(a3.shape == (B, 1, 1), f'R4 PER_TOKEN=false α shape {tuple(a3.shape)} == (B,1,1)')

    # R5 — 로깅 stats(모달별 mean/std_token, detach).
    r4 = StateRouter(dim=C, num_modalities=M, hidden=H)
    with torch.no_grad():
        for m in range(M):
            r4(torch.randn(B, N, C), m)
    st = r4._last_alpha_stats
    check(all(v is not None and 0 <= v <= 1 for v in st['mean'])
          and all(v is not None and v >= 0 for v in st['std_token']),
          f'R5 _last_alpha_stats 모달별 기록 mean={tuple(round(v, 3) for v in st["mean"])}')


# ── [3] 통합(tiny ViT, build_reliadino) ─────────────────────────────────────
def _tiny_cfg(extra_model: dict | None = None, size: int = 128) -> dict:
    """smoke_elora.tiny_cfg 와 같은 레시피(router/M2F/P39 on) — LoRA 모드만 갈아끼운다."""
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE': 'P56C-tiny',
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


def _load_head_encoder():
    """git HEAD 시점의 encoder.py 를 임시 모듈로 로드(불가 시 None → 스킵)."""
    import importlib.util
    import subprocess
    import tempfile
    try:
        src = subprocess.run(
            ['git', '-C', str(_REPO), 'show', 'HEAD:semseg/models/reliadino/encoder.py'],
            capture_output=True, check=True).stdout
    except Exception as e:                                   # noqa: BLE001
        print(f"  [SKIP] HEAD encoder 로드 불가: {e!r}")
        return None
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'encoder_head.py'
        p.write_bytes(src)
        spec = importlib.util.spec_from_file_location('encoder_head', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def test_integration():
    print("\n[3] 통합 (tiny ViT, build_reliadino)")
    from semseg.models.reliadino import build_reliadino
    size = 128
    inputs = [torch.randn(1, 3, size, size) for _ in range(M)]

    # B1 — off 등가(현재 코드 안): 키 없음 == 명시 per_modal.
    torch.manual_seed(1234)
    m_off = build_reliadino(_tiny_cfg(), 25)
    torch.manual_seed(1234)
    m_pm = build_reliadino(_tiny_cfg({'LORA_MODE': 'per_modal'}), 25)
    check(_sd_hash(m_off) == _sd_hash(m_pm), 'B1a off(키 없음) == per_modal state_dict hash 일치')
    wrappers = [blk.attn.qkv for blk in m_off.encoder.backbone.blocks]
    check(all(type(w_) is MultiModalLoRAQKV for w_ in wrappers),
          'B1b off 전 블록 래퍼 = 손대지 않은 MultiModalLoRAQKV')
    check(all('p56c' not in k for k in m_off.state_dict()),
          'B1c off state_dict 에 p56c 키 없음')

    # B2 — off 등가(변경 전과): HEAD encoder 로 같은 seed 생성 → bitwise 일치.
    head = _load_head_encoder()
    if head is not None:
        kw = dict(backbone='vit_tiny_patch16_224', fallback='vit_tiny_patch16_224',
                  pretrained=False, img_size=64, num_modalities=M, lora_r=8)
        torch.manual_seed(77)
        e_head = head.FrozenViTEncoder(**kw)
        torch.manual_seed(77)
        e_new = FrozenViTEncoder(**kw)
        sd_h, sd_n = e_head.state_dict(), e_new.state_dict()
        check(set(sd_h) == set(sd_n) and all(torch.equal(sd_h[k], sd_n[k]) for k in sd_h),
              'B2a HEAD(변경 전) per_modal encoder state_dict 전 키·값 bitwise 일치')
        x = torch.randn(1, 3, 64, 64)
        e_head.set_modality(2), e_new.set_modality(2)
        check(torch.equal(e_head(x, 2), e_new(x, 2)),
              'B2b HEAD(변경 전) per_modal encoder forward 출력 bitwise 일치')

    # B3 — state_routed 빌드: 블록 경계, [P56-C] 로그, forward+backward, α 주입.
    import contextlib
    import io
    torch.manual_seed(7)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = build_reliadino(_tiny_cfg({
            'LORA_MODE': 'state_routed',
            'LORA_ROUTER': {'SHARED_R': 8, 'BLOCK_FRAC': 0.25, 'HIDDEN': 32,
                            'PER_TOKEN': True, 'APPLY_FROM_BLOCK': 7}}), 25)
    check('[P56-C] router params=' in buf.getvalue() and 'apply_from_block=7' in buf.getvalue()
          and 'block_frac=0.25' in buf.getvalue(), 'B3a [P56-C] 기동 로그 1행 출력')
    n_blk = len(model.encoder.backbone.blocks)
    below = [blk.attn.qkv for blk in model.encoder.backbone.blocks[:6]]
    above = [blk.attn.qkv for blk in model.encoder.backbone.blocks[6:]]
    check(all(type(w_) is MultiModalLoRAQKV for w_ in below)
          and all(type(w_) is StateRoutedLoRAQKV for w_ in above)
          and len(model.encoder.state_routed_layers) == n_blk - 6,
          f'B3b 블록 1~6 per_modal · 7~{n_blk} StateRouted '
          f'({len(model.encoder.state_routed_layers)}개)')
    check(model._p56c_block == 3, 'B3c BLOCK_FRAC 0.25 → 12블록 백본에서 라우터 블록 3')
    # 라우터·LoRA 를 0에서 떼어 전 파라미터 grad 관통 확인(init 국면 b/head[-1] 만).
    with torch.no_grad():
        for p in model.p56c_router.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for n_, p in model.named_parameters():
            if '.attn.qkv.' in n_ and '.base.' not in n_:
                p.add_(torch.randn_like(p) * 0.1)
    model.train()
    out = model(inputs, gt_mask=None)
    loss = out[0].float().sum()
    loss.backward()
    check(torch.isfinite(loss), f'B3d forward+backward OK (loss={float(loss):.3e})')
    got = {n_ for n_, p in model.named_parameters() if 'p56c_router' in n_
           and p.grad is not None and float(p.grad.abs().sum()) > 0}
    all_r = {f'p56c_router.{n_}' for n_, _ in model.p56c_router.named_parameters()}
    check(got == all_r,
          f'B3e 라우터 전 파라미터 grad 도달 ({len(got)}/{len(all_r)})')
    lora_got = {n_ for n_, p in model.named_parameters()
                if '.attn.qkv.' in n_ and '.base.' not in n_
                and p.grad is not None and float(p.grad.abs().sum()) > 0}
    lora_all = {n_ for n_, p in model.named_parameters()
                if '.attn.qkv.' in n_ and '.base.' not in n_ and p.requires_grad}
    check(lora_got == lora_all,
          f'B3f qkv 어댑터 전 파라미터 grad 도달 ({len(lora_got)}/{len(lora_all)})')

    # B4 — α 주입 실증 + 모달 전환 초기화.
    a = model.encoder.state_routed_layers[0].alpha
    check(a is not None and a.dim() == 3 and a.shape[-1] == 1
          and a.shape[-2] == (size // 16) ** 2 + model.encoder.num_prefix_tokens,
          f'B4a forward 후 α (B,N,1) 주입됨 {tuple(a.shape)} (마지막 모달)')
    st = model.p56c_router._last_alpha_stats['mean']
    check(all(v is not None for v in st), f'B4b α stats 4모달 기록 {tuple(round(v, 3) for v in st)}')
    model.encoder.set_modality(0)
    check(all(w_.alpha is None for w_ in model.encoder.state_routed_layers),
          'B4c set_modality 후 α=None 초기화(이전 모달 α 세척)')


def main() -> int:
    print("=" * 72)
    print("[P56-C] state-routed LoRA 테스트 — shared/per-sensor 혼합 + StateRouter")
    print("=" * 72)
    test_wrapper()
    test_router()
    try:
        test_integration()
    except Exception as e:                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, f'[3] 통합 테스트 실행 자체 실패: {e!r}')
    print("\n" + "=" * 72)
    if _fails:
        print(f"❌ {len(_fails)}/{_n} FAIL: {_fails}")
        return 1
    print(f"✅ ALL PASS ({_n}/{_n})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
