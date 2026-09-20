"""[R1/R2] BoundaryRefine + ComponentLoss 스모크 — CPU, ~1분.

검사 항목:
  (a) BOUNDARY_REFINE off ⇒ 키 없는 baseline 과 state_dict 키·forward 출력 byte-동일
      (off 계약). self.boundary_refine 미생성(모듈 생성조차 안 함) assert.
  (b) BOUNDARY_REFINE on ⇒ 파라미터 수 증가량 출력, γ·k·τ 에 grad 도달, γ init 0.1.
  (c) 에지 맵 형상 (B,1,H,W) · e 범위 [0,1].
  (d) ComponentLoss: 성분 1개 / 성분 없음(전부 ignore) / 다중 클래스 케이스에서
      값 범위 [0,1] · grad 흐름 · ignore 처리.
  (e) BOUNDARY_REFINE on 모델 forward + 그 출력에 ComponentLoss 동시 적용 정상.

실행:
    python tools/smoke_boundary_refine.py
timm(+torch) 이 없으면 SKIP 한다(모델 부분). ComponentLoss(d)는 timm 무관하게 돈다.
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
    """tiny 학습 config. BOUNDARY_REFINE 는 호출부가 덧붙인다."""
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
    torch.manual_seed(20260920)
    return fn()


def _component_checks():
    """(d) ComponentLoss 검사 — timm 무관."""
    from semseg.models.reliadino.component_loss import ComponentLoss, _HAS_SCIPY
    print("\n[d] ComponentLoss — 성분 1개 / 없음 / 다중 클래스")
    if not _HAS_SCIPY:
        check("d: scipy 존재(연결 성분 라벨링)", False, "scipy.ndimage 부재")
        return
    K, H, W = NUM_CLASSES, 32, 32
    fn = ComponentLoss(num_classes=K, ignore_index=255)

    # 성분 1개: 중앙 사각형 하나가 class 2.
    logits = torch.randn(1, K, H, W, requires_grad=True)
    tgt = torch.full((1, H, W), 255, dtype=torch.long)
    tgt[0, 8:16, 8:16] = 2
    depth = torch.rand(1, 3, H, W)
    loss1, nc1 = fn(logits, tgt, depth=depth)
    check("d1: 성분 1개 loss ∈ [0,1]", 0.0 <= float(loss1) <= 1.0, f"loss={float(loss1):.4f}")
    check("d1: n_comp == 1", nc1 == 1, f"n_comp={nc1}")
    loss1.backward()
    check("d1: logits grad 흐름", logits.grad is not None and float(logits.grad.norm()) > 0.0,
          f"‖grad‖={float(logits.grad.norm()) if logits.grad is not None else 'None'}")

    # 성분 없음: 전부 ignore.
    logits2 = torch.randn(1, K, H, W, requires_grad=True)
    tgt2 = torch.full((1, H, W), 255, dtype=torch.long)
    loss2, nc2 = fn(logits2, tgt2, depth=depth)
    check("d2: 성분 없음 loss ∈ [0,1]", 0.0 <= float(loss2) <= 1.0, f"loss={float(loss2):.4f}")
    check("d2: n_comp == 0", nc2 == 0, f"n_comp={nc2}")
    loss2.backward()   # grad 연결(0) — 예외 없이 통과해야 한다.
    check("d2: backward 예외 없음(grad 연결된 0)", True)

    # 다중 클래스 + 다중 성분 + ignore 섞임.
    logits3 = torch.randn(1, K, H, W, requires_grad=True)
    tgt3 = torch.full((1, H, W), 255, dtype=torch.long)
    tgt3[0, 2:6, 2:6] = 1
    tgt3[0, 2:6, 20:24] = 1        # 같은 class 1 의 두 번째 성분(분리)
    tgt3[0, 20:28, 10:26] = 3
    loss3, nc3 = fn(logits3, tgt3, depth=depth)
    check("d3: 다중 클래스 loss ∈ [0,1]", 0.0 <= float(loss3) <= 1.0, f"loss={float(loss3):.4f}")
    check("d3: n_comp == 3 (class1 두 성분 + class3 한 성분)", nc3 == 3, f"n_comp={nc3}")
    loss3.backward()
    check("d3: logits grad 흐름", float(logits3.grad.norm()) > 0.0, f"‖grad‖={float(logits3.grad.norm()):.4e}")

    # dist_weight off 경로(depth=None) 도 정상.
    fn_nod = ComponentLoss(num_classes=K, ignore_index=255)
    logits4 = torch.randn(1, K, H, W, requires_grad=True)
    loss4, nc4 = fn_nod(logits4, tgt3, depth=None)
    check("d4: depth=None(거리 가중 off) loss ∈ [0,1]", 0.0 <= float(loss4) <= 1.0,
          f"loss={float(loss4):.4f} n_comp={nc4}")


def main():
    # (d) ComponentLoss 는 timm 무관 — 먼저 돈다.
    _component_checks()

    try:
        import timm  # noqa: F401
    except Exception as e:
        print(f"\n[SKIP] timm 부재 — 모델(a/b/c/e) 스모크 건너뜀: {e}")
        return 1 if _FAILS else 0

    x = _inputs(S=64)

    # ── (a) BOUNDARY_REFINE off ⇒ baseline(키 없음) byte-동일 ────────────────
    print("\n[a] BOUNDARY_REFINE off ⇒ baseline byte-동일 (state_dict 키 + forward 출력)")
    cfg_base = _base_cfg()                                  # BOUNDARY_REFINE 키 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['BOUNDARY_REFINE'] = {'ENABLE': False, 'LEVELS': [4],
                                           'K_INIT': 10.0, 'GAMMA_INIT': 0.1}
    m_base = _seeded(lambda: _build(cfg_base)).eval()
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    k_base = list(m_base.state_dict().keys())
    k_off = list(m_off.state_dict().keys())
    check("off: state_dict 키가 baseline 과 동일", k_base == k_off,
          f"base={len(k_base)} off={len(k_off)} diff={set(k_base) ^ set(k_off)}")
    check("off: self.boundary_refine 미생성", m_off.boundary_refine is None)
    with torch.no_grad():
        o_base, _ = m_base(x, True)
        o_off, _ = m_off(x, True)
    check("off: forward 출력 byte-동일", torch.equal(o_base, o_off),
          f"max|Δ|={(o_base - o_off).abs().max().item() if o_base.shape == o_off.shape else 'shape≠'}")

    # ── (b) BOUNDARY_REFINE on ⇒ 파라미터 증가 + γ/k/τ grad + γ init 0.1 ──────
    print("\n[b] BOUNDARY_REFINE on ⇒ 파라미터 증가 + γ/k/τ grad>0 + γ init 0.1")
    cfg_on = _base_cfg()
    cfg_on['MODEL']['BOUNDARY_REFINE'] = {'ENABLE': True, 'LEVELS': [4],
                                          'K_INIT': 10.0, 'GAMMA_INIT': 0.1}
    m_on = _seeded(lambda: _build(cfg_on)).train()
    check("on: self.boundary_refine 생성", m_on.boundary_refine is not None)
    n_base = sum(p.numel() for p in m_base.parameters())
    n_on = sum(p.numel() for p in m_on.parameters())
    n_br = sum(p.numel() for p in m_on.boundary_refine.parameters())
    print(f"    [INFO] BoundaryRefine params (LEVELS=[4], fpn_dim=64) = {n_br:,} "
          f"(모델 총 {n_base:,} → {n_on:,}, Δ={n_on - n_base:,})")
    check("on: 파라미터 증가량 == boundary_refine 파라미터 수", n_on - n_base == n_br,
          f"Δ={n_on - n_base:,} vs br={n_br:,}")
    check("on: γ init == 0.1", torch.allclose(m_on.boundary_refine.gamma.detach(),
          torch.full_like(m_on.boundary_refine.gamma, 0.1)),
          f"gamma={m_on.boundary_refine.gamma.detach().tolist()}")
    check("on: k init == 10.0", abs(float(m_on.boundary_refine.k) - 10.0) < 1e-6)
    check("on: tau init == 0.0(=평균)", abs(float(m_on.boundary_refine.tau)) < 1e-6)
    gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))
    out = m_on(x, True, gt)
    logits = out[0]
    check("on: 출력 shape 가 off 와 동일", logits.shape == o_off.shape,
          f"{tuple(logits.shape)} vs {tuple(o_off.shape)}")
    check("on: 출력 유한", torch.isfinite(logits).all().item())
    logits.float().sum().backward()
    gg = m_on.boundary_refine.gamma.grad
    gk = m_on.boundary_refine.k.grad
    gtau = m_on.boundary_refine.tau.grad
    check("on: γ grad 도달(≠None, 노름>0)",
          gg is not None and float(gg.norm()) > 0.0,
          f"‖γ.grad‖={float(gg.norm()) if gg is not None else 'None'}")
    check("on: k grad 도달", gk is not None and abs(float(gk)) > 0.0,
          f"k.grad={float(gk) if gk is not None else 'None'}")
    check("on: τ grad 도달", gtau is not None and abs(float(gtau)) > 0.0,
          f"tau.grad={float(gtau) if gtau is not None else 'None'}")
    gconv = sum(float(p.grad.norm()) for p in m_on.boundary_refine.refine.parameters()
                if p.grad is not None)
    check("on: 정제 conv grad 도달", gconv > 0.0, f"Σ‖grad‖={gconv:.4e}")

    # ── (c) 에지 맵 형상 + e 범위 ────────────────────────────────────────────
    print("\n[c] 에지 맵 형상 (B,1,H,W) + e 범위 [0,1]")
    depth = x[1]                                       # MODALS[1] == 'depth'
    br = m_on.boundary_refine
    with torch.no_grad():
        g_full = br._edge_magnitude(depth)
        check("c: 에지 맵 shape (B,1,H,W)", tuple(g_full.shape) == (1, 1, 64, 64),
              str(tuple(g_full.shape)))
        check("c: 에지 맵 유한 + 비음수", torch.isfinite(g_full).all().item()
              and float(g_full.min()) >= 0.0)
        # e = sigmoid(k·(g_std − τ)) 범위 확인(표준화 후).
        gm = torch.nn.functional.adaptive_avg_pool2d(g_full, (16, 16))
        gm = (gm - gm.mean()) / (gm.std() + 1e-6)
        e = torch.sigmoid(br.k * (gm - br.tau))
        check("c: e 범위 [0,1]", float(e.min()) >= 0.0 and float(e.max()) <= 1.0,
              f"[{float(e.min()):.4f}, {float(e.max()):.4f}]")

    # ── (e) BOUNDARY_REFINE on + ComponentLoss 동시 ─────────────────────────
    print("\n[e] BOUNDARY_REFINE on 모델 출력에 ComponentLoss 동시 적용")
    from semseg.models.reliadino.component_loss import ComponentLoss, _HAS_SCIPY
    if not _HAS_SCIPY:
        check("e: scipy 존재", False, "scipy.ndimage 부재")
    else:
        m_e = _seeded(lambda: _build(cfg_on)).train()
        out_e = m_e(x, True, gt)[0]
        comp_fn = ComponentLoss(num_classes=NUM_CLASSES, ignore_index=255)
        c_loss, c_nc = comp_fn(out_e, gt.unsqueeze(0) if gt.dim() == 2 else gt, depth=x[1])
        check("e: 동시 forward + component loss ∈ [0,1]",
              0.0 <= float(c_loss) <= 1.0, f"loss={float(c_loss):.4f} n_comp={c_nc}")
        (out_e.float().sum() + c_loss).backward()
        gbr = sum(float(p.grad.norm()) for p in m_e.boundary_refine.parameters()
                  if p.grad is not None)
        check("e: 두 토글 동시 grad 흐름(boundary_refine)", gbr > 0.0, f"Σ‖grad‖={gbr:.4e}")

    print("\n" + ("=" * 60))
    if _FAILS:
        print(f"[BREFINE/COMP SMOKE] FAIL {len(_FAILS)}건: {_FAILS}")
        return 1
    print("[BREFINE/COMP SMOKE] 전 항목 PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
