"""[P51-CMLC] 합성 스모크 — Cross-modal LoRA Coupling (인코딩-시간 결합).

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_p51.py                 # 1~5
    python tools/smoke_p51.py --device cuda

검사 항목
  1. 가드          : CMLC.ENABLE=False 로 빌드한 모델이 CMLC 키 자체가 없는
                     기존 계보 빌드와 출력 수치·state_dict 키가 동일한가
                     (같은 seed — ENABLE 토글이 off 일 때 forward 가 기존과
                     안 갈리는지, RNG 스트림 오염 없음).
  2. 등가          : ENABLE=True 지만 γ=0 강제 → 결합 항등. (a) 모듈 단위
                     cmlc(t) == t (b) 모델 forward 가 같은 가중치의 순차
                     경로(cmlc_enable 토글 off)와 allclose.
  3. grad 흐름     : ENABLE=True, train fwd+bwd → γ · C · down/up projection
                     전부 grad not-None & 유한 & 노름>0 (init γ=1 — 결합이
                     첫 스텝부터 살아 있어야 한다).
  4. 배치-모달 등가: forward_coupled(γ=0) 의 per-modal 출력 == 기존 순차
                     forward 의 per-modal 출력 (allclose rtol 1e-4) —
                     (M*B) 배치-쌓기 + modality_ids gather 가 순차 인코딩과
                     수치 일치하는지가 forward_coupled 의 핵심 계약.
  5. 파라미터 증분 : CMLC on vs base trainable 파라미터 수 차이 == cmlc
     모듈 실측 == 공식 P·M²·r² + P·2·D·r + M².
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402

K = 19                                   # MUSES 클래스 수 (smoke_p47 base_cfg 계승)
MODALS = ['img', 'lidar', 'event', 'radar']
SIZE = 128                               # tiny ViT-16 -> 8x8 토큰
BS = 2
D_EMB = 192                              # vit_tiny_patch16_224 embed_dim
N_TOK = (SIZE // 16) ** 2 + 1            # 64 patch 토큰 + 1 cls
COUPLE_LAYERS = [3, 6, 9]                # 본학습 예 [6,12,18] 을 12블록 tiny 에 맞게
RANK = 8


def base_cfg(cmlc=None):
    """smoke_p47.base_cfg (P39.1-rank 4모달) 를 그대로 계승 + CMLC 키만 얹는다."""
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE_TIMM': 'vit_tiny_patch16_224',
        'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
        'PRETRAINED_BACKBONE': False,
        'LORA_R': 4, 'FPN_DIM': 64,
        'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 3, 'AUX_HIDDEN': 32,
                   'AUX_CE_WEIGHT': 0.5, 'ATTN_BIAS': {'ENABLE': False}},
        'CONSISTENCY': {'ENABLE': False},
        'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
        'CALIBRATION': {'ENABLE': False},
        'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
        'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
        'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 2, 'DIM': 64,
                'NUM_HEADS': 4, 'POINTS': 256, 'SRC': 'modal',
                'ANCHORED': True, 'POINT_QUOTA': 8, 'LOSS_W': 0.5},
        'P39': {'TRUNK_EXP': True, 'ARBITER': True, 'TRUNK_MODE': 'gated_mlp',
                'TRUNK_HIDDEN': 32,
                'VICREG': {'ENABLE': True, 'TOKENS': 128}},
        'MODAL_DROPOUT': {'ENABLE': False},
    }
    if cmlc is not None:
        m['CMLC'] = cmlc
    return {'MODEL': m, 'DATASET': {'MODALS': MODALS},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


def cmlc_cfg(enable=True):
    return {'ENABLE': enable, 'COUPLE_LAYERS': list(COUPLE_LAYERS), 'RANK': RANK}


def make_batch(device, seed=0, n=len(MODALS)):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g).to(device) for _ in range(n)]
    y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g).to(device)
    y[:, :8, :8] = 255                                # ignore 영역도 섞는다
    return x, y


def build(device, seed=0, enable=True):
    torch.manual_seed(seed)
    return build_reliadino(base_cfg(cmlc_cfg(enable)), K).to(device)


def trainable_n(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_loss(model, x, y):
    """train_reliadino.py 의 total 조립을 축약한 것 (전부 pre-scaled)."""
    logits, m_feat, aux = model(x, True, gt_mask=y)
    z = logits.new_zeros(())
    return (F.cross_entropy(logits, y, ignore_index=255)
            + aux.get('m2f_loss', z) + aux.get('vicreg', z)), logits, aux


# ── 1. 가드: ENABLE=False == CMLC 키 없는 기존 계보 ──────────────────────────
def check_guard(device):
    rows, ok = [], True
    torch.manual_seed(0)
    legacy = build_reliadino(base_cfg(), K).to(device)          # CMLC 키 자체 없음
    torch.manual_seed(0)
    off = build(device, seed=0, enable=False)                    # ENABLE: False
    same_keys = (set(legacy.state_dict().keys()) == set(off.state_dict().keys()))
    ok &= same_keys
    rows.append(('1', 'state_dict 키 집합 동일(키 없음 vs ENABLE=False)',
                 str(same_keys), 'PASS' if same_keys else 'FAIL'))
    legacy.eval(); off.eval()
    x, _ = make_batch(device, seed=2)
    with torch.no_grad():
        o_legacy, _ = legacy(x, True)
        o_off, _ = off(x, True)
    d = float((o_legacy - o_off).abs().max())
    good = torch.allclose(o_legacy, o_off, rtol=1e-4)
    ok &= good
    rows.append(('1', '|Δ|max (ENABLE=False vs 키 없음)', f"{d:.3e}",
                 'PASS' if good else 'FAIL'))
    rows.append(('1', '순차 경로 사용(off)', f"cmlc={off.cmlc}", '-'))
    return rows, ok


# ── 2. 등가: γ=0 → 결합 항등 ─────────────────────────────────────────────────
def check_gamma_zero(device):
    rows, ok = [], True
    model = build(device, seed=3)
    model.eval()
    with torch.no_grad():
        model.cmlc.gamma.zero_()                    # 결합 게이트 0 강제
        gmax = float(model.cmlc.gamma.abs().max())
        # (a) 모듈 단위: γ=0 이면 모든 결합점에서 정확히 항등
        t = torch.randn(len(MODALS), BS, N_TOK, D_EMB, device=device)
        for p in range(len(COUPLE_LAYERS)):
            ident = torch.allclose(model.cmlc(t, point_idx=p), t, rtol=1e-4)
            ok &= ident
            rows.append(('2', f'모듈 항등 γ=0 (point {p})', str(ident),
                         'PASS' if ident else 'FAIL'))
        # (b) 모델 수준: coupled 경로(γ=0) == 같은 가중치의 기존 순차 경로
        x, _ = make_batch(device, seed=4)
        o_coupled, _ = model(x, True)
        model.cmlc_enable = False                   # eval-time 토글: 순차 루프로
        o_seq, _ = model(x, True)
        model.cmlc_enable = True
    d = float((o_coupled - o_seq).abs().max())
    good = torch.allclose(o_coupled, o_seq, rtol=1e-4)
    ok &= good and gmax == 0.0
    rows.append(('2', 'γ max (=0 강제)', f"{gmax:.3e}",
                 'PASS' if gmax == 0.0 else 'FAIL'))
    rows.append(('2', '|Δ|max (γ=0 coupled vs 순차)', f"{d:.3e}",
                 'PASS' if good else 'FAIL'))
    return rows, ok


# ── 3. grad 흐름: γ, C, down/up 전부 not-None & 유한 ─────────────────────────
def check_grad(device):
    rows, ok = [], True
    model = build(device, seed=6)
    model.train(); model._current_epoch = 10
    x, y = make_batch(device, seed=7)
    loss, logits, _ = total_loss(model, x, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    def gstat(g, name):
        if g is None:
            rows.append(('3', name, 'grad None', 'FAIL'))
            return False
        finite = bool(torch.isfinite(g).all())
        n = float(g.detach().float().norm())
        good = finite and n > 0.0
        rows.append(('3', name, f"norm={n:.3e} finite={finite}",
                     'PASS' if good else 'FAIL'))
        return good

    ok &= gstat(model.cmlc.gamma.grad, 'γ.grad (init 1.0 → 즉시 흐름)')
    # c[p] 는 Parameter 슬라이스(비-리프)라 .grad 가 안 채워진다 — 리프인
    # c.grad 를 슬라이스해 결합점별로 본다.
    g_c = model.cmlc.c.grad
    if g_c is None:
        rows.append(('3', 'C.grad (결합행렬)', 'grad None', 'FAIL'))
        ok = False
    else:
        for p in range(len(COUPLE_LAYERS)):
            ok &= gstat(g_c[p], f'C[{p}].grad (결합행렬)')
    for p in range(len(COUPLE_LAYERS)):
        ok &= gstat(model.cmlc.down[p].weight.grad, f'down[{p}].weight.grad')
        ok &= gstat(model.cmlc.up[p].weight.grad, f'up[{p}].weight.grad')
    return rows, ok


# ── 4. 배치-모달 등가: forward_coupled(γ=0) == 순차 forward ──────────────────
def check_batch_equivalence(device):
    rows, ok = [], True
    model = build(device, seed=5)
    model.eval()
    x, _ = make_batch(device, seed=1)
    with torch.no_grad():
        model.cmlc.gamma.zero_()            # 결합 항등 — 배치-쌓기 자체의 수치만 본다
        stack = torch.stack(x, dim=0)       # (M, B, C, H, W)
        feats_c = model.encoder.forward_coupled(stack, model.cmlc,
                                                model.cmlc_layers)
        feats_s = [model.encoder(x[i], i) for i in range(len(x))]
    worst = 0.0
    for i, (a, b) in enumerate(zip(feats_c, feats_s)):
        d = float((a - b).abs().max())
        worst = max(worst, d)
        good = torch.allclose(a, b, rtol=1e-4)
        ok &= good
        rows.append(('4', f'per-modal[{i}] ({MODALS[i]}) |Δ|max', f"{d:.3e}",
                     'PASS' if good else 'FAIL'))
    rows.append(('4', 'shape 일치',
                 str(tuple(feats_c[0].shape)), '-'))
    return rows, ok


# ── 5. 파라미터 증가분 ───────────────────────────────────────────────────────
def check_param_delta(device):
    rows, ok = [], True
    torch.manual_seed(0)
    base = build_reliadino(base_cfg(), K).to(device)
    torch.manual_seed(0)
    on = build(device, seed=0, enable=True)
    n_base, n_on = trainable_n(base), trainable_n(on)
    n_cmlc = sum(p.numel() for p in on.cmlc.parameters())
    M, P, r = len(MODALS), len(COUPLE_LAYERS), RANK
    n_formula = P * M * M * r * r + P * 2 * D_EMB * r + M * M   # C + down/up + γ
    good = (n_on - n_base == n_cmlc == n_formula)
    ok &= good
    rows.append(('5', 'base trainable', f"{n_base/1e6:.4f}M", '-'))
    rows.append(('5', 'CMLC on trainable', f"{n_on/1e6:.4f}M", '-'))
    rows.append(('5', '증가분 == cmlc 실측 == 공식',
                 f"{n_on-n_base} == {n_cmlc} == {n_formula}",
                 'PASS' if good else 'FAIL'))
    rows.append(('5', '본학습 규모 추정(D=1024,M=4,r=16,P=3)',
                 f"{(3*16*16*1024*16 + 3*2*1024*16 + 16*16)/1e6:.3f}M", '-'))
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"[P51-smoke] device={device} torch={torch.__version__} "
          f"couple_layers={COUPLE_LAYERS} rank={RANK}")

    all_rows, all_ok = [], True
    for name, fn in (('1 가드 (ENABLE=False 불변)', check_guard),
                     ('2 등가 (γ=0 → 항등)', check_gamma_zero),
                     ('3 grad 흐름', check_grad),
                     ('4 배치-모달 등가', check_batch_equivalence),
                     ('5 파라미터 증분', check_param_delta)):
        rows, ok = fn(device)
        all_rows.extend(rows)
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n" + "=" * 92)
    print(f"{'검사':<6} {'항목':<44} {'값':<26} {'판정'}")
    print("-" * 92)
    for grp, item, val, verdict in all_rows:
        print(f"{grp:<6} {item:<44} {str(val):<26} {verdict}")
    print("=" * 92)
    print(f"RESULT: {'ALL PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
