"""[P47-2 Uni-modal Balance] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_p47.py                 # A~F
    python tools/smoke_p47.py --ddp           # 위 + gloo 2-proc DDP

검사 항목
  A. 빌드/디스패치 : off / on(all modals) / on(img only) 3케이스 1-step fwd+bwd
  B. 키1 gradient  : uni-aux 손실을 **단독 backward** 했을 때 per-modal LoRA
                     (b_q)에 gradient가 실제로 도달하는지 norm으로 확인.
                     ⚠️ **모달 슬라이스별로** 본다 — 'img only'에서 img 슬라이스만
                     0이 아니어야 "모달별 독립"이 실제로 성립한 것이다.
  C. 추론 등가성   : eval() 출력이 off ↔ on 사이에 |Δ|max == 0 (P39.1 경로 불변)
  D. head 독립성   : 모달별 head 파라미터가 공유되지 않는지 (id 중복 없음)
  E. 부수효과 없음 : P47_2 on/off 가 학습 forward의 **다른** 손실 항을 바꾸지 않는지
                     + forward 호출 횟수가 늘지 않는지(추가 forward 없음 = ISSUE-028 무관)
  F. OGM-GE        : 앞선 모달의 LoRA 슬라이스 gradient만 감쇠(k<1)되고 나머지는 1.0
  G. DDP           : find_unused_parameters=True 아래에서 2-rank gradient 동기 (--ddp)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402
from semseg.models.reliadino import p47 as P47                       # noqa: E402

K = 19                      # MUSES 클래스 수
MODALS = ['img', 'lidar', 'event', 'radar']      # 4모달 (P47-MUB 전장)
SIZE = 128                  # tiny ViT-16 -> 8x8 토큰
BS = 2


def base_cfg(p47=False, modals='all', head='linear', warmup=0, lam=0.4):
    """P39.1-rank 4모달 seed2 config의 토글 구성을 tiny 스케일로 옮긴 것."""
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
    if p47:
        m['P47_2'] = {'ENABLE': True, 'LAMBDA_U': lam, 'MODALS': modals,
                      'HEAD': head, 'HIDDEN': 32, 'WARMUP_EP': warmup,
                      'GT_DIV': 4, 'REDUCE': 'mean'}
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


# LoRA는 b(up-proj)가 zero-init이라 step 0의 a_q grad는 정의상 0이다
# (dq=(x·aqᵀ)·bqᵀ, ∂/∂aq ∝ bq = 0) → 도달 판정은 b_q로 한다 (smoke_p46과 동일 규약).
def lora_slice_grad(model, num_modalities=len(MODALS)):
    """모달 슬라이스별 LoRA b_q/b_v gradient L2 — (M,) 리스트."""
    tot = [0.0] * num_modalities
    for n, p in model.named_parameters():
        if not (p.requires_grad and n.endswith(('.b_q', '.b_v'))):
            continue
        if p.grad is None or p.shape[0] != num_modalities:
            continue
        for m in range(num_modalities):
            tot[m] += float(p.grad[m].detach().pow(2).sum())
    return [v ** 0.5 for v in tot]


def group_grad(model, pred):
    tot = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and pred(n) and p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


# ── A/B. 3케이스 1-step + 키1 gradient ──────────────────────────────────────
def check_cases(device):
    rows, ok = [], True
    cases = [('off', {}, None),
             ('on all', dict(p47=True, modals='all'), list(range(len(MODALS)))),
             ('on img', dict(p47=True, modals=['img']), [0])]
    for name, kw, active in cases:
        model = build(device, **kw)
        model.train()
        model._current_epoch = 10
        x, y = make_batch(device)
        logits, m_feat, aux = model(x, True, gt_mask=y)
        has = 'p47_2_uni' in aux
        want = active is not None
        if has != want:
            rows.append((name, 'dispatch', 'MISSING' if want else 'UNEXPECTED',
                         '', '', 'FAIL'))
            ok = False
            del model
            continue
        # (1) 전체 스텝: 주 손실 + 모든 aux 를 한 번에
        total = F.cross_entropy(logits, y, ignore_index=255)
        for k in ('aux_ce', 'm2f_loss', 'router_reg', 'router_ce', 'vicreg',
                  'p47_2_uni'):
            if k in aux:
                total = total + aux[k]
        model.zero_grad(set_to_none=True)
        total.backward(retain_graph=want)
        n_grad = sum(1 for p in model.parameters()
                     if p.requires_grad and p.grad is not None
                     and float(p.grad.abs().sum()) > 0)
        rows.append((name, '(full step)', f"{float(total):.4f}", '',
                     f"{n_grad} params w/ grad", 'OK' if n_grad > 0 else 'FAIL'))
        ok &= n_grad > 0
        if want:
            # (2) 🔴 키1: uni-aux **단독** backward → per-modal LoRA 슬라이스에 도달?
            model.zero_grad(set_to_none=True)
            aux['p47_2_uni'].backward()
            g = lora_slice_grad(model)
            g_head = group_grad(model, lambda n: n.startswith('p47_2.'))
            # 활성 모달만 >0, 비활성은 정확히 0 이어야 한다(모달별 독립의 실증).
            good = all((g[m] > 0) == (m in active) for m in range(len(MODALS)))
            good &= g_head > 0
            ok &= good
            rows.append((name, 'uni_aux solo', f"{float(aux['p47_2_uni']):.4f}",
                         " ".join(f"{MODALS[m]}:{g[m]:.2e}" for m in range(len(MODALS))),
                         f"head |g|={g_head:.2e}", 'OK' if good else 'FAIL'))
        model.zero_grad(set_to_none=True)
        del model
    return rows, ok


# ── C. 추론 등가성 (P39.1 경로 불변) ────────────────────────────────────────
def check_eval_identity(device):
    out = {}
    a = build(device, seed=1234)                                  # P47-2 off
    a.eval()
    x, _ = make_batch(device, seed=7)
    with torch.no_grad():
        la, _ = a(x, True)
    for tag, kw in (('on all', dict(p47=True, modals='all')),
                    ('on img', dict(p47=True, modals=['img']))):
        b = build(device, seed=1234, **kw)
        # off 모델의 가중치를 그대로 얹는다(P47-2 head는 b에만 있으므로 strict=False).
        missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
        b.eval()
        with torch.no_grad():
            lb, _ = b(x, True)
        # 남는 키가 P47-2 head 뿐인지도 함께 본다(다른 파라미터가 흔들리면 seed 재현이 깨진다).
        only_p47 = all(k.startswith('p47_2.') for k in missing) and not unexpected
        out[tag] = (torch.equal(la, lb), float((la - lb).abs().max()),
                    only_p47, len(missing))
        del b
    return out


# ── D. head 독립성 ──────────────────────────────────────────────────────────
def check_head_independence(device):
    m = build(device, p47=True, modals='all')
    ids, per = set(), {}
    for n, p in m.named_parameters():
        if n.startswith('p47_2.heads.'):
            mi = n.split('.')[2]
            per.setdefault(mi, []).append(n)
            if id(p) in ids:
                return False, per, 'shared tensor'
            ids.add(id(p))
    ok = len(per) == len(MODALS) and all(len(v) > 0 for v in per.values())
    return ok, {k: len(v) for k, v in per.items()}, ''


# ── E. 부수효과 없음 (다른 손실 불변 + 추가 forward 없음) ───────────────────
def check_no_side_effects(device):
    a = build(device, seed=555)
    b = build(device, seed=555, p47=True, modals='all')
    b.load_state_dict(a.state_dict(), strict=False)
    for m in (a, b):
        m.train()
        m._current_epoch = 10
    x, y = make_batch(device, seed=3)
    # encoder 호출 횟수 계수 — P47-2가 forward를 추가하지 않는다는 실증
    counts = {}
    for tag, m in (('off', a), ('on', b)):
        c = {'n': 0}
        h = m.encoder.register_forward_hook(lambda *_a, _c=c: _c.__setitem__('n', _c['n'] + 1))
        torch.manual_seed(99)
        _, _, aux = m(x, True, gt_mask=y)
        h.remove()
        counts[tag] = c['n']
        m._smoke_aux = {k: (float(v) if torch.is_tensor(v) and v.dim() == 0 else None)
                        for k, v in aux.items()}
    shared = [k for k in a._smoke_aux
              if k in b._smoke_aux and a._smoke_aux[k] is not None]
    diffs = {k: (a._smoke_aux[k], b._smoke_aux[k]) for k in shared
             if abs(a._smoke_aux[k] - b._smoke_aux[k]) > 1e-6}
    return (not diffs and counts['off'] == counts['on'],
            counts, sorted(shared), diffs)


# ── F. OGM-GE ───────────────────────────────────────────────────────────────
def check_ogm(device):
    m = build(device, p47=True, modals='all')
    m.train()
    m._current_epoch = 10
    ogm = P47.OGMGE(m, num_modalities=len(MODALS), alpha=0.5, ema=0.0, min_k=0.1)
    x, y = make_batch(device)
    logits, _, aux = m(x, True, gt_mask=y)
    (F.cross_entropy(logits, y, ignore_index=255) + aux['p47_2_uni']).backward()
    # img가 홀로 앞서 가는 상황을 주입 (나머지는 동률)
    ogm.observe([0.90, 0.30, 0.30, 0.30])
    before = lora_slice_grad(m)
    k = ogm.apply_()
    after = lora_slice_grad(m)
    ratio = [(after[i] / before[i]) if before[i] > 0 else float('nan')
             for i in range(len(MODALS))]
    ok = (k[0] < 0.999 and all(abs(k[i] - 1.0) < 1e-9 for i in (1, 2, 3))
          and abs(ratio[0] - k[0]) < 1e-4
          and all(abs(ratio[i] - 1.0) < 1e-4 for i in (1, 2, 3)))
    return ok, k, ratio


# ── G. DDP ──────────────────────────────────────────────────────────────────
def _ddp_worker(rank, ws, ret):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29597')
    dist.init_process_group('gloo', rank=rank, world_size=ws)
    dev = torch.device('cpu')
    core = build(dev, seed=99, p47=True, modals='all', warmup=5)
    core.train()
    err = ''
    try:
        ddp = DDP(core, find_unused_parameters=True)     # P39.1 기본(보조 branch 없음)
        ogm = P47.OGMGE(core, num_modalities=len(MODALS), alpha=0.5, ema=0.0)
        for step in range(2):
            # step 0 = warmup 미만(head 미사용 = unused param), step 1 = 사용
            core._current_epoch = 0 if step == 0 else 10
            x, y = make_batch(dev, seed=rank * 10 + step)
            logits, _, aux = ddp(x, True, gt_mask=y)
            total = F.cross_entropy(logits, y, ignore_index=255) \
                + aux.get('m2f_loss', 0) + aux.get('p47_2_uni', 0)
            core.zero_grad(set_to_none=True)
            total.backward()
            # rank마다 **다른** 점수를 넣는다 → all_reduce가 없으면 k가 갈린다.
            ogm.observe([0.9 if rank == 0 else 0.1, 0.3, 0.3, 0.3])
            k = ogm.apply_()
        kt = torch.tensor(k)
        ref = kt.clone()
        dist.broadcast(ref, 0)
        k_same = bool(torch.allclose(kt, ref, atol=1e-9))
        g = torch.cat([p.grad.flatten() for _, p in sorted(core.named_parameters())
                       if p.requires_grad and p.grad is not None])
        gref = g.clone()
        dist.broadcast(gref, 0)
        # gradient는 OGM 변조 후이므로 k가 같아야 grad도 같다 = 두 검사가 결합돼 있다
        same = bool(torch.allclose(g, gref, atol=1e-6))
    except Exception as e:                                    # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        same = k_same = False
    ret[rank] = (same, k_same, err)
    dist.destroy_process_group()


def check_ddp():
    import torch.multiprocessing as mp
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_ddp_worker, args=(2, ret), nprocs=2, join=True)
    return dict(ret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ddp', action='store_true', help='gloo 2-proc DDP 검증 추가')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)

    print("=" * 104)
    print("A/B. 3케이스 1-step fwd+bwd + 키1 gradient (uni_aux 단독 backward → 모달별 LoRA 슬라이스)")
    print("=" * 104)
    rows, ok = check_cases(dev)
    print(f"{'case':<8} {'term':<13} {'value':>10}  {'|g| LoRA b_q/b_v per modal':<58} "
          f"{'':<18} {'':>6}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<13} {r[2]:>10}  {r[3]:<58} {r[4]:<18} {r[5]:>6}")

    print()
    print("=" * 104)
    print("C. 추론 등가성 (동일 가중치, model.eval() — P39.1 경로 불변)")
    for tag, (same, dmax, only, nmiss) in check_eval_identity(dev).items():
        print(f"    {tag:<7} eval logits identical = {same}   max|diff| = {dmax:.3e}   "
              f"신규키 {nmiss}개 전부 p47_2.* = {only}   {'OK' if (same and only) else 'FAIL'}")
        ok &= same and only

    print()
    print("=" * 104)
    print("D. head 모달별 독립 (파라미터 공유 금지)")
    hok, per, msg = check_head_independence(dev)
    print(f"    modal->param count {per} {msg}   {'OK' if hok else 'FAIL'}")
    ok &= hok

    print()
    print("=" * 104)
    print("E. 부수효과 없음 (다른 aux 손실 불변 + encoder forward 횟수 불변)")
    eok, counts, shared, diffs = check_no_side_effects(dev)
    print(f"    encoder.forward 호출 off={counts['off']} on={counts['on']} "
          f"(4모달 × 1 forward = 4 기대; 추가 forward 없음)")
    print(f"    공통 aux 항 {shared}")
    print(f"    차이 = {diffs if diffs else '없음'}   {'OK' if eok else 'FAIL'}")
    ok &= eok

    print()
    print("=" * 104)
    print("F. OGM-GE (img만 앞서는 점수 주입 → img LoRA 슬라이스만 감쇠)")
    gok, k, ratio = check_ogm(dev)
    print(f"    k = " + " ".join(f"{n}:{v:.4f}" for n, v in zip(MODALS, k)))
    print(f"    실제 grad 비율 = " + " ".join(f"{n}:{v:.4f}" for n, v in zip(MODALS, ratio))
          + f"   {'OK' if gok else 'FAIL'}")
    ok &= gok

    if a.ddp:
        print()
        print("=" * 104)
        print("G. DDP(gloo 2-proc) — find_unused_parameters + OGM k의 rank 간 일치")
        for rk, (same, ksame, err) in sorted(check_ddp().items()):
            print(f"    rank{rk}: grads_synced={same} ogm_k_synced={ksame} {err}")
            ok &= same and ksame

    print()
    print("=" * 104)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
