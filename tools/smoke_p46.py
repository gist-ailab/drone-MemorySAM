"""[P46-CTR] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계 의무 검사.

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_p46.py                 # 전 토글 조합 1-step fwd+bwd + grad assert
    python tools/smoke_p46.py --ddp           # 위 + gloo 2-proc DDP 다중-forward 검증

검사 항목
  A. 빌드/디스패치  : C1/C2/C3 각 on/off 조합에서 build + 1 step forward+backward
  B. 키1 gradient   : 각 aux 손실을 **단독 backward** 했을 때 feature 경로
                      (LoRA a_q / fusion / FPN / head) 파라미터에 grad>0 이 실제로 도달
  C. 추론 등가성    : P46 전부 on ↔ 전부 off 모델의 eval() 출력이 **완전 동일**
                      (P46는 학습 전용 — 추론 경로 불변)
  D. C-1 샘플러     : rcs_base_prob이 희소 클래스를 실제로 up-weight 하는지,
                      ClassLossEMA blend가 고-loss 클래스를 추가로 올리는지
  E. DDP            : 보조 branch(2번째 forward)가 있어도 reducer가 살아남고
                      gradient가 정확히 1회 all-reduce 되는지 (--ddp)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402
from semseg.models.reliadino import p46 as P46                       # noqa: E402

K = 25                      # DELIVER 클래스 수
MODALS = ['img', 'depth', 'event', 'lidar']
SIZE = 128                  # tiny ViT-16 -> 8x8 토큰
BS = 2

# feature 경로 대표 파라미터 — 이 이름들에 grad가 닿아야 "주손실 직결"(키1)이다.
# ⚠️ LoRA는 b(up-proj)가 zero-init이라 **step 0에서 a_q의 grad는 정의상 0**이다
#    (dq = (x·aqᵀ)·bqᵀ, ∂/∂aq ∝ bq = 0). baseline seg 손실도 마찬가지 → a_q는
#    합격 기준이 될 수 없다. LoRA 도달 여부는 b_q로 판정한다.
PROBE = {
    'lora_b_q': lambda n: n.endswith('.b_q'),
    'fusion':   lambda n: n.startswith('fusion.layers'),
    'fpn':      lambda n: n.startswith('fpn.'),
    'head':     lambda n: n.startswith('head.fuse'),
}


def base_cfg(c1=False, c2=False, c3=False, cross_view=True):
    return {
        'MODEL': {
            'NAME': 'ReliaDINO',
            'BACKBONE_TIMM': 'vit_tiny_patch16_224',
            'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
            'PRETRAINED_BACKBONE': False,
            'LORA_R': 4, 'FPN_DIM': 64,
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 3, 'AUX_HIDDEN': 32,
                       'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False},
            'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': False},
            'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
            # ANCHORED=True는 num_queries > num_classes를 요구한다 (m2f_head.py)
            'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 2, 'DIM': 64,
                    'NUM_HEADS': 4, 'POINTS': 256, 'SRC': 'modal',
                    'ANCHORED': True, 'POINT_QUOTA': 8, 'LOSS_W': 0.5},
            'P39': {'TRUNK_EXP': True, 'ARBITER': True, 'TRUNK_MODE': 'gated_mlp',
                    'TRUNK_HIDDEN': 32,
                    'VICREG': {'ENABLE': True, 'TOKENS': 128}},
            'P46': {
                'C1_RCS': {'ENABLE': c1},
                'C2_MCC': {'ENABLE': c2, 'MASK_RATIO': 0.5, 'PATCH': 32,
                           'CONF_THRESH': 0.0, 'WARMUP_EP': 0},
                'C3_PROTO': {'ENABLE': c3, 'LAMBDA': 0.1, 'PIXELS': 512,
                             'WARMUP_EP': 0, 'CROSS_VIEW': cross_view},
            },
            'MODAL_DROPOUT': {'ENABLE': False},
        },
        'DATASET': {'MODALS': MODALS},
        'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]},
    }


def make_batch(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g).to(device) for _ in MODALS]
    y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g).to(device)
    y[:, :8, :8] = 255                          # ignore 영역도 섞는다
    return x, y


def build(device, seed=0, **kw):
    torch.manual_seed(seed)
    m = build_reliadino(base_cfg(**kw), K).to(device)
    return m


def grads_reached(model, loss, tag):
    """`loss`만 단독 backward → PROBE 그룹별 grad L2 (키1 검사)."""
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    out = {}
    for name, pred in PROBE.items():
        tot = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and pred(n) and p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        out[name] = tot ** 0.5
    model.zero_grad(set_to_none=True)
    return out


# ── A/B. 토글별 1-step + gradient ───────────────────────────────────────────
def check_toggles(device):
    rows = []
    ok_all = True
    for name, kw in [('all-off', {}),
                     ('C1 only', dict(c1=True)),
                     ('C2 only', dict(c2=True)),
                     ('C3 only', dict(c3=True, cross_view=False)),
                     ('C3+xview', dict(c3=True, cross_view=True)),
                     ('C1+C2+C3', dict(c1=True, c2=True, c3=True))]:
        model = build(device, **kw)
        model.train()
        model._current_epoch = 10
        x, y = make_batch(device)
        logits, m_feat, aux = model(x, True, gt_mask=y)
        base = F.cross_entropy(logits, y, ignore_index=255)
        terms = {'seg': base}
        if 'p46_proto' in aux:
            terms['C3 proto'] = aux['p46_proto']
        # train_reliadino.py와 **동일한** 보조 branch 결선을 재현한다
        # (스타일 2-view → 패치 마스킹, 하나의 forward 공유).
        do_mcc = bool(kw.get('c2'))
        do_xview = bool(kw.get('c3') and kw.get('cross_view', True))
        if do_mcc or do_xview:
            xb = list(x)
            if do_xview:
                xb[0] = P46.style_jitter_normalized(xb[0])
                assert not torch.equal(xb[0], x[0]), "style_jitter가 아무것도 안 바꿨다"
            mask = None
            if do_mcc:
                mask = P46.patch_mask(BS, SIZE, SIZE, 0.5, 32, device)
                xb = P46.apply_patch_mask(xb, mask)
            model._p46_replay_path = True
            bl, _, baux = model(xb, True, gt_mask=y)
            model._p46_replay_path = False
            if do_mcc:
                teacher = P46.EMATeacher(model, alpha=0.999)
                with torch.no_grad():
                    tl = teacher(x)
                cons, _ = P46.masked_consistency_loss(bl, tl, mask, conf_thresh=0.0)
                terms['C2 mcc'] = cons
            if do_xview and 'p46_proto' in baux:
                terms['C3 xview'] = baux['p46_proto']
        for tname, t in terms.items():
            if not torch.is_tensor(t) or not t.requires_grad:
                rows.append((name, tname, 'NO GRAPH', '', '', '', 'FAIL'))
                ok_all = False
                continue
            g = grads_reached(model, t, tname)
            good = all(v > 0 for v in g.values())
            ok_all &= good
            rows.append((name, tname, f"{float(t):.4f}",
                         f"{g['lora_b_q']:.3e}", f"{g['fusion']:.3e}",
                         f"{g['head']:.3e}", 'OK' if good else 'FAIL'))
        # 전체 합산 1 step fwd+bwd (실제 학습 스텝과 동일 결선)
        total = sum(t for t in terms.values())
        model.zero_grad(set_to_none=True)
        total.backward()
        n_grad = sum(1 for p in model.parameters()
                     if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0)
        rows.append((name, '(full step)', f"{float(total):.4f}", '', '',
                     f"{n_grad} params w/ grad", 'OK' if n_grad > 0 else 'FAIL'))
        del model
    return rows, ok_all


# ── C. 추론 등가성 ──────────────────────────────────────────────────────────
def check_eval_identity(device):
    a = build(device, seed=1234)                                  # P46 전부 off
    b = build(device, seed=1234, c1=True, c2=True, c3=True)       # P46 전부 on
    b.load_state_dict(a.state_dict(), strict=False)               # 가중치 동일화
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=7)
    with torch.no_grad():
        la, _ = a(x, True)
        lb, _ = b(x, True)
    same = torch.equal(la, lb)
    return same, float((la - lb).abs().max())


# ── D. C-1 샘플러 ───────────────────────────────────────────────────────────
def check_rcs():
    rng = np.random.default_rng(0)
    pix = np.array([1e8] * 5 + [1e6] * 15 + [1e3] * 5, dtype=np.float64)  # 마지막 5개가 희소
    p = P46.rcs_base_prob(pix, temperature=0.01, mode='daformer')
    rare_share = float(p[-5:].sum())
    freq_share = float(p[:5].sum())
    class_files = [rng.choice(400, size=40, replace=False).astype(np.int64) for _ in range(25)]
    ema = P46.ClassLossEMA(25, momentum=0.5)
    s = P46.RareClassSampler(class_files, p, num_samples=4000, loss_ema=ema,
                             blend_w=1.0, refresh=16)
    _ = list(iter(s))
    hist0 = s.last_class_hist / s.last_class_hist.sum()
    # ⚠️ blend는 base 확률이 비교 가능한 클래스들 사이에서 의미가 있다. DAFormer
    #    식은 T=0.01에서 극도로 뾰족해 빈발 클래스의 base 확률이 사실상 0이므로
    #    (그래도 그 클래스들은 rare 이미지 안에 함께 담겨 충분히 보인다),
    #    난이도 blend는 **희소 클래스 집합 내부**에서 검증한다.
    tgt = 24                                   # 희소 5개 중 하나
    ema.val[:] = 1.0
    ema.seen[:] = True
    ema.val[tgt] = 50.0
    s.set_epoch(1)
    _ = list(iter(s))
    hist1 = s.last_class_hist / s.last_class_hist.sum()
    return {
        'rare5_prob': rare_share, 'freq5_prob': freq_share,
        'rare_up': rare_share > freq_share,
        'tgt': tgt, 'before': float(hist0[tgt]), 'after': float(hist1[tgt]),
        'blend_up': hist1[tgt] > hist0[tgt] * 1.5,
    }


# ── F. C-1 라벨 디코딩/캐시 (합성 DELIVER 라벨 PNG) ─────────────────────────
def check_label_scan():
    """`compute_class_stats`가 DELIVER 라벨 규약(원본 1-25 + 255 → 0-24 + 255)을
    deliver.py `__getitem__`과 **동일하게** 디코딩하는지 실제 PNG로 확인한다.
    (conventions §"데이터 로더/디코더": 상수 가정 대신 실측 후 커밋.)"""
    import shutil
    import tempfile

    from torchvision import io as tvio
    root = Path(tempfile.mkdtemp(prefix='p46smoke_'))
    files = []
    # 이미지 i에는 원본 라벨값 (i+1)만 채운다 → 학습 라벨은 i 가 되어야 한다.
    for i in range(4):
        d_img = root / 'img' / 'cloud' / 'train' / 'seq'
        d_lbl = root / 'semantic' / 'cloud' / 'train' / 'seq'
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        lab = torch.full((1, 16, 16), i + 1, dtype=torch.uint8)
        lab[0, 0, :] = 255                             # 원본 255 → 학습 255(ignore)
        tvio.write_png(lab, str(d_lbl / f"{i:04d}_semantic.png"))
        tvio.write_png(torch.zeros(3, 16, 16, dtype=torch.uint8),
                       str(d_img / f"{i:04d}_rgb.png"))
        files.append(str(d_img / f"{i:04d}_rgb.png"))
    stub = type('DELIVER', (), {'__len__': lambda s: len(s.files)})()
    stub.files = files
    cache = root / 'cache'
    pix, cfiles = compute = P46.compute_class_stats(
        stub, K, str(cache), min_pixels=1, num_workers=0, verbose=False)
    ok = True
    for i in range(4):
        ok &= (int(pix[i]) == 16 * 16 - 16)            # 첫 행 16px은 ignore
        ok &= (list(cfiles[i]) == [i])                 # 이미지 i만 클래스 i를 담는다
    ok &= all(len(cfiles[c]) == 0 for c in range(4, K))
    cached = list(cache.glob('rcs_DELIVER_*.npz'))
    ok &= len(cached) == 1
    pix2, _ = P46.compute_class_stats(stub, K, str(cache), num_workers=0, verbose=False)
    ok &= bool((pix == pix2).all())                    # 캐시 재사용 일치
    shutil.rmtree(root, ignore_errors=True)
    return ok, [int(v) for v in pix[:6]]


# ── E. DDP 다중-forward ─────────────────────────────────────────────────────
def _ddp_worker(rank, ws, ret):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29591')
    dist.init_process_group('gloo', rank=rank, world_size=ws)
    dev = torch.device('cpu')
    core = build(dev, seed=99, c2=True, c3=True)
    core.train()
    core._current_epoch = 10
    # train_reliadino.py와 동일: 보조 branch가 있으면 broadcast_buffers=False.
    # (True면 2번째 forward의 버퍼 브로드캐스트가 1번째 그래프의 empty_weight를
    #  in-place로 갈아엎어 backward가 죽는다 — 이 스모크가 잡아낸 실제 버그.)
    ddp = DDP(core, find_unused_parameters=True, broadcast_buffers=False)
    teacher = P46.EMATeacher(core, alpha=0.999)
    err = ''
    try:
        for step in range(2):
            x, y = make_batch(dev, seed=rank * 10 + step)
            logits, _, aux = ddp(x, True, gt_mask=y)
            total = F.cross_entropy(logits, y, ignore_index=255) \
                + aux.get('m2f_loss', 0) + aux.get('p46_proto', 0)
            mask = P46.patch_mask(BS, SIZE, SIZE, 0.5, 32, dev)
            xb = P46.apply_patch_mask(list(x), mask)
            core._p46_replay_path = True
            bl, _, baux = ddp(xb, True, gt_mask=y)
            core._p46_replay_path = False
            with torch.no_grad():
                tl = teacher(x)
            cons, _ = P46.masked_consistency_loss(bl, tl, mask, conf_thresh=0.0)
            total = total + cons + baux.get('p46_proto', 0)
            core.zero_grad(set_to_none=True)
            total.backward()
            teacher.update(step + 1)
        # rank 간 gradient 동일성 = all-reduce가 정확히 1회 돈 증거
        g = torch.cat([p.grad.flatten() for _, p in sorted(core.named_parameters())
                       if p.requires_grad and p.grad is not None])
        ref = g.clone()
        dist.broadcast(ref, 0)
        same = bool(torch.allclose(g, ref, atol=1e-6))
    except Exception as e:                                    # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        same = False
    ret[rank] = (same, err)
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

    print("=" * 96)
    print("A/B. 토글별 1-step fwd+bwd + 키1 gradient (각 손실 단독 backward)")
    print("=" * 96)
    rows, ok = check_toggles(dev)
    print(f"{'config':<10} {'loss term':<12} {'value':>10} {'|g| lora_b_q':>14} "
          f"{'|g| fusion':>12} {'|g| head':>22} {'':>6}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<12} {r[2]:>10} {r[3]:>14} {r[4]:>12} {r[5]:>22} {r[6]:>6}")

    print()
    print("=" * 96)
    print("C. 추론 등가성 (P46 all-on vs all-off, 동일 가중치, model.eval())")
    same, dmax = check_eval_identity(dev)
    print(f"    eval logits identical = {same}   max|diff| = {dmax:.3e}   "
          f"{'OK' if same else 'FAIL'}")

    print()
    print("=" * 96)
    print("D. C-1 RCS 샘플러")
    d = check_rcs()
    print(f"    희소 5클래스 총 확률 {d['rare5_prob']:.4f} vs 빈발 5클래스 "
          f"{d['freq5_prob']:.4f}  -> rare up-weight {'OK' if d['rare_up'] else 'FAIL'}")
    print(f"    class{d['tgt']} 샘플비 {d['before']:.4f} -> (loss EMA 50배 후) "
          f"{d['after']:.4f}  -> blend {'OK' if d['blend_up'] else 'FAIL'}")
    ok &= d['rare_up'] and d['blend_up'] and same

    print()
    print("=" * 96)
    print("F. C-1 라벨 스캔/디코딩/캐시 (합성 DELIVER 라벨 PNG)")
    lok, lpix = check_label_scan()
    print(f"    per-class pixel counts[:6] = {lpix} (기대: 240 4개 + 0 2개)   "
          f"{'OK' if lok else 'FAIL'}")
    ok &= lok

    if a.ddp:
        print()
        print("=" * 96)
        print("E. DDP(gloo 2-proc) — 보조 branch 2번째 forward + 단일 backward")
        r = check_ddp()
        for rk, (s, err) in sorted(r.items()):
            print(f"    rank{rk}: grads_synced={s} {err}")
        ok &= all(s for s, _ in r.values())

    print()
    print("=" * 96)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
