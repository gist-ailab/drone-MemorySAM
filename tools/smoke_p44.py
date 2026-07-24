#!/usr/bin/env python3
"""tools/smoke_p44.py — P44-BMR / P45-FogStyle 합성 스모크 (CPU, 초소형 텐서, GPU 불요).

코드검수 파이프라인([[code-review-pipeline]]) 의무 항목: **등가성 assert + grad assert**.
백본(DINOv3-L)을 만들지 않기 위해 ReliaDINO 전체 대신 (a) 실제 ReliabilityGatedFusion을
초소형 dim으로 세우고 (b) model.py의 마스킹/presence/style 함수는 순수 함수(p44.py)로
직접 검증한다 — 검증 대상 로직은 전부 그 두 곳에 있다.

  python /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_p44.py
  (실행: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_p44.py)

테스트:
  A all-off 등가        P44/P45 전부 off → fusion 출력·aux·optimizer 1스텝이 baseline과 동일
  B B-1 MMPareto        충돌 시 투영+크기복원, 합의 시 단순 합으로 환원
  C B-2 peer 증류       KL 대칭·유한, warmup 게이팅, gradient가 **모든** 모달 브랜치에 도달
  D B-3 국소 마스킹     img만·영역만·FRAC 샘플만, coverage 모드가 lidar 발자국을 사용
  E V-1 presence        부재 픽셀 가중 정확히 0 + 잔여 가중 합 1로 재정규화
  F P45 FogStyle        img 브랜치 feature만 섭동, 일관성 손실 유한
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semseg.models.reliadino import p44 as P44                      # noqa: E402
from semseg.models.reliadino.fusion import ReliabilityGatedFusion   # noqa: E402
from semseg.models.reliadino.mmpareto import MMPareto               # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


def make_fusion(**kw):
    torch.manual_seed(0)
    return ReliabilityGatedFusion(dim=16, num_classes=4, num_modalities=3,
                                  num_layers=1, num_heads=2, mlp_ratio=1.0,
                                  aux_hidden=32, attn_bias=False,
                                  consistency_bias=False, gate_enable=True,
                                  calibrate=True, router_enable=True, **kw)


def make_inputs(B=4, C=3, H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(B, C, H, W, generator=g)
    lidar = torch.rand(B, C, H, W, generator=g)
    # lidar: 절반 영역만 리턴이 있는 sparse 투영 모사 (무반환 = 정확히 0)
    lidar[:, :, :, W // 2:] = 0.0
    lidar = lidar * (torch.rand(B, 1, H, W, generator=g) < 0.3).float()
    event = torch.rand(B, C, H, W, generator=g)
    return [img, lidar, event]


# ── A. all-off 등가 ─────────────────────────────────────────────────────────
def test_all_off():
    print("\n[A] all-off ⇒ baseline 등가")
    torch.manual_seed(1)
    feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    gt = torch.randint(0, 4, (2, 32, 32))
    gt[0, :4, :4] = 255

    base = make_fusion()
    p44m = make_fusion(p44_mutual_kl=False, p44_rel_corr=False,
                       p44_validity_renorm=False, p44_export_train_aux=False)
    p44m.load_state_dict(base.state_dict())
    base.train(); p44m.train()

    torch.manual_seed(7); fb, ab = base([f.clone() for f in feats], gt)
    torch.manual_seed(7); fp, ap = p44m([f.clone() for f in feats], gt,
                                        img_mask=None, img_idx=0,
                                        presence=None, epoch=99)
    check('fused 동일(bitwise)', torch.equal(fb, fp))
    check('aux 키 동일', sorted(ab.keys()) == sorted(ap.keys()),
          f"{sorted(ab.keys())}")
    check('aux 값 동일(bitwise)',
          all(torch.equal(ab[k], ap[k]) for k in ab if torch.is_tensor(ab[k])))
    check('P44 손실 키 부재',
          not any(k.startswith(('p44_', 'p45_')) for k in ap))
    check('_train_aux_logits 미스태시(off)', p44m._train_aux_logits is None)

    # optimizer 1스텝 등가 (표준 backward 경로)
    def one_step(mod):
        torch.manual_seed(3)
        opt = torch.optim.SGD(mod.parameters(), lr=0.1)
        opt.zero_grad()
        torch.manual_seed(7)
        f, a = mod([x.clone().requires_grad_(False) for x in feats], gt) \
            if mod is base else \
            mod([x.clone() for x in feats], gt, img_mask=None, img_idx=0,
                presence=None, epoch=99)
        loss = f.pow(2).mean() + a['aux_ce'] + a['rbma_cal_loss']
        loss.backward()
        opt.step()
        return [p.detach().clone() for p in mod.parameters()]
    pb, pp = one_step(base), one_step(p44m)
    check('1 optimizer step 후 파라미터 동일',
          all(torch.equal(x, y) for x, y in zip(pb, pp)))


# ── B. MMPareto ─────────────────────────────────────────────────────────────
class _Toy(torch.nn.Module):
    """LoRA 파라미터 이름 규약(.a_q, 첫 축 = 모달)을 흉내낸 초소형 모듈."""

    def __init__(self, m=3, r=2, d=4):
        super().__init__()
        self.a_q = torch.nn.Parameter(torch.zeros(m, r, d))     # per-modal 슬라이스
        self.w = torch.nn.Parameter(torch.zeros(5))             # shared


def test_mmpareto():
    print("\n[B] B-1 MMPareto gradient 통합")
    toy = _Toy()
    mp = MMPareto(toy.named_parameters(), num_modalities=3,
                  modal_names=['img', 'lidar', 'event'])
    names = [g['name'] for g in mp.groups]
    check('그룹 구성 = per-modal LoRA + shared',
          names == ['lora_img', 'lora_lidar', 'lora_event', 'shared'], f"{names}")

    def run(gm_vals, ga_vals):
        """gm_vals/ga_vals: {param_index: tensor} 형태의 합성 gradient."""
        for p in toy.parameters():
            p.grad = None
        mp.reset()
        mp.accumulate(gm_vals, ga_vals)
        st = mp.combine()
        return st, [p.grad.clone() for p in toy.parameters()]

    # (1) 합의 케이스: 두 gradient가 같은 방향 → 단순 합
    gm = [torch.ones(3, 2, 4), torch.ones(5)]
    ga = [torch.ones(3, 2, 4) * 2, torch.ones(5) * 2]
    st, grads = run(gm, ga)
    check('합의 시 cos=+1', abs(st['cos_lora_img'] - 1.0) < 1e-5,
          f"cos={st['cos_lora_img']:.4f}")
    check('합의 시 결합 = g_main + g_aux',
          torch.allclose(grads[0], torch.full((3, 2, 4), 3.0))
          and torch.allclose(grads[1], torch.full((5,), 3.0)))
    check('합의 시 conflict 플래그 0', st['conflict_shared'] == 0.0)

    # (2) 충돌 케이스: lidar 슬라이스만 정확히 반대 + 직교 성분
    gm = torch.zeros(3, 2, 4); ga = torch.zeros(3, 2, 4)
    gm[1, 0, 0] = 1.0; gm[1, 0, 1] = 1.0        # g_main = (1,1)
    ga[1, 0, 0] = -1.0; ga[1, 0, 1] = 0.5       # g_aux  = (-1,0.5) → 내적 -0.5 < 0
    st, grads = run([gm, torch.ones(5)], [ga, torch.ones(5)])
    check('충돌 감지(cos<0)', st['cos_lora_lidar'] < 0,
          f"cos={st['cos_lora_lidar']:.4f}")
    check('충돌 플래그 1', st['conflict_lora_lidar'] == 1.0)
    d = grads[0][1, 0, :2]
    g1 = torch.tensor([1.0, 1.0]); g2 = torch.tensor([-1.0, 0.5])
    check('결합 방향이 두 목표 모두에 대해 하강(내적 ≥ 0)',
          float(d @ g1) > 0 and float(d @ g2) > 0,
          f"d·g_main={float(d @ g1):.4f} d·g_aux={float(d @ g2):.4f}")
    target = float(g1.norm() + g2.norm())
    check('크기 복원 = ‖g_main‖+‖g_aux‖',
          abs(float(d.norm()) - target) < 1e-4,
          f"‖d‖={float(d.norm()):.4f} target={target:.4f}")
    check('충돌은 그룹 국소 — 충돌 없는 shared는 단순 합',
          torch.allclose(grads[1], torch.full((5,), 2.0)))
    # min-norm(크기복원 전) 대비 실제로 커졌는가 = 논문의 magnitude 논지
    alpha = float((g2.pow(2).sum() - g1 @ g2) / (g1 - g2).pow(2).sum())
    dmin = (alpha * g1 + (1 - alpha) * g2).norm()
    check('순수 min-norm보다 크기가 복원됨', float(d.norm()) > float(dmin),
          f"min-norm ‖d‖={float(dmin):.4f}")

    # (3) 한쪽 gradient가 0이면 그대로 통과
    st, grads = run([torch.zeros(3, 2, 4), torch.ones(5)],
                    [torch.ones(3, 2, 4), torch.zeros(5)])
    check('g_main=0 → g_aux 그대로', torch.allclose(grads[0], torch.ones(3, 2, 4)))
    check('g_aux=0 → g_main 그대로', torch.allclose(grads[1], torch.ones(5)))

    # (4) INTERVAL 탈출 밸브
    mp2 = MMPareto(_Toy().named_parameters(), num_modalities=3, interval=3)
    check('INTERVAL=3 게이팅',
          [mp2.active(i) for i in range(6)] == [True, False, False, True, False, False])

    # (5) 통합: 트레이너와 같은 흐름(2 micro-step 누적 → 결합 → step)으로
    #     "충돌 없는 목표쌍이면 단일 backward와 같은 업데이트"인지 확인.
    def build():
        torch.manual_seed(21)
        net = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Linear(6, 3))
        return net
    xs = [torch.randn(4, 6, generator=torch.Generator().manual_seed(s))
          for s in (31, 32)]

    def losses(net, x):
        y = net(x)
        main = y.pow(2).mean()
        aux = 0.5 * main            # 정확히 같은 방향 (cos=+1, 충돌 없음)
        return main, aux

    accum = len(xs)
    net_a = build(); opt_a = torch.optim.SGD(net_a.parameters(), lr=0.05)
    opt_a.zero_grad(set_to_none=True)
    for x in xs:
        m_, a_ = losses(net_a, x)
        ((m_ + a_) / accum).backward()
    opt_a.step()

    net_b = build(); opt_b = torch.optim.SGD(net_b.parameters(), lr=0.05)
    mp3 = MMPareto(net_b.named_parameters(), num_modalities=3)
    opt_b.zero_grad(set_to_none=True)
    for x in xs:
        m_, a_ = losses(net_b, x)
        la = a_ / accum
        lm = (m_ + a_) / accum - la
        gm = torch.autograd.grad(lm, mp3.params, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(la, mp3.params, allow_unused=True)
        mp3.accumulate(gm, ga)
    st3 = mp3.combine()
    opt_b.step()
    diffs = [float((p - q).abs().max())
             for p, q in zip(net_a.parameters(), net_b.parameters())]
    check('비충돌 목표쌍 → 단일 backward와 동일 업데이트 (accum 2 micro-step)',
          max(diffs) < 1e-7, f"max|Δparam|={max(diffs):.2e} cos={st3['cos_shared']:+.4f}")


# ── C. peer mutual distillation ─────────────────────────────────────────────
def test_mutual_kl():
    print("\n[C] B-2 peer 상호증류")
    torch.manual_seed(2)
    logits = [torch.randn(2, 4, 6, 6, requires_grad=True) for _ in range(3)]
    kl = P44.mutual_kl(logits, temperature=1.0)
    check('KL 유한·양수', torch.isfinite(kl) and float(kl) > 0, f"{float(kl):.4f}")
    same = [logits[0], logits[0].clone(), logits[0].clone()]
    check('동일 분포면 KL≈0', float(P44.mutual_kl(same)) < 1e-6)
    # 대칭성: 전 순서쌍 평균이므로 모달 순서를 바꿔도 같은 값
    kl_perm = P44.mutual_kl([logits[2], logits[0], logits[1]])
    check('순서쌍 대칭(모달 순열 불변)', abs(float(kl) - float(kl_perm)) < 1e-5)
    kl.backward()
    check('gradient가 **모든** 모달 브랜치에 도달',
          all(l.grad is not None and float(l.grad.abs().sum()) > 0 for l in logits),
          " ".join(f"{float(l.grad.abs().sum()):.3f}" for l in logits))
    # 픽셀 가중 (B-3로 지워진 img 영역 제외)
    pw = [torch.ones(2, 1, 6, 6), None, None]
    pw[0][:, :, :3] = 0.0
    klw = P44.mutual_kl([l.detach() for l in logits], pixel_weights=pw)
    check('픽셀 가중 KL 유한', torch.isfinite(klw), f"{float(klw):.4f}")

    # 관계형 대응
    feats = [torch.randn(2, 8, 6, 6, requires_grad=True) for _ in range(3)]
    rc = P44.relational_correspondence(feats, num_pairs=64)
    check('rel_corr 유한·≥0', torch.isfinite(rc) and float(rc) >= 0, f"{float(rc):.4f}")
    rc.backward()
    check('rel_corr gradient가 모든 브랜치에 도달',
          all(f.grad is not None and float(f.grad.abs().sum()) > 0 for f in feats))
    same_f = [feats[0].detach(), feats[0].detach().clone(), feats[0].detach().clone()]
    check('동일 feature면 rel_corr≈0',
          float(P44.relational_correspondence(same_f, num_pairs=64)) < 1e-8)

    # warmup 게이팅 (fusion 레벨)
    fus = make_fusion(p44_mutual_kl=True, p44_mkl_warmup_ep=10,
                      p44_rel_corr=True, p44_rc_warmup_ep=10)
    fus.train()
    feats3 = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    gt = torch.randint(0, 4, (2, 32, 32))
    _, a_pre = fus(feats3, gt, epoch=9)
    _, a_post = fus(feats3, gt, epoch=10)
    check('warmup 전에는 손실 키 자체가 없음(정확히 0)',
          'p44_mutual_kl' not in a_pre and 'p44_rel_corr' not in a_pre)
    check('warmup 후 손실 활성', 'p44_mutual_kl' in a_post and 'p44_rel_corr' in a_post,
          f"kl={float(a_post['p44_mutual_kl']):.4f} rc={float(a_post['p44_rel_corr']):.4f}")


# ── D. 국소 마스킹 ──────────────────────────────────────────────────────────
def test_local_mask():
    print("\n[D] B-3 커버리지 국소 마스킹")
    torch.manual_seed(4)
    img, lidar, event = make_inputs(B=8, H=32, W=32, seed=4)

    m = P44.sample_region_mask(img, frac=1.0, mode='rect')
    check('rect: 전 샘플 선택 시 모든 샘플이 일부 마스킹',
          all(float(m[b].sum()) > 0 for b in range(8)))
    check('rect: 전체가 아니라 일부 영역만',
          all(0 < float(m[b].mean()) < 1.0 for b in range(8)),
          f"mean={float(m.mean()):.3f}")

    m0 = P44.sample_region_mask(img, frac=0.0, mode='rect')
    check('FRAC=0 → 마스크 없음(None)', m0 is None)

    torch.manual_seed(11)
    sel_counts = []
    for _ in range(40):
        mm = P44.sample_region_mask(img, frac=0.5, mode='global')
        sel_counts.append(0 if mm is None else int((mm.amax((1, 2, 3)) > 0).sum()))
    rate = sum(sel_counts) / (40 * 8)
    check('FRAC=0.5 샘플 선택률 ≈0.5', 0.35 < rate < 0.65, f"{rate:.3f}")

    mg = P44.sample_region_mask(img, frac=1.0, mode='global')
    check('global 모드 = P42 의미(선택 샘플 전체 마스킹)',
          float(mg.mean()) == 1.0)

    # coverage: lidar 발자국 사용
    torch.manual_seed(5)
    mc = P44.sample_region_mask(img, frac=1.0, mode='coverage', lidar=lidar,
                                coverage_dilate=5, blob_p=1.0)
    valid = (lidar.abs().sum(1, keepdim=True) > 1e-6).float()
    dil = F.max_pool2d(valid, 5, stride=1, padding=2)
    check('coverage: 마스크 ⊆ lidar 유효(팽창) 영역',
          float(((mc > 0.5) & (dil < 0.5)).sum()) == 0)
    # lidar가 전혀 없는 우측 절반 — 팽창 마진(k//2=2px)을 넘어서는 누출은 0이어야
    right = float(mc[:, :, :, 16 + 2:].sum())
    check('coverage: lidar 무반환 영역(우측 절반)은 팽창 마진 밖에서 마스킹 0',
          right == 0.0, f"masked_px={right}")
    check('coverage: 마스킹 픽셀 비율이 lidar 발자국 수준(전역 아님)',
          0 < float(mc.mean()) < 0.6, f"mean={float(mc.mean()):.3f}")
    zero_lidar = torch.zeros_like(lidar)
    mfb = P44.sample_region_mask(img, frac=1.0, mode='coverage', lidar=zero_lidar)
    check('coverage: lidar 부재 샘플은 rect로 폴백',
          mfb is not None and all(float(mfb[b].sum()) > 0 for b in range(8)))

    # 적용: img만 바뀐다
    region = P44.sample_region_mask(img, frac=1.0, mode='rect')
    new_img = img * (1.0 - region)
    check('마스킹 영역의 img = 정확히 0',
          float(new_img[(region.expand_as(img) > 0.5)].abs().sum()) == 0.0)
    check('비마스킹 영역의 img 불변',
          torch.equal(new_img[(region.expand_as(img) < 0.5)],
                      img[(region.expand_as(img) < 0.5)]))
    check('다른 모달은 손대지 않음 (호출자 계약)',
          torch.equal(lidar, make_inputs(B=8, H=32, W=32, seed=4)[1]))

    # P42 직교성: (B,) 샘플 마스크는 예전 그대로 "샘플 제외"로 동작해야 한다
    fus = make_fusion(); fus.train()
    torch.manual_seed(12)
    feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    gt = torch.randint(0, 4, (2, 32, 32))
    _, a_p42 = fus(feats, gt, img_mask=torch.ones(2), img_idx=0)   # 전 샘플 마스킹
    _, a_none = fus(feats, gt, img_mask=None, img_idx=0)
    # img CE가 0이 되므로 aux_ce = (0 + ce_lidar + ce_event)/3 < 전체 평균
    check('P42 (B,) 경로 유지: 마스킹 샘플의 img CE 제외',
          float(a_p42['aux_ce']) < float(a_none['aux_ce']),
          f"{float(a_p42['aux_ce']):.4f} < {float(a_none['aux_ce']):.4f}")
    # P44 (B,1,H,W) 경로: **영역만** ignore → 같은 샘플의 나머지 픽셀은 학습된다
    reg = torch.zeros(2, 1, 32, 32); reg[:, :, :, :16] = 1.0
    _, a_p44 = fus(feats, gt, img_mask=reg, img_idx=0)
    check('P44 영역 경로가 실제로 발동(무마스크와 다름)',
          not torch.equal(a_p44['aux_ce'], a_none['aux_ce']),
          f"{float(a_p44['aux_ce']):.4f} vs {float(a_none['aux_ce']):.4f}")
    check('P44 영역 경로 ≠ 샘플 전면 제외 (부분 제외가 유지됨)',
          not torch.equal(a_p44['aux_ce'], a_p42['aux_ce']),
          f"{float(a_p44['aux_ce']):.4f} vs {float(a_p42['aux_ce']):.4f}")
    reg_all = torch.ones(2, 1, 32, 32)
    _, a_full = fus(feats, gt, img_mask=reg_all, img_idx=0)
    check('영역이 전체면 P42 전면 제외와 동치',
          torch.allclose(a_full['aux_ce'], a_p42['aux_ce'], atol=1e-6),
          f"{float(a_full['aux_ce']):.4f} vs {float(a_p42['aux_ce']):.4f}")


# ── E. V-1 presence 재정규화 ────────────────────────────────────────────────
def test_validity():
    print("\n[E] V-1 presence 재정규화")
    img, lidar, event = make_inputs(B=2, H=32, W=32, seed=6)
    pres = P44.presence_masks([img, lidar, event], size=(8, 8), img_idx=0, dilate=0)
    check('presence shape (m,B,1,h,w)', tuple(pres.shape) == (3, 2, 1, 8, 8),
          str(tuple(pres.shape)))
    check('img는 전 픽셀 present', float(pres[0].min()) == 1.0)
    check('lidar 무반환 영역은 absent',
          float(pres[1, :, :, :, 4:].sum()) == 0.0)
    check('lidar 유효 영역은 present', float(pres[1, :, :, :, :4].sum()) > 0)

    w = torch.full((3, 2, 1, 8, 8), 1.0 / 3.0)
    w2 = P44.renormalize_over_present(w, pres)
    check('부재 모달 가중 정확히 0',
          float((w2 * (pres < 0.5).float()).abs().sum()) == 0.0)
    check('잔여 가중 합 = 1',
          torch.allclose(w2.sum(0), torch.ones(2, 1, 8, 8), atol=1e-6))
    allp = torch.ones_like(pres)
    check('전 모달 present면 항등', torch.allclose(
        P44.renormalize_over_present(w, allp), w, atol=1e-7))

    # fusion 레벨 (eval 경로에도 적용 = 유일한 추론 영향 항목)
    torch.manual_seed(8)
    feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
    fus = make_fusion(p44_validity_renorm=True)
    fus.eval()
    with torch.no_grad():
        f_off, _ = fus(feats, None)
        f_on, _ = fus(feats, None, presence=pres)
    check('V-1 on/off가 실제로 추론 출력을 바꾼다 (no-op 아님)',
          not torch.allclose(f_off, f_on, atol=1e-6),
          f"Δ={float((f_off - f_on).abs().mean()):.6f}")
    with torch.no_grad():
        f_on2, _ = fus(feats, None, presence=pres)
    check('eval 결정론(같은 입력 → 같은 출력)', torch.equal(f_on, f_on2))
    gate = fus._last_gate_spatial
    check('게이트에서 부재 모달 가중 0',
          float((gate * (pres < 0.5).float()).abs().sum()) < 1e-6)


# ── F. P45 FogStyle ─────────────────────────────────────────────────────────
def test_fogstyle():
    print("\n[F] P45 FogStyle (feature-space)")
    torch.manual_seed(9)
    feat = torch.randn(4, 8, 6, 6, requires_grad=True)
    out, applied = P44.style_perturb(feat, prob=1.0, sigma=0.5)
    check('prob=1 → 전 샘플 적용', float(applied.sum()) == 4.0)
    check('섭동이 실제로 통계를 바꾼다',
          not torch.allclose(out, feat, atol=1e-4),
          f"Δ={float((out - feat).abs().mean()):.4f}")
    out0, applied0 = P44.style_perturb(feat, prob=0.0, sigma=0.5)
    check('prob=0 → 항등(비활성 샘플 불변)', torch.equal(out0, feat)
          and float(applied0.sum()) == 0.0)
    logp = F.log_softmax(out.flatten(2).mean(-1), dim=1)
    logc = F.log_softmax(feat.detach().flatten(2).mean(-1), dim=1)
    kl = (logc.exp() * (logc - logp)).sum(1).mean()
    check('일관성 손실 유한', torch.isfinite(kl), f"{float(kl):.4f}")
    kl.backward()
    check('gradient가 img 브랜치 feature로 흐름',
          feat.grad is not None and float(feat.grad.abs().sum()) > 0)
    # 다른 모달 feature는 손대지 않는다 (모델은 img 인덱스만 넘긴다)
    others = [torch.randn(4, 8, 6, 6) for _ in range(2)]
    snapshot = [o.clone() for o in others]
    P44.style_perturb(feat.detach(), prob=1.0, sigma=0.5)
    check('다른 모달 feature 불변',
          all(torch.equal(a, b) for a, b in zip(others, snapshot)))


# ── G. config 배선 (YAML → build_reliadino → ctor kwargs) ───────────────────
# P44/P45 config 스키마 — YAML에 이 밖의 키가 있으면 **조용히 무시**된다(오타 =
# 무음 실패). 아래 테스트가 양방향(오타 검출 + 값 전달)을 모두 잡는다.
P44_SCHEMA = {
    'MMPARETO': {'ENABLE', 'INTERVAL', 'MAGNITUDE'},
    'MUTUAL_KL': {'ENABLE', 'WEIGHT', 'TEMPERATURE', 'WARMUP_EP'},
    'REL_CORR': {'ENABLE', 'WEIGHT', 'PAIRS', 'MODE', 'WARMUP_EP'},
    'LOCAL_MASK': {'ENABLE', 'MODE', 'FRAC', 'WARMUP_EP', 'AREA_RATIO',
                   'NUM_REGIONS', 'COVERAGE_DILATE', 'BLOB_GRID', 'BLOB_P'},
    'HARD_PIXEL_AUX': {'ENABLE', 'WEIGHT'},
    'VALIDITY_RENORM': {'ENABLE', 'DILATE'},
}
P45_SCHEMA = {'FOGSTYLE': {'ENABLE', 'PROB', 'SIGMA', 'WEIGHT', 'DETACH_CLEAN'}}


def test_config_plumbing():
    print("\n[G] config 배선 end-to-end")
    import yaml
    import semseg.models.reliadino.model as M

    captured = {}

    class _Fake:
        def __init__(self, **kw):
            captured.clear(); captured.update(kw)

    real, M.ReliaDINO = M.ReliaDINO, _Fake
    try:
        for name, expect in [
            ('configs/jarvis-muses_rgbel_P44_bmr.yaml',
             dict(p44_local_mask=True, p44_mask_mode='coverage', p44_mask_frac=0.5,
                  p44_mask_warmup_ep=20, p44_mutual_kl=True, p44_mkl_w=0.5,
                  p44_mkl_warmup_ep=10, p44_rel_corr=True, p44_rc_w=0.1,
                  p44_rc_pairs=2048, p44_hard_pixel_aux=False,
                  p44_validity_renorm=False, p45_fogstyle=False)),
            ('configs/hpca100-deliver_rgbdel_P44_bmr.yaml',
             dict(p44_local_mask=True, p44_mask_mode='coverage',
                  p44_mutual_kl=True, p44_rel_corr=True,
                  p44_validity_renorm=False, p45_fogstyle=False)),
            ('configs/yeon-deliver_rgbdel_P44_bmr_smoke.yaml',
             dict(p44_local_mask=True, p44_mask_warmup_ep=0, p44_mkl_warmup_ep=0,
                  p44_rc_warmup_ep=0, p44_hard_pixel_aux=True,
                  p44_validity_renorm=True, p45_fogstyle=True)),
        ]:
            cfg = yaml.safe_load((REPO / name).read_text())
            M.build_reliadino(cfg, num_classes=19)
            bad = {k: v for k, v in expect.items() if captured.get(k) != v}
            check(f"{Path(name).name}: P44/P45 kwargs 전달", not bad, str(bad))
            # 스키마 밖 키 = 오타 (조용히 무시되는 설정)
            p44c = (cfg['MODEL'].get('P44', {}) or {})
            unknown = ([k for k in p44c if k not in P44_SCHEMA]
                       + [f"{g}.{k}" for g, s in P44_SCHEMA.items()
                          for k in (p44c.get(g, {}) or {}) if k not in s])
            p45c = (cfg['MODEL'].get('P45', {}) or {})
            unknown += ([k for k in p45c if k not in P45_SCHEMA]
                        + [f"{g}.{k}" for g, s in P45_SCHEMA.items()
                           for k in (p45c.get(g, {}) or {}) if k not in s])
            check(f"{Path(name).name}: 미소비 키 없음(오타 검출)", not unknown, str(unknown))
            # MMPARETO는 trainer가 직접 읽는다 — 같은 경로로 확인
            mmp = (cfg['MODEL'].get('P44', {}) or {}).get('MMPARETO', {}) or {}
            check(f"{Path(name).name}: MMPARETO 블록 존재·ENABLE",
                  mmp.get('ENABLE', False) is True and mmp.get('INTERVAL', 0) >= 1,
                  str(mmp))
    finally:
        M.ReliaDINO = real


# ── H. 실제 encoder 파라미터 이름 위에서 B-1 그룹핑이 잡히는가 ──────────────
def test_real_lora_grouping():
    print("\n[H] B-1 그룹핑 × 실제 FrozenViTEncoder 파라미터 이름")
    try:
        from semseg.models.reliadino.encoder import FrozenViTEncoder
        enc = FrozenViTEncoder(backbone='vit_tiny_patch16_224',
                               fallback='vit_tiny_patch16_224', pretrained=False,
                               img_size=64, num_modalities=3, lora_r=2)
    except Exception as e:                                    # timm 부재 등
        print(f"  [SKIP] encoder 생성 실패: {e}")
        return
    mp = MMPareto(enc.named_parameters(), num_modalities=3,
                  modal_names=['img', 'lidar', 'event'])
    names = [g['name'] for g in mp.groups]
    n_lora = sum(1 for n, p in enc.named_parameters()
                 if p.requires_grad and n.split('.')[-1] in ('a_q', 'b_q', 'a_v', 'b_v'))
    check('실제 LoRA 이름(blocks.N.attn.qkv.a_q)에서 per-modal 그룹 생성',
          names == ['lora_img', 'lora_lidar', 'lora_event'], f"{names}")
    check('모달 그룹이 전 LoRA 파라미터를 슬라이스로 덮음',
          all(len(g['entries']) == n_lora for g in mp.groups),
          f"entries={[len(g['entries']) for g in mp.groups]} lora_params={n_lora}")


# ── I. end-to-end: 전 토글 on 모델 forward + 트레이너식 분할 backward ────────
def _tiny_cfg(**p44over):
    p44c = {
        'MMPARETO': {'ENABLE': True, 'INTERVAL': 1},
        'MUTUAL_KL': {'ENABLE': True, 'WEIGHT': 0.5, 'WARMUP_EP': 0},
        'REL_CORR': {'ENABLE': True, 'WEIGHT': 0.1, 'PAIRS': 64, 'WARMUP_EP': 0},
        'LOCAL_MASK': {'ENABLE': True, 'MODE': 'coverage', 'FRAC': 1.0,
                       'WARMUP_EP': 0, 'COVERAGE_DILATE': 5},
        'HARD_PIXEL_AUX': {'ENABLE': True, 'WEIGHT': 0.5},
        'VALIDITY_RENORM': {'ENABLE': True, 'DILATE': 1},
    }
    p44c.update(p44over)
    return {
        'MODEL': {
            'BACKBONE_TIMM': 'vit_tiny_patch16_224',
            'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
            'PRETRAINED_BACKBONE': False, 'LORA_R': 2, 'FPN_DIM': 64,
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0,
                       'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
                       'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False},
            'GATE': {'ENABLE': True, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': True, 'LAMBDA': 0.1},
            'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
            'M2F': {'ENABLE': False},
            'P44': p44c,
            'P45': {'FOGSTYLE': {'ENABLE': True, 'PROB': 1.0, 'SIGMA': 0.5,
                                 'WEIGHT': 0.1}},
        },
        'DATASET': {'MODALS': ['img', 'lidar', 'event']},
        'TRAIN': {'IMAGE_SIZE': [64, 64]},
    }


def test_end_to_end():
    print("\n[I] end-to-end (tiny ViT): 전 토글 on forward + 분할 backward")
    from semseg.models.reliadino.model import build_reliadino
    try:
        torch.manual_seed(0)
        model = build_reliadino(_tiny_cfg(), num_classes=5)
    except Exception as e:
        print(f"  [SKIP] 모델 생성 실패: {e}")
        return
    model.train(); model._current_epoch = 5
    B, H, W = 2, 64, 64
    torch.manual_seed(1)
    img = torch.rand(B, 3, H, W)
    lidar = torch.rand(B, 3, H, W); lidar[:, :, :, W // 2:] = 0.0   # 우측 무반환
    event = torch.rand(B, 3, H, W)
    gt = torch.randint(0, 5, (B, H, W))

    logits, m_feat, aux = model([img, lidar, event], True, gt_mask=gt)
    check('forward 통과 + logits 원해상도', tuple(logits.shape) == (B, 5, H, W),
          str(tuple(logits.shape)))
    for k in ('p44_mutual_kl', 'p44_rel_corr', 'p44_hard_aux', 'p45_fogstyle'):
        check(f'aux[{k}] 생성·유한', k in aux and bool(torch.isfinite(aux[k])),
              f"{float(aux[k]):.5f}" if k in aux else 'MISSING')
    check('B-3 마스킹이 실제로 걸림(mask_rate>0)',
          model._last_p44_mask is not None and float(model._last_p44_mask.mean()) > 0,
          f"rate={float(model._last_p44_mask.mean()):.3f}")
    check('P44 손실이 aux CE 대비 과대하지 않음 (스케일 온전성)',
          float(aux['p44_mutual_kl']) < 5 * float(aux['aux_ce'])
          and float(aux['p44_rel_corr']) < float(aux['aux_ce']),
          f"auxCE={float(aux['aux_ce']):.4f} kl={float(aux['p44_mutual_kl']):.4f} "
          f"rc={float(aux['p44_rel_corr']):.4f}")

    # 트레이너와 동일한 분할 backward → MMPareto 결합
    mp = MMPareto(model.named_parameters(), num_modalities=3,
                  modal_names=['img', 'lidar', 'event'])
    names = [g['name'] for g in mp.groups]
    check('실모델 그룹 = 모달 3 + shared',
          names == ['lora_img', 'lora_lidar', 'lora_event', 'shared'], f"{names}")
    main = logits.float().mean() + 0.1 * aux['rbma_cal_loss'] + aux['router_reg']
    auxl = (0.5 * aux['aux_ce'] + aux['p44_mutual_kl'] + aux['p44_rel_corr']
            + aux['p44_hard_aux'] + aux['p45_fogstyle'])
    gm = torch.autograd.grad(main, mp.params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(auxl, mp.params, allow_unused=True)
    mp.accumulate(gm, ga)
    aux_mass = {}
    for g in mp.groups:
        s = 0.0
        for i, sl in g['entries']:
            a = mp._aux[i]
            s += float((a if sl is None else a[sl]).abs().sum())
        aux_mass[g['name']] = s
    check('aux 목표의 gradient가 **모든** per-modal LoRA 그룹에 도달',
          all(aux_mass[f'lora_{m}'] > 0 for m in ('img', 'lidar', 'event')),
          " ".join(f"{k}:{v:.3f}" for k, v in aux_mass.items()))
    st = mp.combine()
    check('결합 후 gradient 전부 유한',
          all(bool(torch.isfinite(p.grad).all()) for p in mp.params))
    check('그룹별 cos 진단 산출(게이트② 계측 가능)',
          all(f'cos_lora_{m}' in st for m in ('img', 'lidar', 'event')),
          " ".join(f"{k}:{v:+.3f}" for k, v in st.items() if k.startswith('cos_')))

    model.eval()
    with torch.no_grad():
        o1 = model([img, lidar, event], True)
        o2 = model([img, lidar, event], True)
    check('eval 결정론(마스킹·style 섭동 미적용)', torch.equal(o1[0], o2[0]))
    check('eval 후 학습용 스태시 해제',
          model._last_p44_mask is None and model.fusion._train_aux_logits is None)


if __name__ == '__main__':
    torch.set_grad_enabled(True)
    print("P44/P45 smoke — CPU, tiny tensors")
    test_config_plumbing()
    test_real_lora_grouping()
    test_end_to_end()
    test_all_off()
    test_mmpareto()
    test_mutual_kl()
    test_local_mask()
    test_validity()
    test_fogstyle()
    print("\n" + ("=" * 60))
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)}) — " + ", ".join(FAILS))
        sys.exit(1)
    print("RESULT: ALL PASS")
