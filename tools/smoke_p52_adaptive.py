"""[P52 adaptive 컨트롤러] 합성 스모크 — C3-adaptive + UniBal-adaptive.

실행 (GPU 불필요, tiny ViT + 합성 배치):
    python tools/smoke_p52_adaptive.py

검사 항목 (지시문 스펙 §4 그대로)
  A. off 가드(①)    : ENABLE false/off↔on 조합 — state_dict 키 3종 완전 동일
                       (컨트롤러 buffer는 persistent=False) + 공통 aux 수치
                       불변 + encoder forward 횟수 불변 + off 가중치로 eval
                       logits |Δ|max == 0 (추론 그래프 불변)
  B. C3 붕괴 반응(②) : 인위 붕괴(클래스 3 GT를 전부 5로 예측) 주입 → 그 클래스
                       λ_c만 상승, 건강 클래스 ≈0. 흩어진 오답(집중도 낮음)은
                       같은 recall이라도 λ가 훨씬 작다(지표 설계 실증).
                       WARMUP_EP 미만 λ_c≡0, 미출현 클래스 s_c=0.
  C. UniBal 반응(③)  : laziness 주입(radar 입력 0, 나머지 모달은 GT 상관 신호)
                       으로 짧은 학습 → radar의 λ_u,m만 상승, 살아있는 모달 ≈0.
                       컨트롤러 수학(observe 직접 주입)도 함께 검증.
  D. 통합 무결성(④)  : 두 adaptive on — train fwd/bwd 유한 + eval 2회 호출 결정론.
  E. ckpt 호환(⑤)    : 기존 ckpt(adaptive 없던 조합) 로드 — strict=True 성공
                       (컨트롤러 buffer가 키에 없음) / UB 단독은 missing이 전부
                       p47_2.* 이고 크래크 없음.
  F. 로깅 형식(⑥)    : [C3-ADPT]/[UB-ADPT] 라인 실제 출력 + 형식 검증.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402
from semseg.models.reliadino import p52 as P52                       # noqa: E402

K = 19                      # MUSES 클래스 수
MODALS = ['img', 'lidar', 'event', 'radar']      # radar = 지시문 §2의 하드코딩 금지 사례
SIZE = 128                  # tiny ViT-16 -> 8x8 토큰
BS = 2
CLASS_NAMES = [f'c{i}' for i in range(K)]


def base_cfg(p46_c3=False, p47=False, c3_ad=False, ub_ad=False,
             c3_warmup=5, ub_warmup=0, ub_ema=0.99, c3_ema=0.99):
    """smoke_p47의 tiny 구성에 [P52] 토글만 얹은 것."""
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
    if p46_c3:
        m['P46'] = {'C3_PROTO': {'ENABLE': True, 'LAMBDA': 0.1,
                                 'WARMUP_EP': 0, 'PIXELS': 512,
                                 'EMA': 0.9}}
    if p47:
        m['P47_2'] = {'ENABLE': True, 'LAMBDA_U': 0.4, 'MODALS': 'all',
                      'HEAD': 'linear', 'HIDDEN': 32, 'WARMUP_EP': 0,
                      'GT_DIV': 4, 'REDUCE': 'mean'}
    if c3_ad:
        m['C3_ADAPTIVE'] = {'ENABLE': True, 'LAMBDA_MAX': 0.1, 'TAU': 0.25,
                            'EMA_M': c3_ema, 'WARMUP_EP': c3_warmup}
    if ub_ad:
        m['UNIBAL_ADAPTIVE'] = {'ENABLE': True, 'LAMBDA_U_MAX': 0.4, 'CAP': 2.0,
                                'EMA_M': ub_ema, 'WARMUP_EP': ub_warmup,
                                'WARMUP_SMALL': 0.05}
    return {'MODEL': m, 'DATASET': {'MODALS': MODALS},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


def make_batch(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g).to(device) for _ in MODALS]
    y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g).to(device)
    y[:, :8, :8] = 255                          # ignore 영역도 섞는다
    return x, y


# [C용] 클래스 풀 — 19개 전체 대신 4개로 두면 신호 간격이 커져 tiny head가
# 몇 스텝 만에 구분한다(스모크 관심사는 "게으른 모달만 CE가 남는가"이지
# 19클래스 난이도가 아니다).
CLS_POOL = [0, 6, 12, 18]
SIG = 0.25          # 신호 = cls*0.25 → {0, 1.5, 3.0, 4.5}


def lazy_batch(device, seed=0):
    """[C용] radar만 0(정보 없음), 나머지 모달은 GT 상관 신호(채널 0 상수 —
    ViT 토큰 LN을 통과해 살아남는 방향 성분)를 담은 배치."""
    g = torch.Generator().manual_seed(seed)
    cls = torch.tensor(CLS_POOL, dtype=torch.long)[
        torch.randint(0, len(CLS_POOL), (BS,), generator=g)]
    xs, gt = [], torch.empty(BS, SIZE, SIZE, dtype=torch.long)
    for name in MODALS:
        if name == 'radar':
            xs.append(torch.zeros(BS, 3, SIZE, SIZE))
        else:
            x = torch.randn(BS, 3, SIZE, SIZE, generator=g)
            x[:, 0] += (cls.float() * SIG).view(BS, 1, 1)
            xs.append(x)
    for b in range(BS):
        gt[b] = int(cls[b])
    gt[:, :8, :8] = 255
    return [x.to(device) for x in xs], gt.to(device)


def build(device, seed=0, **kw):
    torch.manual_seed(seed)
    return build_reliadino(base_cfg(**kw), K).to(device)


def _aux_scalars(model, x, y):
    torch.manual_seed(99)          # P39 path-dropout 등 확률 경로 고정
    _, _, aux = model(x, True, gt_mask=y)
    return {k: float(v) for k, v in aux.items()
            if torch.is_tensor(v) and v.dim() == 0}


# ── A. off 가드 — 키 불변 + 공통 aux 불변 + 추가 forward 없음 + eval 등가 ────
def check_off_guard(device):
    ok = True
    lines = []
    a = build(device, seed=1234)                                    # 완전 off
    b = build(device, seed=1234, p46_c3=True, p47=True)             # 기존 토글 on
    c = build(device, seed=1234, p46_c3=True, p47=True,
              c3_ad=True, ub_ad=True)                               # + P52 on
    ka, kb, kc = (sorted(m.state_dict().keys()) for m in (a, b, c))
    # 완전-off(a)와 P46/P47-on(b)의 키 차이는 기존 토글이 만든 것(스모크 p46/p47
    # 범위). [P52] 가드가 보장해야 하는 것은 **adaptive가 키를 하나도 늘리지
    # 않는 것** = b ↔ c 완전 동일.
    keys_same = (kb == kc)
    p52_extra = [k for k in kc if k not in set(kb)]
    lines.append(f"    state_dict: 기존-on {len(kb)} keys ↔ adaptive-on "
                 f"{len(kc)} keys 완전 동일 = {keys_same} "
                 f"(adaptive 신규 키 = {p52_extra or '없음'})")
    ok &= keys_same
    # 컨트롤러 buffer는 named_buffers 에는 있으나(DDP broadcast 무해) state_dict
    # 에는 없다 → 기존 ckpt 저장/로드 불변.
    buf_names = {n for n, _ in c.named_buffers()
                 if n.startswith(('c3_adaptive.', 'unibal_adaptive.'))}
    in_sd = {n for n in buf_names if n in c.state_dict()}
    lines.append(f"    컨트롤러 buffer named_buffers={len(buf_names)}개, "
                 f"state_dict 포함={len(in_sd)}개(0이어야 함)")
    ok &= len(buf_names) > 0 and not in_sd
    ctrl_none = (getattr(b, 'c3_adaptive', 'X') is None
                 and getattr(b, 'unibal_adaptive', 'X') is None)
    ctrl_some = (c.c3_adaptive is not None and c.unibal_adaptive is not None)
    lines.append(f"    off 모델 컨트롤러 None = {ctrl_none} / "
                 f"on 모델 컨트롤러 존재 = {ctrl_some}")
    ok &= ctrl_none and ctrl_some
    # 공통 aux 수치 불변 + encoder forward 횟수 불변(smoke_p47 E 방식)
    b.load_state_dict(a.state_dict(), strict=False)
    for m in (a, b):
        m.train()
        m._current_epoch = 10
    x, y = make_batch(device, seed=3)
    counts, auxes = {}, {}
    for tag, m in (('off', a), ('on', b)):
        cnt = {'n': 0}
        h = m.encoder.register_forward_hook(
            lambda *_a, _c=cnt: _c.__setitem__('n', _c['n'] + 1))
        auxes[tag] = _aux_scalars(m, x, y)
        h.remove()
        counts[tag] = cnt['n']
    common = sorted(set(auxes['off']) & set(auxes['on']))
    diffs = {k: (auxes['off'][k], auxes['on'][k]) for k in common
             if abs(auxes['off'][k] - auxes['on'][k]) > 1e-6}
    fwd_same = counts['off'] == counts['on']
    lines.append(f"    encoder.forward 호출 off={counts['off']} on={counts['on']} "
                 f"(4모달×1회=4 기대, 추가 forward 없음) = {fwd_same}")
    lines.append(f"    공통 aux({common}) 수치 차이 = {diffs if diffs else '없음'}")
    ok &= (not diffs) and fwd_same
    # off 가중치 → adaptive-on 모델의 eval 출력은 |Δ|max == 0
    c.load_state_dict(a.state_dict(), strict=False)
    for m in (a, c):
        m.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        la, _ = a(x, True)
        torch.manual_seed(7)
        lc, _ = c(x, True)
    same = torch.equal(la, lc)
    lines.append(f"    eval logits off ↔ adaptive-on identical = {same} "
                 f"(max|diff|={float((la - lc).abs().max()):.3e})")
    ok &= same
    return ok, lines, a, b, c


# ── B. C3-adaptive — 인위 붕괴 주입 → 그 클래스 λ_c만 상승 ─────────────────
def check_c3(device):
    ctl = P52.C3Adaptive(K, lambda_max=0.1, tau=0.25, warmup_ep=0).to(device)
    n = 64                                    # 클래스당 픽셀 수
    g = torch.arange(K, device=device).repeat(n)
    logits = torch.full((n * K, K), -10.0, device=device)
    logits[torch.arange(n * K, device=device), g] = 10.0    # 기본 전부 정답
    m3 = g == 3
    logits[m3, 3] = -10.0
    logits[m3, 5] = 10.0                       # 3 → 5 집중 흡수(붕괴 주입)
    m4 = g == 4
    idx4 = torch.nonzero(m4).squeeze(1)
    scatter_t = 6 + torch.arange(idx4.numel(), device=device) % 13
    logits[idx4, 4] = -10.0
    logits[idx4, scatter_t] = 10.0             # 4 → {6..18} 균등 흩어짐(recall 동일,
    # argmax 타이브레이크가 한 클래스로 몰지 않도록 픽셀마다 순환 배정)
    for _ in range(5):                         # EMA 안정화(fresh는 즉시 초기화)
        ctl.observe(logits.view(1, n * K, K).permute(0, 2, 1), g)
    lam = ctl.lambdas(epoch=10)
    healthy_max = max(float(lam[i]) for i in range(K) if i not in (3, 4))
    ok = (float(lam[3]) > 0.099            # s_3 = (1-0)·1 = 1 → λ = LAMBDA_MAX
          and healthy_max < 1e-6           # 건강 클래스(recall=1) → s=0 → λ=0
          and float(lam[4]) < 0.5 * float(lam[3]))   # 흩어진 오답은 집중 흡수의 절반 이하
    # 미출현 클래스(한 번도 GT로 관측 안 됨) → s_c = 0
    ctl2 = P52.C3Adaptive(K, warmup_ep=0).to(device)
    g2 = torch.zeros(1024, dtype=torch.long, device=device)
    ctl2.observe(torch.zeros(1, K, 1024, device=device), g2)
    s2 = ctl2.scores()
    unseen_zero = bool((s2[1:] == 0).all())
    # WARMUP_EP 미만은 전 클래스 λ_c ≡ 0
    ctl3 = P52.C3Adaptive(K, warmup_ep=2).to(device)
    ctl3.observe(logits.view(1, n * K, K).permute(0, 2, 1), g)
    warm0 = float(ctl3.lambdas(epoch=1).abs().max()) == 0.0
    warm1 = float(ctl3.lambdas(epoch=2)[3]) > 0.09
    ok &= unseen_zero and warm0 and warm1
    top = ctl.log_lines(10, CLASS_NAMES)
    lines = [f"    λ_c[c3(집중 흡수)]={float(lam[3]):.4f}  "
             f"λ_c[c4(흩어짐)]={float(lam[4]):.4f}  "
             f"λ_c[건강 max]={healthy_max:.2e}",
             f"    미출현 클래스 s_c≡0 = {unseen_zero} | "
             f"WARMUP(2ep) 중 λ≡0 = {warm0} → ep2 반응 = {warm1}"]
    return ok, lines, top


# ── C. UniBal-adaptive — 인위 laziness(radar 0) → 그 모달 λ_u만 상승 ────────
def check_unibal(device):
    ok, lines = True, []
    # (1) 컨트롤러 수학 — observe 직접 주입(radar만 4배 높은 CE)
    ctl = P52.UniBalAdaptive(len(MODALS), active=list(range(len(MODALS))),
                             lambda_u_max=0.4, cap=2.0, momentum=0.9,
                             warmup_ep=0).to(device)
    for _ in range(10):
        ctl.observe([1.0, 1.0, 1.0, 4.0])
    lam = ctl.lambdas(epoch=3)
    # mean_k = (1+1+1+4)/4 = 1.75 → g_radar = 4/1.75−1 ≈ 1.286 (CAP=2 미만)
    # → λ_u,radar = 0.4·1.286/2 ≈ 0.257. 나머지는 평균 이하 → g=0 → λ=0.
    ok &= (float(lam[3]) > 0.2
           and max(float(lam[i]) for i in (0, 1, 2)) < 1e-6)
    lines.append(f"    [수학] 주입 L_ema=[1,1,1,4] → λ_u = "
                 + " ".join(f"{n}:{float(v):.4f}" for n, v in zip(MODALS, lam)))
    # (2) warmup 중 균일 소값(0.05)
    ctlw = P52.UniBalAdaptive(len(MODALS), active=list(range(len(MODALS))),
                              warmup_ep=3, momentum=0.9).to(device)
    ctlw.observe([1.0, 1.0, 1.0, 4.0])
    lamw = ctlw.lambdas(epoch=1)
    ok &= bool(torch.allclose(lamw, torch.full_like(lamw, 0.05)))
    lines.append(f"    [warmup] ep1<3 λ_u 균일 0.05 = "
                 + " ".join(f"{v:.3f}" for v in lamw.tolist()))
    # (3) end-to-end — radar 입력 0, 나머지는 GT 상관 신호 → 짧은 학습 후 반응
    m = build(device, seed=77, p47=True, ub_ad=True, ub_ema=0.7)
    m.train()
    m._current_epoch = 0
    opt = torch.optim.Adam(
        [p for p in m.parameters() if p.requires_grad], lr=3e-3)
    ces = {n: [] for n in MODALS}
    for step in range(80):
        x, y = lazy_batch(device, seed=step)
        logits, _, aux = m(x, True, gt_mask=y)
        loss = F.cross_entropy(logits, y, ignore_index=255) \
            + aux.get('p47_2_uni', 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for i, n in enumerate(MODALS):
            c = m.p47_2.last_ce[i]
            if c is not None:
                ces[n].append(float(c))
    lam_m = m.unibal_adaptive.lambdas(epoch=0)
    ce_last = {n: (lambda w: sum(w) / len(w))(v[-3:] or v)
               for n, v in ces.items() if v}
    radar_up = float(lam_m[3]) > 0.01
    others_down = max(float(lam_m[i]) for i in (0, 1, 2)) < float(lam_m[3])
    live_ce_gap = ce_last['radar'] > ce_last['img']
    ok &= radar_up and others_down and live_ce_gap
    lines.append("    [e2e] 80스텝 후 λ_u = "
                 + " ".join(f"{n}:{float(v):.4f}" for n, v in zip(MODALS, lam_m))
                 + f"  (radar>0={radar_up}, radar가 최대={others_down}, "
                   f"최근 CE radar:{ce_last['radar']:.3f} > img:{ce_last['img']:.3f}"
                   f"={live_ce_gap})")
    return ok, lines, m.unibal_adaptive.log_lines(0, MODALS)


# ── D. 통합 — 두 adaptive on fwd/bwd 유한 + eval 결정론 ─────────────────────
def check_integrated(device, model):
    model.train()
    model._current_epoch = 10
    x, y = make_batch(device, seed=5)
    logits, _, aux = model(x, True, gt_mask=y)
    total = F.cross_entropy(logits, y, ignore_index=255)
    for k in ('aux_ce', 'm2f_loss', 'router_reg', 'router_ce', 'vicreg',
              'p46_proto', 'p47_2_uni'):
        if k in aux:
            total = total + aux[k]
    model.zero_grad(set_to_none=True)
    total.backward()
    loss_fin = bool(torch.isfinite(total))
    grad_fin = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                   if p.grad is not None)
    n_grad = sum(1 for p in model.parameters()
                 if p.requires_grad and p.grad is not None
                 and float(p.grad.abs().sum()) > 0)
    # eval 결정론 — 2회 호출 동일 + train 중 쌓인 컨트롤러 EMA와 무관
    model.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        l1, _ = model(x, True)
        torch.manual_seed(11)
        l2, _ = model(x, True)
    det = torch.equal(l1, l2)
    ok = loss_fin and grad_fin and det and n_grad > 0
    lines = [f"    train total={float(total):.4f} finite={loss_fin} | "
             f"grads finite={grad_fin} ({n_grad} params w/ grad)",
             f"    eval 2회 호출 logits identical = {det} "
             f"(max|Δ|={float((l1 - l2).abs().max()):.3e})",
             "    aux 항: " + " ".join(sorted(aux.keys()))]
    model.zero_grad(set_to_none=True)
    return ok, lines


# ── E. 기존 ckpt 로드 호환 ──────────────────────────────────────────────────
def check_ckpt(device, legacy, legacy_bare):
    ok, lines = True, []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'legacy.pth'
        torch.save(legacy.state_dict(), p)
        # (a) 동일 토글 + adaptive on — 컨트롤러 buffer가 state_dict 키에 없으므로
        #     strict=True 로드가 성공해야 한다(기존 RESUME/eval 경로 무손상).
        try:
            m1 = build(device, seed=1, p46_c3=True, p47=True,
                       c3_ad=True, ub_ad=True)
            m1.load_state_dict(torch.load(p, map_location=device), strict=True)
            m1.train()
            m1._current_epoch = 10
            x, y = make_batch(device, seed=6)
            _, _, aux = m1(x, True, gt_mask=y)
            strict_ok = 'p46_proto' in aux and 'p47_2_uni' in aux
        except Exception as e:                                     # noqa: BLE001
            strict_ok = False
            lines.append(f"    (a) strict=True 로드 예외: {type(e).__name__}: {e}")
        lines.append(f"    (a) legacy ckpt → adaptive-on 모델 strict=True 로드 + "
                     f"train forward = {strict_ok}")
        ok &= strict_ok
        # (b) P47_2 off 였던 legacy(bare) → UNIBAL_ADAPTIVE 단독(on) :
        #     strict=False 관례(train_reliadino.py RESUME과 동일)에서 missing은
        #     모델이 새로 만든 p47_2 head 뿐, 크래크 없음.
        p2 = Path(td) / 'legacy_bare.pth'
        torch.save(legacy_bare.state_dict(), p2)
        m2 = build(device, seed=1, ub_ad=True)          # P47_2.ENABLE은 false
        miss, unexpected = m2.load_state_dict(
            torch.load(p2, map_location=device), strict=False)
        only_p47 = all(k.startswith('p47_2.') for k in miss) and not unexpected
        m2.eval()
        with torch.no_grad():
            x, _ = make_batch(device, seed=6)
            lo, _ = m2(x, True)
        fwd_ok = bool(torch.isfinite(lo).all())
        lines.append(f"    (b) UB 단독: missing {len(miss)}개 전부 p47_2.* = "
                     f"{only_p47}, unexpected = {len(unexpected)}, "
                     f"eval forward finite = {fwd_ok}")
        ok &= only_p47 and fwd_ok and len(miss) > 0
    return ok, lines


# ── F. 로깅 라인 형식 ───────────────────────────────────────────────────────
def check_logformat(c3_lines, ub_lines):
    ok = True
    for ln in c3_lines + ub_lines:
        print("        " + ln)
    ok &= all(ln.startswith('[C3-ADPT] ') for ln in c3_lines)
    ok &= any('λ_c top5' in ln for ln in c3_lines) \
        and any('s_c' in ln for ln in c3_lines)
    ok &= all(ln.startswith('[UB-ADPT] ') for ln in ub_lines)
    ok &= any('λ_u' in ln for ln in ub_lines) \
        and any('L_ema' in ln for ln in ub_lines)
    ok &= any(f'{n}' in ln for ln in ub_lines for n in MODALS)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)
    ok = True

    print("=" * 104)
    print("A. off 가드 — state_dict 키 불변 + 공통 aux 불변 + 추가 forward 없음 + eval 등가")
    aok, alines, a, b, c = check_off_guard(dev)
    for ln in alines:
        print(ln)
    print(f"    → {'OK' if aok else 'FAIL'}")
    ok &= aok
    del a, b

    print()
    print("=" * 104)
    print("B. C3-adaptive — 인위 붕괴(3→5 집중 흡수) 주입 → 그 클래스 λ_c만 상승")
    bok, blines, btop = check_c3(dev)
    for ln in blines:
        print(ln)
    print(f"    → {'OK' if bok else 'FAIL'}")
    ok &= bok

    print()
    print("=" * 104)
    print("C. UniBal-adaptive — 인위 laziness(radar 입력 0) → 그 모달 λ_u만 상승")
    cok, clines, ctop = check_unibal(dev)
    for ln in clines:
        print(ln)
    print(f"    → {'OK' if cok else 'FAIL'}")
    ok &= cok

    print()
    print("=" * 104)
    print("D. 통합 — 두 adaptive on: train fwd/bwd 유한 + eval 결정론")
    dok, dlines = check_integrated(dev, c)
    for ln in dlines:
        print(ln)
    print(f"    → {'OK' if dok else 'FAIL'}")
    ok &= dok

    print()
    print("=" * 104)
    print("E. 기존 ckpt 로드 호환 — 컨트롤러 buffer는 키에 없다")
    legacy = build(dev, seed=1, p46_c3=True, p47=True)     # (a)용: adaptive 없던 조합
    legacy_bare = build(dev, seed=1)                       # (b)용: P47_2도 없던 조합
    eok, elines = check_ckpt(dev, legacy, legacy_bare)
    for ln in elines:
        print(ln)
    print(f"    → {'OK' if eok else 'FAIL'}")
    ok &= eok
    del legacy, c

    print()
    print("=" * 104)
    print("F. 로깅 라인 형식 — train.log 에 남을 [C3-ADPT]/[UB-ADPT] 라인")
    fok = check_logformat(btop, ctop)
    print(f"    → {'OK' if fok else 'FAIL'}")
    ok &= fok

    print()
    print("=" * 104)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
