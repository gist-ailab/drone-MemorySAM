"""[P49-AIR] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

실행 (GPU 불필요, tiny ViT + 작은 ConvNeXt + 합성 배치):
    python tools/smoke_p49.py

검사 항목
  A. gradient 흐름  : γ(주입/피라미드) · ViT 말단 블록 · 보조 ConvNeXt · 헤드
                      전부에 gradient 가 도달하는가.
                      🔴 두 국면을 **나눠서** 본다 —
                        A1 init(γ=0) + VICReg on : ConvNeXt 의 gradient 출구는
                           VICReg 다(task 경로는 γ=0 이라 정의상 0). γ 자신은
                           ∂L/∂γ=⟨∂L/∂out, Δ⟩≠0 이라 **여기서 이미 움직인다**.
                        A2 γ=0.1(=1스텝 뒤 상태) + VICReg off : task 손실만으로
                           ConvNeXt 까지 gradient 가 닿는가 = 주입 경로의 도통 확인.
                        A3 γ=0 + VICReg off : ConvNeXt grad == 0 (C/D 와 같은 사실)
  B. 비대칭 등가성  : INJECT:false(4모달) 출력 == RGB 단독(1모달) 출력, |Δ|max == 0
  C. init identity  : γ=0 에서 보조 **입력을 바꿔도** 출력 불변, |Δ|max == 0
  D. 무영향 가드    : P49 키 없는 config 로 기존 P39.1 모델이 그대로 빌드/forward
                      (클래스가 ReliaDINO 인지 + shape/finite)
  E. eval 결정론    : 같은 입력 2회 → 동일 출력
  F. 규모           : 파라미터 수(구성요소별) + 활성화 메모리 실측/추정
  G. 토글 가드      : DEFORM:true 는 NotImplementedError 로 죽는다(조용한 폴백 금지)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402
from semseg.models.reliadino.p49 import P49AIR                       # noqa: E402

K = 25                                   # DELIVER 클래스 수
MODALS = ['img', 'depth', 'event', 'lidar']
SIZE = 128                               # tiny ViT-16 -> 8x8 토큰
BS = 2


def pick_aux_backbone() -> str:
    """스모크용으로 가장 작은 ConvNeXt 변종을 고른다 (timm 버전 방어)."""
    import timm
    avail = set(timm.list_models())
    for n in ('convnext_atto', 'convnext_femto', 'convnext_pico',
              'convnext_nano', 'convnext_tiny', 'convnext_small'):
        if n in avail:
            return n
    return 'convnext_small'


AUX_BB = None      # main()에서 채운다


def p49_cfg(inject=True, rgb_ft=True, ms_head=True, aux='convnext',
            vicreg=True, proto=False, deform=False, modals=None,
            head_mode='pixel'):
    """본학습 config(configs/deliver/deliver_rgbdel_P49_air.yaml)의 tiny 판."""
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE': 'P49AIR-tiny',
        'BACKBONE_TIMM': 'vit_tiny_patch16_224',
        'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
        'PRETRAINED_BACKBONE': False,
        'LORA_R': 4, 'LORA_ALPHA': None, 'FPN_DIM': 64,
        'P49': {
            'ENABLE': True,
            'RGB_FT': rgb_ft,
            'AUX_ENCODER': aux,
            'AUX_BACKBONE': AUX_BB,
            'AUX_PRETRAINED': False,
            'INJECT': inject,
            'DEFORM': deform,
            'NUM_BLOCKS': 4,
            'DIM': 64, 'ATTN_DIM': 64, 'NUM_HEADS': 4, 'MLP_RATIO': 2.0,
            'KV_GRID': 8, 'KV_GRID_FLOOR': 2,
            'MS_HEAD': ms_head,
            'HEAD_MODE': head_mode,
            'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 2,
                    'DIM': 64, 'NUM_HEADS': 4, 'MLP_RATIO': 2.0,
                    'POINTS': 256, 'DEEP_SUPERVISION': True, 'LOSS_W': 0.5,
                    'ANCHORED': True, 'POINT_QUOTA': 8},
            'VICREG': {'ENABLE': vicreg, 'LVAR': 0.1, 'LCOV': 0.01,
                       'TOKENS': 128, 'LEVEL': 2,
                       'LIDAR_W': 1.0, 'OTHER_W': 0.25},
        },
    }
    if proto:
        m['P46'] = {'C3_PROTO': {'ENABLE': True, 'FEATURE': 'mfeat',
                                 'LAMBDA': 0.2, 'WARMUP_EP': 0,
                                 'PIXELS': 256, 'CROSS_VIEW': False}}
    return {'MODEL': m,
            'DATASET': {'MODALS': list(modals or MODALS), 'IGNORE_LABEL': 255},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


def p391_cfg():
    """P49 키가 **없는** 기존 계보 config (무영향 가드용). smoke_p47 의 base와 동형."""
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
    return {'MODEL': m, 'DATASET': {'MODALS': MODALS, 'IGNORE_LABEL': 255},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


def make_batch(device, seed=0, n=len(MODALS)):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g).to(device) for _ in range(n)]
    y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g).to(device)
    y[:, :8, :8] = 255                                # ignore 영역도 섞는다
    return x, y


def build(device, seed=0, **kw):
    torch.manual_seed(seed)
    return build_reliadino(p49_cfg(**kw), K).to(device)


def gnorm(model, pred) -> float:
    tot = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and pred(n) and p.grad is not None:
            tot += float(p.grad.detach().float().pow(2).sum())
    return tot ** 0.5


def total_loss(model, x, y):
    """train_reliadino.py 의 total 조립을 그대로 축약한 것 (전부 pre-scaled)."""
    logits, m_feat, aux = model(x, True, gt_mask=y)
    z = logits.new_zeros(())
    return (F.cross_entropy(logits, y, ignore_index=255)
            + aux.get('m2f_loss', z) + aux.get('vicreg', z)
            + aux.get('p46_proto', z)), logits, aux


# ── A. gradient 흐름 ─────────────────────────────────────────────────────────
def check_grad(device):
    rows, ok = [], True
    groups = {
        'gamma_inj':  lambda n: n.endswith('gamma_inj'),
        'gamma_pyr':  lambda n: n.endswith('gamma_pyr'),
        'vit_last':   lambda n: '.blocks.11.' in n,
        'aux_convnext': lambda n: n.startswith('aux_enc.') and '.net.' in n,
        'injector':   lambda n: n.startswith('injectors.'),
        'extractor':  lambda n: n.startswith('extractors.'),
        'head':       lambda n: n.startswith('head.'),
        'fpn':        lambda n: n.startswith('fpn.'),
    }

    def one(gamma_val, vicreg):
        model = build(device, vicreg=vicreg)
        model.train()
        model._current_epoch = 10
        if gamma_val != 0.0:
            with torch.no_grad():
                model.gamma_inj.fill_(gamma_val)
                model.gamma_pyr.fill_(gamma_val)
        x, y = make_batch(device)
        loss, _, aux = total_loss(model, x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        return {k: gnorm(model, f) for k, f in groups.items()}, aux

    # A1 — init(γ=0) + VICReg on.
    #   🔴 zero-init 게이트의 정의상, γ=0 인 **딱 그 스텝**에는 게이트 뒤쪽 모듈
    #   (injector/extractor 가중치)의 gradient 가 0 이다 (∂L/∂W = ∂L/∂out·γ·∂Δ/∂W).
    #   중요한 건 **γ 자신**이 gradient 를 받는다는 것이고(∂L/∂γ=⟨∂L/∂out, Δ⟩),
    #   γ 가 1스텝 움직이면 그 뒤 전부가 살아난다(A2 가 그 상태를 검사한다).
    #   보조 인코더는 VICReg 덕에 γ=0 에서도 산다.
    g1, aux1 = one(0.0, True)
    live_at_init = ['gamma_inj', 'gamma_pyr', 'vit_last', 'aux_convnext', 'head', 'fpn']
    for k in live_at_init:
        good = g1[k] > 0
        ok &= good
        rows.append(('A1 init γ=0 +VICReg', k, f"{g1[k]:.3e}", 'PASS' if good else 'FAIL'))
    for k in ('injector', 'extractor'):
        good = g1[k] == 0.0                      # 게이트 뒤 = init 에서 0 (설계 계약)
        ok &= good
        rows.append(('A1 init γ=0 +VICReg', f'{k} (=0 기대)', f"{g1[k]:.3e}",
                     'PASS' if good else 'FAIL'))
    rows.append(('A1', 'vicreg항 존재', str('vicreg' in aux1),
                 'PASS' if 'vicreg' in aux1 else 'FAIL'))
    ok &= ('vicreg' in aux1)

    # A2 — γ=0.1 (=1스텝 뒤 상태) + VICReg off → task 손실만으로 **전 구성요소** 도통
    g2, _ = one(0.1, False)
    for k in ['gamma_inj', 'gamma_pyr', 'vit_last', 'aux_convnext', 'head', 'fpn',
              'injector', 'extractor']:
        good = g2[k] > 0
        ok &= good
        rows.append(('A2 γ=0.1 -VICReg', k, f"{g2[k]:.3e}", 'PASS' if good else 'FAIL'))

    # A3 — γ=0 + VICReg off → 보조 인코더 task gradient 는 정의상 0,
    #      그래도 γ 는 gradient 를 받는다(= 키1 "흡수"가 아니라는 직접 증거).
    g3, _ = one(0.0, False)
    zero_ok = g3['aux_convnext'] == 0.0
    ok &= zero_ok
    rows.append(('A3 γ=0 -VICReg', 'aux_convnext(=0 기대)', f"{g3['aux_convnext']:.3e}",
                 'PASS' if zero_ok else 'FAIL'))
    gamma_moves = g3['gamma_inj'] > 0 and g3['gamma_pyr'] > 0
    ok &= gamma_moves
    rows.append(('A3 γ=0 -VICReg', 'gamma(>0 기대, 키1 반증)',
                 f"{g3['gamma_inj']:.3e}/{g3['gamma_pyr']:.3e}",
                 'PASS' if gamma_moves else 'FAIL'))
    return rows, ok


# ── B. 비대칭 등가성: INJECT:false == RGB 단독 ────────────────────────────────
def check_inject_off(device):
    a = build(device, seed=7, inject=False)                       # 4모달 + 주입 off
    b = build(device, seed=7, inject=True, modals=['img'])        # RGB 단독
    a.eval(); b.eval()
    x, _ = make_batch(device, seed=3)
    with torch.no_grad():
        oa, _ = a(x, True)
        ob, _ = b([x[0]], True)
    d = float((oa - ob).abs().max())
    same_params = (sum(p.numel() for p in a.parameters())
                   == sum(p.numel() for p in b.parameters()))
    ok = (d == 0.0) and same_params
    return [('B', '|Δ|max (INJECT:false vs RGB단독)', f"{d:.3e}",
             'PASS' if d == 0.0 else 'FAIL'),
            ('B', '파라미터 수 동일', str(same_params),
             'PASS' if same_params else 'FAIL')], ok


# ── C. init identity: γ=0 에서 보조 입력 무관 ────────────────────────────────
def check_identity(device):
    model = build(device, seed=11)
    model.eval()
    x1, _ = make_batch(device, seed=5)
    x2 = [x1[0]] + [torch.randn_like(t) * 3.0 + 1.0 for t in x1[1:]]   # 보조만 교체
    with torch.no_grad():
        o1, _ = model(x1, True)
        o2, _ = model(x2, True)
    d = float((o1 - o2).abs().max())
    gmax = float(model.gamma_inj.abs().max()) + float(model.gamma_pyr.abs().max())
    ok = (d == 0.0) and (gmax == 0.0)
    return [('C', 'γ init 합', f"{gmax:.3e}", 'PASS' if gmax == 0.0 else 'FAIL'),
            ('C', '|Δ|max (보조 입력 교체)', f"{d:.3e}",
             'PASS' if d == 0.0 else 'FAIL')], ok


# ── D. 무영향 가드: P49 키 없는 기존 모델 ────────────────────────────────────
def check_untouched(device):
    torch.manual_seed(0)
    model = build_reliadino(p391_cfg(), K).to(device)
    is_relia = type(model).__name__ == 'ReliaDINO'
    model.train(); model._current_epoch = 10
    x, y = make_batch(device)
    logits, m_feat, aux = model(x, True, gt_mask=y)
    shape_ok = tuple(logits.shape) == (BS, K, SIZE, SIZE)
    fin_ok = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(m_feat).all())
    model.eval()
    with torch.no_grad():
        eo, _ = model(x, True)
    eval_ok = tuple(eo.shape) == (BS, K, SIZE, SIZE) and bool(torch.isfinite(eo).all())
    ok = is_relia and shape_ok and fin_ok and eval_ok
    return [('D', 'P49 키 없음 -> ReliaDINO', type(model).__name__,
             'PASS' if is_relia else 'FAIL'),
            ('D', 'train logits shape', str(tuple(logits.shape)),
             'PASS' if shape_ok else 'FAIL'),
            ('D', 'train finite', str(fin_ok), 'PASS' if fin_ok else 'FAIL'),
            ('D', 'eval shape+finite', str(eval_ok), 'PASS' if eval_ok else 'FAIL'),
            ('D', 'aux keys', ",".join(sorted(aux.keys()))[:48] or '-', 'PASS')], ok


# ── E. eval 결정론 ───────────────────────────────────────────────────────────
def check_determinism(device):
    rows, ok = [], True
    for mode in ('pixel', 'query'):
        model = build(device, seed=21, head_mode=mode)
        model.eval()
        # 학습된 상태를 흉내내려고 γ 를 깨운다 (γ=0 이면 보조 경로가 무의미하게 통과)
        with torch.no_grad():
            model.gamma_inj.fill_(0.3)
            model.gamma_pyr.fill_(0.3)
        x, _ = make_batch(device, seed=9)
        with torch.no_grad():
            o1, _ = model(x, True)
            o2, _ = model(x, True)
        d = float((o1 - o2).abs().max())
        good = d == 0.0
        ok &= good
        rows.append(('E', f'|Δ|max 2회 forward (HEAD_MODE={mode})', f"{d:.3e}",
                     'PASS' if good else 'FAIL'))
    return rows, ok


# ── F. 규모 ──────────────────────────────────────────────────────────────────
def check_size(device):
    model = build(device, seed=1, proto=True)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def cnt(pred):
        return sum(p.numel() for n, p in model.named_parameters() if pred(n))
    parts = [
        ('ViT 백본(RGB 주경로)', cnt(lambda n: n.startswith('encoder.'))),
        ('보조 인코더', cnt(lambda n: n.startswith('aux_enc.') or n.startswith('aux_shared.'))),
        ('injector', cnt(lambda n: n.startswith('injectors.'))),
        ('extractor', cnt(lambda n: n.startswith('extractors.'))),
        ('γ (inj/pyr)', cnt(lambda n: n.endswith(('gamma_inj', 'gamma_pyr')))),
        ('FPN+pixel head', cnt(lambda n: n.startswith(('fpn.', 'head.', 'pyr_proj.')))),
        ('M2F-lite', cnt(lambda n: n.startswith('m2f.'))),
    ]
    rows = [('F', f'params {k}', f"{v/1e6:.3f}M", '-') for k, v in parts]
    rows.append(('F', 'params total/trainable',
                 f"{tot/1e6:.3f}M / {tr/1e6:.3f}M ({100.0*tr/max(tot,1):.1f}%)", '-'))

    model.train(); model._current_epoch = 10
    x, y = make_batch(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    loss, _, _ = total_loss(model, x, y)
    loss.backward()
    if device.type == 'cuda':
        rows.append(('F', 'peak alloc (BS2·128²·tiny)',
                     f"{torch.cuda.max_memory_allocated()/2**20:.1f} MiB", '-'))
    else:
        rows.append(('F', 'peak alloc', 'n/a (CPU 실행)', '-'))
    # 본학습 규모 추정 (선형 외삽 아님 — 어디까지나 자릿수 감각용)
    rows.append(('F', '본학습 규모 메모',
                 'ViT-L 24층 full-FT ~304M + ConvNeXt-S ×3 ~150M -> trainable ≈ 0.5B '
                 '(제안 §3 "≈300M" 은 백본만 센 수치)', '-'))
    return rows, True


# ── G. 토글 가드 ─────────────────────────────────────────────────────────────
def check_guards(device):
    rows, ok = [], True
    try:
        build(device, deform=True)
        rows.append(('G', 'DEFORM:true', 'no raise', 'FAIL')); ok = False
    except NotImplementedError as e:
        rows.append(('G', 'DEFORM:true -> NotImplementedError',
                     str(e).split('.')[0][:44], 'PASS'))
    except Exception as e:                       # 다른 예외면 의도한 가드가 아니다
        rows.append(('G', 'DEFORM:true', f"{type(e).__name__}", 'FAIL')); ok = False
    # AUX_CNN:false -> vit_lora 팔이 실제로 다른 인코더를 만드는가
    torch.manual_seed(0)
    cfg = p49_cfg()
    cfg['MODEL']['P49'].pop('AUX_ENCODER')
    cfg['MODEL']['P49']['AUX_CNN'] = False
    cfg['MODEL']['P49']['MS_HEAD'] = True
    try:
        m = build_reliadino(cfg, K).to(device)
        kind = m.aux_encoder_kind
        good = kind == 'vit_lora'
        ok &= good
        rows.append(('G', 'AUX_CNN:false -> AUX_ENCODER', kind,
                     'PASS' if good else 'FAIL'))
        m.eval()
        x, _ = make_batch(device, seed=2)
        with torch.no_grad():
            o, _ = m(x, True)
        good2 = tuple(o.shape) == (BS, K, SIZE, SIZE)
        ok &= good2
        rows.append(('G', 'vit_lora 팔 forward shape', str(tuple(o.shape)),
                     'PASS' if good2 else 'FAIL'))
    except Exception as e:
        rows.append(('G', 'vit_lora 팔', f"{type(e).__name__}: {e}"[:60], 'FAIL'))
        ok = False
    # MS_HEAD:false 도 돈다 (단일스케일 공급)
    try:
        m = build(device, seed=3, ms_head=False)
        m.train(); m._current_epoch = 10
        x, y = make_batch(device, seed=4)
        loss, lg, _ = total_loss(m, x, y)
        loss.backward()
        good = tuple(lg.shape) == (BS, K, SIZE, SIZE) and bool(torch.isfinite(loss))
        ok &= good
        rows.append(('G', 'MS_HEAD:false fwd+bwd', str(tuple(lg.shape)),
                     'PASS' if good else 'FAIL'))
    except Exception as e:
        rows.append(('G', 'MS_HEAD:false', f"{type(e).__name__}: {e}"[:60], 'FAIL'))
        ok = False
    # RGB_FT:false (frozen + LoRA 팔)
    try:
        m = build(device, seed=5, rgb_ft=False)
        n_bb_train = sum(p.numel() for n, p in m.named_parameters()
                         if n.startswith('encoder.backbone.') and p.requires_grad)
        n_lora = sum(p.numel() for n, p in m.named_parameters()
                     if n.endswith(('.a_q', '.b_q', '.a_v', '.b_v')))
        good = n_lora > 0 and n_bb_train == n_lora
        ok &= good
        rows.append(('G', 'RGB_FT:false -> 백본 trainable==LoRA만',
                     f"{n_bb_train} vs lora {n_lora}", 'PASS' if good else 'FAIL'))
    except Exception as e:
        rows.append(('G', 'RGB_FT:false', f"{type(e).__name__}: {e}"[:60], 'FAIL'))
        ok = False
    return rows, ok


def main():
    global AUX_BB
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    AUX_BB = pick_aux_backbone()
    print(f"[P49-smoke] device={device} aux_backbone={AUX_BB} "
          f"torch={torch.__version__}")

    all_rows, all_ok = [], True
    for name, fn in (('A gradient 흐름', check_grad),
                     ('B 비대칭 등가성', check_inject_off),
                     ('C init identity', check_identity),
                     ('D 무영향 가드', check_untouched),
                     ('E eval 결정론', check_determinism),
                     ('F 규모', check_size),
                     ('G 토글 가드', check_guards)):
        rows, ok = fn(device)
        all_rows.extend(rows)
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n" + "=" * 92)
    print(f"{'검사':<20} {'항목':<38} {'값':<24} {'판정'}")
    print("-" * 92)
    for grp, item, val, verdict in all_rows:
        print(f"{grp:<20} {item:<38} {str(val):<24} {verdict}")
    print("=" * 92)
    print(f"RESULT: {'ALL PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
