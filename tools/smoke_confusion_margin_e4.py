"""[E4] 혼동 쌍 margin 손실 합성 스모크 (CPU, tiny ViT, 1분 이내).

카드 = .claude_logs/decisions/2026-09-07-daily-cycle-experiment-cards.md §1 "E4".
구현 = MODEL.P46.CONFUSION_MARGIN({ENABLE, PAIRS, MARGIN, LAMBDA, VAL_IMAGES, SOURCE}).

실행 (GPU 불필요):
    python tools/smoke_confusion_margin_e4.py

검사 항목 (지시문 §4 그대로)
  (a) off(ENABLE=false / 키 없음) ⇒ state_dict 키·값 동일(cm 버퍼 미생성) +
      train/eval 출력 |Δ|max==0 (byte-동일). ENABLE=true는 cm 버퍼가 생겨야 함.
  (b) 명시 PAIRS ⇒ 손실 유한 + feature/LoRA 경로 gradient 도달 +
      hinge 방향성(sim(f,p_c) > sim(f,p_j)+m 이면 손실 0, 반대면 >0).
  (c) auto 모드의 혼동행렬 → top-k 선택 로직(confusion_pairs_from_cm)을 합성
      혼동행렬로 검증(대각 제외·행정규화·내림차순·행합0 제외).
  (d) 쌍 클래스가 배치에 안 나오거나 전부-ignore ⇒ NaN 없음(손실 0/None).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402
from semseg.models.reliadino import p46 as P46                       # noqa: E402

K = 25                                    # DELIVER 클래스 수
MODALS = ['img', 'depth', 'event', 'lidar']       # DELIVER 4센서
SIZE = 128                                # tiny ViT-16 -> 8x8 토큰
BS = 2


def base_cfg(cm=None, c3=True, c3_src='mfeat'):
    """tiny 구성 + C3_PROTO(margin 앵커) + 선택적 CONFUSION_MARGIN(cm dict)."""
    c3d = {'ENABLE': c3, 'LAMBDA': 0.1, 'WARMUP_EP': 0, 'PIXELS': 256,
           'EMA': 0.9, 'TEMPERATURE': 0.1, 'FEATURE': c3_src}
    p46 = {'C3_PROTO': c3d}
    if cm is not None:
        p46['CONFUSION_MARGIN'] = cm
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
        'P46': p46,
        'MODAL_DROPOUT': {'ENABLE': False},
    }
    return {'MODEL': m, 'DATASET': {'MODALS': MODALS},
            'TRAIN': {'IMAGE_SIZE': [SIZE, SIZE]}}


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return build_reliadino(base_cfg(**kw), K)


def make_batch(seed=0, single_class=None, all_ignore=False):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(BS, 3, SIZE, SIZE, generator=g) for _ in MODALS]
    if all_ignore:
        y = torch.full((BS, SIZE, SIZE), 255, dtype=torch.long)
    elif single_class is not None:
        y = torch.full((BS, SIZE, SIZE), single_class, dtype=torch.long)
        y[:, :8, :8] = 255
    else:
        y = torch.randint(0, K, (BS, SIZE, SIZE), generator=g)
        y[:, :8, :8] = 255
    return x, y


def _train_eval(model, x, y, seed=99):
    model.train()
    model._current_epoch = 10
    torch.manual_seed(seed)
    _, _, aux = model(x, True, gt_mask=y)
    proto = float(aux['p46_proto']) if 'p46_proto' in aux else None
    model.eval()
    with torch.no_grad():
        torch.manual_seed(7)
        lo, _ = model(x, True)
    return proto, lo


# ── (a) off ⇒ byte-동일 / on ⇒ cm 버퍼 생성 ──────────────────────────────────
def check_off_byte_identical():
    ok, lines = True, []
    a = build(seed=1234, cm=None)                                # 키 없음
    b = build(seed=1234, cm={'ENABLE': False})                   # ENABLE=false
    ka, kb = sorted(a.state_dict()), sorted(b.state_dict())
    keys_same = ka == kb
    vals_same = (keys_same and all(torch.equal(a.state_dict()[k], b.state_dict()[k])
                                   for k in ka))
    no_cm_keys = not any(('_cm_pairs' in k or '_cm_armed' in k) for k in ka)
    b.load_state_dict(a.state_dict())
    x, y = make_batch(seed=3)
    pa, la = _train_eval(a, x, y)
    pb, lb = _train_eval(b, x, y)
    proto_same = (pa is not None and abs(pa - pb) < 1e-6)
    logits_same = torch.equal(la, lb)
    off_ok = (keys_same and vals_same and no_cm_keys and proto_same and logits_same)
    lines.append(f"    off: keys={keys_same} vals={vals_same} cm키없음={no_cm_keys} "
                 f"proto|Δ|={abs(pa - pb):.2e} eval|Δ|max={float((la - lb).abs().max()):.2e}"
                 f" → {off_ok}")
    # on ⇒ cm 버퍼 생성(byte-동일 아님이 정상)
    c = build(seed=1234, cm={'ENABLE': True, 'PAIRS': 'auto_val_k5',
                             'SOURCE': 'proto'})
    has_cm = any(('_cm_pairs' in k or '_cm_armed' in k) for k in c.state_dict())
    lines.append(f"    on: cm 버퍼 생성={has_cm} (byte-동일 아님이 정상)")
    ok = off_ok and has_cm
    return ok, lines


# ── (b) 명시 PAIRS ⇒ 손실 유한 + gradient + hinge 방향성 ──────────────────────
def check_explicit_and_direction():
    ok, lines = True, []
    # (b-1) 전 모델 경로: 명시 pairs arm → 손실 유한 + feature/LoRA grad 도달
    m = build(seed=77, cm={'ENABLE': True, 'PAIRS': [[5, 6], [10, 12]],
                           'MARGIN': 0.5, 'LAMBDA': 0.05, 'SOURCE': 'proto'})
    # 트레이너의 명시-리스트 해석(정수라 그대로) → arm
    m.set_confusion_pairs(torch.tensor([[5, 6], [10, 12]], dtype=torch.long))
    armed = m.p46_cm_armed()
    feats_ref = []

    def _hook(_mod, _inp, out):
        out.retain_grad()
        feats_ref.append(out)

    h = m.encoder.register_forward_hook(_hook)
    m.train()
    m._current_epoch = 10
    x, y = make_batch(seed=5)          # 랜덤 배치: 5·6·10·12 모두 등장
    torch.manual_seed(99)
    _, _, aux = m(x, True, gt_mask=y)
    h.remove()
    has_cm = 'p46_confmargin' in aux
    cm_fin = has_cm and bool(torch.isfinite(aux['p46_confmargin']))
    if has_cm and float(aux['p46_confmargin'].detach().abs()) > 0:
        m.zero_grad(set_to_none=True)
        aux['p46_confmargin'].backward()
        n_enc = sum(1 for p in m.encoder.parameters()
                    if p.requires_grad and p.grad is not None
                    and float(p.grad.abs().sum()) > 0)
        feat_grad = any(f.grad is not None and float(f.grad.abs().sum()) > 0
                        for f in feats_ref)
        grad_ok = n_enc > 0 and feat_grad
    else:
        n_enc, grad_ok = 0, (float(aux.get('p46_confmargin', torch.zeros(())).detach()) == 0)
    b1 = armed and has_cm and cm_fin and grad_ok
    lines.append(f"    (b-1) armed={armed} cm항존재={has_cm} 유한={cm_fin} "
                 f"encoder grad={n_enc} feature grad도달={grad_ok} → {b1}")

    # (b-2) hinge 방향성 — pair_margin을 손수 만든 bank로 직접 검증
    bank = P46.PrototypeBank(num_classes=4, dim=8, pixels=0, temperature=0.1)
    with torch.no_grad():
        bank.proto.zero_()
        bank.proto[0, 0] = 1.0            # class 0 prototype = e0
        bank.proto[1, 1] = 1.0            # class 1 prototype = e1 (직교)
        bank.inited[0] = 1.0
        bank.inited[1] = 1.0
    margin = 0.5
    # feat 전 픽셀 = e0 방향(class 0 정답에 정렬), gt 전부 class 0
    feat_c = torch.zeros(1, 8, 2, 2); feat_c[:, 0] = 1.0
    gt0 = torch.zeros(1, 2, 2, dtype=torch.long)
    l_aligned, n_a = bank.pair_margin(feat_c, gt0, [(0, 1)], margin)
    aligned_zero = (n_a == 1 and abs(float(l_aligned)) < 1e-6)   # sim_c=1 > sim_j=0 + m → 0
    # feat 전 픽셀 = e1 방향(혼동 class 1 쪽), gt 전부 class 0 → 손실 > 0
    feat_j = torch.zeros(1, 8, 2, 2); feat_j[:, 1] = 1.0
    l_wrong, n_w = bank.pair_margin(feat_j, gt0, [(0, 1)], margin)
    wrong_pos = (n_w == 1 and float(l_wrong) > 0)               # relu(0.5-(0-1))=1.5
    # 미초기화 prototype 낀 쌍은 건너뜀
    _, n_skip = bank.pair_margin(feat_c, gt0, [(0, 3)], margin)  # class 3 미초기화
    skip_ok = (n_skip == 0)
    b2 = aligned_zero and wrong_pos and skip_ok
    lines.append(f"    (b-2) 정렬(c>j+m)→손실{float(l_aligned):.3f}(=0? {aligned_zero}) "
                 f"역정렬→손실{float(l_wrong):.3f}(>0? {wrong_pos}) "
                 f"미초기화쌍 skip={skip_ok} → {b2}")
    ok = b1 and b2
    return ok, lines


# ── (c) auto 혼동행렬 → top-k 선택 로직 ──────────────────────────────────────
def check_auto_topk():
    ok, lines = True, []
    # 합성 혼동행렬(4클래스): 대각 크게 + 특정 오분류 주입
    cm = np.array([
        [50, 30, 10,  0],   # GT0: 0.30이 →1, 0.10이 →2  (최상위 후보 0→1)
        [ 5, 40, 20,  0],   # GT1: →2 비율 20/65≈0.31   (0→1의 0.375 다음)
        [ 0,  0, 60,  0],   # GT2: 혼동 거의 없음
        [ 0,  0,  0,  0],   # GT3: 행합 0 → 후보 제외
    ], dtype=np.float64)
    pairs = P46.confusion_pairs_from_cm(cm, k=3)
    # 행정규화 후 비율: (0→1)=30/90=0.333, (1→2)=20/65=0.308, (0→2)=10/90=0.111
    expect = [(0, 1), (1, 2), (0, 2)]
    order_ok = pairs == expect
    # 대각 미포함
    no_diag = all(c != j for c, j in pairs)
    # 행합0 클래스(3) 미포함
    no_empty = all(c != 3 and j != 3 for c, j in pairs)
    # k 상한
    pairs2 = P46.confusion_pairs_from_cm(cm, k=1)
    k_ok = (len(pairs2) == 1 and pairs2[0] == (0, 1))
    ok = order_ok and no_diag and no_empty and k_ok
    lines.append(f"    top-3={pairs} (기대 {expect}? {order_ok}) 대각없음={no_diag} "
                 f"행합0제외={no_empty} k=1→{pairs2}({k_ok})")
    return ok, lines


# ── (d) 쌍 클래스 미출현 / 전부-ignore ⇒ NaN 없음 ────────────────────────────
def check_no_nan():
    ok, lines = True, []
    m = build(seed=33, cm={'ENABLE': True, 'PAIRS': [[5, 6], [10, 12]],
                           'MARGIN': 0.5, 'SOURCE': 'proto'})
    m.set_confusion_pairs(torch.tensor([[5, 6], [10, 12]], dtype=torch.long))
    m.train()
    m._current_epoch = 10
    cases = [('쌍 미출현(단일 class=3)', dict(single_class=3)),
             ('전부 ignore(255)', dict(all_ignore=True))]
    for name, kw in cases:
        x, y = make_batch(seed=1, **kw)
        torch.manual_seed(99)
        _, _, aux = m(x, True, gt_mask=y)
        cm_loss = aux.get('p46_confmargin', None)
        if cm_loss is None:
            fin, gfin = True, True                 # 쌍 c 미출현 → 항 없음(정상)
            val = 0.0
        else:
            val = float(cm_loss)
            fin = bool(torch.isfinite(cm_loss))
            if cm_loss.requires_grad and abs(val) > 0:
                m.zero_grad(set_to_none=True)
                cm_loss.backward()
                gfin = all(bool(torch.isfinite(p.grad).all())
                           for p in m.parameters() if p.grad is not None)
            else:
                gfin = True
        good = fin and gfin
        ok &= good
        lines.append(f"    {name}: cm={val:.4f} 유한={fin} grad유한={gfin} → {good}")
    return ok, lines


def main():
    ok = True
    for title, fn in [
        ("(a) off ⇒ byte-동일 / on ⇒ cm 버퍼 생성", check_off_byte_identical),
        ("(b) 명시 PAIRS ⇒ 손실 유한 + gradient + hinge 방향성", check_explicit_and_direction),
        ("(c) auto 혼동행렬 → top-k 선택 로직", check_auto_topk),
        ("(d) 쌍 미출현 / 전부-ignore ⇒ NaN 없음", check_no_nan),
    ]:
        print("=" * 96)
        print(title)
        sok, lines = fn()
        for ln in lines:
            print(ln)
        print(f"    → {'OK' if sok else 'FAIL'}\n")
        ok &= sok
    print("=" * 96)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
