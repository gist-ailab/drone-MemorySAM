"""[E3/C3-M] 센서별 클래스 prototype 합성 스모크 (CPU, tiny ViT).

카드 = .claude_logs/decisions/2026-09-07-daily-cycle-experiment-cards.md §1 "E3".
구현 = MODEL.P46.C3_PROTO.SRC: permodal (센서별 PrototypeBank M개) + AGREE_LAMBDA.

실행 (GPU 불필요, 합성 배치, 1분 이내):
    python tools/smoke_c3_permodal_e3.py

검사 항목 (지시문 §4 그대로)
  (a) SRC 기존값(mfeat/fused) ⇒ SRC 키 유무와 무관하게 state_dict 키·값 동일 +
      train p46_proto·eval logits |Δ|max==0 + permodal bank 미생성(byte-동일).
  (b) permodal ⇒ 센서별 손실이 각각 유한 / 각 센서 feature 경로에 gradient 도달 /
      encoder(LoRA) 파라미터 grad 도달 / bank 개수 = 센서 수.
  (c) AGREE_LAMBDA>0 ⇒ 손실이 (agree off 대비) 커지고 gradient 도달.
  (d) 클래스 미출현/전부-ignore 배치에서 NaN 없음.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semseg.models.reliadino import build_reliadino                  # noqa: E402

K = 25                                    # DELIVER 클래스 수
MODALS = ['img', 'depth', 'event', 'lidar']       # DELIVER 4센서
SIZE = 128                                # tiny ViT-16 -> 8x8 토큰
BS = 2


def base_cfg(c3_src=None, c3_feature='mfeat', agree=0.0, warmup=0):
    """smoke_p52의 tiny 구성 위에 C3_PROTO(SRC/AGREE_LAMBDA)만 얹은 것."""
    c3 = {'ENABLE': True, 'LAMBDA': 0.1, 'WARMUP_EP': warmup,
          'PIXELS': 128, 'EMA': 0.9, 'TEMPERATURE': 0.1,
          'FEATURE': c3_feature, 'AGREE_LAMBDA': agree}
    if c3_src is not None:
        c3['SRC'] = c3_src
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
        'P46': {'C3_PROTO': c3},
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


def _proto_and_eval(model, x, y, seed=99):
    """train p46_proto(스칼라) + eval logits — 확률 경로 고정."""
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


# ── (a) 기존값(mfeat/fused): SRC 키 유무 무관 byte-동일 + permodal 미생성 ──────
def check_existing_byte_identical():
    ok, lines = True, []
    for feat in ('mfeat', 'fused'):
        a = build(seed=1234, c3_src=None, c3_feature=feat)      # 기존 config(SRC 없음)
        b = build(seed=1234, c3_src=feat, c3_feature=feat)      # SRC 명시(현행값)
        ka, kb = sorted(a.state_dict()), sorted(b.state_dict())
        keys_same = ka == kb
        # 값 동일(같은 시드 → permodal 분기 미진입 → RNG 스트림 동일)
        vals_same = all(torch.equal(a.state_dict()[k], b.state_dict()[k])
                        for k in ka) if keys_same else False
        # permodal bank 미생성
        no_permodal = (a.p46_proto_permodal is None
                       and b.p46_proto_permodal is None
                       and a.p46_proto is not None)
        no_pm_keys = not any('p46_proto_permodal' in k for k in ka)
        # train p46_proto + eval logits |Δ|max == 0
        b.load_state_dict(a.state_dict())
        x, y = make_batch(seed=3)
        pa, la = _proto_and_eval(a, x, y)
        pb, lb = _proto_and_eval(b, x, y)
        proto_same = (pa is not None and abs(pa - pb) < 1e-6)
        logits_same = torch.equal(la, lb)
        good = (keys_same and vals_same and no_permodal and no_pm_keys
                and proto_same and logits_same)
        ok &= good
        lines.append(
            f"    [{feat}] keys={keys_same} vals={vals_same} "
            f"permodal미생성={no_permodal} permodal키없음={no_pm_keys} "
            f"proto|Δ|={abs(pa - pb):.2e} eval|Δ|max="
            f"{float((la - lb).abs().max()):.2e} → {good}")
    return ok, lines


# ── (b) permodal: 센서별 손실 유한 + 각 센서 feature/LoRA 경로 gradient 도달 ──
def check_permodal():
    ok, lines = True, []
    m = build(seed=77, c3_src='permodal')
    # bank 개수 = 센서 수
    n_bank = len(m.p46_proto_permodal)
    bank_ok = n_bank == len(MODALS)
    # encoder 출력(feats[i])을 붙잡아 retain_grad — 센서별 feature 경로 gradient 확인
    feats_ref = []

    def _hook(_mod, _inp, out):
        out.retain_grad()
        feats_ref.append(out)

    h = m.encoder.register_forward_hook(_hook)
    m.train()
    m._current_epoch = 10
    x, y = make_batch(seed=5)
    torch.manual_seed(99)
    _, _, aux = m(x, True, gt_mask=y)
    proto = aux['p46_proto']
    m.zero_grad(set_to_none=True)
    proto.backward()                     # p46_proto 항만 격리 backward
    h.remove()
    # 센서별 손실 스냅샷 유한
    snap = m._last_permodal_proto or {}
    snap_ok = (len(snap) == len(MODALS)
               and all(torch.isfinite(torch.tensor(v)) for v in snap.values()))
    # 각 센서 feature.grad 가 nonzero(= 센서별 경로에 gradient 도달)
    feat_grads = [(f.grad is not None and float(f.grad.abs().sum()) > 0)
                  for f in feats_ref[:len(MODALS)]]
    feat_ok = len(feat_grads) == len(MODALS) and all(feat_grads)
    # encoder(LoRA) 파라미터 grad 도달
    n_enc_grad = sum(1 for p in m.encoder.parameters()
                     if p.requires_grad and p.grad is not None
                     and float(p.grad.abs().sum()) > 0)
    enc_ok = n_enc_grad > 0
    proto_fin = bool(torch.isfinite(proto))
    ok = bank_ok and snap_ok and feat_ok and enc_ok and proto_fin
    lines.append(f"    bank 개수={n_bank}(={len(MODALS)}? {bank_ok}) | "
                 f"proto 유한={proto_fin}")
    lines.append(f"    센서별 손실 스냅샷 = "
                 + " ".join(f"{k}={v:.4f}" for k, v in snap.items()))
    lines.append(f"    센서별 feature.grad nonzero = "
                 + " ".join(f"{MODALS[i]}={feat_grads[i]}"
                            for i in range(len(feat_grads)))
                 + f" | encoder grad params={n_enc_grad}")
    return ok, lines


# ── (c) AGREE_LAMBDA>0: 손실 증가 + gradient 도달 ────────────────────────────
def check_agree():
    ok, lines = True, []
    m0 = build(seed=55, c3_src='permodal', agree=0.0)
    m1 = build(seed=55, c3_src='permodal', agree=0.1)
    m1.load_state_dict(m0.state_dict())          # 동일 가중치 → 순수 agree 효과
    x, y = make_batch(seed=9)
    for m in (m0, m1):
        m.train()
        m._current_epoch = 10
    torch.manual_seed(99)
    _, _, a0 = m0(x, True, gt_mask=y)
    torch.manual_seed(99)
    _, _, a1 = m1(x, True, gt_mask=y)
    p0, p1 = float(a0['p46_proto']), float(a1['p46_proto'])
    larger = p1 > p0                              # agree 항(비음)이 더해져 커진다
    agree_snap = float(m1._last_proto_agree)
    agree_pos = agree_snap > 0
    # gradient 도달
    m1.zero_grad(set_to_none=True)
    a1['p46_proto'].backward()
    n_grad = sum(1 for p in m1.encoder.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0)
    grad_ok = n_grad > 0
    ok = larger and agree_pos and grad_ok
    lines.append(f"    proto(agree=0)={p0:.4f} < proto(agree=0.1)={p1:.4f} "
                 f"→ {larger} | agree 스냅샷={agree_snap:.4f}(>0? {agree_pos})")
    lines.append(f"    agree backward encoder grad params={n_grad}(>0? {grad_ok})")
    return ok, lines


# ── (d) 클래스 미출현 / 전부-ignore 배치에서 NaN 없음 ────────────────────────
def check_no_nan():
    ok, lines = True, []
    m = build(seed=33, c3_src='permodal', agree=0.1)
    m.train()
    m._current_epoch = 10
    cases = [('단일 클래스(class=7)', dict(single_class=7)),
             ('전부 ignore(255)', dict(all_ignore=True))]
    for name, kw in cases:
        x, y = make_batch(seed=1, **kw)
        torch.manual_seed(99)
        _, _, aux = m(x, True, gt_mask=y)
        proto = aux.get('p46_proto', torch.zeros(()))
        fin = bool(torch.isfinite(proto))
        m.zero_grad(set_to_none=True)
        if proto.requires_grad and float(proto.detach().abs()) > 0:
            proto.backward()
            gfin = all(bool(torch.isfinite(p.grad).all())
                       for p in m.parameters() if p.grad is not None)
        else:
            gfin = True                            # 손실 0(그래프 없음) — grad 없음이 정상
        good = fin and gfin
        ok &= good
        lines.append(f"    {name}: p46_proto={float(proto):.4f} 유한={fin} "
                     f"grad유한={gfin} → {good}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')     # CPU 전용 스모크
    ap.parse_args()
    ok = True

    print("=" * 96)
    print("(a) SRC 기존값(mfeat/fused) — SRC 키 유무 무관 byte-동일 + permodal 미생성")
    aok, al = check_existing_byte_identical()
    for ln in al:
        print(ln)
    print(f"    → {'OK' if aok else 'FAIL'}")
    ok &= aok

    print("\n" + "=" * 96)
    print("(b) permodal — 센서별 손실 유한 + 각 센서 feature/LoRA gradient + bank 수")
    bok, bl = check_permodal()
    for ln in bl:
        print(ln)
    print(f"    → {'OK' if bok else 'FAIL'}")
    ok &= bok

    print("\n" + "=" * 96)
    print("(c) AGREE_LAMBDA>0 — 손실 증가 + gradient 도달")
    cok, cl = check_agree()
    for ln in cl:
        print(ln)
    print(f"    → {'OK' if cok else 'FAIL'}")
    ok &= cok

    print("\n" + "=" * 96)
    print("(d) 클래스 미출현 / 전부-ignore 배치 — NaN 없음")
    dok, dl = check_no_nan()
    for ln in dl:
        print(ln)
    print(f"    → {'OK' if dok else 'FAIL'}")
    ok &= dok

    print("\n" + "=" * 96)
    print(f"RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
