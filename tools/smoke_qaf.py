#!/usr/bin/env python3
"""[P54-QAF] 품질 인지 융합 CPU 스모크 — 데이터·ckpt 불필요.

검사(전부 assert):
  (a) MODEL.QAF off + TRAIN.QAF off ⇒ 키 없는 baseline 과 state_dict 키·forward
      출력 byte-동일. self.fusion.quality_head / qaf_self_layer 미생성.
  (b) MODEL.QAF on ⇒ η̂ 형상·범위, key 마스크가 MASK_MODALS 의 key 에만 적용됨
      (비대상 모달 key 로짓 변화 정확히 0), η̂=0 이면 마스크 없음과 출력 동일(항등),
      가중 평균 η̂=0 이면 산술 평균과 동일.
  (c) 두 패스 학습 스텝 1회(작은 모델, 교사 없음/있음 둘 다): 손실 항 분리
      (clean→identity, degrade→quality) · grad 흐름 · held-out 연산자 미호출.
  (d) SEVERITY_SCHEDULE 단조(_qaf_sev_cap) + cap 적용(_qaf_apply_cap)이 결측 0 유지.
  (e) fp32 유지(half 입력에도 η̂ float32).

실행:  python tools/smoke_qaf.py
timm 부재 시 모델(a/c)만 SKIP, 순수 torch 항목(b/d/e)은 그대로 돈다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from semseg.datasets import degrade as D                          # noqa: E402
from semseg.datasets.degrade import Degrader                      # noqa: E402
from semseg.models.reliadino.fusion import ReliabilityGatedFusion  # noqa: E402

MODALS = ['img', 'depth', 'event', 'lidar']
NUM_CLASSES = 6
_FAILS = []


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ::  {extra}" if extra else ""))
    if not ok:
        _FAILS.append(name)


# ===========================================================================
# (b)/(e) 순수 융합 모듈 검사 — timm 불필요
# ===========================================================================
def _fusion(qaf_enable=True, key_mask=True, wmean=True, self_attn=True,
            dim=32, m=4):
    return ReliabilityGatedFusion(
        dim=dim, num_classes=NUM_CLASSES, num_modalities=m,
        num_layers=1, num_heads=4, mlp_ratio=1.0, aux_hidden=32,
        attn_bias=False, consistency_bias=False, gate_enable=False,
        veto_floor=False, calibrate=False,
        qaf_enable=qaf_enable, qaf_mask_modals=['depth', 'lidar'],
        qaf_self_attn=self_attn, qaf_key_mask=key_mask, qaf_weighted_mean=wmean,
        qaf_head_hidden=16, qaf_modal_names=MODALS)


class _ZeroQH(nn.Module):
    """η̂=0 스텁 품질 헤드(항등·산술평균 검사용)."""
    def forward(self, feats):
        B = feats[0].shape[0]
        h, w = feats[0].shape[-2:]
        M = len(feats)
        return {'eta_scalar': torch.zeros(B, M),
                'eta_token': torch.zeros(B, M, h, w),
                'presence_logit': torch.zeros(B, M)}


def test_b_qaf_forward():
    print("\n[b] MODEL.QAF on — η̂ 형상/범위 · key 마스크 국소 · η̂=0 항등")
    B, dim, h, w = 2, 32, 6, 6
    N = h * w
    torch.manual_seed(0)
    fus = _fusion(dim=dim).eval()
    feats = [torch.randn(B, dim, h, w) for _ in MODALS]

    pred = fus.quality_head(feats)
    check("(b) eta_scalar shape", tuple(pred['eta_scalar'].shape) == (B, 4))
    check("(b) eta_token shape", tuple(pred['eta_token'].shape) == (B, 4, h, w))
    check("(b) eta 범위 [0,1]",
          bool((pred['eta_scalar'] >= 0).all() and (pred['eta_scalar'] <= 1).all()
               and (pred['eta_token'] >= 0).all() and (pred['eta_token'] <= 1).all()))
    check("(b) mask_idx == depth,lidar(1,3)", fus.qaf_mask_idx == [1, 3],
          str(fus.qaf_mask_idx))

    # key 마스크 캡처: 공유 층 forward 를 래핑해 key_bias 를 가로챈다.
    captured = {}
    _orig = fus.layers[0].forward

    def _cap(x, kv, key_bias):
        captured['kb'] = key_bias
        return _orig(x, kv, key_bias)

    fus.layers[0].forward = _cap
    with torch.no_grad():
        fus(feats)
    fus.layers[0].forward = _orig
    kb = captured['kb']
    check("(b) key_bias 존재(마스크 활성)", kb is not None)
    if kb is not None:
        kb_m = kb.reshape(B, len(MODALS), N)              # concat 순서 = 모달 major
        eps = fus.qaf_eps
        for j in range(len(MODALS)):
            sl = kb_m[:, j]
            if j in fus.qaf_mask_idx:
                exp = torch.log((1.0 - pred['eta_token'][:, j].reshape(B, N)).clamp_min(eps))
                check(f"(b) modal[{j}] key 마스크 = log(1−η̂)",
                      torch.allclose(sl, exp, atol=1e-5),
                      f"max|Δ|={(sl - exp).abs().max().item():.2e}")
            else:
                check(f"(b) 비대상 modal[{j}] key 로짓 변화 0",
                      bool((sl == 0).all()), f"max|kb|={sl.abs().max().item():.2e}")

    # η̂=0 → 마스크 없음과 출력 동일(항등) + 가중 평균 = 산술 평균.
    torch.manual_seed(1)
    fus2 = _fusion(dim=dim).eval()
    fus2.quality_head = _ZeroQH()
    with torch.no_grad():
        out_maskon, _ = fus2(feats)
        fus2.qaf_key_mask = False
        out_maskoff, _ = fus2(feats)
    check("(b) η̂=0 → key 마스크 항등(마스크 on==off)",
          torch.allclose(out_maskon, out_maskoff, atol=1e-6),
          f"max|Δ|={(out_maskon - out_maskoff).abs().max().item():.2e}")
    with torch.no_grad():
        fus2.qaf_key_mask = True
        out_wmean, _ = fus2(feats)          # η̂=0 → wts 전부 1 → 산술 평균
        fus2.qaf_weighted_mean = False
        out_mean, _ = fus2(feats)           # sum/m
    check("(b) η̂=0 → 가중 평균 == 산술 평균",
          torch.allclose(out_wmean, out_mean, atol=1e-6),
          f"max|Δ|={(out_wmean - out_mean).abs().max().item():.2e}")


def test_e_fp32():
    print("\n[e] fp32 유지 — half 입력에도 η̂ float32")
    B, dim, h, w = 2, 32, 6, 6
    fus = _fusion(dim=dim).eval()
    feats = [torch.randn(B, dim, h, w).half() for _ in MODALS]
    pred = fus.quality_head(feats)
    check("(e) eta_scalar fp32", pred['eta_scalar'].dtype == torch.float32,
          str(pred['eta_scalar'].dtype))
    check("(e) eta_token fp32", pred['eta_token'].dtype == torch.float32,
          str(pred['eta_token'].dtype))


# ===========================================================================
# (d) 두 패스 헬퍼 검사 — train_reliadino 모듈에서 import
# ===========================================================================
def test_d_schedule():
    print("\n[d] SEVERITY_SCHEDULE 단조 + cap 적용(결측 0 유지)")
    try:
        from train_reliadino import _qaf_sev_cap, _qaf_apply_cap
    except Exception as e:
        check("(d) train_reliadino 헬퍼 import", False, f"{type(e).__name__}: {e}")
        return
    sched = [[0, 0.3], [10, 0.6], [20, 1.0]]
    caps = [_qaf_sev_cap(sched, e) for e in range(0, 40)]
    mono = all(caps[i + 1] >= caps[i] - 1e-9 for i in range(len(caps) - 1))
    check("(d) cap 단조 비감소", mono, f"caps[0,10,20]={caps[0]},{caps[10]},{caps[20]}")
    check("(d) cap 경계값", caps[0] == 0.3 and caps[10] == 0.6 and caps[20] == 1.0)

    # cap 적용: 결측(presence=0)은 0 유지, 존재-열화는 진폭 축소.
    B, M, H, W = 2, len(MODALS), 8, 8
    clean = [torch.randn(B, 3, H, W) for _ in MODALS]
    deg, lab = Degrader(cfg={'p_per_modal': 1.0}, seed=2)(clean, MODALS)
    out, lab2 = _qaf_apply_cap(clean, deg, lab, cap=0.5)
    ok_missing = True
    pres = lab['presence']
    for m in range(M):
        miss = pres[:, m] < 0.5
        if bool(miss.any()):
            ok_missing = ok_missing and bool((out[m][miss] == deg[m][miss]).all())
    check("(d) cap<1 결측 표본 0 유지", ok_missing)
    check("(d) cap<1 severity 스케일(≤ 원본)",
          bool((lab2['severity'] <= lab['severity'] + 1e-6).all()))


# ===========================================================================
# (a)/(c) 전체 모델 검사 — timm 필요
# ===========================================================================
def _base_cfg(img=64):
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
            'ROUTER': {'ENABLE': False},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
            'M2F': {'ENABLE': False},
            'P39': {'TRUNK_EXP': True, 'ARBITER': False, 'TRUNK_MODE': 'gated_mlp',
                    'TRUNK_HIDDEN': 64, 'VICREG': {'ENABLE': False}},
        },
        'DATASET': {'MODALS': list(MODALS)},
        'TRAIN': {'IMAGE_SIZE': [img, img]},
    }


def _build(cfg):
    from semseg.models.reliadino.model import build_reliadino
    return build_reliadino(cfg, NUM_CLASSES)


def _seeded(fn):
    torch.manual_seed(20260921)
    return fn()


def _inputs(B=1, S=64):
    g = torch.Generator().manual_seed(1234)
    return [torch.randn(B, 3, S, S, generator=g) for _ in MODALS]


def test_a_off_byte_identical():
    print("\n[a] MODEL.QAF off ⇒ baseline byte-동일 (state_dict 키 + forward 출력)")
    cfg_base = _base_cfg()                                  # QAF 키 없음
    cfg_off = _base_cfg()
    cfg_off['MODEL']['QAF'] = {'ENABLE': False, 'MASK_MODALS': ['depth', 'lidar']}
    m_base = _seeded(lambda: _build(cfg_base)).eval()
    m_off = _seeded(lambda: _build(cfg_off)).eval()
    kb = list(m_base.state_dict().keys())
    ko = list(m_off.state_dict().keys())
    check("(a) state_dict 키 동일", kb == ko,
          f"base={len(kb)} off={len(ko)} diff={set(kb) ^ set(ko)}")
    check("(a) quality_head 미생성", m_off.fusion.quality_head is None)
    check("(a) qaf_self_layer 미생성", m_off.fusion.qaf_self_layer is None)
    x = _inputs()
    with torch.no_grad():
        o_base, _ = m_base(x, True)
        o_off, _ = m_off(x, True)
    check("(a) forward 출력 byte-동일", torch.equal(o_base, o_off),
          f"max|Δ|={(o_base - o_off).abs().max().item() if o_base.shape == o_off.shape else 'shape≠'}")


def _qaf_cfg():
    cfg = _base_cfg()
    cfg['MODEL']['QAF'] = {'ENABLE': True, 'MASK_MODALS': ['depth', 'lidar'],
                           'SELF_ATTN': True, 'KEY_MASK': True,
                           'WEIGHTED_MEAN': True, 'HEAD_HIDDEN': 16}
    return cfg


def test_c_two_pass():
    print("\n[c] 두 패스 학습 스텝 — 손실 분리 · grad 흐름 · held-out 미호출 · KD")
    from train_reliadino import _qaf_kd_kl
    model = _seeded(lambda: _build(_qaf_cfg())).train()
    check("(c) quality_head 생성", model.fusion.quality_head is not None)
    x = _inputs()
    gt = torch.randint(0, NUM_CLASSES, (1, 64, 64))

    # clean 패스 (qaf_labels None) → identity 손실만
    _, _, aux_c = model(x, True, gt_mask=gt)
    check("(c) clean 패스 identity 손실 존재",
          'qaf_identity_loss' in aux_c and 'qaf_quality_loss' not in aux_c,
          f"keys={[k for k in aux_c if k.startswith('qaf')]}")

    # 열화 패스 (Degrader 라벨) → quality 손실만. held-out 미호출 감시.
    orig_g, orig_s = D.gaussian_noise, D.salt_pepper

    def boom(*a, **k):
        raise RuntimeError("held-out 함수가 학습 열화 패스에서 호출됨")

    D.gaussian_noise, D.salt_pepper = boom, boom
    heldout_ok = True
    try:
        deg, lab = Degrader(cfg={'p_per_modal': 1.0}, seed=0)([s.detach() for s in x], MODALS)
    except RuntimeError:
        heldout_ok = False
    finally:
        D.gaussian_noise, D.salt_pepper = orig_g, orig_s
    check("(c) 열화 패스 held-out 미호출", heldout_ok)

    _, d_logits_dummy, aux_d = model(deg, True, gt_mask=gt, qaf_labels=lab)
    check("(c) 열화 패스 quality 손실 존재",
          'qaf_quality_loss' in aux_d and 'qaf_identity_loss' not in aux_d,
          f"keys={[k for k in aux_d if k.startswith('qaf')]}")

    # 결합 backward → quality_head grad 흐름
    loss = aux_c['qaf_identity_loss'] + aux_d['qaf_quality_loss']
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in model.fusion.quality_head.parameters() if p.grad is not None)
    check("(c) quality_head grad 흐름(norm>0)", gnorm > 0, f"gnorm={gnorm:.3e}")

    # 교사 있는 경우: KD KL 유한 + grad 흐름
    model2 = _seeded(lambda: _build(_qaf_cfg())).train()
    teacher = _seeded(lambda: _build(_qaf_cfg())).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        t_logits = teacher(x, True)[0]
    d_logits, _, aux_d2 = model2(deg, True, gt_mask=gt, qaf_labels=lab)
    kd = _qaf_kd_kl(t_logits, d_logits, 2.0)
    check("(c) KD KL 유한", bool(torch.isfinite(kd)), f"kd={float(kd):.4f}")
    (aux_d2['qaf_quality_loss'] + 0.5 * kd).backward()
    g2 = sum(p.grad.abs().sum().item()
             for p in model2.fusion.quality_head.parameters() if p.grad is not None)
    check("(c) 교사 경로 quality_head grad 흐름", g2 > 0, f"g2={g2:.3e}")


def main():
    print("== smoke_qaf ==")
    test_b_qaf_forward()
    test_e_fp32()
    test_d_schedule()
    try:
        import timm  # noqa: F401
        _has_timm = True
    except Exception as e:
        print(f"\n[SKIP] timm 부재 — 모델(a/c) 스모크 건너뜀: {e}")
        _has_timm = False
    if _has_timm:
        test_a_off_byte_identical()
        test_c_two_pass()
    print()
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
