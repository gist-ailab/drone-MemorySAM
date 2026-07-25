#!/usr/bin/env python3
"""tools/smoke_p43.py — CPU synthetic smoke for [P43] PanopticDual.

Runs the REAL ReliaDINO (tiny random-init timm ViT, 64x64 inputs, 2 modalities,
5 classes, 8 queries) so the wiring under test is the wiring that ships. No GPU,
no dataset, no pretrained download; a few seconds.

Asserts, per the code-review pipeline ([[code-review-pipeline]]):
  A  forward + backward produce finite losses
  B  gradient reaches queries / decoder / mask-embed / mask-feature proj /
     lateral projections AND still reaches LoRA, fusion and the pixel head
  B' the mask loss ALONE reaches the shared trunk but NOT the pixel head —
     i.e. the two heads are genuinely independent (no residual coupling, the
     P38 failure this design exists to avoid)
  C  P43.M2F_HEAD=false + LATERAL=false is byte-identical to a config without a
     P43 block at all (same state_dict keys, allclose logits)
  D  the lambda(t) warmup schedule takes the specified values
  E  panoptic_inference emits a valid segment map + segments_info
  F  the P43 toggles resolve in tools/module_ablation.make_toggles

Usage:
  /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_p43.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# tools/module_ablation.py imports val.py, which imports the vendored SAM2.
sys.path.insert(0, str(REPO / 'semseg' / 'models' / 'sam2'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402

from semseg.models.reliadino import build_reliadino                 # noqa: E402

B, HW, K, NM = 2, 64, 5, 2
FAILURES = []


def check(name, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ''))
    if not cond:
        FAILURES.append(name)


def make_cfg(p43=None):
    cfg = {
        'DEVICE': 'cpu',
        'MODEL': {
            'NAME': 'ReliaDINO',
            'BACKBONE': 'smoke',
            # tiny random-init ViT: 12 blocks, dim 192, patch 16 -> 4x4 tokens
            'BACKBONE_TIMM': 'vit_tiny_patch16_224',
            'BACKBONE_FALLBACK': '',
            'PRETRAINED_BACKBONE': False,
            'LORA_R': 4,
            'FPN_DIM': 64,
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 8, 'MLP_RATIO': 2.0,
                       'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
                       'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False},
            'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': False},
            'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False},
            'CLASS_TOKEN': {'ENABLE': False},
            'M2F': {'ENABLE': False},
        },
        'DATASET': {'MODALS': ['img', 'lidar']},
        'TRAIN': {'IMAGE_SIZE': [HW, HW]},
    }
    if p43 is not None:
        cfg['MODEL']['P43'] = p43
    return cfg


P43_ON = {'M2F_HEAD': True, 'LATERAL': True, 'NUM_TAPS': 3, 'NUM_QUERIES': 8,
          'DEC_LAYERS': 3, 'DIM': 32, 'NUM_HEADS': 8, 'NUM_POINTS': 32,
          'LAMBDA': 1.0, 'LAMBDA_WARMUP_EP': 5, 'THING_IDS': [3, 4]}


def build(p43=None, seed=0):
    torch.manual_seed(seed)
    return build_reliadino(make_cfg(p43), K)


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(B, 3, HW, HW, generator=g) for _ in range(NM)]
    gt = torch.randint(0, K, (B, HW, HW), generator=g)
    gt[:, :4, :4] = 255                       # ignore region must be tolerated
    return x, gt


def grad_norm(p):
    return 0.0 if p.grad is None else float(p.grad.abs().sum())


def main():
    torch.set_num_threads(4)
    print("=" * 68)
    print("P43 PanopticDual — CPU synthetic smoke")
    print("=" * 68)

    # ── A/B: train step, finite losses, gradient coverage ───────────────────
    print("\n[A/B] training step: finite losses + gradient coverage")
    model = build(P43_ON)
    model.train()
    model._current_epoch = 5                  # lambda at full strength
    x, gt = inputs()
    logits, m_feat, aux = model(x, True, gt_mask=gt)
    loss_seg = F.cross_entropy(logits.float(), gt, ignore_index=255)
    p43_loss = aux['p43_mask_loss']
    check('A1 p43_mask_loss present in aux', 'p43_mask_loss' in aux)
    check('A2 losses finite',
          bool(torch.isfinite(loss_seg)) and bool(torch.isfinite(p43_loss)),
          f"seg={float(loss_seg):.4f} mask={float(p43_loss):.4f}")
    check('A3 logits shape == input res', tuple(logits.shape) == (B, K, HW, HW),
          str(tuple(logits.shape)))
    (loss_seg + p43_loss + aux.get('aux_ce', logits.new_zeros(()))).backward()

    probes = {
        'p43.query': model.p43.query,
        'p43.decoder(l0.cq)': model.p43.layers[0].cq.weight,
        'p43.mask_mlp[0]': model.p43.mask_mlp[0].weight,
        'p43.mask_feat_proj': model.p43.mask_feat_proj.weight,
        'p43.cls_head': model.p43.cls_head.weight,
        'p43.level_embed': model.p43.level_embed,
        'p43_lateral[0].conv': model.p43_lateral[0][1].weight,
        'p43_lateral[2].conv': model.p43_lateral[2][1].weight,
        # LoRA up-projections only: a_q/a_v are provably grad-free at step 0
        # (b is zero-init, so dL/dA = 0) — probing them would be a false alarm.
        'lora.b_q(block0)': model.encoder.lora_layers[0].b_q,
        'lora.b_v(block-1)': model.encoder.lora_layers[-1].b_v,
        'fusion.aux_decoders[1]': next(model.fusion.aux_decoders[1].parameters()),
        'pixel head.cls': model.head.cls.weight,
        'fpn.lateral[0]': model.fpn.lateral[0][0].weight,
    }
    for name, p in probes.items():
        check(f"B {name} receives grad", grad_norm(p) > 0,
              f"|g|={grad_norm(p):.3e}")
    n_nan = sum(1 for p in model.parameters()
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()))
    check('B* no non-finite gradients', n_nan == 0, f"{n_nan} bad tensors")

    # ── B': independence — mask loss must not touch the pixel head ─────────
    print("\n[B'] head independence (no residual coupling)")
    model.zero_grad(set_to_none=True)
    logits, _, aux = model(x, True, gt_mask=gt)
    aux['p43_mask_loss'].backward()
    check("B'1 mask loss reaches the shared trunk (LoRA)",
          grad_norm(model.encoder.lora_layers[0].b_q) > 0,
          f"|g|={grad_norm(model.encoder.lora_layers[0].b_q):.3e}")
    check("B'2 mask loss reaches the lateral projections",
          grad_norm(model.p43_lateral[0][1].weight) > 0)
    check("B'3 mask loss reaches the SimpleFPN trunk",
          grad_norm(model.fpn.lateral[0][0].weight) > 0)
    check("B'4 mask loss does NOT reach the pixel head classifier",
          grad_norm(model.head.cls.weight) == 0.0,
          f"|g|={grad_norm(model.head.cls.weight):.3e}")
    check("B'5 mask loss does NOT reach the pixel head trunk",
          grad_norm(model.head.fuse[0].weight) == 0.0)
    model.zero_grad(set_to_none=True)
    logits, _, aux = model(x, True, gt_mask=gt)
    F.cross_entropy(logits.float(), gt, ignore_index=255).backward()
    check("B'6 pixel loss does NOT reach the query decoder",
          grad_norm(model.p43.layers[0].cq.weight) == 0.0)

    # ── C: OFF config == baseline ──────────────────────────────────────────
    print("\n[C] P43 off == baseline (no P43 block)")
    base = build(None, seed=7)
    # M2F_HEAD:false ALONE must be the full-off config (LATERAL follows it) —
    # the spec's byte-identity requirement is stated on M2F_HEAD.
    off = build({'M2F_HEAD': False}, seed=7)
    kb, ko = set(base.state_dict().keys()), set(off.state_dict().keys())
    check('C1 state_dict keys identical', kb == ko,
          f"+{sorted(ko - kb)[:3]} -{sorted(kb - ko)[:3]}")
    check('C2 no P43 modules built',
          off.p43 is None and off.p43_lateral is None
          and len(off.encoder.tap_layers) == 0)
    base.eval(); off.eval()
    with torch.no_grad():
        ob, _ = base(x, True)
        oo, _ = off(x, True)
    check('C3 logits allclose', torch.allclose(ob, oo, atol=0, rtol=0),
          f"max|Δ|={float((ob - oo).abs().max()):.3e}")
    n_base = sum(p.numel() for p in base.parameters() if p.requires_grad)
    n_off = sum(p.numel() for p in off.parameters() if p.requires_grad)
    n_on = sum(p.numel() for p in build(P43_ON, seed=7).parameters() if p.requires_grad)
    check('C4 P43-off adds zero trainable params, P43-on adds some',
          n_off == n_base and n_on > n_base,
          f"base={n_base/1e6:.3f}M off={n_off/1e6:.3f}M on={n_on/1e6:.3f}M")
    lat_only = build({'M2F_HEAD': False, 'LATERAL': True, 'NUM_TAPS': 3}, seed=7)
    check('C5 LATERAL is independently togglable (no mask head)',
          lat_only.p43 is None and lat_only.p43_lateral is not None
          and lat_only.encoder.tap_layers == [2, 5, 8],
          f"taps={lat_only.encoder.tap_layers}")
    lat_only.eval()
    with torch.no_grad():
        o_lat, _ = lat_only(x, True)
    check('C6 lateral-only run is live and finite',
          bool(torch.isfinite(o_lat).all())
          and not torch.allclose(o_lat, ob, atol=1e-6))

    # ── C-AMP: the real training path is bf16 autocast ─────────────────────
    print("\n[C-AMP] bfloat16 autocast (TRAIN.AMP_DTYPE: bfloat16)")
    amp = build(P43_ON, seed=11)
    amp.train()
    amp._current_epoch = 3
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        lg, _, a2 = amp(x, True, gt_mask=gt)
        ls = F.cross_entropy(lg.float(), gt, ignore_index=255)
    ml = a2['p43_mask_loss']
    check('C-AMP1 mask loss stays fp32 under autocast', ml.dtype == torch.float32,
          str(ml.dtype))
    check('C-AMP2 losses finite under autocast',
          bool(torch.isfinite(ml)) and bool(torch.isfinite(ls)),
          f"seg={float(ls):.4f} mask={float(ml):.4f}")
    (ls + ml).backward()
    check('C-AMP3 grads finite under autocast',
          all(bool(torch.isfinite(p.grad).all())
              for p in amp.parameters() if p.grad is not None))

    # ── D: lambda warmup schedule ──────────────────────────────────────────
    print("\n[D] lambda(t) warmup 0.1 -> 1.0 over LAMBDA_WARMUP_EP=5")
    want = {0: 0.10, 1: 0.28, 2: 0.46, 5: 1.00, 12: 1.00}
    for ep, w in want.items():
        model._current_epoch = ep
        got = model._p43_lambda_now()
        check(f"D ep{ep} lambda == {w:.2f}", abs(got - w) < 1e-6, f"got {got:.4f}")
    model.p43_lambda = 0.5
    model._current_epoch = 99
    check('D scaled by LAMBDA (0.5)', abs(model._p43_lambda_now() - 0.5) < 1e-6)
    model.p43_lambda = 1.0

    # ── E: panoptic_inference ──────────────────────────────────────────────
    print("\n[E] panoptic_inference")
    model.eval()
    res = model.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0,
                                   size=(HW, HW))
    check('E1 one result per image', len(res) == B)
    pan, segs = res[0]
    check('E2 segment map shape/dtype', tuple(pan.shape) == (HW, HW)
          and pan.dtype == torch.int32, f"{tuple(pan.shape)} {pan.dtype}")
    ids = [s['id'] for s in segs]
    check('E3 segment ids unique and >0',
          len(ids) == len(set(ids)) and all(i > 0 for i in ids), f"{len(ids)} segments")
    check('E4 categories in range',
          all(0 <= s['category_id'] < K for s in segs))
    check('E5 isthing follows THING_IDS',
          all(s['isthing'] == (s['category_id'] in (3, 4)) for s in segs))
    check('E6 painted pixels belong to a declared segment',
          set(torch.unique(pan).tolist()) - {0} <= set(ids))
    sem = model.semantic_from_queries(x, size=(HW, HW))
    check('E7 semantic_from_queries shape', tuple(sem.shape) == (B, K, HW, HW),
          str(tuple(sem.shape)))
    check('E8 eval semantic path untouched by the head (SEM_SOURCE=pixel)',
          model.p43_sem_source == 'pixel')
    with torch.no_grad():
        o1, _ = model(x, True)
        model.p43_eval_head = True
        o2, _ = model(x, True)
        model.p43_eval_head = False
    check('E9 EVAL_HEAD does not change the semantic output',
          torch.allclose(o1, o2, atol=0, rtol=0),
          f"max|Δ|={float((o1 - o2).abs().max()):.3e}")

    # ── F: module_ablation toggles ─────────────────────────────────────────
    print("\n[F] tools/module_ablation toggles")
    from tools.module_ablation import make_toggles
    T = make_toggles(model)
    check('F1 p43_lateral_off registered', 'p43_lateral_off' in T, str(sorted(T)))
    check('F2 p43_m2f_off NOT registered when SEM_SOURCE=pixel '
          '(guaranteed-zero row would be misread as a dead module)',
          'p43_m2f_off' not in T)
    restore = T['p43_lateral_off']()
    with torch.no_grad():
        o_off, _ = model(x, True)
    restore()
    with torch.no_grad():
        o_on, _ = model(x, True)
    check('F3 p43_lateral_off actually changes the output (module is live)',
          not torch.allclose(o_on, o_off, atol=1e-6),
          f"max|Δ|={float((o_on - o_off).abs().max()):.3e}")
    check('F4 toggle restores exactly',
          torch.allclose(o_on, o1, atol=0, rtol=0))
    qmodel = build(dict(P43_ON, SEM_SOURCE='query'), seed=3)
    qmodel.eval()
    check('F5 p43_m2f_off registered when SEM_SOURCE=query',
          'p43_m2f_off' in make_toggles(qmodel))

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("RESULT: PASS — all checks green")
    return 0


if __name__ == '__main__':
    sys.exit(main())
