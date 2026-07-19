"""Did the module actually do anything? — detection module ablation.

The det counterpart of tools/module_ablation.py, same contract: a toggle registry
where every entry AUTO-SKIPS when its attribute is absent, so one command works on
any P-version (and on future ones without editing this file, as long as the module
follows the repo's naming).

Why this exists: four consecutive detection/seg mechanisms shipped as zero-init
residuals and turned out to be no-ops (P36 router, P37a CEFR, P37b classtoken,
P38 m2f). A module is only "working" if switching it off MOVES the predictions.

Verdict per toggle (mirrors the P39 pre-registered gate):
  NO-OP      |ΔmAP50| < --delta-thresh AND agreement > --agree-thresh
  ACTIVE     otherwise (report the sign: helping or hurting)

  python tools/det_module_ablation.py --cfg <cfg> --ckpt <ckpt> \
      --toggles auto --out analysis/P37a_ablation
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _det_common import (DEFAULT_LOWLIGHT_CLIPS, agreement, build_detector,  # noqa: E402
                         build_loader, detection_signature, eval_overall, load_cfg,
                         load_det_checkpoint, run_inference, split_by_clip, write_outputs)


def make_toggles(model):
    """Registry: name -> callable that disables one module and returns a restore fn.

    Each entry probes for its attribute and is skipped when absent, so this list is
    a superset across P-versions. `seg` is the ReliaDINO (or SAM2) backbone module.
    """
    seg = getattr(model, 'seg_model', model)
    fus = getattr(seg, 'fusion', None)
    m2f = getattr(seg, 'm2f', None)
    toggles = {}

    def _attr(name, obj, attr, value, is_param=False):
        if obj is None or not hasattr(obj, attr):
            return
        cur = getattr(obj, attr)
        if cur is None:
            return                  # module present but disabled -> nothing to toggle
        if is_param and not torch.is_tensor(cur):
            return                  # expected a Parameter/tensor; skip rather than crash
        def apply():
            old = getattr(obj, attr)
            if is_param:
                with torch.no_grad():
                    saved = old.detach().clone()
                    old.fill_(value) if old.numel() else None
                def restore():
                    with torch.no_grad():
                        getattr(obj, attr).copy_(saved)
            else:
                setattr(obj, attr, value)
                def restore():
                    setattr(obj, attr, old)
            return restore
        toggles[name] = apply

    # ── det-side seams (what the detection path actually reads) ─────────────────
    _attr('p36_router_det_off', seg, 'det_router_alpha', 0.0, is_param=True)
    _attr('p37b_classtoken_det_off', seg, 'det_classtoken_alpha', 0.0, is_param=True)
    # ── backbone / fusion mechanisms ────────────────────────────────────────────
    _attr('p36_router_off', fus, 'router_alpha', 0.0, is_param=True)
    _attr('p37a_cefr_off', getattr(fus, 'cefr', None), 'a', -20.0, is_param=True)  # sigma(a)~0
    _attr('attn_bias_off', fus, 'lambda1', 0.0, is_param=True)
    _attr('consistency_off', fus, 'lambda2', 0.0, is_param=True)
    # ── P38 / P39 query head ────────────────────────────────────────────────────
    _attr('p38_m2f_beta_off', m2f, 'beta', 0.0, is_param=True)
    _attr('p39_query_off', seg, 'p39_query_off', True)
    _attr('p39_trunkexp_off', seg, 'p39_trunkexp_off', True)
    _attr('p39_modalsrc_off', m2f, 'use_modal_src', False)     # V2 -> fused-only
    _attr('p39_anchored_off', m2f, 'anchored', False)          # V3 -> free queries only
    return toggles


def _verdict(d50, agr, dt, at):
    if abs(d50) < dt and agr > at:
        return 'NO-OP'
    return 'ACTIVE(+)' if d50 < 0 else 'ACTIVE(-)'   # off hurts => module helps


def _md(name, base, rows, dt, at) -> str:
    parts = [f'baseline mAP50 **{base["all"]["AP50"]:.4f}**']
    if base.get('night'):
        parts.append(f'night {base["night"]["AP50"]:.4f}')
    if base.get('normal'):
        parts.append(f'normal {base["normal"]["AP50"]:.4f}')
    L = [f'# Detection module ablation — {name}', '', ' · '.join(parts), '',
         f'NO-OP criterion: |ΔmAP50| < {dt} AND top-10 detection agreement > {at}', '',
         '| toggle (module OFF) | mAP50 | ΔmAP50 | Δnight | Δnormal | agreement | verdict |',
         '|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f'| {r["toggle"]} | {r["AP50"]:.4f} | {r["dAP50"]:+.4f} | '
                 f'{r.get("dNight", float("nan")):+.4f} | {r.get("dNormal", float("nan")):+.4f} | '
                 f'{r["agreement"]:.4f} | **{r["verdict"]}** |')
    L += ['', 'ACTIVE(+) = turning the module off *hurts* -> the module contributes.',
          'ACTIVE(-) = turning it off *helps* -> the module is a net negative.',
          'NO-OP = the module changes essentially nothing; it is dead weight.']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--toggles', default='auto', help='comma list, or "auto" for all available')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--lowlight-clips', default=','.join(DEFAULT_LOWLIGHT_CLIPS))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1,
                    help='evaluate every Nth image (spans all clips; use for cheap runs)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--delta-thresh', type=float, default=0.005, help='|ΔmAP50| no-op bound')
    ap.add_argument('--agree-thresh', type=float, default=0.99, help='agreement no-op bound')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, args.mode, args.workers)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    load_det_checkpoint(model, args.ckpt, dev)

    ann = cfg['DATASET'][f'ANNOTATION_{args.mode.upper()}']
    clips = [c for c in args.lowlight_clips.split(',') if c]

    def evaluate():
        preds, id2file = run_inference(model, ds, loader, cfg, dev, args.score_thresh, args.limit, args.stride)
        nid, mid = split_by_clip(id2file, clips)
        return ({'all': eval_overall(ann, preds, list(id2file)),
                 'night': eval_overall(ann, preds, nid) if nid else None,
                 'normal': eval_overall(ann, preds, mid) if mid else None},
                detection_signature(preds))

    base, base_sig = evaluate()
    print(f"[ablation] baseline mAP50 {base['all']['AP50']:.4f}")

    registry = make_toggles(model)
    want = sorted(registry) if args.toggles == 'auto' else \
        [t for t in args.toggles.split(',') if t]
    avail = [t for t in want if t in registry]
    skipped = [t for t in want if t not in registry]
    print(f"[ablation] available={avail} skipped={skipped}")

    rows = []
    for t in avail:
        restore = registry[t]()
        try:
            m, sig = evaluate()
        finally:
            restore()
        agr = agreement(base_sig, sig)
        row = {'toggle': t, 'AP50': m['all']['AP50'],
               'dAP50': m['all']['AP50'] - base['all']['AP50'], 'agreement': agr,
               'verdict': _verdict(m['all']['AP50'] - base['all']['AP50'], agr,
                                   args.delta_thresh, args.agree_thresh)}
        # split deltas only when that split exists in BOTH runs (a night-only or
        # normal-only dataset, or a --limit smoke, legitimately has just one).
        if base.get('night') and m.get('night'):
            row['dNight'] = m['night']['AP50'] - base['night']['AP50']
        if base.get('normal') and m.get('normal'):
            row['dNormal'] = m['normal']['AP50'] - base['normal']['AP50']
        rows.append(row)
        print(f"[ablation] {t}: mAP50 {row['AP50']:.4f} ({row['dAP50']:+.4f}) "
              f"agree {agr:.4f} -> {row['verdict']}")

    name = args.name or os.path.basename(args.cfg).replace('.yaml', '')
    write_outputs(args.out,
                  {'model': name, 'cfg': args.cfg, 'ckpt': args.ckpt, 'baseline': base,
                   'available': avail, 'skipped': skipped, 'results': rows,
                   'criterion': {'delta': args.delta_thresh, 'agreement': args.agree_thresh}},
                  _md(name, base, rows, args.delta_thresh, args.agree_thresh))


if __name__ == '__main__':
    main()
