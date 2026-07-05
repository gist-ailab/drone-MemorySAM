#!/usr/bin/env python3
"""
tools/eval_reliability_auroc.py — Phase 0 diagnostic for P32-B (CoRB).

Question it answers (roadmap 23_seg_arch_proposals_P32.md §3/§7, GATE #1):
  RBMA's reliability is currently self-entropy  rel_i = 1 - H(softmax(D_i(f_i)))/logC,
  whose per-modal correctness-AUROC is [img .77 / depth .62 / event .30 / lidar .22]
  (16_failure_analysis §7) — event/LiDAR are anti-calibrated (<0.5).

  P32-B replaces the signal's *meaning*: instead of self-confidence, measure how much
  each modality's per-pixel posterior is CORROBORATED by the leave-one-out consensus of
  the others (training-free). This tool measures the corroboration signal's AUROC and
  compares it head-to-head with self-entropy, per modality/condition, on EXISTING ckpts.

  GATE: if event/LiDAR corroboration-AUROC > 0.5 (self-entropy is <0.5), the signal is
  repaired -> GO to P32-B training. Else the signal needs redesign (temp/decoder capacity).

Signals compared (all training-free, same correctness target = per-modal argmax == GT):
  - selfent : 1 - H(p_i)/logC                                  (baseline, reproduces §7)
  - corr_bc : Bhattacharyya coefficient  Σ_c √(p_i · p̄_{-i})    ∈[0,1]  (agreement w/ consensus)
  - corr_js : 1 - JSD(p_i ‖ p̄_{-i})/log2                        ∈[0,1]
  where p̄_{-i} = mean_{j≠i} p_j  (leave-one-out consensus, uniform weights = 1st approx).

Reuses the exact protocol of tools/module_diagnostics.py (V.load_model, per-modal outputs
via core._last_per_modal_outputs, histogram-AUROC of reliability->correct). Model-agnostic:
works for any LoRA_Sam_PXX that exposes _last_per_modal_outputs (P26..P31).

Usage:
  python tools/eval_reliability_auroc.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --conditions cloud,fog,night,rain,sun \
    --max-imgs 120 --gpu 0 --out <prefix>
"""
import argparse, os, sys, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V

WR = 256  # common working resolution for per-pixel stats


def rs_nn(t, hw):   # nearest resize a (H,W) long/uint tensor -> (hw,hw)
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='nearest')[0, 0].long()


def rs_bl(t, hw):   # bilinear resize a (H,W) float tensor
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='bilinear', align_corners=False)[0, 0]


def auroc_from_hist(neg, pos):
    # neg[b], pos[b] = counts of incorrect/correct in signal-bin b (bins ascending).
    # Rank-AUROC via the standard mid-rank formula on the binned histogram.
    N, P = neg.sum(), pos.sum()
    if N == 0 or P == 0:
        return float('nan')
    below = 0.0
    auc = 0.0
    for b in range(len(neg)):
        auc += pos[b] * (below + 0.5 * neg[b])
        below += neg[b]
    return float(auc / (N * P))


def corroboration_signals(logits_list):
    """logits_list: list of M tensors (C,H,W) = per-modal decoder logits (native res, on device).
    Returns dict of per-modal signal lists (each entry a (H,W) tensor) + argmax preds:
      selfent  : 1 - H(p_i)/logC                                  (per-modal self-confidence)
      corr_bc  : Bhattacharyya coeff Σ√(p_i·p̄_{-i})               (corroboration, leave-one-out)
      corr_js  : 1 - JSD(p_i‖p̄_{-i})/log2
      corr_veto: unique-info veto blend  g_i·selfent_i + (1-g_i)·corr_bc_i
                 g_i = clamp(selfent_i - max_{j≠i} selfent_j, 0, 1)  (threshold-free: how much
                 MORE confident modality i is than the best OTHER → protect uniquely-confident
                 workhorse; roadmap §3 unique-info veto, soft form)
      corr_max : max(corr_bc_i, selfent_i)                        (per-pixel take-the-higher)
    corr_veto uses corr_bc as the corroboration base (matches P31 consistency_bias formulation)."""
    M = len(logits_list)
    C = logits_list[0].shape[0]
    logC = math.log(C)
    log2 = math.log(2.0)
    ps = [F.softmax(lg.float(), dim=0) for lg in logits_list]          # each (C,H,W)
    preds = [p.argmax(0) for p in ps]
    p_sum = torch.stack(ps, 0).sum(0)                                   # (C,H,W)
    se, bc, js = [], [], []
    for i in range(M):
        p = ps[i]
        H = -(p * (p + 1e-8).log()).sum(0)                             # (H,W)
        se.append((1.0 - H / logC).clamp(0, 1))
        if M >= 2:
            cons = (p_sum - p) / (M - 1)                               # leave-one-out consensus (C,H,W)
            cons = cons.clamp_min(0)
            bc_i = (p * cons).clamp_min(0).sqrt().sum(0)               # Bhattacharyya coeff Σ√(p·q) ∈[0,1]
            m = 0.5 * (p + cons)
            kl_pm = (p * ((p + 1e-8).log() - (m + 1e-8).log())).sum(0)
            kl_qm = (cons * ((cons + 1e-8).log() - (m + 1e-8).log())).sum(0)
            jsd = 0.5 * kl_pm + 0.5 * kl_qm                            # ∈[0, log2]
            js_i = 1.0 - jsd / log2
        else:
            bc_i = torch.ones_like(H)
            js_i = torch.ones_like(H)
        bc.append(bc_i.clamp(0, 1))
        js.append(js_i.clamp(0, 1))
    # unique-info veto blend + per-pixel max (need all se[] first)
    veto, mx = [], []
    for i in range(M):
        if M >= 2:
            others_max = torch.stack([se[j] for j in range(M) if j != i], 0).amax(0)
            g = (se[i] - others_max).clamp(0, 1)                       # unique-confidence gate ∈[0,1]
        else:
            g = torch.ones_like(se[i])
        veto.append(g * se[i] + (1.0 - g) * bc[i])
        mx.append(torch.maximum(bc[i], se[i]))
    return {'selfent': se, 'corr_bc': bc, 'corr_js': js,
            'corr_veto': veto, 'corr_max': mx}, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='cloud,fog,night,rain,sun')
    ap.add_argument('--max-imgs', type=int, default=120, help='cap images/condition')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root:
        ds_cfg['ROOT'] = args.dataset_root
    if isinstance(ds_cfg.get('PHYSAUG'), dict):
        ds_cfg['PHYSAUG']['ENABLE'] = False
    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(isz, dataset_cfg=ds_cfg)
    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model
    NB = 20  # signal bins

    SIGNALS = ['selfent', 'corr_bc', 'corr_js', 'corr_veto', 'corr_max']
    CORR_FAMILY = ['corr_bc', 'corr_js', 'corr_veto', 'corr_max']
    report = {'model': Path(args.model_path).stem, 'signals': SIGNALS, 'conditions': {}}
    # accumulate for the cross-condition mean
    agg = {s: None for s in SIGNALS}
    Mn = None
    modal_names = None

    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, 'test', transform, 'test', macvi=False, eval_day=False)
        C = len(dataset.CLASSES)
        ign = getattr(dataset, 'ignore_label', 255)
        n = min(args.max_imgs, len(dataset))
        hist = None  # per-signal: (M, NB, 2) neg/pos histogram
        for idx in range(n):
            images, label, _ = dataset[idx]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            with torch.no_grad():
                _ = model(imgs, multimask_output=True)
            pmo = getattr(core, '_last_per_modal_outputs', None)
            if pmo is None:
                raise RuntimeError("model exposes no _last_per_modal_outputs; not an RBMA model (P26+)")
            M = len(imgs)
            logits = [pmo[i][0].to(device).float() for i in range(M)]  # (C,H,W) each
            if Mn is None:
                Mn = M
                modal_names = getattr(core, 'modals', None) or [f'm{i}' for i in range(M)]
            if hist is None:
                hist = {s: np.zeros((M, NB, 2), np.int64) for s in SIGNALS}
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy()
            valid = gt != ign
            gg = gt[valid]
            sig_maps, preds = corroboration_signals(logits)
            for i in range(M):
                pred_i = rs_nn(preds[i], WR).cpu().numpy()
                cc = (pred_i[valid] == gg).astype(int)                 # per-modal correctness target
                for s in SIGNALS:
                    sv = rs_bl(sig_maps[s][i].detach(), WR).cpu().numpy()
                    rv = np.clip((sv[valid] * NB).astype(int), 0, NB - 1)
                    np.add.at(hist[s][i], (rv, cc), 1)
        # finalize condition: AUROC per signal per modality
        cond_out = {'n': n, 'M': Mn, 'modals': modal_names}
        for s in SIGNALS:
            cond_out[s] = [round(auroc_from_hist(hist[s][i, :, 0], hist[s][i, :, 1]), 3) for i in range(Mn)]
            agg[s] = hist[s] if agg[s] is None else agg[s] + hist[s]
        report['conditions'][cond] = cond_out
        line = '  '.join(f"{s}={cond_out[s]}" for s in SIGNALS)
        print(f"[relAUROC] {cond}: {line}", flush=True)

    # cross-condition mean (pooled histogram)
    report['mean'] = {'modals': modal_names}
    for s in SIGNALS:
        report['mean'][s] = [round(auroc_from_hist(agg[s][i, :, 0], agg[s][i, :, 1]), 3) for i in range(Mn)]
    # per-signal robustness across modalities: min (worst modality) + mean.
    # The whole point: NO modality should stay anti-calibrated -> rank forms by worst-modality AUROC.
    stats = {}
    for s in SIGNALS:
        v = report['mean'][s]
        stats[s] = {'min': round(min(v), 3), 'mean': round(sum(v) / len(v), 3)}
    report['signal_stats'] = stats
    best_form = max(SIGNALS, key=lambda s: stats[s]['min'])   # form that lifts the WORST modality highest
    report['best_form'] = best_form
    # gate verdict: does the best corroboration-family form lift event/lidar above 0.5?
    gate = {}
    for i, mname in enumerate(modal_names):
        best_corr = max(report['mean'][s][i] for s in CORR_FAMILY)
        gate[mname] = {
            'selfent': report['mean']['selfent'][i],
            'corr_best': round(best_corr, 3),
            'crosses_0.5': bool(best_corr > 0.5),
            'delta': round(best_corr - report['mean']['selfent'][i], 3),
        }
    report['gate'] = gate

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    print("\n[relAUROC] === cross-condition mean (modals=%s) ===" % modal_names)
    for s in SIGNALS:
        st = stats[s]
        print(f"  {s:9s}: {report['mean'][s]}   min={st['min']:.3f} mean={st['mean']:.3f}")
    print(f"[relAUROC] BEST FORM (max worst-modality AUROC) = {best_form}  "
          f"(min={stats[best_form]['min']:.3f}, mean={stats[best_form]['mean']:.3f})")
    print("[relAUROC] GATE (best corroboration-family vs self-entropy, per modality):")
    for mname, g in gate.items():
        flag = 'PASS>0.5' if g['crosses_0.5'] else 'fail'
        print(f"    {mname:8s} selfent={g['selfent']:.3f} -> corr={g['corr_best']:.3f} (Δ{g['delta']:+.3f}) [{flag}]")
    print(f"[relAUROC] wrote {args.out}.json")


if __name__ == '__main__':
    main()
