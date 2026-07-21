#!/usr/bin/env python3
"""
tools/adapter_health.py — MODEL-AGNOSTIC LoRA/adapter health probe (static, from checkpoint).

Fills the biggest gap in the seg analysis toolkit: "is the adapter actually adapting?".
Works on ANY checkpoint of BOTH model families with NO forward pass, NO GPU, NO base
weights, NO dataset — it reads the state_dict directly and does matrix math on CPU.

Covers:
  - Plain LoRA (`_LoRA_qkv`): keys `*.linear_a_q.weight` + `*.linear_b_q.weight`
    (and `_v`). Present in SAM2 P8/P9-style plain-LoRA blocks AND SAM3-RBMA
    (`sam3_lora_rbma.inject_plain_lora`, B init 0).
  - SoftMoE LoRA (`SoftMoE_LoRA_Layer`): keys `*.experts_a.{i}.weight`,
    `*.experts_b.{i}.weight`, `*.gate.weight` (SAM2 P8+ SoftMoE blocks).

For each injected qkv site it computes the effective delta-weight  dW = B @ A  and reports:
  - ||dW||_F                     : how much the frozen qkv is actually shifted
  - ||B||_F, ||A||_F             : B is init-0; ||B||~0 after training == dead adapter
  - dead flag                    : ||B||_F < --dead-thresh (default 1e-4)
  - ratio ||dW||/||W_base||      : ONLY if the frozen base qkv weight is in the ckpt
For MoE sites also: per-expert ||dW_e||, gate weight norm, and expert-usage spread
(coefficient of variation of per-expert ||dW||) to flag expert collapse.

Output: console table (tabulate if available) + a JSON summary with per-layer + aggregate
health, so the driver / logs can consume it.

Usage:
  python tools/adapter_health.py --ckpt <model.pth> [--out health.json] [--dead-thresh 1e-4]
"""
import argparse, json, sys, re
from pathlib import Path
import torch


def load_state(ckpt_path):
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # already a raw state_dict?
        if all(isinstance(v, torch.Tensor) for v in obj.values() if v is not None):
            return obj
    raise ValueError(f"could not find a state_dict inside {ckpt_path}")


def fro(t):
    return float(t.detach().float().norm().item()) if t is not None else float('nan')


def collect_plain_lora(sd):
    """Pair linear_a_{q,v}.weight with linear_b_{q,v}.weight by shared prefix."""
    sites = {}
    pat = re.compile(r'^(.*)\.linear_(a|b)_(q|v)\.weight$')
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            continue
        prefix, ab, qv = m.group(1), m.group(2), m.group(3)
        d = sites.setdefault(prefix, {})
        d[f'{ab}_{qv}'] = v
    out = []
    for prefix, d in sorted(sites.items()):
        for qv in ('q', 'v'):
            A, B = d.get(f'a_{qv}'), d.get(f'b_{qv}')
            if A is None or B is None:
                continue
            # A: (r, dim)  B: (dim, r)  -> dW: (dim, dim)
            dW = B.float() @ A.float()
            base = sd.get(f'{prefix}.qkv.weight')
            base_slice = None
            if base is not None and base.shape[0] % 3 == 0:
                dim = base.shape[0] // 3
                base_slice = base[0:dim] if qv == 'q' else base[2 * dim:3 * dim]
            out.append({
                'layer': f'{prefix}[{qv}]', 'type': 'plain_lora',
                'dW_norm': fro(dW), 'A_norm': fro(A), 'B_norm': fro(B),
                'rank': int(A.shape[0]),
                'ratio_to_base': (fro(dW) / fro(base_slice)) if base_slice is not None else None,
            })
    return out


def collect_softmoe(sd):
    """Pair experts_a.{i}.weight with experts_b.{i}.weight; also gate.weight."""
    sites = {}
    pat = re.compile(r'^(.*)\.experts_(a|b)\.(\d+)\.weight$')
    gate = {}
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            prefix, ab, i = m.group(1), m.group(2), int(m.group(3))
            sites.setdefault(prefix, {}).setdefault(i, {})[ab] = v
        gm = re.match(r'^(.*)\.gate\.weight$', k)
        if gm:
            gate[gm.group(1)] = v
    out = []
    for prefix, experts in sorted(sites.items()):
        per_expert = []
        for i in sorted(experts):
            A, B = experts[i].get('a'), experts[i].get('b')
            if A is None or B is None:
                continue
            dW = B.float() @ A.float()
            per_expert.append(fro(dW))
        if not per_expert:
            continue
        t = torch.tensor(per_expert)
        mean = float(t.mean()); std = float(t.std(unbiased=False))
        out.append({
            'layer': prefix, 'type': 'softmoe_lora',
            'n_experts': len(per_expert),
            'dW_norm_mean': mean,
            'dW_norm_per_expert': per_expert,
            'B_norm': None, 'A_norm': None, 'rank': None,
            'expert_cv': (std / mean) if mean > 0 else None,   # collapse if ~0
            'gate_norm': fro(gate.get(prefix)) if prefix in gate else None,
            'ratio_to_base': None,
            'dW_norm': mean,  # unified field for aggregate
        })
    return out


def collect_multimodal_lora(sd, modals=('img', 'depth', 'event', 'lidar')):
    """[P34 ReliaDINO] MultiModalLoRAQKV: batched per-modality params
    `*.qkv.a_q` (M,r,in) + `*.qkv.b_q` (M,attn,r) (and _v). dW_m = B[m]@A[m] —
    항목① '모달별 adapter 적응도'를 정적으로 직접 답한다 (per-modality per-site)."""
    sites = {}
    pat = re.compile(r'^(.*)\.(a|b)_(q|v)$')
    for k, v in sd.items():
        m = pat.match(k)
        if m is None or v.dim() != 3:
            continue
        prefix, ab, qv = m.group(1), m.group(2), m.group(3)
        sites.setdefault(prefix, {})[f'{ab}_{qv}'] = v
    out = []
    for prefix, d in sorted(sites.items()):
        for qv in ('q', 'v'):
            A, B = d.get(f'a_{qv}'), d.get(f'b_{qv}')  # (M,r,in), (M,attn,r)
            if A is None or B is None or A.shape[0] != B.shape[0]:
                continue
            M = A.shape[0]
            per_modal = [fro(B[m].float() @ A[m].float()) for m in range(M)]
            names = list(modals)[:M] + [f'mod{i}' for i in range(len(modals), M)]
            t = torch.tensor(per_modal)
            out.append({
                'layer': f'{prefix}[{qv}]', 'type': 'mm_lora',
                'dW_norm': float(t.mean()),
                'dW_norm_per_modality': {names[m]: round(per_modal[m], 4) for m in range(M)},
                'B_norm': float(B.float().norm()), 'A_norm': float(A.float().norm()),
                'rank': int(A.shape[1]),
                'modal_cv': float(t.std(unbiased=False) / t.mean()) if t.mean() > 0 else None,
                'ratio_to_base': None,
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None, help='write JSON summary here')
    ap.add_argument('--dead-thresh', type=float, default=1e-4,
                    help='||B||_F below this == dead adapter (B is init-0)')
    ap.add_argument('--modals', default='img,depth,event,lidar',
                    help="per-modality 라벨 순서 (DATASET.MODALS와 동일하게). "
                         "MUSES 3모달=img,lidar,event · MUSES 4모달=img,lidar,event,radar. "
                         "틀리면 dW가 엉뚱한 모달 이름으로 보고된다.")
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    plain = collect_plain_lora(sd)
    moe = collect_softmoe(sd)
    modals = tuple(m.strip() for m in args.modals.split(',') if m.strip())
    mm = collect_multimodal_lora(sd, modals=modals)
    layers = plain + moe + mm

    if not layers:
        print(f"[adapter_health] NO LoRA/adapter sites found in {args.ckpt}. "
              f"(keys scanned: linear_a/b_{{q,v}}, experts_a/b.{{i}}). "
              f"Model may be full-finetune or a family not covered.")
        summary = {'ckpt': str(args.ckpt), 'n_sites': 0, 'layers': []}
    else:
        # dead flag: plain via ||B||, moe via near-zero mean dW
        for L in layers:
            if L['type'] == 'plain_lora':
                L['dead'] = (L['B_norm'] < args.dead_thresh)
            elif L['type'] == 'mm_lora':
                L['dead'] = (L['B_norm'] < args.dead_thresh)
            else:
                L['dead'] = (L['dW_norm_mean'] < args.dead_thresh)
        dW = torch.tensor([L['dW_norm'] for L in layers])
        n_dead = sum(1 for L in layers if L['dead'])
        summary = {
            'ckpt': str(args.ckpt),
            'modals': list(modals),
            'n_sites': len(layers),
            'n_dead': n_dead,
            'dead_frac': n_dead / len(layers),
            'dW_norm_min': float(dW.min()), 'dW_norm_max': float(dW.max()),
            'dW_norm_mean': float(dW.mean()), 'dW_norm_median': float(dW.median()),
            'plain_lora_sites': len(plain), 'softmoe_sites': len(moe),
            'mm_lora_sites': len(mm),
            'layers': layers,
        }
        if mm:  # [항목①] 전 site 평균 per-modality ||dW|| — 모달별 적응 총량 헤드라인
            keys = list(mm[0]['dW_norm_per_modality'].keys())
            summary['mm_per_modality_dW_mean'] = {
                k: round(float(torch.tensor(
                    [L['dW_norm_per_modality'][k] for L in mm]).mean()), 4)
                for k in keys}
            print(f"[adapter_health] per-modality mean ||dW|| (mm_lora): "
                  f"{summary['mm_per_modality_dW_mean']}")
        # console report
        try:
            from tabulate import tabulate
            rows = [[L['layer'][-48:], L['type'].replace('_lora', ''),
                     f"{L['dW_norm']:.4g}",
                     (f"{L['B_norm']:.3g}" if L.get('B_norm') is not None else '-'),
                     (f"{L['ratio_to_base']:.3g}" if L.get('ratio_to_base') is not None else '-'),
                     (f"{L['expert_cv']:.2f}" if L.get('expert_cv') is not None else '-'),
                     'DEAD' if L['dead'] else 'ok'] for L in layers]
            print(tabulate(rows, headers=['layer', 'type', '||dW||', '||B||',
                                          'dW/W', 'expCV', 'health'], tablefmt='github'))
        except Exception:
            for L in layers:
                print(f"  {L['layer'][-48:]:50s} {L['type']:12s} "
                      f"||dW||={L['dW_norm']:.4g} {'DEAD' if L['dead'] else 'ok'}")
        print(f"\n[adapter_health] {len(layers)} sites | dead={n_dead} "
              f"({100*summary['dead_frac']:.0f}%) | "
              f"||dW|| min/med/max = {summary['dW_norm_min']:.3g}/"
              f"{summary['dW_norm_median']:.3g}/{summary['dW_norm_max']:.3g}")
        if n_dead:
            print(f"  ⚠️  {n_dead} adapter site(s) effectively unchanged from init "
                  f"(||B||<{args.dead_thresh}) — those layers are NOT adapting.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[adapter_health] wrote {args.out}")


if __name__ == '__main__':
    main()
