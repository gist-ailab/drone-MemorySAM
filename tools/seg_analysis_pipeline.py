#!/usr/bin/env python3
"""
tools/seg_analysis_pipeline.py — ONE model-agnostic driver for segmentation analysis.

Runs the four analysis dimensions over ANY DELIVER seg checkpoint (SAM2 family
P8..P33 AND SAM3-RBMA), auto-detecting the model family and which diagnostic hooks
the checkpoint actually exposes, then running each dimension with graceful skip +
a logged capability matrix so a blank panel is never mistaken for "covered".

Dimensions:
  D1 per-class / per-domain metrics   -> tools/eval_per_domain.py + tools/analyze_per_domain.py
  D2 per-modality encoder info        -> tools/viz_features.py (R2) + tools/module_diagnostics.py (B)
  D3 adapter / LoRA / router health   -> tools/adapter_health.py (static, always) + module_diagnostics (C/E/F)
  D4 post-fusion feature info         -> tools/viz_features.py (R2/R3/R5) + module_diagnostics (D)

Capability probe: builds the model, forwards ONE image, records which of
  _last_per_modal_feats / _last_per_modal_outputs / _last_uamm_spatial /
  _last_amf_weights / _last_moe_gates / _last_reliab
are non-None. Hook-dependent stages that would be blank are SKIPPED and the reason logged.
D1 and D3(adapter_health) are model-agnostic (GT metrics / static weights) and always run.

Usage:
  python tools/seg_analysis_pipeline.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --out-dir <dir> --gpu 0 \
    [--stages D1,D2,D3,D4] [--conditions cloud,fog,night,rain,sun] \
    [--viz-case sun --viz-contains RailTrack --viz-num 2] [--max-imgs 120] [--skip-per-domain]

Writes <out-dir>/report.md, <out-dir>/capability.json, and each stage's own artifacts.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / 'tools'

HOOKS = ['_last_per_modal_feats', '_last_per_modal_outputs', '_last_uamm_spatial',
         '_last_amf_weights', '_last_moe_gates', '_last_reliab', '_last_aux_logits']


def sh(cmd, log_path):
    """Run a subprocess, tee stdout/stderr to log_path, return (rc, tail)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    tail = ''
    try:
        tail = ''.join(open(log_path).readlines()[-15:])
    except Exception:
        pass
    return p.returncode, tail


def probe_capability(cfg_path, model_path, dataset_root, gpu):
    """Build model, forward 1 image, report family + which _last_* hooks are live."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    import yaml, torch
    sys.path.insert(0, str(REPO))
    import val as V
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    family = cfg.get('MODEL', {}).get('LORA_MODEL', '?')
    modals = cfg.get('DATASET', {}).get('MODALS', [])
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds = cfg['DATASET']
    if dataset_root:
        ds['ROOT'] = dataset_root
    if isinstance(ds.get('PHYSAUG'), dict):
        ds['PHYSAUG']['ENABLE'] = False
    cap = {'family': family, 'modals': modals, 'hooks': {h: False for h in HOOKS},
           'forward_ok': False, 'error': None}
    try:
        device = torch.device(cfg['DEVICE']); V.setup_cudnn()
        isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
        tf = V.get_val_augmentation(isz, dataset_cfg=ds)
        model = V.load_model(cfg, Path(model_path), device); model.eval()
        core = model.module if hasattr(model, 'module') else model
        ds['CASE'] = ds.get('CASE', 'sun')
        dset, _ = V.create_dataset(ds, 'test', tf, 'test', macvi=False, eval_day=False)
        images, _, _ = dset[0]
        imgs = [im.unsqueeze(0).to(device) for im in images]
        with torch.no_grad():
            model(imgs, multimask_output=True)
        cap['forward_ok'] = True
        for h in HOOKS:
            cap['hooks'][h] = getattr(core, h, None) is not None
    except Exception as e:
        cap['error'] = f'{type(e).__name__}: {e}'
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--stages', default='D1,D2,D3,D4')
    ap.add_argument('--conditions', default='cloud,fog,night,rain,sun')
    ap.add_argument('--max-imgs', type=int, default=120)
    ap.add_argument('--viz-case', default='sun')
    ap.add_argument('--viz-contains', default=None)
    ap.add_argument('--viz-num', type=int, default=2)
    ap.add_argument('--skip-per-domain', action='store_true',
                    help='D1: skip the heavy 5-condition eval, adapter_health/diagnostics still run')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stages = [s.strip().upper() for s in args.stages.split(',') if s.strip()]
    py = sys.executable
    results = {}  # stage -> {status, log, note}
    t0 = time.time()

    # ---- capability probe (drives graceful skip) ----
    print('[pipeline] probing model capability ...')
    cap = probe_capability(args.cfg, args.model_path, args.dataset_root, args.gpu)
    (out / 'capability.json').write_text(json.dumps(cap, indent=2))
    print(f"[pipeline] family={cap['family']} forward_ok={cap['forward_ok']} "
          f"live_hooks={[h for h, v in cap['hooks'].items() if v]}")
    if cap['error']:
        print(f"[pipeline] capability probe error (stages needing forward may fail): {cap['error']}")

    def run(stage, cmd, note=''):
        print(f"[pipeline] {stage}: {' '.join(str(c) for c in cmd)}")
        rc, tail = sh(cmd, out / f'{stage}.log')
        results[stage] = {'status': 'ok' if rc == 0 else f'rc={rc}',
                          'log': f'{stage}.log', 'note': note, 'tail': tail}

    def skip(stage, reason):
        print(f"[pipeline] {stage}: SKIP ({reason})")
        results[stage] = {'status': 'skipped', 'note': reason}

    # ---- D3a: adapter_health (ALWAYS, static, model-agnostic) ----
    if 'D3' in stages:
        run('D3_adapter_health',
            [py, str(TOOLS / 'adapter_health.py'), '--ckpt', args.model_path,
             '--out', str(out / 'adapter_health.json')],
            note='static LoRA dW/dead-adapter — family-agnostic, no forward')

    # ---- D1: per-class + per-domain ----
    if 'D1' in stages:
        if args.skip_per_domain:
            skip('D1_per_domain', 'user requested --skip-per-domain')
        else:
            ed = out / 'per_domain'
            run('D1_eval_per_domain',
                [py, str(TOOLS / 'eval_per_domain.py'), '--cfg', args.cfg,
                 '--ckpt', f'best={args.model_path}',
                 *(['--dataset-root', args.dataset_root] if args.dataset_root else []),
                 '--gpu', args.gpu, '--out-dir', str(ed)],
                note='5-condition per-class IoU (GT-based, model-agnostic)')
            if results['D1_eval_per_domain']['status'] == 'ok':
                run('D1_analyze_per_domain',
                    [py, str(TOOLS / 'analyze_per_domain.py'),
                     '--logs-dir', str(ed), '--out', str(out / 'per_domain_analysis.md')],
                    note='per-domain x per-class matrix + failure-class classification')

    # ---- D2/D3/D4 numeric: module_diagnostics (needs SAM2-style hooks) ----
    needs = ['_last_per_modal_outputs', '_last_uamm_spatial']
    if any(s in stages for s in ('D2', 'D3', 'D4')):
        if not cap['forward_ok']:
            skip('module_diagnostics', 'forward failed in capability probe')
        elif not any(cap['hooks'][h] for h in needs):
            skip('module_diagnostics',
                 f'no per-modal/UAMM hooks live (family={cap["family"]}); '
                 f'reliability-only families expose only _last_reliab')
        else:
            run('module_diagnostics',
                [py, str(TOOLS / 'module_diagnostics.py'), '--cfg', args.cfg,
                 '--model_path', args.model_path,
                 *(['--dataset-root', args.dataset_root] if args.dataset_root else []),
                 '--conditions', args.conditions, '--max-imgs', str(args.max_imgs),
                 '--gpu', args.gpu, '--out', str(out / 'module_diag')],
                note='B modal-competence, C reliability-AUROC, D UAMM alloc, E drop-dMIoU, F MoE')

    # ---- D2/D4 visual: viz_features (needs per-modal/fused hooks) ----
    if any(s in stages for s in ('D2', 'D4')):
        if not cap['forward_ok']:
            skip('viz_features', 'forward failed in capability probe')
        elif not (cap['hooks']['_last_per_modal_feats'] or cap['hooks']['_last_reliab']
                  or cap['hooks']['_last_uamm_spatial']):
            skip('viz_features', f'no per-modal/fused/reliability hooks live (family={cap["family"]})')
        else:
            run('viz_features',
                [py, str(TOOLS / 'viz_features.py'), '--cfg', args.cfg,
                 '--model_path', args.model_path,
                 *(['--dataset-root', args.dataset_root] if args.dataset_root else []),
                 '--case', args.viz_case,
                 *(['--contains', args.viz_contains] if args.viz_contains else []),
                 '--num', str(args.viz_num), '--gpu', args.gpu,
                 '--out-dir', str(out / 'viz')],
                note='per-modal encoder PCA (R2), reliability maps (R3), UAMM weights (R5)')

    # ---- consolidated report ----
    dt = time.time() - t0
    lines = [
        f"# Seg analysis pipeline report",
        f"- model: `{Path(args.model_path).name}`  cfg: `{Path(args.cfg).name}`",
        f"- family: **{cap['family']}**  modals: {cap['modals']}",
        f"- forward_ok: {cap['forward_ok']}  live hooks: "
        f"{[h for h, v in cap['hooks'].items() if v] or 'none'}",
        f"- wall: {dt:.0f}s   stages requested: {stages}",
        "",
        "## Capability matrix (which hooks the checkpoint exposes)",
        "| hook | live |", "|---|---|",
        *[f"| `{h}` | {'✅' if v else '—'} |" for h, v in cap['hooks'].items()],
        "",
        "## Stage results",
        "| stage | status | note |", "|---|---|---|",
        *[f"| {s} | {r['status']} | {r.get('note', '')} |" for s, r in results.items()],
        "",
        "## Key artifacts",
        "- `capability.json` — family + live-hook map",
        "- `adapter_health.json` — per-layer LoRA ||dW||, dead-adapter flags (D3, always)",
        "- `per_domain_analysis.md` — per-domain × per-class IoU + failure classes (D1)",
        "- `module_diag.json` — modal-competence / reliability-AUROC / UAMM alloc / drop-dMIoU (D2-4)",
        "- `viz/panel_*.png` — per-modal encoder + fused + reliability + UAMM panels (D2/D4)",
        "",
        "## How to read (what each dimension answers)",
        "- **D1** per-class/per-domain: which classes drop, whether it's domain shift (GT-based, any model).",
        "- **D2** per-modality encoder: what each img/depth/event/lidar encoder captures (PCA + per-class recall).",
        "- **D3** adapter health: `adapter_health.json` is model-agnostic (||dW||/dead per LoRA site); "
        "module_diag C/E/F add reliability-AUROC + drop-ΔmIoU (dead-modality) + MoE usage.",
        "- **D4** post-fusion: fused-feature PCA, reliability(1−H) maps, UAMM allocation + misallocation.",
    ]
    (out / 'report.md').write_text('\n'.join(lines))
    print(f"\n[pipeline] done in {dt:.0f}s -> {out/'report.md'}")
    for s, r in results.items():
        print(f"  {s:26s} {r['status']}")


if __name__ == '__main__':
    main()
