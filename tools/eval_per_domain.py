#!/usr/bin/env python3
"""
Per-domain (condition) DELIVER evaluation runner — reusable across P28/P29/P30/....

Why generic: it loads the *model's own* config (keeping its MODEL/QUALITY_GATE block,
so any LoRA_Sam_PXX builds correctly) and only overrides eval-specific fields
(dataset ROOT, per-condition CASE, resume off, PHYSAUG off, eval batch). It then
invokes val.py per (checkpoint, condition).

NOTE: uses `--macvi` (metrics path) because val.py's per-image visualization panel
crashes on 4-modal DELIVER (row-width mismatch). --macvi skips that viz but still
computes + prints the full per-class IoU table from GT (DELIVER always has GT).

Examples
--------
# P28 (two checkpoints, all 5 weather conditions, test split, GPU0)
python tools/eval_per_domain.py \
  --cfg configs/deliver/b200-deliver_rgbdel_P28_physaug.yaml \
  --ckpt ep178=ckpt_P28/test_epoch178_55.27_top1_checkpoint.pth \
  --ckpt ep100=ckpt_P28/epoch100_63.4_top1_checkpoint.pth \
  --dataset-root /ailab_mat2/dataset/DELIVER \
  --gpu 0 --out-dir ~/eval_P28_out

# P29 / P30 — identical call, just swap --cfg and --ckpt
python tools/eval_per_domain.py --cfg configs/deliver/b200-deliver_rgbdel_P30_physaug.yaml \
  --ckpt best=outputs/.../P30_best_checkpoint.pth \
  --dataset-root /ailab_mat2/dataset/DELIVER --gpu 1 --out-dir ~/eval_P30_out
"""
import argparse, copy, os, subprocess, sys
from pathlib import Path
import yaml

DEFAULT_CONDITIONS = "cloud,fog,night,rain,sun"

def build_eval_cfg(base, dataset_root, case, batch):
    cfg = copy.deepcopy(base)
    cfg.setdefault('MODEL', {})
    cfg['MODEL']['RESUME_ENABLE'] = False
    cfg['MODEL']['RESUME_PATH'] = ''
    ds = cfg.setdefault('DATASET', {})
    ds['ROOT'] = dataset_root
    ds['CASE'] = case                       # val.py: DELIVER filters by this condition
    if isinstance(ds.get('PHYSAUG'), dict):
        ds['PHYSAUG']['ENABLE'] = False     # clean eval (no train-time aug)
    cfg.setdefault('EVAL', {})['BATCH_SIZE'] = batch
    cfg.setdefault('TEST', {})['FILE'] = dataset_root
    return cfg

def grep_miou(log_path):
    val = None
    for ln in open(log_path, encoding='utf-8', errors='ignore'):
        if ln.startswith('mIoU:'):
            val = ln.split('mIoU:')[1].split()[0]
    return val

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True, help="base model config (its MODEL block is reused as-is)")
    ap.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH',
                    help="checkpoint as label=path; repeatable")
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--conditions', default=DEFAULT_CONDITIONS)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--repo', default='.', help="cwd for val.py (repo root)")
    ap.add_argument('--val-script', default='val.py')
    ap.add_argument('--no-macvi', action='store_true', help="disable --macvi (will hit viz bug on 4-modal)")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
    ckpts = []
    for spec in args.ckpt:
        if '=' not in spec:
            ap.error(f"--ckpt must be label=path, got: {spec}")
        lbl, path = spec.split('=', 1)
        ckpts.append((lbl, path))

    out = Path(args.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    with open(args.cfg) as f:
        base = yaml.safe_load(f)

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    summary = []
    for lbl, ckpt in ckpts:
        for case in conds:
            cfg = build_eval_cfg(base, args.dataset_root, case, args.batch)
            cfg_path = out / f"cfg_{lbl}_{case}.yaml"
            with open(cfg_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            log_path = out / f"{lbl}__{case}.log"
            cmd = [sys.executable, args.val_script, '--cfg', str(cfg_path),
                   '--mode', args.split, '--model_path', ckpt,
                   '--save_dir', str(out / f"masks_{lbl}_{case}")]
            if not args.no_macvi:
                cmd.append('--macvi')
            print(f"[eval] {lbl} / {case} -> {log_path.name}", flush=True)
            with open(log_path, 'w') as lf:
                rc = subprocess.call(cmd, cwd=args.repo, env=env, stdout=lf, stderr=subprocess.STDOUT)
            miou = grep_miou(log_path)
            print(f"   rc={rc}  mIoU={miou}", flush=True)
            summary.append((lbl, case, miou, rc))

    print("\n=== SUMMARY (per-domain mIoU) ===")
    for lbl, case, miou, rc in summary:
        print(f"{lbl:24s} {case:8s} mIoU={miou} rc={rc}")
    print(f"\nLogs in {out}.  Analyze with: tools/analyze_per_domain.py --logs-dir {out} ...")
    # 감사 2026-07-21: 조건별 실패(rc≠0 또는 mIoU 파싱 실패)가 exit 0으로
    # 삼켜져 파이프라인이 D1을 'ok'로 오기록하던 문제 — 실패 전파.
    n_fail = sum(1 for _, _, miou, rc in summary if rc != 0 or miou is None)
    if n_fail:
        print(f"[eval_per_domain] ⚠️ {n_fail}/{len(summary)} runs failed — exit 1")
        sys.exit(1)

if __name__ == '__main__':
    main()
