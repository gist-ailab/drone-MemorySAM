"""One-command detection analysis for any model in this repo.

Mirrors tools/seg_analysis_pipeline.py: stage-based, shells out to the individual
tools so each stage stays runnable on its own.

  D1  breakdown  overall mAP/mAP50/mAP75 + per-class AP + night/normal split
  D2  ablation   per-module toggle -> ΔmAP50 + agreement -> ACTIVE / NO-OP verdict

  python tools/det_analysis_pipeline.py --cfg <cfg> --ckpt <ckpt> \
      --out-dir analysis/P37a --gpu 0
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, log_path):
    print(f"$ {' '.join(cmd)}")
    with open(log_path, 'w') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f"  !! failed (rc={p.returncode}) — see {log_path}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--stages', default='D1,D2')
    ap.add_argument('--mode', default='val', choices=['val', 'test'])
    ap.add_argument('--toggles', default='auto')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    name = args.name or os.path.basename(args.cfg).replace('.yaml', '')
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    common = ['--cfg', args.cfg, '--ckpt', args.ckpt, '--mode', args.mode, '--name', name]
    if args.limit:
        common += ['--limit', str(args.limit)]
    t0 = time.time()
    rc = {}
    for st in stages:
        if st == 'D1':
            cmd = [sys.executable, os.path.join(HERE, 'det_eval_breakdown.py'),
                   *common, '--out', os.path.join(args.out_dir, f'{name}_breakdown')]
        elif st == 'D2':
            cmd = [sys.executable, os.path.join(HERE, 'det_module_ablation.py'),
                   *common, '--toggles', args.toggles,
                   '--out', os.path.join(args.out_dir, f'{name}_ablation')]
        else:
            print(f"  ?? unknown stage {st}"); continue
        os.environ.update(env)
        rc[st] = sh(cmd, os.path.join(args.out_dir, f'{name}_{st}.log'))
    print(f"\n[det-analysis-pipeline] {name}: {rc} in {time.time()-t0:.0f}s -> {args.out_dir}")


if __name__ == '__main__':
    main()
