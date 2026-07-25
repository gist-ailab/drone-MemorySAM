#!/usr/bin/env python
"""Train/eval split leakage check — certification "학습·평가 분리" proof.

Proves the train and eval (test) COCO annotations are disjoint, so the certified
eval numbers carry no training-data leakage. Checks two levels:

  1. frame-level : no image file appears in both splits (the hard leakage test)
  2. clip-level  : no capture session (video clip) is shared between the splits
     — the stronger "no adjacent-frame leakage" guarantee, since consecutive
       video frames are near-duplicates; a clip-holdout keeps whole sessions on
       one side only.

No model, no GPU — pure annotation comparison, deterministic.

  python certification/check_split.py --cfg configs/det/det_D1_vitsp_jarvis.yaml
  python certification/check_split.py --train <train.json> --test <test.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

GRN, RED, YEL, BOLD, RST = '\033[32m', '\033[31m', '\033[33m', '\033[1m', '\033[0m'


def _clip(fname: str) -> str:
    # poongsan file_name = "capture_YYYYMMDD_hhmmss/rgb/xxx.png" -> clip = 1st part
    return fname.replace('\\', '/').split('/')[0]


def _load(path: str):
    with open(path) as f:
        d = json.load(f)
    imgs = d.get('images', [])
    files = [i['file_name'] for i in imgs]
    return {
        'path': path,
        'n_img': len(imgs),
        'n_box': len(d.get('annotations', [])),
        'files': set(files),
        'ids': [int(i['id']) for i in imgs],
        'clips': Counter(_clip(f) for f in files),
        'cats': {c['id']: c['name'] for c in d.get('categories', [])},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', default=None, help='read ANNOTATION_TRAIN/VAL from this det config')
    ap.add_argument('--train', default=None, help='train COCO json (overrides --cfg)')
    ap.add_argument('--test', default=None, help='test/eval COCO json (overrides --cfg)')
    ap.add_argument('--data-root', default=None, help='rebase the config paths to this mount')
    ap.add_argument('--max-list', type=int, default=10, help='how many overlaps to print')
    args = ap.parse_args()

    train, test = args.train, args.test
    if (train is None or test is None) and args.cfg:
        import yaml
        with open(args.cfg) as f:
            c = yaml.safe_load(f)
        tr = c['DATASET']['ANNOTATION_TRAIN']
        te = c['DATASET']['ANNOTATION_VAL']
        if args.data_root:
            r = args.data_root.rstrip('/')
            tr = f'{r}/_final_ann/{os.path.basename(tr)}'
            te = f'{r}/_final_ann/{os.path.basename(te)}'
        train, test = train or tr, test or te
    if not train or not test:
        ap.error('need --train + --test, or --cfg')
    for p in (train, test):
        if not os.path.exists(p):
            ap.error(f'annotation not found: {p}')

    tr, te = _load(train), _load(test)

    # ---- overlaps -------------------------------------------------------------
    file_ov = sorted(tr['files'] & te['files'])
    clip_ov = sorted(set(tr['clips']) & set(te['clips']))
    id_ov = sorted(set(tr['ids']) & set(te['ids']))       # informational only

    print(f"\n{BOLD}╔═══ Train / Eval split leakage check ═══╗{RST}")
    print(f"  {BOLD}TRAIN{RST}: {tr['n_img']:>6d} images  {tr['n_box']:>6d} boxes  "
          f"{len(tr['clips'])} clips  {os.path.basename(train)}")
    for c, n in sorted(tr['clips'].items()):
        print(f"         {c}: {n}")
    print(f"  {BOLD}EVAL {RST}: {te['n_img']:>6d} images  {te['n_box']:>6d} boxes  "
          f"{len(te['clips'])} clips  {os.path.basename(test)}")
    for c, n in sorted(te['clips'].items()):
        print(f"         {c}: {n}")

    print(f"\n  {BOLD}── overlap ──{RST}")
    print(f"  frame-level (shared image files) : {len(file_ov)}")
    for f in file_ov[:args.max_list]:
        print(f"       {RED}LEAK{RST} {f}")
    if len(file_ov) > args.max_list:
        print(f"       … +{len(file_ov) - args.max_list} more")
    print(f"  clip-level  (shared capture clips): {len(clip_ov)}  "
          f"{('-> ' + ', '.join(clip_ov)) if clip_ov else '(disjoint)'}")
    print(f"  image_id collisions (informational; ids are per-split): {len(id_ov)}")

    # ---- category consistency (both splits must share the class space) --------
    same_cats = tr['cats'] == te['cats']
    print(f"  categories : train {len(tr['cats'])} / eval {len(te['cats'])}  "
          f"{'identical' if same_cats else RED + 'MISMATCH' + RST}")

    # ---- verdict --------------------------------------------------------------
    frame_ok = len(file_ov) == 0
    clip_ok = len(clip_ov) == 0
    ok = frame_ok and same_cats
    print(f"\n  {BOLD}Verdict{RST}: "
          + (f"{GRN}PASS{RST}" if ok else f"{RED}FAIL{RST}")
          + f"  — frame-disjoint={frame_ok}, clip-disjoint={clip_ok}, cats-consistent={same_cats}")
    if frame_ok and clip_ok:
        print(f"  {GRN}Clean capture-holdout: whole video clips go entirely to train OR eval,{RST}")
        print(f"  {GRN}so there is no frame-level and no adjacent-frame leakage.{RST}")
    elif frame_ok and not clip_ok:
        print(f"  {YEL}Frame-disjoint but clips are shared — adjacent-frame near-duplicates{RST}")
        print(f"  {YEL}may exist across the split. Certification usually wants clip-holdout.{RST}")
    else:
        print(f"  {RED}LEAKAGE: {len(file_ov)} images appear in both splits.{RST}")

    out = {'train': train, 'test': test,
           'train_images': tr['n_img'], 'test_images': te['n_img'],
           'train_clips': sorted(tr['clips']), 'test_clips': sorted(te['clips']),
           'frame_overlap': len(file_ov), 'frame_overlap_files': file_ov[:100],
           'clip_overlap': clip_ov, 'image_id_collisions': len(id_ov),
           'categories_consistent': same_cats,
           'verdict': 'PASS' if ok else 'FAIL',
           'clip_holdout': frame_ok and clip_ok}
    op = os.environ.get('SPLIT_CHECK_OUT', 'split_check.json')
    with open(op, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  report: {op}{BOLD}╚════════════════════════════════════════╝{RST}\n")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
