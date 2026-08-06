#!/usr/bin/env python3
"""tools/eval_pq.py — panoptic quality (PQ/SQ/RQ) for ReliaDINO checkpoints.

Runs the active mask-classification head (P43 MaskClsHead, else the P38/P39
MaskQueryLiteHead that the P39.1-rank production checkpoints carry) through
`model.panoptic_inference()`, writes the predictions in the **MUSES AUPQ
format** (see tools/pq_format.py for the format's provenance, line by line),
and scores them against the panoptic GT.

Geometry (MUSES) — this is the part that silently invalidates PQ if it is wrong:
  semseg/datasets/muses.py letterboxes every 1080x1920 frame to a 1920^2 square
  (label padded with ignore=255) and the val transform then resizes that square
  to EVAL.IMAGE_SIZE (1024). The trainer's mIoU is computed in that letterboxed
  1024^2 space, but the panoptic GT PNGs are native 1080x1920. So the default
  `--geometry native` inverts the letterbox on the MASK LOGITS:
      stride-4 logits -> bilinear to 1024^2 -> crop the pad band
      (tools/eval_muses_official.letterbox_valid_box, round-trip proven there)
      -> bilinear to 1080x1920 -> sigmoid -> argmax/0.5 threshold.
  Resampling before the threshold matters: upsampling an already-binarised
  segment map changes which query owns a pixel and shreds thin segments.
  `--geometry letterbox` keeps the 1024^2 working space; it CANNOT be scored
  against native GT, so that mode only dumps predictions.

MUSES test is not supported: the benchmark withholds test GT (see
semseg/datasets/muses.py), so `--split test` is refused rather than faked.

Example:
  python tools/eval_pq.py \
    --cfg configs/jarvis-muses_rgbel_P39_1_rank_seed2.yaml \
    --model_path outputs/ReliaDINO/.../epochNNN_..._checkpoint.pth \
    --dataset muses --split val --out /path/to/pq_out --gpu 0
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --gpu must land before torch initialises CUDA (same convention as
# tools/eval_muses_official.py / tools/module_diagnostics.py).
if '--gpu' in sys.argv:
    _gi = sys.argv.index('--gpu')
    if _gi + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_gi + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import yaml                                                         # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402
from tqdm import tqdm                                               # noqa: E402

from semseg.augmentations_mm import get_val_augmentation            # noqa: E402
from semseg.models.reliadino.model import build_reliadino           # noqa: E402
from tools import pq_format                                         # noqa: E402
from tools.eval_muses_official import letterbox_valid_box           # noqa: E402

# suffixes that decorate the same scene id across MUSES folders; stripped to
# get the join key between our val list and the panoptic GT.
_SUFFIXES = ('_frame_camera', '_gt_panoptic', '_panoptic', '_gt_labelTrainIds',
             '_gt_labelIds', '_gt_semantic', '_leftImg8bit', '_rgb')


def base_key(name: str) -> str:
    stem = Path(name).name
    for ext in ('.png', '.jpg', '.jpeg'):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if stem.endswith(suf):
                stem, changed = stem[:-len(suf)], True
    return stem


def load_model(cfg, ckpt_path, device, n_classes):
    model = build_reliadino(cfg, n_classes)
    # weights_only=False: our ckpts carry optimizer/scaler state (torch>=2.6
    # would otherwise refuse them).
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise SystemExit(f"state_dict mismatch: missing={msg.missing_keys[:5]} "
                         f"unexpected={msg.unexpected_keys[:5]}")
    return model.to(device).eval(), ck


def build_dataset(name, cfg, split, root_override):
    dcfg, ecfg = cfg['DATASET'], cfg['EVAL']
    if root_override:
        dcfg['ROOT'] = root_override
    tf = get_val_augmentation(ecfg['IMAGE_SIZE'], dataset_cfg=dcfg)
    if name == 'muses':
        from semseg.datasets.muses import MUSES
        return MUSES(dcfg['ROOT'], split, tf, dcfg['MODALS'], return_meta=True,
                     proj_dir=dcfg.get('PROJ_DIR', 'projected_to_rgb')), dcfg
    from semseg.datasets.deliver import DELIVER
    return DELIVER(dcfg['ROOT'], split, tf, dcfg['MODALS'], return_meta=True), dcfg


def _match_folder(gt_json, gt_folder):
    """json의 file_name 이 기준으로 삼는 base 디렉터리를 고른다.

    MUSES 배포 val.json 은 split 접두("val/clear/...")가 붙은 경로를
    gt_panoptic/ 기준으로 담고 있는데, gt_folder 는 이미 gt_panoptic/val 을
    가리킬 수 있어 그대로 조인하면 split 이 중복된다("val/val/...").
    어느 규약인지 가정하지 말고 **실제 항목 하나로 검증**한다.
    """
    anns = gt_json.get('annotations') or []
    if not anns:
        return gt_folder
    name = anns[0].get('file_name', '')
    if (gt_folder / name).exists():
        return gt_folder
    if (gt_folder.parent / name).exists():
        return gt_folder.parent
    return gt_folder


def resolve_gt(args, root, split, keys):
    """Return (gt_json, gt_folder) or (None, None).

    Prefers the json shipped with the dataset; `--build-gt-json` derives one
    from the PNGs under the documented Cityscapes convention (self-validating —
    see pq_format.build_gt_json_from_pngs)."""
    gt_folder = Path(args.gt_folder) if args.gt_folder else None
    if gt_folder is None:
        for cand in (Path(root) / 'gt_panoptic' / split, Path(root) / 'gt_panoptic'):
            if cand.is_dir():
                gt_folder = cand
                break
    if args.gt_json:
        with open(args.gt_json) as f:
            gt_json = json.load(f)
        if gt_folder is None:
            gt_folder = Path(args.gt_json).with_suffix('')
        return gt_json, gt_folder
    if gt_folder is None or not gt_folder.is_dir():
        return None, None
    # a shipped json next to / above the folder
    for cand in (gt_folder.with_suffix('.json'),
                 gt_folder.parent / f'{gt_folder.name}.json',
                 Path(root) / f'gt_panoptic_{split}.json'):
        if cand.is_file():
            with open(cand) as f:
                gt_json = json.load(f)
            return gt_json, _match_folder(gt_json, gt_folder)
    if not args.build_gt_json:
        return None, gt_folder
    pngs = sorted(p for p in gt_folder.rglob('*.png'))
    sel = {base_key(p.name): p for p in pngs}
    missing = [k for k in keys if k not in sel]
    if missing:
        raise SystemExit(f"--build-gt-json: {len(missing)} val images have no "
                         f"panoptic PNG under {gt_folder} (e.g. {missing[:3]})")
    paths = [sel[k] for k in keys]
    rel = [str(p.relative_to(gt_folder)) for p in paths]
    gt_json = pq_format.build_gt_json_from_pngs(
        paths, keys, pq_format.cityscapes_categories())
    for ann, r in zip(gt_json['annotations'], rel):
        ann['file_name'] = r
    return gt_json, gt_folder


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True, help='training config (protocol source of truth)')
    ap.add_argument('--model_path', required=True, help='*_checkpoint.pth or raw state_dict')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--dataset', required=True, choices=['muses', 'deliver'])
    ap.add_argument('--obj-thresh', type=float, default=0.8)
    ap.add_argument('--overlap-thresh', type=float, default=0.8)
    ap.add_argument('--out', required=True, help='output dir (predictions + report)')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dataset-root', default=None, help='override DATASET.ROOT')
    ap.add_argument('--geometry', default='native', choices=['native', 'letterbox'])
    ap.add_argument('--thing-ids', default=None,
                    help='comma-separated trainIds treated as things. Default: '
                         'Cityscapes/MUSES 11..18. REQUIRED for DELIVER (25 '
                         'classes, no official panoptic protocol).')
    ap.add_argument('--gt-json', default=None, help='panoptic GT json (COCO format)')
    ap.add_argument('--gt-folder', default=None, help='panoptic GT PNG folder')
    ap.add_argument('--build-gt-json', action='store_true',
                    help='derive the GT json from the PNGs (Cityscapes '
                         'category*1000+instance ids; validated, not assumed)')
    ap.add_argument('--run-aupq', action='store_true',
                    help='also shell out to the official MUSES AUPQ script as a '
                         'cross-check (needs the gt_uncertainty folder)')
    ap.add_argument('--nr-thresholds', type=int, default=2,
                    help='AUPQ threshold grid. Our confidence maps are constant '
                         '255, so every cell is identical and 2 == 16.')
    ap.add_argument('--limit', type=int, default=None, help='stop after N images')
    args = ap.parse_args()

    if args.split == 'test':
        raise SystemExit(
            "MUSES/DELIVER test panoptic GT is not available to us: MUSES "
            "withholds test GT for the public benchmark (semseg/datasets/muses.py "
            "raises on split='test'), so a test PQ number cannot be produced "
            "locally. Use --split val, or submit to the Codabench panoptic "
            "benchmark for a test score.")

    cfg = yaml.safe_load(open(args.cfg))
    dsname = str(cfg['DATASET'].get('NAME', '')).strip().upper()
    if dsname != args.dataset.upper():
        raise SystemExit(f"--dataset {args.dataset} but DATASET.NAME={dsname}")
    device = torch.device(cfg.get('DEVICE', 'cuda')
                          if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ds, dcfg = build_dataset(args.dataset, cfg, args.split, args.dataset_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True)
    model, ck = load_model(cfg, args.model_path, device, ds.n_classes)
    head = 'p43' if model.p43 is not None else ('m2f' if model.m2f is not None else None)
    if head is None:
        raise SystemExit(
            "no mask-classification head in this checkpoint (MODEL.P43.M2F_HEAD "
            "and MODEL.M2F.ENABLE both off) — a per-pixel model cannot produce "
            "panoptic segments.")
    thing_ids = ([int(t) for t in args.thing_ids.split(',')]
                 if args.thing_ids else None)
    thing_ids = model._resolve_thing_ids(thing_ids)
    side = cfg['EVAL']['IMAGE_SIZE'][0]
    print(f"[pq] head={head}  thing_ids={thing_ids}  geometry={args.geometry} "
          f"side={side}  obj={args.obj_thresh} overlap={args.overlap_thresh}",
          flush=True)
    if head == 'p43' and args.geometry == 'native':
        raise SystemExit(
            "the P43 head does not implement un-letterboxing (its 2026-07-25 "
            "behaviour is frozen); use --geometry letterbox for prediction dumps "
            "or evaluate an M2F checkpoint.")

    writer = pq_format.PanopticPredWriter(outdir / 'pred')
    keys, geoms = [], {}
    n_done, n_empty = 0, 0
    with torch.no_grad():
        for images, _label, meta in tqdm(loader, desc='panoptic'):
            images = [x.to(device, non_blocking=True) for x in images]
            H, W = int(meta['orig_h'][0]), int(meta['orig_w'][0])
            if args.geometry == 'native':
                size = (H, W)
                if args.dataset == 'muses':
                    crop, crop_size = letterbox_valid_box(H, W, side), (side, side)
                else:
                    crop, crop_size = None, None       # DELIVER is not letterboxed
            else:
                size, crop, crop_size = (side, side), None, None
            res = model.panoptic_inference(
                images, thing_ids=thing_ids, obj_thresh=args.obj_thresh,
                overlap_thresh=args.overlap_thresh, size=size, crop=crop,
                crop_size=crop_size)
            pan, segs = res[0]
            key = base_key(str(meta['stem'][0]))
            if key in geoms:      # one PNG per key -> a collision loses images
                raise SystemExit(f"duplicate scene key {key!r}; the val list is "
                                 "not uniquely identified by its stems.")
            keys.append(key)
            geoms[key] = {'orig': [H, W], 'pred': list(pan.shape)}
            n_empty += int(len(segs) == 0)
            writer.add(key, f'{key}.png', pan.cpu().numpy().astype(np.uint32), segs)
            n_done += 1
            if args.limit and n_done >= args.limit:
                break
    pred_json_path = writer.close()
    print(f"[pq] wrote {n_done} predictions -> {writer.pred_dir} "
          f"({n_empty} with zero segments)", flush=True)

    report = {
        'cfg': str(args.cfg), 'ckpt': str(args.model_path),
        'epoch': ck.get('epoch', None), 'dataset': args.dataset,
        'split': args.split, 'head': head, 'geometry': args.geometry,
        'n_images': n_done, 'n_images_without_segments': n_empty,
        'thing_ids_trainid': thing_ids,
        'obj_thresh': args.obj_thresh, 'overlap_thresh': args.overlap_thresh,
        'pred_json': str(pred_json_path),
        'sample_geometry': dict(list(geoms.items())[:2]),
        'scored': False,
    }

    if args.geometry != 'native':
        report['note'] = ("letterbox geometry: predictions are in the 1024^2 "
                          "working space and were NOT scored (panoptic GT is "
                          "native resolution).")
    else:
        gt_json, gt_folder = resolve_gt(args, dcfg['ROOT'], args.split, keys)
        if gt_json is None:
            report['note'] = (
                f"no panoptic GT found (looked for a json near {gt_folder}); "
                "predictions written but NOT scored. Pass --gt-json/--gt-folder, "
                "or --build-gt-json to derive one from the GT PNGs.")
            print(f"[pq] {report['note']}", flush=True)
        else:
            # Join on the normalised scene id: the shipped json may key images
            # any way it likes, so try image_id first and fall back to the PNG
            # name. Normalise BEFORE filtering, or --limit would drop everything
            # whenever the json's ids are not scene stems.
            keyset = set(keys)
            gt_json = dict(gt_json)
            anns = []
            for a in gt_json['annotations']:
                a = dict(a)
                cand = base_key(str(a.get('image_id', '')))
                if cand not in keyset:
                    cand = base_key(str(a['file_name']))
                a['image_id'] = cand
                if cand in keyset:
                    anns.append(a)
            gt_json['annotations'] = anns
            missing = keyset - {a['image_id'] for a in anns}
            if missing:
                raise SystemExit(
                    f"{len(missing)} predicted images have no GT annotation "
                    f"(e.g. {sorted(missing)[:3]}). GT ids look like "
                    f"{[a['image_id'] for a in gt_json['annotations'][:3]]}.")
            with open(pred_json_path) as f:
                pred_json = json.load(f)
            results = pq_format.pq_compute(gt_json, pred_json, gt_folder,
                                           writer.pred_dir, progress=tqdm)
            report['scored'] = True
            report['n_scored'] = len(anns)
            report['gt_folder'] = str(gt_folder)
            report['results'] = {k: results[k] for k in ('All', 'Things', 'Stuff')}
            report['per_class'] = results['per_class']
            print("\n" + pq_format.format_table(results))
            if args.run_aupq:
                gtj = outdir / 'gt_panoptic.json'
                with open(gtj, 'w') as f:
                    json.dump(gt_json, f)
                cmd = [sys.executable,
                       str(REPO / 'third_party/MUSES/MUSES/AUPQ/'
                                  'uncertainty_aware_panoptic_quality.py'),
                       '--gt_json_file', str(gtj), '--gt_folder', str(gt_folder),
                       '--pred_json_file', str(pred_json_path),
                       '--pred_folder', str(writer.pred_dir),
                       '--nr_thresholds', str(args.nr_thresholds)]
                print("\n[aupq] " + " ".join(cmd), flush=True)
                report['aupq_returncode'] = subprocess.call(cmd)

    with open(outdir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {outdir}/report.json")


if __name__ == '__main__':
    main()
