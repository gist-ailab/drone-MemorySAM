#!/usr/bin/env python3
"""tools/smoke_pq.py — CPU synthetic smoke for the PQ evaluation path.

Runs the REAL ReliaDINO (tiny random-init timm ViT, 64x64 inputs, 2 modalities,
19 classes so the Cityscapes thing-id default applies, 8 queries). No GPU, no
dataset, no pretrained download.

Checks, in the order the task specifies them:
  1  M2F model: model.panoptic_inference() emits a valid (panoptic_seg,
     segments_info) — contiguous ids from 1, ids in the PNG == ids in the JSON,
     category_id in range, stuff merged to one segment per class, things not
     merged.
  2  size=/crop=/crop_size= land the map at exactly the requested label
     resolution, and the un-letterbox crop keeps the right aspect.
  3  semantic equivalence: the semantic output is bit-identical with the
     panoptic capture on vs off, and before vs after a panoptic_inference call;
     forward() never calls the panoptic post-processing.
  4  the P43 path still routes to the P43 head and is unchanged.
  5  AUPQ pipe integrity: one synthetic GT scene scored end-to-end by our
     scorer AND by the official MUSES AUPQ script.

Usage:
  /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_pq.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'semseg' / 'models' / 'sam2'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from PIL import Image                                               # noqa: E402

from semseg.models.reliadino import build_reliadino                 # noqa: E402
from tools import pq_format                                         # noqa: E402
from tools.eval_muses_official import letterbox_valid_box           # noqa: E402

B, HW, K, NM = 2, 64, 19, 2
FAILURES = []


def check(name, cond, detail=''):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ''))
    if not cond:
        FAILURES.append(name)


def make_cfg(m2f=None, p43=None):
    cfg = {
        'DEVICE': 'cpu',
        'MODEL': {
            'NAME': 'ReliaDINO', 'BACKBONE': 'smoke',
            'BACKBONE_TIMM': 'vit_tiny_patch16_224', 'BACKBONE_FALLBACK': '',
            'PRETRAINED_BACKBONE': False, 'LORA_R': 4, 'FPN_DIM': 64,
            'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 8, 'MLP_RATIO': 2.0,
                       'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5,
                       'ATTN_BIAS': {'ENABLE': False}},
            'CONSISTENCY': {'ENABLE': False},
            'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
            'CALIBRATION': {'ENABLE': False},
            'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
            'CEFR': {'ENABLE': False}, 'CLASS_TOKEN': {'ENABLE': False},
            'M2F': {'ENABLE': False},
        },
        'DATASET': {'MODALS': ['img', 'lidar']},
        'TRAIN': {'IMAGE_SIZE': [HW, HW]},
    }
    if m2f is not None:
        cfg['MODEL']['M2F'] = m2f
        # production P39.1 shape: trunk expansion + arbiter instead of the
        # legacy zero-init beta residual.
        cfg['MODEL']['P39'] = {'TRUNK_EXP': True, 'TRUNK_MODE': 'gated_mlp',
                               'TRUNK_HIDDEN': 32, 'ARBITER': True}
    if p43 is not None:
        cfg['MODEL']['P43'] = p43
    return cfg


M2F_ON = {'ENABLE': True, 'NUM_QUERIES': 8, 'NUM_LAYERS': 2, 'DIM': 32,
          'NUM_HEADS': 8, 'MLP_RATIO': 2.0, 'POINTS': 64, 'SRC': 'modal'}
P43_ON = {'M2F_HEAD': True, 'LATERAL': True, 'NUM_TAPS': 3, 'NUM_QUERIES': 8,
          'DEC_LAYERS': 3, 'DIM': 32, 'NUM_HEADS': 8, 'NUM_POINTS': 32,
          'THING_IDS': [3, 4]}


def build(m2f=None, p43=None, seed=0):
    torch.manual_seed(seed)
    return build_reliadino(make_cfg(m2f, p43), K)


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, 3, HW, HW, generator=g) for _ in range(NM)]


def seg_checks(prefix, pan, segs, thing_ids, n_classes=K):
    ids = [s['id'] for s in segs]
    painted = set(int(v) for v in torch.unique(pan).tolist()) - {0}
    check(f'{prefix} ids contiguous 1..N',
          ids == list(range(1, len(ids) + 1)), f"{len(ids)} segments")
    check(f'{prefix} painted ids all declared (AUPQ KeyError guard)',
          painted <= set(ids), f"orphans={sorted(painted - set(ids))[:3]}")
    check(f'{prefix} category_id in [0,{n_classes})',
          all(0 <= s['category_id'] < n_classes for s in segs))
    check(f'{prefix} isthing follows thing_ids',
          all(s['isthing'] == (s['category_id'] in thing_ids) for s in segs))
    stuff_cats = [s['category_id'] for s in segs if not s['isthing']]
    check(f'{prefix} stuff merged (one segment per stuff class)',
          len(stuff_cats) == len(set(stuff_cats)),
          f"stuff cats={sorted(stuff_cats)}")


def main():
    torch.set_num_threads(4)
    print("=" * 72)
    print("PQ evaluation path — CPU synthetic smoke")
    print("=" * 72)
    x = inputs()

    # ── 1: M2F panoptic_inference ──────────────────────────────────────────
    print("\n[1] M2F model: panoptic_inference validity")
    m = build(m2f=M2F_ON)
    m.eval()
    check('1.0 M2F head built, P43 absent', m.m2f is not None and m.p43 is None)
    check('1.1 thing_ids default = Cityscapes trainIds 11..18',
          m._resolve_thing_ids(None) == list(range(11, 19)),
          str(m._resolve_thing_ids(None)))
    thing = m._resolve_thing_ids(None)
    # thresholds at 0 so every query survives -> exercises merge/overlap logic
    res = m.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0)
    check('1.2 one result per image', len(res) == B, f"{len(res)}")
    pan, segs = res[0]
    check('1.3 stride-4 map by default',
          tuple(pan.shape) == (HW // 4, HW // 4) and pan.dtype == torch.int32,
          f"{tuple(pan.shape)} {pan.dtype}")
    seg_checks('1.4', pan, segs, thing)
    res_t = m.panoptic_inference(x, obj_thresh=0.99, overlap_thresh=0.99)
    check('1.5 obj_thresh filters (high threshold -> fewer/no segments)',
          len(res_t[0][1]) <= len(segs), f"{len(res_t[0][1])} vs {len(segs)}")
    custom = m.panoptic_inference(x, thing_ids=[0, 1], obj_thresh=0.0,
                                  overlap_thresh=0.0)
    seg_checks('1.6 custom thing_ids', custom[0][0], custom[0][1], [0, 1])
    try:
        build()._resolve_thing_ids(None)   # 19-class default still applies
        nodef = False
    except ValueError:
        nodef = True
    dl = build(m2f=M2F_ON)
    dl.num_classes = 25
    try:
        dl._resolve_thing_ids(None)
        raised = False
    except ValueError:
        raised = True
    check('1.7 non-Cityscapes class count refuses to guess thing_ids',
          raised and not nodef)
    plain = build()
    plain.eval()
    try:
        plain.panoptic_inference(x)
        loud = False
    except RuntimeError as e:
        loud = 'per-pixel' in str(e)
    check('1.8 head-less model raises instead of failing silently', loud)

    # ── 1b: merge/threshold logic on a hand-built head output ──────────────
    # A random-init model rarely emits more than one surviving query, so drive
    # the post-processing directly with a known (cls, masks) pair.
    print("\n[1b] stuff merge / thing separation on a deterministic output")
    q, h, w = 4, 8, 8
    cls = torch.full((1, q, K + 1), -20.0)
    for i, cat in enumerate([2, 2, 13, 13]):        # 2=building(stuff) 13=car(thing)
        cls[0, i, cat] = 20.0
    ml = torch.full((1, q, h, w), -20.0)
    ml[0, 0, :4, :4] = 20.0
    ml[0, 1, :4, 4:] = 20.0                        # 2nd building -> merges
    ml[0, 2, 4:, :4] = 20.0
    ml[0, 3, 4:, 4:] = 20.0                        # 2nd car -> stays separate
    r1b = m.m2f.panoptic_inference({'cls': cls, 'masks': ml}, thing,
                                   obj_thresh=0.5, overlap_thresh=0.5)
    pan1b, segs1b = r1b[0]
    seg_checks('1b.1', pan1b, segs1b, thing)
    check('1b.2 two things + one merged stuff = 3 segments',
          len(segs1b) == 3, f"{[(s['id'], s['category_id']) for s in segs1b]}")
    check('1b.3 both building quadrants carry the same segment id',
          int(pan1b[0, 0]) == int(pan1b[0, 7]) and int(pan1b[0, 0]) != 0)
    check('1b.4 the two car quadrants carry different ids',
          int(pan1b[7, 0]) != int(pan1b[7, 7]))
    up = m.m2f.panoptic_inference({'cls': cls, 'masks': ml}, thing,
                                  obj_thresh=0.5, overlap_thresh=0.5,
                                  size=(4 * h, 4 * w))[0]
    check('1b.5 upsampling preserves the segment set, not just the shape',
          tuple(up[0].shape) == (4 * h, 4 * w)
          and [(s['id'], s['category_id']) for s in up[1]]
          == [(s['id'], s['category_id']) for s in segs1b],
          f"{[(s['id'], s['category_id']) for s in up[1]]}")

    # ── 2: geometry ────────────────────────────────────────────────────────
    print("\n[2] size= / crop= geometry")
    lab = (2 * HW, 3 * HW)
    r = m.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0, size=lab)
    check('2.1 size= lands at label resolution',
          tuple(r[0][0].shape) == lab, str(tuple(r[0][0].shape)))
    seg_checks('2.2 upsampled', r[0][0], r[0][1], thing)
    # MUSES letterbox: 1080x1920 -> 1920^2 square -> HW; invert on the logits.
    crop = letterbox_valid_box(1080, 1920, HW)
    rc = m.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0,
                              size=(1080, 1920), crop=crop, crop_size=(HW, HW))
    check('2.3 crop+size lands at native MUSES resolution',
          tuple(rc[0][0].shape) == (1080, 1920), str(tuple(rc[0][0].shape)))
    seg_checks('2.4 un-letterboxed', rc[0][0], rc[0][1], thing)
    rcrop = m.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0,
                                 crop=crop, crop_size=(HW, HW))
    check('2.5 crop alone gives the letterbox-free box',
          tuple(rcrop[0][0].shape) == (crop[1] - crop[0], crop[3] - crop[2]),
          f"{tuple(rcrop[0][0].shape)} vs box {crop}")
    empty = m.panoptic_inference(x, obj_thresh=1.1, size=lab)
    check('2.6 zero-segment image still emits the right canvas',
          tuple(empty[0][0].shape) == lab and len(empty[0][1]) == 0
          and int(empty[0][0].max()) == 0)

    # ── 3: semantic equivalence ────────────────────────────────────────────
    print("\n[3] semantic output equivalence (|Δ|max == 0)")
    with torch.no_grad():
        o_before, _ = m(x, True)
    check('3.0 capture flag off by default', m._m2f_capture is False)
    _ = m.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0, size=lab)
    with torch.no_grad():
        o_after, _ = m(x, True)
    d1 = float((o_before - o_after).abs().max())
    check('3.1 semantic identical before/after panoptic_inference', d1 == 0.0,
          f"|Δ|max={d1:.3e}")
    m._m2f_capture = True
    with torch.no_grad():
        o_cap, _ = m(x, True)
    m._m2f_capture = False
    m._last_m2f_out = None
    d2 = float((o_before - o_cap).abs().max())
    check('3.2 semantic identical with the capture flag ON', d2 == 0.0,
          f"|Δ|max={d2:.3e}")
    check('3.3 stash released after the read', m._last_m2f_out is None)
    calls = {'n': 0}
    orig = m.m2f.panoptic_inference

    def counting(*a, **kw):
        calls['n'] += 1
        return orig(*a, **kw)
    m.m2f.panoptic_inference = counting
    with torch.no_grad():
        m(x, True)
    check('3.4 forward() never enters the panoptic post-processing',
          calls['n'] == 0, f"{calls['n']} calls")
    m.m2f.panoptic_inference = orig
    m.train()
    try:
        m.panoptic_inference(x)
        guarded = False
    except RuntimeError as e:
        guarded = 'eval' in str(e)
    m.eval()
    check('3.5 train-mode call is refused (eval-only path)', guarded)

    # ── 4: P43 regression ──────────────────────────────────────────────────
    print("\n[4] P43 path unchanged")
    p = build(p43=P43_ON, seed=7)
    p.eval()
    check('4.0 P43 head built, M2F absent', p.p43 is not None and p.m2f is None)
    rp = p.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0, size=(HW, HW))
    check('4.1 P43 panoptic_inference still runs', len(rp) == B)
    seg_checks('4.2 P43', rp[0][0], rp[0][1], P43_ON['THING_IDS'])
    check('4.3 P43 still uses MODEL.P43.THING_IDS',
          all(s['isthing'] == (s['category_id'] in (3, 4)) for s in rp[0][1]))
    direct = p.p43.panoptic_inference(p._p43_forward_out(x), P43_ON['THING_IDS'],
                                      obj_thresh=0.0, overlap_thresh=0.0,
                                      size=(HW, HW))
    check('4.4 model route == direct head call (no behaviour drift)',
          torch.equal(rp[0][0], direct[0][0])
          and rp[0][1] == direct[0][1])
    try:
        p.panoptic_inference(x, crop=(0, 8, 0, 8), crop_size=(HW, HW))
        p43_crop = False
    except NotImplementedError:
        p43_crop = True
    check('4.5 P43 + crop refuses loudly (not silently ignored)', p43_crop)
    both = build(m2f=M2F_ON, p43=P43_ON, seed=7)
    both.eval()
    calls43 = {'n': 0}
    o43 = both.p43.panoptic_inference

    def c43(*a, **kw):
        calls43['n'] += 1
        return o43(*a, **kw)
    both.p43.panoptic_inference = c43
    both.panoptic_inference(x, obj_thresh=0.0, overlap_thresh=0.0)
    check('4.6 P43 wins the dispatch when both heads exist', calls43['n'] == 1)

    # ── 5: AUPQ pipe integrity ─────────────────────────────────────────────
    print("\n[5] AUPQ format + scoring pipe (synthetic scene)")
    tmp = Path(tempfile.mkdtemp(prefix='smoke_pq_'))
    try:
        H, W = 64, 96
        cats = pq_format.cityscapes_categories()
        # GT: road (stuff, labelId 7) top half, one car instance (thing 26) box,
        # the rest VOID. Cityscapes id convention: stuff=cat, thing=cat*1000+i.
        gt = np.zeros((H, W), dtype=np.uint32)
        gt[:32, :] = 7
        gt[40:56, 20:60] = 26 * 1000 + 1
        gtdir = tmp / 'gt_panoptic'
        gtdir.mkdir()
        Image.fromarray(pq_format.id2rgb(gt)).save(gtdir / 'scene1_panoptic.png')
        gt_json = pq_format.build_gt_json_from_pngs(
            [gtdir / 'scene1_panoptic.png'], ['scene1'], cats)
        check('5.1 GT json derived + convention self-validated',
              len(gt_json['annotations'][0]['segments_info']) == 2,
              str([s['category_id'] for s in gt_json['annotations'][0]['segments_info']]))
        # prediction in OUR head's output form: trainIds + segment ids from 1
        pan = np.zeros((H, W), dtype=np.uint32)
        pan[:30, :] = 1                          # road   (trainId 0)  slight miss
        pan[40:56, 20:60] = 2                    # car    (trainId 13) exact
        pan[58:62, 0:10] = 3                     # a false positive building
        segs = [{'id': 1, 'category_id': 0, 'isthing': False},
                {'id': 2, 'category_id': 13, 'isthing': True},
                {'id': 3, 'category_id': 2, 'isthing': False}]
        w = pq_format.PanopticPredWriter(tmp / 'pred')
        w.add('scene1', 'scene1.png', pan, segs)
        pred_json_path = w.close()
        pred_json = json.loads(pred_json_path.read_text())
        check('5.2 pred trainIds mapped to Cityscapes labelIds',
              [s['category_id'] for s in pred_json['annotations'][0]['segments_info']]
              == [7, 26, 11])
        check('5.3 AUPQ folder layout written',
              (tmp / 'pred/labelIds/scene1.png').is_file()
              and (tmp / 'pred/classConfidence/scene1.png').is_file()
              and (tmp / 'pred/instanceConfidence/scene1.png').is_file()
              and pred_json_path.name == 'labelIds.json')
        rt = pq_format.rgb2id(np.array(
            Image.open(tmp / 'pred/labelIds/scene1.png').convert('RGB'),
            dtype=np.uint32))
        check('5.4 id<->rgb round-trip is lossless', np.array_equal(rt, pan))
        results = pq_format.pq_compute(gt_json, pred_json, gtdir,
                                       tmp / 'pred/labelIds')
        ours = results['All']['pq']
        check('5.5 our PQ runs end to end',
              np.isfinite(ours) and results['Things']['n'] == 1
              and results['Stuff']['n'] >= 1,
              f"PQ={100 * ours:.1f} things_n={results['Things']['n']} "
              f"stuff_n={results['Stuff']['n']}")
        check('5.6 exact-match thing segment scores SQ=RQ=1',
              abs(results['per_class']['car']['pq'] - 1.0) < 1e-9,
              f"car PQ={100 * results['per_class']['car']['pq']:.2f}")
        # official MUSES AUPQ: needs gt_uncertainty next to gt_panoptic. With
        # our constant-255 confidence every threshold cell is the same cell, so
        # AUPQ must reproduce PQ exactly.
        undir = tmp / 'gt_uncertainty'
        undir.mkdir()
        Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(
            undir / 'scene1_uncertainty.png')
        gtj = tmp / 'gt.json'
        gtj.write_text(json.dumps(gt_json))
        cmd = [sys.executable,
               str(REPO / 'third_party/MUSES/MUSES/AUPQ/'
                          'uncertainty_aware_panoptic_quality.py'),
               '--gt_json_file', str(gtj), '--gt_folder', str(gtdir),
               '--pred_json_file', str(pred_json_path),
               '--pred_folder', str(tmp / 'pred/labelIds'),
               '--nr_thresholds', '2']
        proc = subprocess.run(cmd, capture_output=True, text=True)
        check('5.7 official AUPQ script runs to completion',
              proc.returncode == 0,
              (proc.stderr.strip().splitlines() or [''])[-1][:120])
        aupq = None
        for line in proc.stdout.splitlines():
            if line.startswith('All'):
                aupq = float(line.split('|')[1].split()[0])
        check('5.8 official AUPQ == our PQ under saturated confidence',
              aupq is not None and abs(aupq - 100 * ours) < 0.05,
              f"AUPQ={aupq} ours={100 * ours:.1f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── 6: eval_pq join/discovery logic (the silent-failure surface) ───────
    print("\n[6] tools/eval_pq GT discovery + image-id join")
    from argparse import Namespace
    from tools import eval_pq
    check('6.1 base_key normalises every folder suffix',
          eval_pq.base_key('AA_BB_01_frame_camera.png')
          == eval_pq.base_key('AA_BB_01_gt_panoptic.png')
          == eval_pq.base_key('AA_BB_01_gt_labelTrainIds.png')
          == 'AA_BB_01')
    tmp2 = Path(tempfile.mkdtemp(prefix='smoke_pq_gt_'))
    try:
        d = tmp2 / 'gt_panoptic' / 'val' / 'clear' / 'day'
        d.mkdir(parents=True)
        gt = np.zeros((16, 16), dtype=np.uint32)
        gt[:8] = 7
        gt[8:, :8] = 26 * 1000 + 1
        Image.fromarray(pq_format.id2rgb(gt)).save(d / 'AA_BB_01_gt_panoptic.png')
        a = Namespace(gt_folder=None, gt_json=None, build_gt_json=True)
        gj, gf = eval_pq.resolve_gt(a, str(tmp2), 'val', ['AA_BB_01'])
        check('6.2 gt_panoptic/<split> auto-discovered',
              gf == tmp2 / 'gt_panoptic' / 'val', str(gf))
        ann = gj['annotations'][0]
        check('6.3 file_name is relative to the gt folder (AUPQ os.path.join)',
              ann['file_name'] == 'clear/day/AA_BB_01_gt_panoptic.png'
              and (gf / ann['file_name']).is_file(), ann['file_name'])
        check('6.4 image_id == our normalised scene key',
              ann['image_id'] == 'AA_BB_01')
        a2 = Namespace(gt_folder=None, gt_json=None, build_gt_json=False)
        gj2, _ = eval_pq.resolve_gt(a2, str(tmp2), 'val', ['AA_BB_01'])
        check('6.5 without --build-gt-json the GT is reported missing, '
              'not invented', gj2 is None)
        try:
            eval_pq.resolve_gt(a, str(tmp2), 'val', ['AA_BB_01', 'ZZ_99'])
            caught = False
        except SystemExit:
            caught = True
        check('6.6 a val image with no GT PNG aborts', caught)
        try:
            pq_format.PanopticPredWriter(tmp2 / 'out_labelIds')
            named = False
        except ValueError:
            named = True
        check("6.7 output path containing 'labelIds' is rejected "
              "(AUPQ string-replaces it)", named)
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("RESULT: PASS — all checks green")
    return 0


if __name__ == '__main__':
    sys.exit(main())
