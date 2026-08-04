#!/usr/bin/env python3
"""tools/pq_format.py — COCO/MUSES panoptic I/O + a standard PQ scorer.

Everything here is derived from ONE source of truth that is vendored in this
repo: `third_party/MUSES/MUSES/AUPQ/uncertainty_aware_panoptic_quality.py`
(the official MUSES AUPQ scorer, itself a fork of panopticapi's evaluation.py).
Nothing about the format is guessed; the paragraph below cites the lines.

── What the AUPQ scorer actually eats ────────────────────────────────────────
  * `--gt_json_file` / `--pred_json_file`, both COCO-panoptic JSONs:
      gt   {'categories': [{'id','isthing',...}, ...],
            'annotations': [{'image_id', 'file_name',
                             'segments_info': [{'id','category_id','area',
                                                'iscrowd'}]}]}          (L402-434)
      pred {'annotations': [{'image_id','file_name',
                             'segments_info': [{'id','category_id'}]}]}
      -> `area` is NOT read from the pred json; the scorer recounts it from the
         PNG (L280-288). `iscrowd` is read from the GT only (L306, L351).
  * `--gt_folder` / `--pred_folder`, defaulting to the json path minus '.json'
    (L408-411). Each holds one **RGB PNG per image**, segment id encoded as
    `id = R + 256*G + 256*256*B` (`rgb2id`, L119-121). `id == 0` is VOID (L17).
  * every id painted in the pred PNG must appear in the pred json and vice
    versa, and every pred category_id must exist in the GT `categories`
    (L156-175) — the scorer raises KeyError otherwise.
  * category ids are Cityscapes **labelIds**, not trainIds: the module-level
    constants are STUFF = (7,8,11,12,13,17,19,20,21,22,23) and
    THINGS = (24,25,26,27,28,31,32,33) (L19-20), i.e. trainIds 0..10 and 11..18.
  * AUPQ-only extras: GT uncertainty PNGs at
    `gt_folder.replace('gt_panoptic','gt_uncertainty')` with
    `_panoptic.png -> _uncertainty.png` (L178-183), and per-prediction
    confidence PNGs at `pred_folder.replace('labelIds','classConfidence')` /
    `...('labelIds','instanceConfidence')` (L374-382). Hence the pred folder
    MUST be named `labelIds`.

── Why this module also carries its own scorer ───────────────────────────────
`pq_compute` below is the AUPQ body (L279-368) with the uncertainty machinery
removed, which is exactly what AUPQ degenerates to when the predicted
confidence is saturated: `unconf = conf < t` is False for every t in
`linspace(0,255,n)` when conf == 255, so all n^2 threshold cells collapse to the
same plain-PQ cell and AUPQ == PQ, AUSQ == SQ, AURQ == RQ. Running the official
script on constant-255 confidence maps is therefore a cross-check of this code,
not a different metric — tools/eval_pq.py writes those maps and can shell out to
it with --run-aupq. The local scorer exists because the official one hard-
requires the `gt_uncertainty` folder, which a gt_panoptic-only download of MUSES
does not have.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

VOID = 0
OFFSET = 256 * 256 * 256

# Cityscapes trainId -> labelId. Verified against the AUPQ STUFF/THINGS tuples:
# trainIds 0..10 map onto STUFF, trainIds 11..18 onto THINGS, element for
# element and in order.
TRAINID_TO_LABELID = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23,
                      24, 25, 26, 27, 28, 31, 32, 33)
AUPQ_STUFF = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23)
AUPQ_THINGS = (24, 25, 26, 27, 28, 31, 32, 33)
CITYSCAPES_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole",
                    "traffic light", "traffic sign", "vegetation", "terrain",
                    "sky", "person", "rider", "car", "truck", "bus", "train",
                    "motorcycle", "bicycle")


def cityscapes_categories() -> List[Dict]:
    """`categories` block in AUPQ/panopticapi form, keyed by Cityscapes labelId."""
    return [{'id': lid, 'name': name, 'isthing': int(lid in AUPQ_THINGS)}
            for lid, name in zip(TRAINID_TO_LABELID, CITYSCAPES_NAMES)]


def rgb2id(color: np.ndarray) -> np.ndarray:
    color = color.astype(np.uint32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def id2rgb(idmap: np.ndarray) -> np.ndarray:
    idmap = idmap.astype(np.uint32)
    return np.stack([idmap % 256, (idmap // 256) % 256,
                     (idmap // 65536) % 256], axis=-1).astype(np.uint8)


# ── prediction writer ────────────────────────────────────────────────────────
class PanopticPredWriter:
    """Writes an AUPQ-shaped prediction directory.

        <root>/labelIds/<name>.png             panoptic PNG (rgb2id segment ids)
        <root>/labelIds.json                   pred json (upq_compute default:
                                               pred_folder = json minus '.json')
        <root>/classConfidence/<name>.png      uint8, constant `confidence`
        <root>/instanceConfidence/<name>.png   uint8, constant `confidence`

    The folder is named `labelIds` because AUPQ derives the two confidence
    folders by string-replacing that literal in the pred folder path; a saturated
    (255) confidence makes AUPQ reduce to plain PQ (see module docstring).
    """

    def __init__(self, root, category_map: Sequence[int] = TRAINID_TO_LABELID,
                 confidence: int = 255, write_confidence: bool = True):
        self.root = Path(root)
        if 'labelIds' in str(self.root):
            raise ValueError(
                f"output root {self.root} contains the literal 'labelIds'; AUPQ "
                "string-replaces it to find the confidence folders, so the path "
                "above the pred dir must not use that word.")
        self.pred_dir = self.root / 'labelIds'
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.write_confidence = write_confidence
        if write_confidence:
            (self.root / 'classConfidence').mkdir(parents=True, exist_ok=True)
            (self.root / 'instanceConfidence').mkdir(parents=True, exist_ok=True)
        self.category_map = list(category_map)
        self.confidence = int(confidence)
        self.annotations: List[Dict] = []

    def add(self, image_id: str, file_name: str, pan: np.ndarray,
            segments: Sequence[Dict]) -> Dict:
        """`pan`/`segments` come straight out of model.panoptic_inference();
        `segments[*]['category_id']` is a **trainId** and is mapped here."""
        pan = np.asarray(pan)
        painted = set(int(v) for v in np.unique(pan)) - {VOID}
        declared = set(int(s['id']) for s in segments)
        # AUPQ raises on either mismatch (L156-175); catching it at write time
        # gives a useful message instead of a KeyError 250 images later.
        if painted - declared:
            raise ValueError(f"{image_id}: ids in PNG but not in JSON: "
                             f"{sorted(painted - declared)}")
        seg_info = []
        for s in segments:
            if int(s['id']) not in painted:
                continue          # fully overwritten by a later segment -> drop
            tid = int(s['category_id'])
            if not 0 <= tid < len(self.category_map):
                raise ValueError(f"{image_id}: category trainId {tid} outside "
                                 f"0..{len(self.category_map) - 1}")
            seg_info.append({'id': int(s['id']),
                             'category_id': int(self.category_map[tid])})
        Image.fromarray(id2rgb(pan)).save(self.pred_dir / file_name)
        if self.write_confidence:
            conf = np.full(pan.shape, self.confidence, dtype=np.uint8)
            Image.fromarray(conf).save(self.root / 'classConfidence' / file_name)
            Image.fromarray(conf).save(self.root / 'instanceConfidence' / file_name)
        ann = {'image_id': image_id, 'file_name': file_name,
               'segments_info': seg_info}
        self.annotations.append(ann)
        return ann

    def close(self, categories: Optional[List[Dict]] = None) -> Path:
        payload = {'annotations': self.annotations}
        if categories is not None:
            payload['categories'] = categories
        path = self.root / 'labelIds.json'
        with open(path, 'w') as f:
            json.dump(payload, f)
        return path


# ── ground truth ─────────────────────────────────────────────────────────────
def build_gt_json_from_pngs(png_paths: Sequence[Path], image_ids: Sequence[str],
                            categories: Sequence[Dict]) -> Dict:
    """Derive a panoptic GT json from Cityscapes-convention panoptic PNGs.

    Convention assumed: the encoded id is `category_id * 1000 + instance_id` for
    thing segments and a bare `category_id` for stuff (the Cityscapes/
    panopticapi `createPanopticImgs` convention). This is NOT read off the
    MUSES SDK — MUSES ships the json alongside gt_panoptic, so PREFER the
    shipped one and treat this as a fallback. It self-validates: every derived
    category must exist in `categories`, otherwise the convention is wrong for
    this dataset and we raise instead of scoring garbage.
    """
    valid = {int(c['id']) for c in categories}
    annotations = []
    for p, image_id in zip(png_paths, image_ids):
        pan = rgb2id(np.array(Image.open(p).convert('RGB'), dtype=np.uint32))
        ids, counts = np.unique(pan, return_counts=True)
        segs = []
        for sid, area in zip(ids.tolist(), counts.tolist()):
            if sid == VOID:
                continue
            cat = sid // 1000 if sid >= 1000 else sid
            if cat not in valid:
                raise ValueError(
                    f"{p}: segment id {sid} -> category {cat}, which is not in "
                    f"the category table {sorted(valid)}. The "
                    "'category*1000+instance' convention does not hold for this "
                    "GT; use the json shipped with the dataset (--gt-json).")
            segs.append({'id': int(sid), 'category_id': int(cat),
                         'area': int(area), 'iscrowd': 0})
        annotations.append({'image_id': image_id, 'file_name': Path(p).name,
                            'segments_info': segs})
    return {'categories': list(categories), 'annotations': annotations}


# ── scorer (AUPQ body with the uncertainty branch removed == standard PQ) ─────
class _PQStatCat:
    def __init__(self):
        self.iou, self.tp, self.fp, self.fn = 0.0, 0, 0, 0

    def __iadd__(self, o):
        self.iou += o.iou
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn
        return self


class PQStat:
    def __init__(self):
        self.per_cat = defaultdict(_PQStatCat)

    def __getitem__(self, i):
        return self.per_cat[i]

    def __iadd__(self, o):
        for k, v in o.per_cat.items():
            self.per_cat[k] += v
        return self

    def average(self, categories: Dict[int, Dict], isthing: Optional[bool]):
        pq = sq = rq = 0.0
        n = 0
        per_class = {}
        for label, info in categories.items():
            if isthing is not None and (info['isthing'] == 1) != isthing:
                continue
            s = self.per_cat[label]
            if s.tp + s.fp + s.fn == 0:
                continue
            n += 1
            pq_c = s.iou / (s.tp + 0.5 * s.fp + 0.5 * s.fn)
            sq_c = s.iou / s.tp if s.tp else 0.0
            rq_c = s.tp / (s.tp + 0.5 * s.fp + 0.5 * s.fn)
            per_class[label] = {'pq': pq_c, 'sq': sq_c, 'rq': rq_c,
                                'tp': s.tp, 'fp': s.fp, 'fn': s.fn}
            pq += pq_c
            sq += sq_c
            rq += rq_c
        d = max(n, 1)
        return {'pq': pq / d, 'sq': sq / d, 'rq': rq / d, 'n': n}, per_class


def pq_compute_single(gt_ann: Dict, pred_ann: Dict, pan_gt: np.ndarray,
                      pan_pred: np.ndarray, categories: Dict[int, Dict],
                      stat: PQStat) -> None:
    """One image. Line-for-line the AUPQ inner loop (L279-368) with the
    uncertainty replacement disabled (no VOID substitution, no oracle
    instances) — i.e. panopticapi's standard PQ."""
    gt_segms = {int(el['id']): dict(el) for el in gt_ann['segments_info']}
    pred_segms = {int(el['id']): dict(el) for el in pred_ann['segments_info']}

    pred_labels_set = set(pred_segms)
    for label in np.unique(pan_pred):
        label = int(label)
        if label == VOID:
            continue
        if label not in pred_segms:
            raise KeyError(f"{gt_ann['image_id']}: segment {label} in PNG, "
                           "not in JSON")
        if pred_segms[label]['category_id'] not in categories:
            raise KeyError(f"{gt_ann['image_id']}: segment {label} has unknown "
                           f"category_id {pred_segms[label]['category_id']}")
        pred_labels_set.discard(label)
    if pred_labels_set:
        raise KeyError(f"{gt_ann['image_id']}: segments {sorted(pred_labels_set)} "
                       "in JSON, not in PNG")

    labels, counts = np.unique(pan_pred, return_counts=True)
    new_pred = {}
    for label, cnt in zip(labels.tolist(), counts.tolist()):
        if label == VOID:
            continue
        pred_segms[label]['area'] = cnt
        new_pred[label] = pred_segms[label]
    pred_segms = new_pred
    # GT areas: use the json's when present (panopticapi contract), else count.
    if any('area' not in s for s in gt_segms.values()):
        gl, gc = np.unique(pan_gt, return_counts=True)
        gt_area = dict(zip(gl.tolist(), gc.tolist()))
        for sid, s in gt_segms.items():
            s.setdefault('area', int(gt_area.get(sid, 0)))
    for s in gt_segms.values():
        s.setdefault('iscrowd', 0)

    pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
    gt_pred_map = {}
    for label, inter in zip(*np.unique(pan_gt_pred, return_counts=True)):
        gt_pred_map[(int(label) // OFFSET, int(label) % OFFSET)] = int(inter)

    gt_matched, pred_matched = set(), set()
    for (gt_id, pred_id), inter in gt_pred_map.items():
        if gt_id not in gt_segms or pred_id not in pred_segms:
            continue
        if gt_segms[gt_id]['iscrowd'] == 1:
            continue
        if gt_segms[gt_id]['category_id'] != pred_segms[pred_id]['category_id']:
            continue
        union = (pred_segms[pred_id]['area'] + gt_segms[gt_id]['area'] - inter
                 - gt_pred_map.get((VOID, pred_id), 0))
        iou = inter / union if union > 0 else 0.0
        if iou > 0.5:
            cat = gt_segms[gt_id]['category_id']
            stat[cat].tp += 1
            stat[cat].iou += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    crowd = {}
    for gt_id, info in gt_segms.items():
        if gt_id in gt_matched:
            continue
        if info['iscrowd'] == 1:
            crowd[info['category_id']] = gt_id
            continue
        stat[info['category_id']].fn += 1

    for pred_id, info in pred_segms.items():
        if pred_id in pred_matched:
            continue
        inter = gt_pred_map.get((VOID, pred_id), 0)
        if info['category_id'] in crowd:
            inter += gt_pred_map.get((crowd[info['category_id']], pred_id), 0)
        if info['area'] and inter / info['area'] > 0.5:
            continue
        stat[info['category_id']].fp += 1


def pq_compute(gt_json: Dict, pred_json: Dict, gt_folder, pred_folder,
               progress=None) -> Dict:
    """Standard PQ over a matched (gt_json, pred_json) pair. Returns
    {'All'|'Things'|'Stuff': {pq,sq,rq,n}, 'per_class': {...}}."""
    gt_folder, pred_folder = Path(gt_folder), Path(pred_folder)
    categories = {int(el['id']): el for el in gt_json['categories']}
    pred_by_id = {a['image_id']: a for a in pred_json['annotations']}
    stat = PQStat()
    anns = gt_json['annotations']
    it = progress(anns) if progress is not None else anns
    for gt_ann in it:
        image_id = gt_ann['image_id']
        if image_id not in pred_by_id:
            raise KeyError(f"no prediction for image {image_id}")
        pred_ann = pred_by_id[image_id]
        pan_gt = rgb2id(np.array(Image.open(gt_folder / gt_ann['file_name'])
                                 .convert('RGB'), dtype=np.uint32))
        pan_pred = rgb2id(np.array(Image.open(pred_folder / pred_ann['file_name'])
                                   .convert('RGB'), dtype=np.uint32))
        if pan_gt.shape != pan_pred.shape:
            raise ValueError(
                f"{image_id}: GT {pan_gt.shape} vs pred {pan_pred.shape} — the "
                "prediction geometry does not match the panoptic GT (letterbox "
                "not undone?). Scoring this would be meaningless.")
        pq_compute_single(gt_ann, pred_ann, pan_gt, pan_pred, categories, stat)

    out = {}
    for name, isthing in (('All', None), ('Things', True), ('Stuff', False)):
        res, per_class = stat.average(categories, isthing)
        out[name] = res
        if name == 'All':
            out['per_class'] = {categories[k]['name']: v
                                for k, v in per_class.items()}
    return out


def format_table(results: Dict) -> str:
    lines = ["{:10s}| {:>6s} {:>6s} {:>6s} {:>5s}".format('', 'PQ', 'SQ', 'RQ', 'N'),
             "-" * 38]
    for name in ('All', 'Things', 'Stuff'):
        r = results[name]
        lines.append("{:10s}| {:6.1f} {:6.1f} {:6.1f} {:5d}".format(
            name, 100 * r['pq'], 100 * r['sq'], 100 * r['rq'], r['n']))
    lines += ["", "{:16s} {:>6s} {:>6s} {:>6s} {:>5s} {:>5s} {:>5s}".format(
        'class', 'PQ', 'SQ', 'RQ', 'TP', 'FP', 'FN'), "-" * 54]
    for name, v in results.get('per_class', {}).items():
        lines.append("{:16s} {:6.1f} {:6.1f} {:6.1f} {:5d} {:5d} {:5d}".format(
            name, 100 * v['pq'], 100 * v['sq'], 100 * v['rq'],
            v['tp'], v['fp'], v['fn']))
    return "\n".join(lines)
