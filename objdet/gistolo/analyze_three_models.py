"""3모델 비교 분석 — ViT-L(멀티모달) / GISTOLO-ViT-L(RGB 증류) / YOLO b0(RGB 기본).

세 모델을 같은 test 프레임에 돌려 (a) 예측값을 전부 저장하고, (b) 지표를 비교하고,
(c) **YOLO b0 는 놓쳤는데 GISTOLO 또는 ViT-L 은 잡은 GT** 를 찾아 발표용 시각화를 만든다.
증류가 실제로 무엇을 되찾아 주는지 그림으로 보여주는 것이 목적.

  python analyze_three_models.py --stage infer   # 3모델 추론 → preds/*.json
  python analyze_three_models.py --stage analyze # 차이 분석 + 시각화
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

CLASSES = ['Allies', 'Enemies', 'Casualties', 'Windows', 'Doors', 'Obstacles',
           'Lighting', 'Emergency Exits', 'Fire Extinguishers', 'Landing Markers']


# ─────────────────────────── IoU / 매칭 ───────────────────────────
def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def hit(gt_box, gt_cls, preds, thr=0.5, iou_t=0.5):
    """이 GT 를 잡은 예측이 있으면 (IoU, score) 반환, 없으면 None."""
    best = None
    for p in preds:
        if p['cls'] != gt_cls or p['score'] < thr:
            continue
        i = iou_xywh(gt_box, p['bbox'])
        if i >= iou_t and (best is None or i > best[0]):
            best = (i, p['score'])
    return best


# ─────────────────────────── 추론 단계 ───────────────────────────
def infer_multimodal(cfg_path, ckpt, data_root, out_json, gpu=0):
    """ViT-L (ReliaDINO 3-modal) — cert 추론 스택 재사용."""
    import torch
    from _det_common import (build_detector, build_loader, load_cfg,
                             load_det_checkpoint, run_inference)
    cfg = load_cfg(cfg_path)
    r = data_root.rstrip('/')
    cfg['DATASET']['ROOT'] = r
    cfg['DATASET']['ANNOTATION_VAL'] = f'{r}/_final_ann/instances_test_common.json'
    cfg['DATASET']['REQUIRE_ALL_MODALITIES'] = True
    dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, 'val', workers=4)
    model = build_detector(cfg, dev, cfg['MODEL'].get('N_CLASSES') or ds.n_classes)
    load_det_checkpoint(model, ckpt, dev)
    model.eval()
    preds, id2file = run_inference(model, ds, loader, cfg, dev, score_thresh=0.05)
    cat2idx = {v: k for k, v in {i: c for i, c in enumerate(sorted(
        {a['category_id'] for a in preds}))}.items()} if preds else {}
    out = {}
    ann = json.load(open(cfg['DATASET']['ANNOTATION_VAL']))
    catid2idx = {c['id']: i for i, c in enumerate(sorted(ann['categories'], key=lambda c: c['id']))}
    for p in preds:
        f = id2file[int(p['image_id'])]
        out.setdefault(f, []).append({'bbox': [float(v) for v in p['bbox']],
                                      'score': float(p['score']),
                                      'cls': catid2idx[p['category_id']]})
    json.dump(out, open(out_json, 'w'))
    print(f'[multimodal] {sum(len(v) for v in out.values())} boxes over {len(out)} images -> {out_json}')


def infer_yolo(weights, yolo_dir, data_yaml, img_dir, out_json, gpu=0, conf=0.05):
    """YOLOv5m (RGB) — ultralytics/yolov5 detect API 대신 직접 로드해서 배치 추론."""
    import glob
    import torch
    sys.path.insert(0, yolo_dir)
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    import cv2
    import numpy as np

    dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    m = DetectMultiBackend(weights, device=dev)
    m.eval()
    files = sorted(glob.glob(f'{img_dir}/*.png') + glob.glob(f'{img_dir}/*.jpg'))
    out = {}
    with torch.no_grad():
        for k, f in enumerate(files):
            im0 = cv2.imread(f)
            im = letterbox(im0, 640, stride=32, auto=True)[0]
            im = im.transpose((2, 0, 1))[::-1].copy()
            t = torch.from_numpy(im).to(dev).float() / 255.0
            t = t[None]
            pred = non_max_suppression(m(t), conf, 0.45)[0]
            rows = []
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(t.shape[2:], pred[:, :4], im0.shape).round()
                for *xyxy, sc, cl in pred.tolist():
                    x1, y1, x2, y2 = xyxy
                    rows.append({'bbox': [x1, y1, x2 - x1, y2 - y1],
                                 'score': float(sc), 'cls': int(cl)})
            out[os.path.basename(f)] = rows
            if (k + 1) % 500 == 0:
                print(f'  ..{k+1}/{len(files)}')
    json.dump(out, open(out_json, 'w'))
    print(f'[yolo] {sum(len(v) for v in out.values())} boxes over {len(out)} images -> {out_json}')


# ─────────────────────────── 분석 단계 ───────────────────────────
def flat(fn):
    return fn.replace('/rgb/', '_').replace('/', '_')


def analyze(ann_path, preds_dir, out_dir, score_thr=0.3):
    """YOLO 실패 & (GISTOLO or ViT-L) 성공 GT 를 찾아 정리."""
    ann = json.load(open(ann_path))
    cats = sorted(ann['categories'], key=lambda c: c['id'])
    catid2idx = {c['id']: i for i, c in enumerate(cats)}
    imgs = {im['id']: im for im in ann['images']}
    gt_by_img = {}
    for a in ann['annotations']:
        gt_by_img.setdefault(a['image_id'], []).append(a)

    P = {k: json.load(open(f'{preds_dir}/{k}.json'))
         for k in ('vitl', 'gistolo', 'yolo')}

    def get(model, im):
        """모델별 키 규칙 흡수 — multimodal 은 'clip/rgb/ts.png', yolo 는 flat 파일명."""
        fn = im['file_name']
        d = P[model]
        return d.get(fn) or d.get(flat(fn)) or d.get(os.path.basename(flat(fn))) or []

    recovered = []     # yolo miss, gistolo hit
    mm_only = []       # yolo miss, vitl hit (gistolo 무관)
    both_miss = 0
    stats = {'gt': 0, 'yolo_hit': 0, 'gistolo_hit': 0, 'vitl_hit': 0}
    per_cls = {c['name']: {'gt': 0, 'yolo': 0, 'gistolo': 0, 'vitl': 0} for c in cats}

    for iid, anns in gt_by_img.items():
        im = imgs[iid]
        pv, pg, py = get('vitl', im), get('gistolo', im), get('yolo', im)
        if not (pv or pg or py):
            continue                       # 이 프레임을 아무도 안 본 경우(모달 결손 등)
        for a in anns:
            cls = catid2idx[a['category_id']]
            name = cats[cls]['name']
            stats['gt'] += 1; per_cls[name]['gt'] += 1
            hv = hit(a['bbox'], cls, pv, score_thr)
            hg = hit(a['bbox'], cls, pg, score_thr)
            hy = hit(a['bbox'], cls, py, score_thr)
            for k, h in (('vitl', hv), ('gistolo', hg), ('yolo', hy)):
                if h:
                    stats[f'{k}_hit'] += 1; per_cls[name][k] += 1
            if not hy and (hg or hv):
                # YOLO 가 "정말로 아무것도 못 본" 경우와 "박스만 부정확한" 경우를 구분한다.
                # 발표에서 'YOLO는 놓쳤다'고 말하려면 전자여야 정직하다.
                yolo_near = max((iou_xywh(a['bbox'], p['bbox'])
                                 for p in py if p['cls'] == cls and p['score'] >= score_thr),
                                default=0.0)
                rec = {'image_id': iid, 'file_name': im['file_name'],
                       'night': bool(im.get('low_light')), 'cls': cls, 'cls_name': name,
                       'bbox': a['bbox'],
                       'yolo_best_iou': float(yolo_near),
                       'clean_miss': yolo_near < 0.1,     # 같은 클래스 예측이 사실상 전무
                       'gistolo': {'iou': hg[0], 'score': hg[1]} if hg else None,
                       'vitl': {'iou': hv[0], 'score': hv[1]} if hv else None}
                (recovered if hg else mm_only).append(rec)
            elif not hy and not hg and not hv:
                both_miss += 1

    os.makedirs(out_dir, exist_ok=True)
    json.dump({'stats': stats, 'per_class': per_cls,
               'recovered_by_gistolo': recovered, 'recovered_by_vitl_only': mm_only,
               'all_miss': both_miss, 'score_thr': score_thr},
              open(f'{out_dir}/diff_analysis.json', 'w'), indent=2)

    g = stats['gt']
    print(f"\n{'='*66}\n  GT {g} 개 기준 검출 성공률 (score>{score_thr}, IoU>=0.5)")
    print(f"  YOLO b0        {stats['yolo_hit']:5d}  ({stats['yolo_hit']/g:.3f})")
    print(f"  GISTOLO-ViT-L  {stats['gistolo_hit']:5d}  ({stats['gistolo_hit']/g:.3f})")
    print(f"  ViT-L 멀티모달  {stats['vitl_hit']:5d}  ({stats['vitl_hit']/g:.3f})")
    print(f"\n  YOLO 실패 → GISTOLO 성공 : {len(recovered)} 건  ★ 증류가 되찾은 표적")
    print(f"  YOLO 실패 → ViT-L 만 성공 : {len(mm_only)} 건  (멀티모달만의 이득)")
    print(f"  셋 다 실패                : {both_miss} 건")
    print(f"\n  클래스별 (GT / yolo / gistolo / vitl)")
    for n, d in sorted(per_cls.items(), key=lambda x: -(x[1]['gistolo'] - x[1]['yolo'])):
        if d['gt']:
            print(f"    {n:20s} {d['gt']:5d} / {d['yolo']:5d} / {d['gistolo']:5d} / {d['vitl']:5d}"
                  f"   Δ(gistolo-yolo) {d['gistolo']-d['yolo']:+d}")
    print(f"  -> {out_dir}/diff_analysis.json\n{'='*66}")
    return recovered, mm_only


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['infer-mm', 'infer-yolo', 'analyze'])
    ap.add_argument('--cfg'); ap.add_argument('--ckpt'); ap.add_argument('--data-root')
    ap.add_argument('--weights'); ap.add_argument('--yolo-dir'); ap.add_argument('--img-dir')
    ap.add_argument('--ann'); ap.add_argument('--preds-dir'); ap.add_argument('--out')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    if a.stage == 'infer-mm':
        infer_multimodal(a.cfg, a.ckpt, a.data_root, a.out, a.gpu)
    elif a.stage == 'infer-yolo':
        infer_yolo(a.weights, a.yolo_dir, None, a.img_dir, a.out, a.gpu)
    else:
        analyze(a.ann, a.preds_dir, a.out, a.score_thr)
