"""KD 모델이 baseline 보다 더 잡은 사례를 찾아 발표용 이미지로 만든다.

두 YOLOv5m (구조 동일, 가중치만 다름) 을 같은 test 프레임에 돌려
  - baseline : KD 없이 학습
  - KD       : cross-modal feature distillation (kd_w 0.03, modality-aware)
GT 마다 검출 성공 여부를 IoU 0.5 / conf 0.3 로 판정하고,
'baseline 실패 & KD 성공' 인 표적을 골라 [baseline | KD] 2-패널로 그린다.

  python viz_kd_vs_base.py --base <base.pt> --kd <kd.pt> --yolo-dir ~/yolov5 \
      --img-dir ~/dset/poongsan_v2_yolo_rgb_modal/images/test \
      --label-dir ~/dset/poongsan_v2_yolo_rgb_modal/labels/test \
      --out runs/kd_vs_base --n 24
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

CLASSES = ['Allies', 'Enemies', 'Casualties', 'Windows', 'Doors', 'Obstacles',
           'Lighting', 'Emergency Exits', 'Fire Extinguishers', 'Landing Markers']
BASE_C = (150, 160, 175)      # 회색 — baseline
KD_C = (245, 170, 66)         # 앰버 — KD (주인공)
GT_C = (64, 220, 110)         # 초록 — GT
MISS_C = (240, 74, 60)        # 빨강 점선 — 놓친 GT
BG = (16, 18, 22)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def infer(weights, yolo_dir, files, dev_str, conf=0.25):
    sys.path.insert(0, yolo_dir)
    import torch
    import cv2
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    dev = torch.device(dev_str)
    m = DetectMultiBackend(weights, device=dev)
    m.eval()
    out = {}
    with torch.no_grad():
        for k, f in enumerate(files):
            im0 = cv2.imread(f)
            im = letterbox(im0, 640, stride=32, auto=True)[0]
            im = im.transpose((2, 0, 1))[::-1].copy()
            t = torch.from_numpy(im).to(dev).float() / 255.0
            pred = non_max_suppression(m(t[None]), conf, 0.45)[0]
            rows = []
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(t.shape[1:], pred[:, :4], im0.shape).round()
                for *xyxy, sc, cl in pred.tolist():
                    rows.append({'box': [float(v) for v in xyxy],
                                 'score': float(sc), 'cls': int(cl)})
            out[os.path.basename(f)] = rows
            if (k + 1) % 400 == 0:
                print(f'    ..{k+1}/{len(files)}')
    del m
    torch.cuda.empty_cache()
    return out


def load_gt(label_path, W, H):
    gts = []
    if not os.path.exists(label_path):
        return gts
    for l in open(label_path).read().split('\n'):
        if not l.strip():
            continue
        c, cx, cy, w, h = l.split()
        c = int(c); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        gts.append({'cls': c, 'box': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return gts


def hit(gt, preds, thr=0.3, iou_t=0.5):
    best = None
    for p in preds:
        if p['cls'] != gt['cls'] or p['score'] < thr:
            continue
        i = iou(gt['box'], p['box'])
        if i >= iou_t and (best is None or i > best[0]):
            best = (i, p['score'])
    return best


def _font(sz):
    from PIL import ImageFont
    for pat in ('/usr/share/fonts/**/NanumGothic*.ttf',
                '/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        hits = glob.glob(pat, recursive=True)
        if hits:
            try:
                return ImageFont.truetype(hits[0], sz)
            except Exception:
                continue
    return ImageFont.load_default()


def dashed(d, box, color, width=4, dash=12):
    x1, y1, x2, y2 = box
    for (sx, sy, ex, ey) in ((x1, y1, x2, y1), (x2, y1, x2, y2),
                             (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(max(abs(ex - sx), abs(ey - sy)) / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            d.line([sx + (ex - sx) * t0, sy + (ey - sy) * t0,
                    sx + (ex - sx) * t1, sy + (ey - sy) * t1], fill=color, width=width)


def panel(img, preds, focus, color, title, sub, ok, crop, font, fsm, W=470):
    from PIL import Image, ImageDraw
    im = img.copy()
    d = ImageDraw.Draw(im)
    for p in preds:
        if p['score'] < 0.3:
            continue
        x1, y1, x2, y2 = p['box']
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        d.text((x1 + 3, max(0, y1 - 16)), f"{CLASSES[p['cls']][:12]} {p['score']:.2f}",
               fill=color, font=fsm)
    fx1, fy1, fx2, fy2 = focus
    if ok:
        d.rectangle([fx1 - 3, fy1 - 3, fx2 + 3, fy2 + 3], outline=color, width=5)
    else:
        dashed(d, [fx1 - 3, fy1 - 3, fx2 + 3, fy2 + 3], MISS_C)
    im = im.crop(crop)
    im = im.resize((W, int(im.height * W / im.width)))
    out = Image.new('RGB', (W, im.height + 46), BG)
    out.paste(im, (0, 46))
    d2 = ImageDraw.Draw(out)
    d2.text((8, 5), title, fill=color, font=font)
    d2.text((8, 25), sub, fill=(205, 212, 222) if ok else MISS_C, font=fsm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--kd', required=True)
    ap.add_argument('--yolo-dir', default=os.path.expanduser('~/yolov5'))
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--label-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    from PIL import Image, ImageDraw

    files = sorted(glob.glob(os.path.join(args.img_dir, '*.png')) +
                   glob.glob(os.path.join(args.img_dir, '*.jpg')))
    print(f'[kd-vs-base] test {len(files)} 장')
    print('  baseline 추론...')
    pb = infer(args.base, args.yolo_dir, files, f'cuda:{args.gpu}')
    print('  KD 추론...')
    pk = infer(args.kd, args.yolo_dir, files, f'cuda:{args.gpu}')

    # baseline 실패 & KD 성공 찾기
    cases = []
    n_gt = n_base = n_kd = 0
    for f in files:
        stem = os.path.basename(f)
        im = Image.open(f)
        W, H = im.size
        gts = load_gt(os.path.join(args.label_dir, os.path.splitext(stem)[0] + '.txt'), W, H)
        for g in gts:
            n_gt += 1
            hb, hk = hit(g, pb.get(stem, [])), hit(g, pk.get(stem, []))
            n_base += bool(hb); n_kd += bool(hk)
            if hk and not hb:
                near = max((iou(g['box'], p['box']) for p in pb.get(stem, [])
                            if p['cls'] == g['cls'] and p['score'] >= 0.3), default=0.0)
                cases.append({'file': f, 'stem': stem, 'cls': g['cls'], 'box': g['box'],
                              'kd_iou': hk[0], 'kd_score': hk[1], 'base_near_iou': near,
                              'clean': near < 0.1, 'night': 'capture_20260618_114021' in stem
                              or 'capture_20260618_115624' in stem})
    print(f'  GT {n_gt} — baseline 성공 {n_base} / KD 성공 {n_kd}')
    print(f'  ★ baseline 실패 & KD 성공: {len(cases)} 건 (그중 baseline 완전 미검출 '
          f'{sum(c["clean"] for c in cases)} 건)')

    os.makedirs(args.out, exist_ok=True)
    json.dump({'n_gt': n_gt, 'base_hit': n_base, 'kd_hit': n_kd, 'cases': cases},
              open(f'{args.out}/kd_vs_base.json', 'w'), indent=2)

    # 발표용: 깨끗한 미검출 우선, 야간/클래스 다양하게
    clean = [c for c in cases if c['clean']] or cases
    by_cls = {}
    for c in clean:
        by_cls.setdefault(c['cls'], []).append(c)
    for v in by_cls.values():
        v.sort(key=lambda c: (not c['night'], -c['kd_score']))
    picks, i = [], 0
    while len(picks) < min(args.n, len(clean)):
        added = False
        for k in sorted(by_cls):
            if i < len(by_cls[k]):
                picks.append(by_cls[k][i]); added = True
                if len(picks) >= args.n:
                    break
        if not added:
            break
        i += 1

    font, fsm = _font(15), _font(12)
    made = 0
    for c in picks:
        img = Image.open(c['file']).convert('RGB')
        x1, y1, x2, y2 = c['box']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.7
        crop = (max(0, int(cx - half)), max(0, int(cy - half)),
                min(img.width, int(cx + half)), min(img.height, int(cy + half)))
        p1 = panel(img, pb.get(c['stem'], []), c['box'], BASE_C,
                   'Baseline (KD 없음)', '✗ 미검출', False, crop, font, fsm)
        p2 = panel(img, pk.get(c['stem'], []), c['box'], KD_C,
                   'Cross-modal KD', f"✓ 검출  IoU {c['kd_iou']:.2f} · conf {c['kd_score']:.2f}",
                   True, crop, font, fsm)
        gap = 10
        sheet = Image.new('RGB', (p1.width * 2 + gap, max(p1.height, p2.height) + 34), BG)
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{CLASSES[c['cls']]}   {'야간(저조도)' if c['night'] else '주간'}   {c['stem']}",
               fill=(235, 240, 247), font=font)
        sheet.paste(p1, (0, 30)); sheet.paste(p2, (p1.width + gap, 30))
        sheet.save(f"{args.out}/{CLASSES[c['cls']].replace(' ', '')}_"
                   f"{'night' if c['night'] else 'day'}_{made:02d}.png")
        made += 1
    print(f'  시각화 {made} 장 -> {args.out}')


if __name__ == '__main__':
    main()
