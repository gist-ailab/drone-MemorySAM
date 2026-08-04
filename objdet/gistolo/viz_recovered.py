"""발표용 시각화 — "YOLO 는 놓쳤는데 증류/멀티모달은 잡았다" 를 한 장에.

analyze_three_models.py 가 뽑은 recovered 케이스마다 3-패널을 그린다:
   [ YOLO b0 (실패) | GISTOLO-ViT-L (성공) | ViT-L 멀티모달 ]
GT = 초록 실선, 각 모델 예측 = 해당 패널 색, 놓친 GT 는 빨간 점선으로 강조.
아래에는 그 GT 를 잡았는지 여부와 IoU/score 를 적는다.

  python viz_recovered.py --analysis runs/three_model_analysis \
      --img-root ~/poongsan_v2 --out runs/three_model_analysis/viz --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import random

CLASSES = ['Allies', 'Enemies', 'Casualties', 'Windows', 'Doors', 'Obstacles',
           'Lighting', 'Emergency Exits', 'Fire Extinguishers', 'Landing Markers']
GT_GREEN = (64, 220, 110)
MISS_RED = (240, 74, 60)
YOLO_C = (150, 160, 175)      # 회색 — 못 찾은 쪽
GIST_C = (245, 170, 66)       # 앰버 — 증류 (주인공)
VITL_C = (96, 165, 250)       # 블루 — 멀티모달
BG = (16, 18, 22)


def _font(sz):
    import glob
    from PIL import ImageFont
    for pat in ('/usr/share/fonts/**/NanumGothic*.ttf', '/usr/share/fonts/**/NotoSansCJK*.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        hit = glob.glob(pat, recursive=True)
        if hit:
            try:
                return ImageFont.truetype(hit[0], sz)
            except Exception:
                continue
    return ImageFont.load_default()


def _dashed(d, box, color, width=3, dash=12):
    """놓친 GT 를 점선으로 — 실선(정상 GT)과 구분."""
    x1, y1, x2, y2 = box
    for (sx, sy, ex, ey) in ((x1, y1, x2, y1), (x2, y1, x2, y2),
                             (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(max(abs(ex - sx), abs(ey - sy)) / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            d.line([sx + (ex - sx) * t0, sy + (ey - sy) * t0,
                    sx + (ex - sx) * t1, sy + (ey - sy) * t1], fill=color, width=width)


def panel(img, preds, gt_all, focus, color, title, sub, ok, crop_box, font, fsm):
    """한 모델의 패널 하나 — 예측 박스 + GT + 초점 GT 강조."""
    from PIL import Image, ImageDraw
    im = img.copy()
    d = ImageDraw.Draw(im)
    for g in gt_all:                                   # 전체 GT (옅은 초록 실선)
        x, y, w, h = g['bbox']
        d.rectangle([x, y, x + w, y + h], outline=GT_GREEN, width=2)
    for p in preds:                                    # 모델 예측
        if p['score'] < 0.3:
            continue
        x, y, w, h = p['bbox']
        d.rectangle([x, y, x + w, y + h], outline=color, width=3)
        d.text((x + 3, max(0, y - 16)), f"{CLASSES[p['cls']][:12]} {p['score']:.2f}",
               fill=color, font=fsm)
    fx, fy, fw, fh = focus                             # 초점 GT: 성공=색테두리 / 실패=빨간 점선
    if ok:
        d.rectangle([fx - 3, fy - 3, fx + fw + 3, fy + fh + 3], outline=color, width=5)
    else:
        _dashed(d, [fx - 3, fy - 3, fx + fw + 3, fy + fh + 3], MISS_RED, width=4)

    im = im.crop(crop_box)
    W = 460
    im = im.resize((W, int(im.height * W / im.width)))
    # 헤더 붙이기
    out = Image.new('RGB', (W, im.height + 46), BG)
    out.paste(im, (0, 46))
    d2 = ImageDraw.Draw(out)
    d2.rectangle([0, 0, W, 45], fill=BG)
    d2.text((8, 5), title, fill=color, font=font)
    d2.text((8, 25), sub, fill=(200, 208, 218) if ok else MISS_RED, font=fsm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True, help='diff_analysis.json 이 있는 디렉터리')
    ap.add_argument('--preds-dir', default=None)
    ap.add_argument('--img-root', required=True, help='poongsan_v2 (clip/rgb/*.png)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--pad', type=float, default=1.6, help='초점 박스 주변 확대 배율')
    ap.add_argument('--clean-only', action='store_true', default=True,
                    help='YOLO가 같은 클래스 예측을 전혀 못 낸 케이스만 (발표용 정직성)')
    ap.add_argument('--all-cases', dest='clean_only', action='store_false')
    args = ap.parse_args()
    from PIL import Image

    A = json.load(open(f'{args.analysis}/diff_analysis.json'))
    pdir = args.preds_dir or f'{args.analysis}/preds'
    P = {k: json.load(open(f'{pdir}/{k}.json')) for k in ('vitl', 'gistolo', 'yolo')}
    ann = json.load(open(A.get('ann_path', '')) ) if A.get('ann_path') else None

    cases = A['recovered_by_gistolo']
    # 발표에는 "YOLO 가 그 자리에 아무 예측도 못 낸" 깨끗한 미검출만 쓴다.
    # (박스만 부정확한 케이스를 '못 봤다'고 보여주면 과장이 된다)
    clean = [c for c in cases if c.get('clean_miss', True)]
    if args.clean_only and clean:
        print(f'[viz] 깨끗한 미검출 {len(clean)} / 전체 recovered {len(cases)} 건 중에서 선별')
        cases = clean
    # 야간 우선 + 클래스 다양하게
    random.seed(0)
    by_cls = {}
    for c in cases:
        by_cls.setdefault(c['cls_name'], []).append(c)
    for v in by_cls.values():
        v.sort(key=lambda c: (not c['night'], -(c['gistolo']['score'] if c['gistolo'] else 0)))
    picks, i = [], 0
    while len(picks) < min(args.n, len(cases)):
        added = False
        for k in sorted(by_cls):
            if i < len(by_cls[k]):
                picks.append(by_cls[k][i]); added = True
                if len(picks) >= args.n:
                    break
        if not added:
            break
        i += 1

    os.makedirs(args.out, exist_ok=True)
    font, fsm = _font(15), _font(12)

    def gets(model, fn):
        d = P[model]
        return d.get(fn) or d.get(fn.replace('/rgb/', '_').replace('/', '_')) or []

    made = 0
    for c in picks:
        fn = c['file_name']
        src = os.path.join(args.img_root, fn)
        if not os.path.exists(src):
            continue
        img = Image.open(src).convert('RGB')
        x, y, w, h = c['bbox']
        cx, cy = x + w / 2, y + h / 2
        half = max(w, h) * args.pad
        crop = (max(0, int(cx - half)), max(0, int(cy - half)),
                min(img.width, int(cx + half)), min(img.height, int(cy + half)))
        gt_all = []                                   # 이 프레임의 GT 는 focus 만 표시(간결)
        gi, vi = c.get('gistolo'), c.get('vitl')
        ybi = c.get('yolo_best_iou', 0.0)
        ysub = ('✗ 미검출 (해당 표적 예측 없음)' if ybi < 0.1
                else f'✗ 미검출 (박스 부정확, best IoU {ybi:.2f})')
        p1 = panel(img, gets('yolo', fn), gt_all, c['bbox'], YOLO_C,
                   'YOLO b0 (RGB, 증류 없음)', ysub, False, crop, font, fsm)
        p2 = panel(img, gets('gistolo', fn), gt_all, c['bbox'], GIST_C,
                   'GISTOLO-ViT-L (RGB 증류)',
                   f"✓ 검출  IoU {gi['iou']:.2f} · conf {gi['score']:.2f}" if gi else '✗',
                   bool(gi), crop, font, fsm)
        p3 = panel(img, gets('vitl', fn), gt_all, c['bbox'], VITL_C,
                   'ViT-L 멀티모달 (teacher)',
                   (f"✓ 검출  IoU {vi['iou']:.2f} · conf {vi['score']:.2f}" if vi else '✗ 미검출'),
                   bool(vi), crop, font, fsm)

        gap = 10
        W = p1.width * 3 + gap * 2
        H = max(p1.height, p2.height, p3.height) + 34
        sheet = Image.new('RGB', (W, H), BG)
        from PIL import ImageDraw
        d = ImageDraw.Draw(sheet)
        tag = f"{c['cls_name']}   {'야간(저조도)' if c['night'] else '주간'}   {os.path.basename(fn)}"
        d.text((8, 8), tag, fill=(235, 240, 247), font=font)
        for k, p in enumerate((p1, p2, p3)):
            sheet.paste(p, (k * (p1.width + gap), 30))
        name = f"{c['cls_name'].replace(' ','')}_{'night' if c['night'] else 'day'}_{made:02d}.png"
        sheet.save(f'{args.out}/{name}')
        made += 1

    print(f'[viz] {made} 장 저장 -> {args.out}')
    print(f'      (GISTOLO 가 되찾은 표적 {len(cases)} 건 중 대표 샘플)')


if __name__ == '__main__':
    main()
